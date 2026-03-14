"""
LLM-guided evolution of CCAs evaluated in Remy's framework.

This is the core experiment: evolve Python CCA code (Remy-compatible policies)
using LLM mutations, evaluated on Remy's utility function across ConfigRanges.

The key comparison: Remy trees vs PPO vs LLM-evolved code.
"""

import json
import math
import os
import random
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .remy_eval import (
    RemyMemory, RemyAction, ConfigRange, make_training_range,
    evaluate_policy, run_remy_sim,
    aimd_policy, copa_policy, constant_policy,
)
from dataclasses import dataclass
from .llm_client import create_client, get_response_from_llm


# ── Seed policies (as source code strings) ──────────────────────────

SEED_AIMD = '''
def evolved_policy(memory):
    """AIMD: additive increase per ACK, halve on congestion (high RTT ratio)."""
    from alphacc.remy_eval import RemyAction
    if memory.rtt_ratio > 2.0:
        return RemyAction(window_increment=0, window_multiple=0.5, intersend=0.0)
    return RemyAction(window_increment=1, window_multiple=1.0, intersend=0.0)
'''

SEED_COPA = '''
def evolved_policy(memory):
    """Copa-style: target rate = 1/(delta * queuing_delay).
    Increase cwnd when rtt_ratio < 1+delta, decrease when above."""
    from alphacc.remy_eval import RemyAction
    delta = 0.5
    if memory.rtt_ratio > 1.0 + delta:
        return RemyAction(window_increment=-1, window_multiple=1.0, intersend=0.0)
    return RemyAction(window_increment=1, window_multiple=1.0, intersend=0.0)
'''

SEED_ADAPTIVE = '''
def evolved_policy(memory):
    """Adaptive CCA using send/receive rate signals for bandwidth estimation.
    Tracks state via function attributes for rate-based pacing."""
    from alphacc.remy_eval import RemyAction
    import math

    # Per-sender state (keyed by id(memory) to avoid shared state between senders)
    if not hasattr(evolved_policy, '_states'):
        evolved_policy._states = {}
    mid = id(memory)
    if mid not in evolved_policy._states:
        evolved_policy._states[mid] = {'phase': 'startup', 'last_ratio': 1.0, 'stable_count': 0}
    st = evolved_policy._states[mid]

    # Bandwidth signal: if rec_ewma < send_ewma, we're sending faster than receiving
    if memory.send_ewma > 0 and memory.rec_ewma > 0:
        rate_ratio = memory.rec_ewma / memory.send_ewma  # >1 = bottlenecked
    else:
        rate_ratio = 1.0

    # Phase-based control
    if st['phase'] == 'startup':
        # Exponential growth until RTT inflates or rate saturates
        if memory.rtt_ratio > 1.3 or rate_ratio > 1.5:
            st['phase'] = 'steady'
            return RemyAction(window_increment=0, window_multiple=0.9, intersend=0.0)
        return RemyAction(window_increment=2, window_multiple=1.0, intersend=0.0)

    # Steady state: delay-based with rate awareness
    if memory.rtt_ratio > 2.0:
        # Heavy congestion: halve
        st['stable_count'] = 0
        return RemyAction(window_increment=0, window_multiple=0.5, intersend=0.0)
    elif memory.rtt_ratio > 1.3:
        # Moderate: back off gently
        return RemyAction(window_increment=-1, window_multiple=1.0, intersend=0.0)
    elif rate_ratio < 1.1:
        # Receiving as fast as sending, room to grow
        return RemyAction(window_increment=2, window_multiple=1.0, intersend=0.0)
    else:
        return RemyAction(window_increment=1, window_multiple=1.0, intersend=0.0)
'''

SEED_BBR_LIKE = '''
def evolved_policy(memory):
    """BBR-inspired: estimate BW from receive rate, pace accordingly.
    Uses slow_rec_ewma for stable BW estimate and rtt_ratio for queuing."""
    from alphacc.remy_eval import RemyAction
    import math

    if not hasattr(evolved_policy, '_states'):
        evolved_policy._states = {}
    mid = id(memory)
    if mid not in evolved_policy._states:
        evolved_policy._states[mid] = {'cycle': 0, 'max_bw_est': 0.0, 'probe_rtt_rounds': 0}
    st = evolved_policy._states[mid]

    # BW estimation: inverse of rec_ewma = packets per ms arrival rate
    bw_est = 1.0 / max(0.01, memory.rec_ewma)
    st['max_bw_est'] = max(st['max_bw_est'] * 0.99, bw_est)  # decaying max

    # 8-phase cycle: 1.25x probe, 0.75x drain, 6x cruise
    gains = [1.25, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    gain = gains[st['cycle'] % 8]

    # Advance cycle based on receiving pattern change
    if memory.rtt_ratio < 1.05:
        st['cycle'] = (st['cycle'] + 1) % 8

    # Target cwnd = gain * BDP estimate
    # BDP = bw * min_rtt; target_cwnd = gain * bw * min_rtt
    # intersend = 1 / (gain * bw_est) to pace at desired rate
    target_intersend = 1.0 / (gain * max(0.001, st['max_bw_est']))

    if memory.rtt_ratio > 2.5:
        # Emergency: back off hard
        return RemyAction(window_increment=0, window_multiple=0.5, intersend=target_intersend)
    elif memory.rtt_ratio > 1.5:
        return RemyAction(window_increment=0, window_multiple=0.95, intersend=target_intersend)
    else:
        return RemyAction(window_increment=1, window_multiple=1.0, intersend=target_intersend)
'''

SEED_PACING = '''
def evolved_policy(memory):
    """Rate-based pacing CCA: intersend = 1/(rec_rate * gain).
    Ablation-proven: pacing accounts for 100% of advantage over window-only.
    Phase machine: startup -> drain -> steady with congestion-proportional response."""
    from alphacc.remy_eval import RemyAction
    import math

    if not hasattr(evolved_policy, "_states"):
        evolved_policy._states = {}
    mid = id(memory)
    if mid not in evolved_policy._states:
        evolved_policy._states[mid] = {
            "phase": "startup",
            "rtt_base": max(1.0, float(memory.rtt_ratio) if memory.rtt_ratio > 0 else 1.0),
            "trend": 0.0,
            "util_ema": 1.0,
            "last_intersend": 0.0,
            "cooldown": 0,
        }
    st = evolved_policy._states[mid]

    send = max(1e-6, float(memory.send_ewma) if memory.send_ewma > 0 else 1e-6)
    rec = max(1e-6, float(memory.rec_ewma) if memory.rec_ewma > 0 else 1e-6)
    slow_rec = max(rec, float(memory.slow_rec_ewma) if memory.slow_rec_ewma > 0 else rec)
    rtt = max(1.0, float(memory.rtt_ratio) if memory.rtt_ratio > 0 else 1.0)

    send_rate = 1.0 / send
    rec_rate = 1.0 / rec
    slow_rec_rate = 1.0 / slow_rec

    util = send / rec
    st["util_ema"] = 0.85 * st["util_ema"] + 0.15 * util
    rtt_grad = rtt - st["rtt_base"]
    st["trend"] = 0.8 * st["trend"] + 0.2 * rtt_grad
    st["rtt_base"] = 0.995 * st["rtt_base"] + 0.005 * min(rtt, st["rtt_base"])

    if st["cooldown"] > 0:
        st["cooldown"] -= 1

    if st["phase"] == "startup":
        if rtt > 1.25 or util > 1.20:
            st["phase"] = "drain"
            st["cooldown"] = 2
            target_rate = max(1e-6, rec_rate * 0.95)
            intersend = 1.0 / target_rate
            st["last_intersend"] = intersend
            return RemyAction(window_increment=1, window_multiple=0.92, intersend=intersend)
        target_rate = max(send_rate, rec_rate * 1.30)
        intersend = 1.0 / max(1e-6, target_rate)
        st["last_intersend"] = intersend
        return RemyAction(window_increment=3, window_multiple=1.0, intersend=intersend)

    if st["phase"] == "drain":
        if rtt < 1.12 and util < 1.08:
            st["phase"] = "steady"
        target_rate = max(1e-6, slow_rec_rate * 0.92)
        intersend = 1.0 / target_rate
        intersend = 0.7 * st["last_intersend"] + 0.3 * intersend
        st["last_intersend"] = intersend
        if rtt > 2.0:
            return RemyAction(window_increment=0, window_multiple=0.60, intersend=intersend)
        return RemyAction(window_increment=1, window_multiple=0.96, intersend=intersend)

    congestion = (rtt - 1.0) + 0.5 * max(0.0, st["trend"]) + 0.6 * max(0.0, st["util_ema"] - 1.0)

    if rtt > 2.0:
        target_rate = max(1e-6, slow_rec_rate * 0.80)
        intersend = 1.0 / target_rate
        intersend = 0.6 * st["last_intersend"] + 0.4 * intersend
        st["last_intersend"] = intersend
        st["cooldown"] = 2
        return RemyAction(window_increment=0, window_multiple=0.55, intersend=intersend)

    if rtt < 1.3:
        if congestion < 0.10:
            inc = 3
            gain = 1.18
        elif congestion < 0.22:
            inc = 2
            gain = 1.08
        else:
            inc = 1
            gain = 1.02
        target_rate = max(1e-6, rec_rate * gain)
        intersend = 1.0 / target_rate
        intersend = 0.75 * st["last_intersend"] + 0.25 * intersend
        st["last_intersend"] = intersend
        return RemyAction(window_increment=inc, window_multiple=1.0, intersend=intersend)

    sev = min(1.0, max(0.0, (rtt - 1.3) / 0.7))
    mult = 1.0 - 0.22 * sev
    if st["cooldown"] > 0:
        mult = min(mult, 0.95)
    if sev < 0.5:
        inc = 1
    elif sev < 0.8:
        inc = 0
    else:
        inc = -1

    gain = 1.0 - 0.18 * sev
    target_rate = max(1e-6, slow_rec_rate * gain)
    intersend = 1.0 / target_rate
    intersend = 0.8 * st["last_intersend"] + 0.2 * intersend
    st["last_intersend"] = intersend

    return RemyAction(window_increment=inc, window_multiple=mult, intersend=intersend)
'''

SEED_CONSTANT = '''
def evolved_policy(memory):
    """Minimal seed — barely grows, no congestion response. AlphaZero-style: start from scratch."""
    from alphacc.remy_eval import RemyAction
    # Minimal: grow by 1, never decrease. The LLM must discover everything else.
    return RemyAction(window_increment=1, window_multiple=1.0, intersend=0.0)
'''

SEEDS = {
    'aimd': SEED_AIMD,
    'copa': SEED_COPA,
    'adaptive': SEED_ADAPTIVE,
    'bbr_like': SEED_BBR_LIKE,
    'pacing': SEED_PACING,
    'constant': SEED_CONSTANT,
}


# ── LLM mutation ────────────────────────────────────────────────────

MUTATION_PROMPT = '''You are evolving a congestion control algorithm (CCA) policy function.

## Observation Space (RemyMemory fields)
- `memory.send_ewma`: EWMA of inter-send times (ms). Higher = sending slower.
- `memory.rec_ewma`: EWMA of inter-receive times (ms, fast alpha=1/8). Higher = receiving slower.
- `memory.rtt_ratio`: smoothed RTT / min_RTT. 1.0 = no queuing, >1.5 = congestion.
- `memory.slow_rec_ewma`: EWMA of inter-receive times (ms, slow alpha=1/256).
- `memory.min_rtt`: minimum observed RTT (ms).

## Action Space (RemyAction)
- `window_increment`: integer added to cwnd (positive = grow, negative = shrink)
- `window_multiple`: float multiplied to cwnd (1.0 = no change, 0.5 = halve)
- `intersend`: inter-send time in ms (0 = cwnd-limited only)

New cwnd = old_cwnd * window_multiple + window_increment (clamped to ≥ 0.5)

## CRITICAL RULES
1. When rtt_ratio < 1.3, the policy MUST increase cwnd (window_increment > 0)
2. When rtt_ratio > 2.0, the policy SHOULD decrease cwnd (multiplicative decrease)
3. You can use function attributes for state: `evolved_policy._state = {{}}`
4. You can use `import math` inside the function
5. The import line MUST be: `from alphacc.remy_eval import RemyAction`
6. The function MUST be named `evolved_policy`
7. BW estimation: 1/rec_ewma ≈ receiving rate. Compare to 1/send_ewma for utilization.
8. Useful patterns: slow start (large inc until rtt rises), pacing (intersend based on BW estimate), rate-based control (target_rate = bw_est * gain)

## Current Best Policy
```python
{parent_code}
```

## Performance
{performance_summary}

## Top Performers
{archive_summary}

## Task
Produce an IMPROVED version. Focus on:
- THROUGHPUT: use send/receive rate signals to fill the pipe faster
- DELAY: respond to rtt_ratio changes proportionally, not just threshold-based
- STABILITY: avoid oscillations by using gradual adjustments
- PACING: use intersend for smooth sending (intersend = 1/target_rate_pps where rate is in pkts/ms)

Output ONLY the Python function code, no explanation, no markdown fences.
'''

MUTATION_PROMPT_MULTIPOINT = '''You are evolving a congestion control algorithm (CCA) that must GENERALIZE across a 40x range of link rates.

## Observation Space (RemyMemory fields)
- `memory.send_ewma`: EWMA of inter-send times (ms). Higher = sending slower.
- `memory.rec_ewma`: EWMA of inter-receive times (ms, fast alpha=1/8). Higher = receiving slower.
- `memory.rtt_ratio`: smoothed RTT / min_RTT. 1.0 = no queuing, >1.5 = congestion.
- `memory.slow_rec_ewma`: EWMA of inter-receive times (ms, slow alpha=1/256).
- `memory.min_rtt`: minimum observed RTT (ms).

## Action Space (RemyAction)
- `window_increment`: integer added to cwnd (positive = grow, negative = shrink)
- `window_multiple`: float multiplied to cwnd (1.0 = no change, 0.5 = halve)
- `intersend`: inter-send time in ms (0 = cwnd-limited only)

New cwnd = old_cwnd * window_multiple + window_increment (clamped to ≥ 0.5)

## CRITICAL RULES
1. When rtt_ratio < 1.3, the policy MUST increase cwnd (window_increment > 0)
2. When rtt_ratio > 2.0, the policy SHOULD decrease cwnd (multiplicative decrease)
3. You can use function attributes for state: `evolved_policy._state = {{}}`
4. You can use `import math` inside the function
5. The import line MUST be: `from alphacc.remy_eval import RemyAction`
6. The function MUST be named `evolved_policy`

## GENERALIZATION CHALLENGE
The policy is evaluated at link rates from 2.4 to 95 Mbps (0.237 to 9.49 pkt/ms).
Fixed thresholds like "rtt_ratio > 2.0" are HARMFUL because the BDP-to-RTT relationship
changes with link rate. At 2 Mbps, cwnd=2 fills the pipe. At 95 Mbps, cwnd=95 fills the pipe.

To generalize, the policy MUST:
- Use RELATIVE signals (rtt_ratio, rec_ewma/send_ewma ratio) not absolute ones
- Adapt its aggressiveness based on estimated BDP (via min_rtt and rec_ewma)
- Scale cwnd changes proportionally: at high BDP, increment by more than 1

## KEY INSIGHT FROM ABLATION
Continuous rate-based PACING (intersend = 1/(target_rate)) accounts for 100% of advantage
over window-only policies. Window-only multipoint policies score -1.80 mean; pacing
policies score -0.87. The pacing rate naturally adapts to any link rate because
it's derived from the receiving rate (1/rec_ewma), which reflects the actual bottleneck.
The best target to beat is Remy 10x tree at -0.71 mean across 9 link rates.

## Current Best Policy
```python
{parent_code}
```

## Per-Link-Rate Performance (lower = worse)
{performance_summary}

## Top Performers
{archive_summary}

## Task
Produce an IMPROVED version that performs well across ALL link rates.
The fitness is the MEAN normalized score across 6 rates. You cannot specialize.

Output ONLY the Python function code, no explanation, no markdown fences.
'''


def mutate_policy(
    parent_code: str,
    parent_fitness: float,
    archive_summary: str = "",
    model: str = "gpt-5.3-codex",
    multipoint: bool = False,
    per_rate_scores: Dict = None,
) -> Optional[str]:
    """Use LLM to generate a mutated policy."""
    if multipoint and per_rate_scores:
        perf_lines = [f"Mean normalized score: {parent_fitness:.3f}"]
        for mbps, score in sorted(per_rate_scores.items()):
            perf_lines.append(f"  {mbps:6.1f} Mbps: {score:.3f}")
        perf = '\n'.join(perf_lines)
        template = MUTATION_PROMPT_MULTIPOINT
    else:
        perf = f"Current utility: {parent_fitness:.2f}"
        template = MUTATION_PROMPT

    prompt = template.format(
        parent_code=parent_code.strip(),
        performance_summary=perf,
        archive_summary=archive_summary,
    )

    try:
        client, model_name = create_client(model)
        system_msg = "You are an expert in congestion control algorithm design. Output ONLY Python code, no markdown fences."
        content, _ = get_response_from_llm(
            msg=prompt,
            client=client,
            model=model_name,
            system_message=system_msg,
            temperature=0.8,
        )
        if content:
            code = content.strip()
            # Strip markdown fences if present
            if code.startswith('```'):
                lines = code.split('\n')
                code = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])
            return code
    except Exception as e:
        print(f"  LLM mutation failed: {e}")
    return None


def compile_policy(code: str):
    """Safely compile a policy from source code. Returns callable or None."""
    try:
        namespace = {}
        exec(code, namespace)
        if 'evolved_policy' not in namespace:
            return None
        fn = namespace['evolved_policy']

        # Sanity test 1: basic call works
        mem = RemyMemory()
        mem.rtt_ratio = 1.5
        mem.send_ewma = 10.0
        mem.rec_ewma = 10.0
        mem.slow_rec_ewma = 10.0
        mem.min_rtt = 150.0
        result = fn(mem)
        if not isinstance(result, RemyAction):
            return None

        # Sanity test 2: policy should increase cwnd when no congestion
        mem2 = RemyMemory()
        mem2.rtt_ratio = 1.05  # barely any queuing
        mem2.send_ewma = 5.0
        mem2.rec_ewma = 5.0
        mem2.slow_rec_ewma = 5.0
        mem2.min_rtt = 150.0
        r2 = fn(mem2)
        if not isinstance(r2, RemyAction):
            return None
        # cwnd should grow: either positive increment or multiple > 1
        grows = (r2.window_increment > 0) or (r2.window_multiple > 1.001)
        if not grows:
            print(f"  Compile rejected: policy doesn't grow cwnd (inc={r2.window_increment}, mult={r2.window_multiple})")
            return None

        return fn
    except Exception as e:
        print(f"  Compile failed: {e}")
        return None


# ── Multi-point evaluation ─────────────────────────────────────────

# Remy's exact evaluation link rates (pkt/ms)
REMY_LINK_PPTS = [0.237, 0.376, 0.596, 0.946, 1.500, 2.379, 3.773, 5.983, 9.490]
# Subset for fast training eval (6 points spanning the range)
TRAIN_LINK_PPTS = [0.237, 0.596, 1.500, 3.773, 5.983, 9.490]


def evaluate_multipoint(
    policy,
    link_ppts: List[float] = None,
    rtt_ms: float = 150.0,
    num_senders: int = 2,
    duration_ms: float = 50_000.0,
    seed: int = 42,
) -> Tuple[float, Dict[float, float]]:
    """Evaluate a policy across multiple link rates.

    Returns (mean_normalized, {mbps: normalized_score}).
    """
    if link_ppts is None:
        link_ppts = TRAIN_LINK_PPTS

    scores = {}
    for i, link_ppt in enumerate(link_ppts):
        # Reset any function-attribute state
        if hasattr(policy, '_state'):
            del policy._state
        if hasattr(policy, '_states'):
            del policy._states

        result = run_remy_sim(
            policies=[policy],
            link_ppt=link_ppt,
            rtt_ms=rtt_ms,
            num_senders=num_senders,
            duration_ms=duration_ms,
            seed=seed + i * 100,
        )
        mbps = link_ppt * 10.0
        scores[mbps] = result['normalized_score']

    mean_score = sum(scores.values()) / len(scores)
    return mean_score, scores


# ── Evolution loop ──────────────────────────────────────────────────

@dataclass
class EvolutionConfig:
    """Configuration for evolution run."""
    generations: int = 20
    population_size: int = 5  # candidates per generation
    train_multiplier: float = 1.0  # training ConfigRange multiplier
    test_multipliers: List[float] = None  # for generalization eval
    duration_ms: float = 50_000.0  # sim duration per config
    num_trials: int = 1
    seed: int = 42
    model: str = "gpt-5.3-codex"
    output_dir: str = "output_remy_evolve"
    multipoint: bool = False  # train across multiple link rates
    seed_policy: Optional[str] = None  # evolve from specific seed only

    def __post_init__(self):
        if self.test_multipliers is None:
            self.test_multipliers = [1, 2, 5, 10, 20, 50, 100]


def run_evolution(config: EvolutionConfig) -> Dict:
    """Run LLM-guided evolution of Remy-compatible CCA policies.

    Returns the best policy code, fitness, and evolution history.
    """
    os.makedirs(config.output_dir, exist_ok=True)

    train_cfg = make_training_range(config.train_multiplier)
    train_cfg.duration_ms = config.duration_ms

    # Archive: list of (code, fitness, generation)
    archive = []
    best_code = None
    best_fitness = -float('inf')
    history = []

    # Evaluate seeds
    seeds_to_eval = {config.seed_policy: SEEDS[config.seed_policy]} if config.seed_policy else SEEDS
    print(f"=== Evaluating seed policies ({', '.join(seeds_to_eval.keys())}) ===")
    for name, code in seeds_to_eval.items():
        fn = compile_policy(code)
        if fn is None:
            print(f"  Seed {name}: FAILED to compile")
            continue
        result = evaluate_policy(fn, train_cfg, num_trials=config.num_trials, seed=config.seed)
        fitness = result['mean_normalized']
        print(f"  Seed {name}: utility={result['mean_utility']:.2f} "
              f"normalized={fitness:.3f} "
              f"tput={result['mean_throughput_ppt']:.4f} "
              f"delay={result['mean_delay_ms']:.1f}ms")
        archive.append((code, fitness, 0))
        if fitness > best_fitness:
            best_fitness = fitness
            best_code = code
        history.append({
            'gen': 0, 'name': name, 'fitness': fitness,
            'utility': result['mean_utility'],
            'throughput': result['mean_throughput_ppt'],
            'delay': result['mean_delay_ms'],
        })

    print(f"\nBest seed: normalized={best_fitness:.3f}")

    # Evolution loop
    for gen in range(1, config.generations + 1):
        print(f"\n=== Generation {gen}/{config.generations} ===")
        gen_best_fitness = -float('inf')
        gen_best_code = None

        # Build archive summary for LLM context
        sorted_archive = sorted(archive, key=lambda x: x[1], reverse=True)[:5]
        archive_lines = []
        for i, (code, fit, g) in enumerate(sorted_archive):
            archive_lines.append(f"# Rank {i+1} (gen {g}, normalized={fit:.3f}):")
            # Show just the core logic, not the full code
            for line in code.strip().split('\n'):
                if line.strip() and not line.strip().startswith('#'):
                    archive_lines.append(f"  {line}")
            archive_lines.append("")
        archive_summary = '\n'.join(archive_lines[:30])  # limit context

        # Select parent
        if random.random() < 0.5:
            # Exploit: best
            parent_code = best_code
            parent_fitness = best_fitness
        else:
            # Explore: random from archive
            parent_code, parent_fitness, _ = random.choice(archive)

        for candidate_idx in range(config.population_size):
            print(f"  Candidate {candidate_idx + 1}/{config.population_size}...", end=" ")

            # Mutate
            child_code = mutate_policy(
                parent_code, parent_fitness,
                archive_summary=archive_summary,
                model=config.model,
            )
            if child_code is None:
                print("LLM failed")
                continue

            # Compile
            fn = compile_policy(child_code)
            if fn is None:
                print("compile failed")
                continue

            # Evaluate
            try:
                result = evaluate_policy(
                    fn, train_cfg,
                    num_trials=config.num_trials,
                    seed=config.seed + gen * 100 + candidate_idx,
                )
                fitness = result['mean_normalized']
                print(f"normalized={fitness:.3f} "
                      f"(tput={result['mean_throughput_ppt']:.4f}, "
                      f"delay={result['mean_delay_ms']:.1f}ms)")

                archive.append((child_code, fitness, gen))
                history.append({
                    'gen': gen, 'candidate': candidate_idx,
                    'fitness': fitness,
                    'utility': result['mean_utility'],
                    'throughput': result['mean_throughput_ppt'],
                    'delay': result['mean_delay_ms'],
                })

                if fitness > gen_best_fitness:
                    gen_best_fitness = fitness
                    gen_best_code = child_code

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_code = child_code
                    print(f"    *** NEW BEST: {fitness:.3f} ***")
                    # Save best
                    with open(os.path.join(config.output_dir, 'best_policy.py'), 'w') as f:
                        f.write(child_code)

            except Exception as e:
                print(f"eval failed: {e}")
                traceback.print_exc()

        print(f"  Gen {gen} best: {gen_best_fitness:.3f}, Overall best: {best_fitness:.3f}")

    # Save final results
    with open(os.path.join(config.output_dir, 'best_policy.py'), 'w') as f:
        f.write(best_code)
    with open(os.path.join(config.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(config.output_dir, 'archive.json'), 'w') as f:
        json.dump([(code, fit, gen) for code, fit, gen in archive], f, indent=2)

    # Run generalization evaluation on best
    print("\n=== Generalization Evaluation ===")
    best_fn = compile_policy(best_code)
    gen_results = {}
    if best_fn:
        for mult in config.test_multipliers:
            test_cfg = make_training_range(mult)
            test_cfg.duration_ms = config.duration_ms
            result = evaluate_policy(best_fn, test_cfg, num_trials=config.num_trials, seed=config.seed)
            gen_results[mult] = {
                'throughput': result['mean_throughput_ppt'],
                'delay': result['mean_delay_ms'],
                'utility': result['mean_utility'],
                'normalized': result['mean_normalized'],
            }
            print(f"  {mult:4}x: norm={result['mean_normalized']:.3f} "
                  f"tput={result['mean_throughput_ppt']:.4f} "
                  f"delay={result['mean_delay_ms']:.1f}ms")

    with open(os.path.join(config.output_dir, 'generalization.json'), 'w') as f:
        json.dump(gen_results, f, indent=2)

    return {
        'best_code': best_code,
        'best_fitness': best_fitness,
        'history': history,
        'generalization': gen_results,
    }


def run_evolution_multipoint(config: EvolutionConfig) -> Dict:
    """Run multi-point LLM-guided evolution for generalization.

    Fitness = mean normalized score across 6 link rates.
    LLM receives per-rate feedback to guide improvements.
    """
    os.makedirs(config.output_dir, exist_ok=True)

    archive = []  # (code, fitness, per_rate_scores, generation)
    best_code = None
    best_fitness = -float('inf')
    best_per_rate = {}
    history = []

    # Evaluate seeds
    seeds_to_eval = {config.seed_policy: SEEDS[config.seed_policy]} if config.seed_policy else SEEDS
    print(f"=== Evaluating seed policies (multi-point, {', '.join(seeds_to_eval.keys())}) ===")
    for name, code in seeds_to_eval.items():
        fn = compile_policy(code)
        if fn is None:
            print(f"  Seed {name}: FAILED to compile")
            continue
        try:
            fitness, per_rate = evaluate_multipoint(
                fn, link_ppts=TRAIN_LINK_PPTS,
                duration_ms=config.duration_ms, seed=config.seed,
            )
        except Exception as e:
            print(f"  Seed {name}: eval failed: {e}")
            continue
        print(f"  Seed {name}: mean={fitness:.3f}")
        for mbps, score in sorted(per_rate.items()):
            print(f"    {mbps:6.1f} Mbps: {score:.3f}")
        archive.append((code, fitness, per_rate, 0))
        if fitness > best_fitness:
            best_fitness = fitness
            best_code = code
            best_per_rate = per_rate
        history.append({
            'gen': 0, 'name': name, 'fitness': fitness, 'per_rate': per_rate,
        })

    print(f"\nBest seed: mean={best_fitness:.3f}")

    # Evolution loop
    for gen in range(1, config.generations + 1):
        print(f"\n=== Generation {gen}/{config.generations} (multi-point) ===")
        gen_best_fitness = -float('inf')
        gen_best_code = None

        # Archive summary
        sorted_archive = sorted(archive, key=lambda x: x[1], reverse=True)[:5]
        archive_lines = []
        for i, (code, fit, pr, g) in enumerate(sorted_archive):
            archive_lines.append(f"# Rank {i+1} (gen {g}, mean={fit:.3f}):")
            for line in code.strip().split('\n'):
                if line.strip() and not line.strip().startswith('#'):
                    archive_lines.append(f"  {line}")
            archive_lines.append("")
        archive_summary = '\n'.join(archive_lines[:40])

        # Select parent
        if random.random() < 0.5:
            parent_code = best_code
            parent_fitness = best_fitness
            parent_per_rate = best_per_rate
        else:
            parent_code, parent_fitness, parent_per_rate, _ = random.choice(archive)

        for candidate_idx in range(config.population_size):
            print(f"  Candidate {candidate_idx + 1}/{config.population_size}...", end=" ")

            child_code = mutate_policy(
                parent_code, parent_fitness,
                archive_summary=archive_summary,
                model=config.model,
                multipoint=True,
                per_rate_scores=parent_per_rate,
            )
            if child_code is None:
                print("LLM failed")
                continue

            fn = compile_policy(child_code)
            if fn is None:
                print("compile failed")
                continue

            try:
                fitness, per_rate = evaluate_multipoint(
                    fn, link_ppts=TRAIN_LINK_PPTS,
                    duration_ms=config.duration_ms,
                    seed=config.seed + gen * 100 + candidate_idx,
                )
                worst = min(per_rate.values())
                best_rate = max(per_rate.values())
                print(f"mean={fitness:.3f} (worst={worst:.3f}, best={best_rate:.3f})")

                archive.append((child_code, fitness, per_rate, gen))
                history.append({
                    'gen': gen, 'candidate': candidate_idx,
                    'fitness': fitness, 'per_rate': per_rate,
                })

                if fitness > gen_best_fitness:
                    gen_best_fitness = fitness
                    gen_best_code = child_code

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_code = child_code
                    best_per_rate = per_rate
                    print(f"    *** NEW BEST: {fitness:.3f} ***")
                    with open(os.path.join(config.output_dir, 'best_policy.py'), 'w') as f:
                        f.write(child_code)

            except Exception as e:
                print(f"eval failed: {e}")
                traceback.print_exc()

        print(f"  Gen {gen} best: {gen_best_fitness:.3f}, Overall best: {best_fitness:.3f}")

    # Save results
    with open(os.path.join(config.output_dir, 'best_policy.py'), 'w') as f:
        f.write(best_code)
    with open(os.path.join(config.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2, default=str)

    # Full 9-point generalization eval
    print("\n=== Full Generalization Evaluation (9 link rates) ===")
    best_fn = compile_policy(best_code)
    gen_results = {}
    if best_fn:
        _, full_scores = evaluate_multipoint(
            best_fn, link_ppts=REMY_LINK_PPTS,
            duration_ms=config.duration_ms, seed=config.seed,
        )
        for mbps, score in sorted(full_scores.items()):
            link_ppt = mbps / 10.0
            gen_results[str(link_ppt)] = {'normalized': score, 'link_mbps': mbps}
            print(f"  {mbps:6.1f} Mbps: {score:.3f}")

    with open(os.path.join(config.output_dir, 'generalization.json'), 'w') as f:
        json.dump(gen_results, f, indent=2)

    return {
        'best_code': best_code,
        'best_fitness': best_fitness,
        'best_per_rate': best_per_rate,
        'history': history,
        'generalization': gen_results,
    }


# ── Baseline generalization eval ────────────────────────────────────

def run_baseline_generalization(
    test_multipliers: List[float] = None,
    duration_ms: float = 50_000.0,
    seed: int = 42,
    output_dir: str = "output_remy_evolve",
) -> Dict:
    """Evaluate baseline CCAs across test multipliers for comparison."""
    if test_multipliers is None:
        test_multipliers = [1, 2, 5, 10, 20, 50, 100]

    baselines = {
        'AIMD': aimd_policy,
        'Copa': copa_policy,
        'Constant': constant_policy,
    }

    results = {}
    for name, policy in baselines.items():
        results[name] = {}
        for mult in test_multipliers:
            cfg = make_training_range(mult)
            cfg.duration_ms = duration_ms
            result = evaluate_policy(policy, cfg, num_trials=1, seed=seed)
            results[name][mult] = {
                'throughput': result['mean_throughput_ppt'],
                'delay': result['mean_delay_ms'],
                'utility': result['mean_utility'],
                'normalized': result['mean_normalized'],
            }
            print(f"  {name:12s} @ {mult:4}x: norm={result['mean_normalized']:.3f} "
                  f"tput={result['mean_throughput_ppt']:.4f} "
                  f"delay={result['mean_delay_ms']:.1f}ms")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'baselines.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ── CLI ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='LLM-guided CCA evolution in Remy framework')
    parser.add_argument('--generations', type=int, default=15)
    parser.add_argument('--population', type=int, default=5)
    parser.add_argument('--train-mult', type=float, default=1.0,
                        help='Training ConfigRange multiplier (1=single point, 10=10x range)')
    parser.add_argument('--duration', type=float, default=50_000.0,
                        help='Simulation duration per config (ms)')
    parser.add_argument('--model', type=str, default='gpt-5.3-codex')
    parser.add_argument('--output', type=str, default='output_remy_evolve')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--baselines-only', action='store_true',
                        help='Only run baseline generalization eval')
    parser.add_argument('--multipoint', action='store_true',
                        help='Train across 6 link rates for generalization')
    parser.add_argument('--seed-policy', type=str, default=None,
                        choices=list(SEEDS.keys()),
                        help='Evolve from a specific seed only (AlphaZero-style)')

    args = parser.parse_args()

    if args.baselines_only:
        print("=== Baseline Generalization Evaluation ===")
        run_baseline_generalization(
            duration_ms=args.duration,
            seed=args.seed,
            output_dir=args.output,
        )
    elif args.multipoint:
        cfg = EvolutionConfig(
            generations=args.generations,
            population_size=args.population,
            duration_ms=args.duration,
            model=args.model,
            output_dir=args.output,
            seed=args.seed,
            multipoint=True,
            seed_policy=args.seed_policy,
        )
        results = run_evolution_multipoint(cfg)
        print(f"\n=== DONE ===")
        print(f"Best mean normalized: {results['best_fitness']:.3f}")
        print(f"Output: {args.output}/")
    else:
        cfg = EvolutionConfig(
            generations=args.generations,
            population_size=args.population,
            train_multiplier=args.train_mult,
            duration_ms=args.duration,
            model=args.model,
            output_dir=args.output,
            seed=args.seed,
            seed_policy=args.seed_policy,
        )
        results = run_evolution(cfg)
        print(f"\n=== DONE ===")
        print(f"Best normalized score: {results['best_fitness']:.3f}")
        print(f"Output: {args.output}/")
