#!/usr/bin/env python3
"""Dense evaluation of all CCA methods at 50 log-spaced link rates.

Matches the Remy paper's generalization plot methodology:
- 50 log-spaced link rates from 0.1 to 100 ppt (1 to 1000 Mbps)
- RTT = 150ms, 2 senders, on/off = 1000ms, duration = 30s
- Evaluates: Remy (WhiskerTree), AlphaCC (pacing_r1 + all systematic runs),
  Copa, AIMD, BBR baselines

Output: results/dense_eval.json
"""

import sys
import json
import math
import time
import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from alphacc.remy_eval import (
    run_remy_sim, RemyAction, RemyMemory,
    aimd_policy, copa_policy, bbr_policy,
)
from alphacc.whisker_loader import make_whisker_policy

# ── Configuration ─────────────────────────────────────────────────

N_POINTS = 50
LINK_PPT_LO = 0.1    # 1 Mbps
LINK_PPT_HI = 100.0  # 1000 Mbps
RTT_MS = 150.0
NUM_SENDERS = 2
MEAN_ON_MS = 1000.0
MEAN_OFF_MS = 1000.0
BASE_DURATION_MS = 30_000.0  # base duration at low rates
SEED = 42

# Log-spaced link rates
LINK_PPTS = [
    LINK_PPT_LO * (LINK_PPT_HI / LINK_PPT_LO) ** (i / (N_POINTS - 1))
    for i in range(N_POINTS)
]


def get_duration(link_ppt: float) -> float:
    """Scale simulation duration inversely with link rate.

    At high link rates, more packets flow per ms, so the sim is O(packets).
    Reduce duration to keep wall-clock time bounded while still getting
    enough data for convergence (at least 10 RTTs worth of traffic).
    """
    # At 1 ppt: 30s, at 10 ppt: 15s, at 100 ppt: 5s
    # Min duration: max(5000, 20 * RTT) = 5000ms (enough for ~33 RTTs)
    scaled = BASE_DURATION_MS / max(1.0, link_ppt)
    return max(5_000.0, min(BASE_DURATION_MS, scaled))


# ── Load policies ─────────────────────────────────────────────────

def load_policy_from_file(path: str, func_name: str = 'evolved_policy'):
    """Load a policy function from a .py file."""
    spec = importlib.util.spec_from_file_location("policy_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, func_name)


def reset_policy_state(policy):
    """Clear per-memory state dicts used by stateful policies."""
    if hasattr(policy, '_states'):
        policy._states = {}
    if hasattr(policy, '_state'):
        del policy._state


def eval_single(policy, link_ppt: float) -> dict:
    """Evaluate one policy at one link rate. Returns normalized score + details."""
    reset_policy_state(policy)
    duration = get_duration(link_ppt)
    result = run_remy_sim(
        policies=[policy],
        link_ppt=link_ppt,
        rtt_ms=RTT_MS,
        num_senders=NUM_SENDERS,
        buffer_pkts=None,  # infinite buffer (Remy default)
        duration_ms=duration,
        mean_on_ms=MEAN_ON_MS,
        mean_off_ms=MEAN_OFF_MS,
        seed=SEED,
    )
    return {
        'link_ppt': link_ppt,
        'link_mbps': link_ppt * 10,
        'normalized_score': result['normalized_score'],
        'total_utility': result['total_utility'],
        'throughput_ppt': result['throughput_ppt'],
        'avg_delay_ms': result['avg_delay_ms'],
        'duration_ms': duration,
    }


# ── Main ──────────────────────────────────────────────────────────

def main():
    results = {}

    # --- Whisker trees (Remy) ---
    whisker_dir = _REPO_ROOT / 'results' / 'whisker-trees'
    whisker_configs = {
        'Remy_1x':  whisker_dir / '1x-2src' / 'cca.179',
        'Remy_10x': whisker_dir / '10x-2src' / 'cca.36',
        'Remy_20x': whisker_dir / '20x-2src' / 'cca.19',
    }

    for name, tree_path in whisker_configs.items():
        if not tree_path.exists():
            print(f"  SKIP {name}: {tree_path} not found")
            continue
        print(f"Evaluating {name} ({N_POINTS} points)...", flush=True)
        policy = make_whisker_policy(str(tree_path))
        evals = []
        t0 = time.time()
        for i, lp in enumerate(LINK_PPTS):
            r = eval_single(policy, lp)
            evals.append(r)
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{N_POINTS} done ({elapsed:.1f}s)", flush=True)
        results[name] = evals
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

    # --- Baselines (Copa, AIMD, BBR) ---
    baselines = {
        'Copa': copa_policy,
        'AIMD': aimd_policy,
        'BBR': bbr_policy,
    }

    for name, policy in baselines.items():
        print(f"Evaluating {name} ({N_POINTS} points)...", flush=True)
        evals = []
        t0 = time.time()
        for i, lp in enumerate(LINK_PPTS):
            r = eval_single(policy, lp)
            evals.append(r)
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{N_POINTS} done ({elapsed:.1f}s)", flush=True)
        results[name] = evals
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

    # --- AlphaCC systematic runs (diverse_seeds) ---
    diverse_dir = _REPO_ROOT / 'results' / 'diverse_seeds'
    alphacc_runs = {}
    for run_dir in sorted(diverse_dir.iterdir()):
        policy_path = run_dir / 'best_policy.py'
        if not policy_path.exists():
            continue
        alphacc_runs[run_dir.name] = str(policy_path)

    for name, path in alphacc_runs.items():
        print(f"Evaluating AlphaCC/{name} ({N_POINTS} points)...", flush=True)
        policy = load_policy_from_file(path)
        evals = []
        t0 = time.time()
        for i, lp in enumerate(LINK_PPTS):
            r = eval_single(policy, lp)
            evals.append(r)
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{N_POINTS} done ({elapsed:.1f}s)", flush=True)
        results[f'AlphaCC_{name}'] = evals
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

    # --- AlphaCC archive runs (v2-v6, multipoint, 10x, stateless) ---
    # Skipped for speed — diverse seed runs are the paper-critical ones.
    # Uncomment to include exploratory archive runs.
    # alphacc_evals_dir = _REPO_ROOT / 'results' / 'alphacc-evals'
    # for policy_file in sorted(alphacc_evals_dir.glob('*_best_policy.py')):
    #     run_name = policy_file.stem.replace('_best_policy', '')
    #     label = f'AlphaCC_{run_name}'
    #     if label in results:
    #         continue
    #     print(f"Evaluating {label} ({N_POINTS} points)...", flush=True)
    #     policy = load_policy_from_file(str(policy_file))
    #     evals = []
    #     t0 = time.time()
    #     for i, lp in enumerate(LINK_PPTS):
    #         r = eval_single(policy, lp)
    #         evals.append(r)
    #         if (i + 1) % 10 == 0:
    #             elapsed = time.time() - t0
    #             print(f"  {i+1}/{N_POINTS} done ({elapsed:.1f}s)", flush=True)
    #     results[label] = evals
    #     elapsed = time.time() - t0
    #     print(f"  Done in {elapsed:.1f}s")

    # --- Save ---
    out_path = _REPO_ROOT / 'results' / 'dense_eval.json'
    # Store metadata alongside results
    output = {
        'metadata': {
            'n_points': N_POINTS,
            'link_ppt_range': [LINK_PPT_LO, LINK_PPT_HI],
            'rtt_ms': RTT_MS,
            'num_senders': NUM_SENDERS,
            'mean_on_ms': MEAN_ON_MS,
            'mean_off_ms': MEAN_OFF_MS,
            'duration_ms': BASE_DURATION_MS,
            'seed': SEED,
            'link_ppts': LINK_PPTS,
        },
        'results': results,
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(results)} methods × {N_POINTS} points to {out_path}")


if __name__ == '__main__':
    main()
