#!/usr/bin/env python3
"""RTT sweep evaluation: all policies × 3 RTTs × 3 link rates."""

import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "results" / "python_sim" / "output_remy_evolve"
REGENERATED_DIR = REPO_ROOT / "results" / "python_sim" / "regenerated" / "output_remy_evolve"

sys.path.insert(0, str(REPO_ROOT))

from alphacc.remy_eval import (
    run_remy_sim, RemyMemory, RemyAction,
    copa_policy, aimd_policy, bbr_policy,
)

# Load LLM-evolved policy
spec_path = OUTPUT_DIR / "best_policy.py"
with spec_path.open() as f:
    code = f.read()
exec(compile(code, str(spec_path), 'exec'))
# evolved_policy is now in scope

RTTS = [50, 150, 300]
LINK_RATES = [0.237, 0.946, 5.983]
SEED = 42
DURATION_MS = 30000

policies = {
    'Copa': copa_policy,
    'AIMD': aimd_policy,
    'BBR': bbr_policy,
    'LLM_evolved': evolved_policy,
}

results = {}

for name, policy_fn in policies.items():
    results[name] = {}
    for rtt in RTTS:
        for link in LINK_RATES:
            key = f"{rtt}_{link}"

            # Reset function-attribute state between evals
            if hasattr(policy_fn, '_state'):
                del policy_fn._state

            r = run_remy_sim(
                policies=[policy_fn],
                link_ppt=link,
                rtt_ms=float(rtt),
                num_senders=2,
                duration_ms=DURATION_MS,
                seed=SEED,
            )

            results[name][key] = {
                'normalized': round(r['normalized_score'], 4),
                'tput': round(r['throughput_ppt'], 6),
                'delay': round(r['avg_delay_ms'], 2),
            }
            print(f"{name:12s}  rtt={rtt:3d}  link={link:.3f}  "
                  f"norm={r['normalized_score']:+.3f}  "
                  f"tput={r['throughput_ppt']:.4f}  "
                  f"delay={r['avg_delay_ms']:.1f}ms")

REGENERATED_DIR.mkdir(parents=True, exist_ok=True)
out_path = REGENERATED_DIR / 'rtt_sweep.json'
with out_path.open('w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {out_path}")
