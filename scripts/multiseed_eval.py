#!/usr/bin/env python3
"""Multi-seed evaluation of CCA policies for workshop paper error bars."""

import sys
import json
import math
import time
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PY = REPO_ROOT / "results" / "python_sim" / "output_remy_evolve"
REGENERATED_PY = REPO_ROOT / "results" / "python_sim" / "regenerated" / "output_remy_evolve"
RESULTS_CPP = REPO_ROOT / "results" / "cpp_sim" / "dense_cpp"

sys.path.insert(0, str(REPO_ROOT))

from alphacc.remy_eval import run_remy_sim, aimd_policy, copa_policy

# Import LLM evolved policy
spec_path = RESULTS_PY / "best_policy.py"
with spec_path.open() as f:
    code = f.read()
exec(compile(code, str(spec_path), 'exec'))
# evolved_policy is now in scope

LINK_RATES = [0.237, 0.376, 0.596, 0.946, 1.500, 2.379, 3.773, 5.983, 9.490]
SEEDS = [42, 123, 456, 789, 1000]
DURATION_MS = 30000.0
RTT_MS = 150.0
NUM_SENDERS = 2
MEAN_ON_MS = 1000.0   # Keith's canonical params (was 5000 default)
MEAN_OFF_MS = 1000.0

def eval_policy_multiseed(policy_fn, name, reset_state=False):
    """Evaluate a policy at all link rates with multiple seeds."""
    result = {}
    for lr in LINK_RATES:
        scores = []
        for seed in SEEDS:
            if reset_state and hasattr(policy_fn, '_state'):
                del policy_fn._state
            r = run_remy_sim(
                policies=[policy_fn],
                link_ppt=lr,
                rtt_ms=RTT_MS,
                num_senders=NUM_SENDERS,
                duration_ms=DURATION_MS,
                mean_on_ms=MEAN_ON_MS,
                mean_off_ms=MEAN_OFF_MS,
                seed=seed,
            )
            scores.append(r['normalized_score'])
        mean = sum(scores) / len(scores)
        std = math.sqrt(sum((s - mean) ** 2 for s in scores) / len(scores))
        key = f"{lr:.3f}"
        result[key] = {"mean": round(mean, 6), "std": round(std, 6), "scores": [round(s, 6) for s in scores]}
        print(f"  {name} @ {lr:.3f} ppt: mean={mean:.4f} std={std:.4f}")
    return result


def load_csv_scores(filepath):
    """Load normalized-score columns from archived CSVs."""
    scores_by_lr = {}
    with open(filepath) as f:
        reader = csv.reader(f)
        rows = list(reader)
    if rows and rows[0] and rows[0][0] == 'link_ppt':
        rows = rows[1:]
        value_idx = 2
    else:
        value_idx = 1
    for row in rows:
        if not row:
            continue
        lr = float(row[0])
        ns = float(row[value_idx])
        scores_by_lr[lr] = ns
    closest = {}
    for lr in LINK_RATES:
        best = min(scores_by_lr, key=lambda seen: abs(seen - lr))
        key = f"{lr:.3f}"
        closest[key] = scores_by_lr[best]
    return closest


def main():
    results = {}

    # Copa
    print("Evaluating Copa...")
    t0 = time.time()
    results["Copa"] = eval_policy_multiseed(copa_policy, "Copa")
    print(f"  Copa done in {time.time()-t0:.1f}s\n")

    # AIMD
    print("Evaluating AIMD...")
    t0 = time.time()
    results["AIMD"] = eval_policy_multiseed(aimd_policy, "AIMD")
    print(f"  AIMD done in {time.time()-t0:.1f}s\n")

    # LLM evolved
    print("Evaluating LLM-evolved...")
    t0 = time.time()
    results["LLM"] = eval_policy_multiseed(evolved_policy, "LLM", reset_state=True)
    print(f"  LLM done in {time.time()-t0:.1f}s\n")

    # Remy tree (single CSV, no variance)
    print("Loading Remy tree scores...")
    remy_scores = load_csv_scores(RESULTS_CPP / 'remy-1x.csv')
    results["Remy"] = {}
    for key, ns in remy_scores.items():
        results["Remy"][key] = {"mean": round(ns, 6), "std": 0.0, "scores": [round(ns, 6)]}
        print(f"  Remy @ {key} ppt: score={ns:.4f}")
    print()

    # PPO (6 brain CSVs)
    print("Loading PPO brain scores...")
    brain_files = sorted(RESULTS_CPP.glob('ppo-brain.*.csv'))
    print(f"  Found {len(brain_files)} brain files")
    ppo_by_lr = {}
    for bf in brain_files:
        scores = load_csv_scores(bf)
        for key, ns in scores.items():
            ppo_by_lr.setdefault(key, []).append(ns)

    results["PPO"] = {}
    for key in sorted(ppo_by_lr.keys(), key=lambda x: float(x)):
        scores = ppo_by_lr[key]
        mean = sum(scores) / len(scores)
        std = math.sqrt(sum((s - mean) ** 2 for s in scores) / len(scores))
        results["PPO"][key] = {"mean": round(mean, 6), "std": round(std, 6), "scores": [round(s, 6) for s in scores]}
        print(f"  PPO @ {key} ppt: mean={mean:.4f} std={std:.4f} (n={len(scores)})")
    print()

    # Save
    REGENERATED_PY.mkdir(parents=True, exist_ok=True)
    out_path = REGENERATED_PY / 'multiseed_eval.json'
    with out_path.open('w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
