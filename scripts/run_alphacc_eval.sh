#!/bin/bash
# Evaluate an AlphaCC evolved policy across 9 link rates
# Usage: ./scripts/run_alphacc_eval.sh <policy.py> [--rtt RTT_MS]
# Example: ./scripts/run_alphacc_eval.sh results/alphacc-evals/v5_best_policy.py
set -e

POLICY=${1:?Usage: $0 <policy.py> [--rtt RTT_MS]}
RTT=${3:-150}

echo "=== Evaluating AlphaCC policy: $POLICY (RTT=${RTT}ms) ==="

python3 -c "
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath('$0')), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath('$0')), '..', '..'))
from alphacc.remy_eval import run_remy_sim, RemyMemory, RemyAction

# Load policy by exec'ing the .py file (defines evolved_policy)
with open('$POLICY') as f:
    code = f.read()
ns = {}
exec(compile(code, '$POLICY', 'exec'), ns)
policy = ns['evolved_policy']

results = {}

# 9 link rates from Remy paper (packets/ms)
# Conversion: 1 ppt = 10 Mbps (Remy convention)
for rate_ppt in [0.237, 0.376, 0.596, 0.946, 1.5, 2.379, 3.773, 5.983, 9.49]:
    # Reset stateful policies between runs
    if hasattr(policy, '_state'):
        del policy._state
    r = run_remy_sim(
        policies=[policy],
        link_ppt=rate_ppt,
        rtt_ms=float(${RTT}),
        num_senders=2,
        duration_ms=30000.0,
        mean_on_ms=1000.0,
        mean_off_ms=1000.0,
        seed=42,
    )
    mbps = rate_ppt * 10.0  # 1 ppt = 10 Mbps (Remy convention)
    score = r['normalized_score']
    print(f'  {mbps:6.1f} Mbps: normalized={score:.4f}')
    results[str(rate_ppt)] = {
        'normalized': round(score, 6),
        'link_mbps': round(mbps, 1),
        'throughput_ppt': round(r['throughput_ppt'], 6),
        'avg_delay_ms': round(r['avg_delay_ms'], 2),
    }

# Save
outfile = os.path.splitext('$POLICY')[0] + '_eval.json'
with open(outfile, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved to {outfile}')
"
