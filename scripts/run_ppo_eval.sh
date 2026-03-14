#!/bin/bash
# Evaluate a trained PPO brain across 9 link rates
# Usage: ./scripts/run_ppo_eval.sh <brain_path> <config>
# Example: ./scripts/run_ppo_eval.sh checkpoints/brain.248 remy/configs/link-1x.cfg
set -e

BRAIN=${1:?Usage: $0 <brain_path> <config>}
CONFIG=${2:?Usage: $0 <brain_path> <config>}

echo "=== Evaluating PPO brain: $BRAIN ==="
echo "Config: $CONFIG"

cd remy
# Build if needed (requires libtorch)
if [ ! -f src/neural-evaluator ]; then
    echo "Building remy with neural support..."
    ./autogen.sh && ./configure && make -j$(nproc)
fi

for RATE in 0.237 0.376 0.596 0.946 1.5 2.379 3.773 5.983 9.49; do
    echo "Link rate: ${RATE} ppt"
    ./src/neural-evaluator cf=../$CONFIG if=../$BRAIN link_ppt=$RATE
done
