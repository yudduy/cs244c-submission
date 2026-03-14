#!/bin/bash
# Evaluate a trained Remy WhiskerTree across 9 link rates
# Usage: ./scripts/run_remy_eval.sh <tree_path> <config>
# Example: ./scripts/run_remy_eval.sh results/whisker-trees/1x-2src/cca.179 remy/configs/link-1x.cfg
set -e

TREE=${1:?Usage: $0 <tree_path> <config>}
CONFIG=${2:?Usage: $0 <tree_path> <config>}

echo "=== Evaluating Remy tree: $TREE ==="
echo "Config: $CONFIG"

cd remy
# Build if needed
if [ ! -f src/sender-runner ]; then
    echo "Building remy..."
    ./autogen.sh && ./configure && make -j$(nproc)
fi

# 9 link rates from paper (packets/ms)
for RATE in 0.237 0.376 0.596 0.946 1.5 2.379 3.773 5.983 9.49; do
    echo "Link rate: ${RATE} ppt"
    ./src/sender-runner cf=../$CONFIG if=../$TREE link_ppt=$RATE
done
