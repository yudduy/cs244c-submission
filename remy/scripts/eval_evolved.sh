#!/bin/bash
# Evaluate AlphaCC evolved policy at 9 standard link rates (on=1000/off=1000).
# Output format matches all_trees_on1000.txt for direct comparison.
#
# Usage: ./scripts/eval_evolved.sh [path/to/sender-runner]

SENDER_RUNNER="${1:-./src/sender-runner}"

if [ ! -x "$SENDER_RUNNER" ]; then
    echo "Error: $SENDER_RUNNER not found or not executable"
    echo "Build first: cd src && make sender-runner"
    exit 1
fi

echo "--- AlphaCC_v5_best (evolved single-point) ---"
echo "link_ppt, normalized_score"

for LINK_PPT in 0.237 0.376 0.596 0.946 1.500 2.379 3.773 5.983 9.490; do
    OUTPUT=$("$SENDER_RUNNER" sender=evolved link=$LINK_PPT rtt=100 on=1000 off=1000 nsrc=2 2>/dev/null)
    NORM_SCORE=$(echo "$OUTPUT" | grep "normalized_score" | awk -F= '{print $2}' | tr -d ' ')
    echo "$LINK_PPT, $NORM_SCORE"
done
