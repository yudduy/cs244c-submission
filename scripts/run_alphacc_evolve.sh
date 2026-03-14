#!/bin/bash
# Run AlphaCC evolution (single-point or multipoint)
# Usage: ./scripts/run_alphacc_evolve.sh [--multipoint] [--seed SEED]
# Requires: OPENAI_API_KEY environment variable
set -e

MULTIPOINT=false
SEED=42

while [[ $# -gt 0 ]]; do
    case $1 in
        --multipoint) MULTIPOINT=true; shift ;;
        --seed) SEED=$2; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY must be set"
    exit 1
fi

OUTDIR="output/alphacc-seed-${SEED}"
mkdir -p "$OUTDIR"

if [ "$MULTIPOINT" = true ]; then
    echo "=== AlphaCC multipoint evolution (seed=$SEED) ==="
    python3 -m alphacc.evolve_remy \
        --generations 30 --population 5 \
        --multipoint --seed "$SEED" \
        --output "$OUTDIR"
else
    echo "=== AlphaCC single-point evolution (seed=$SEED) ==="
    python3 -m alphacc.evolve_remy \
        --generations 15 --population 5 \
        --seed "$SEED" \
        --output "$OUTDIR"
fi

echo "Results in: $OUTDIR"
