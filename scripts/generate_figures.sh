#!/bin/bash
# Generate all paper figures from results
# Usage: ./scripts/generate_figures.sh
set -e

echo "=== Generating paper figures ==="

# Main generalization plot (Fig 1)
python3 scripts/plot_paper_v2.py

# Multipoint comparison (Fig 5)
python3 scripts/plot_multipoint.py

# RTT sweep (Fig 2)
python3 scripts/rtt_sweep.py --plot-only

echo "Figures saved to paper/figures/"
