"""
Figure 2: Python Simulator Evaluation.

All methods evaluated consistently in the Python simulator across 50 log-spaced
link rates (1-1000 Mbps). Shows Remy trees, PPO neural nets, and AlphaCC.
"""

import json
import csv
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d

LINK_PPT_TO_MBPS = 10
BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
FIGURES = BASE / "paper" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Load dense eval (Remy 1x, AlphaCC multi-pt best) ──────────────────────

with open(RESULTS / "dense_eval.json") as f:
    dense = json.load(f)

link_ppts = np.array(dense["metadata"]["link_ppts"])
link_mbps = link_ppts * LINK_PPT_TO_MBPS

def extract_scores(results_list):
    return np.array([r["normalized_score"] for r in results_list])

remy_1x = extract_scores(dense["results"]["Remy_1x"])
alphacc_multi = extract_scores(dense["results"]["AlphaCC_pacing_r1"])

# ── Load AlphaCC single-point ─────────────────────────────────────────────

with open(RESULTS / "alphacc_singlepoint_dense.json") as f:
    alphacc_sp_data = json.load(f)

alphacc_sp = extract_scores(alphacc_sp_data)

# ── Load PPO brains (6 seeds), interpolate to dense grid ──────────────────

ppo_csvs = sorted(glob.glob(str(RESULTS / "ppo-evals" / "brain-*.csv")))
ppo_all = []

for csv_path in ppo_csvs:
    rows = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append([float(x) for x in row])
    rows = np.array(rows)
    ppo_link_ppts = rows[:, 0]
    # col 1 is already per-sender average (verified numerically)
    ppo_scores = rows[:, 1]

    # Interpolate to dense grid (log-space)
    interp_fn = interp1d(
        np.log10(ppo_link_ppts), ppo_scores,
        kind='linear', fill_value='extrapolate'
    )
    ppo_interp = interp_fn(np.log10(link_ppts))
    ppo_all.append(ppo_interp)

ppo_all = np.array(ppo_all)  # (6, 50)
ppo_mean = ppo_all.mean(axis=0)
ppo_std = ppo_all.std(axis=0)

# ── Plot ──────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
})

fig, ax = plt.subplots(figsize=(8, 5))

# Colors
BLUE = '#2166ac'
RED = '#c0392b'
RED_FILL = '#e74c3c'
PURPLE = '#7b2d8e'
GRAY = '#888888'

# 1. Remy 1x (reference)
ax.plot(link_mbps, remy_1x, color=BLUE, linewidth=2.2, linestyle='-',
        label='Remy 1x', zorder=3)

# 2. PPO mean +/- 1 sigma
ax.fill_between(link_mbps, ppo_mean - ppo_std, ppo_mean + ppo_std,
                color=RED_FILL, alpha=0.12, zorder=1, linewidth=0)
ax.plot(link_mbps, ppo_mean, color=RED, linewidth=2.2, linestyle='-',
        label=r'PPO (mean $\pm$ 1$\sigma$, n=6)', zorder=3)

# 3. AlphaCC single-point
ax.plot(link_mbps, alphacc_sp, color=PURPLE, linewidth=2.2, linestyle='-',
        label='AlphaCC single-pt', zorder=3)

# 4. AlphaCC multi-point best
ax.plot(link_mbps, alphacc_multi, color=PURPLE, linewidth=1.8,
        linestyle='--', dashes=(5, 3), label='AlphaCC multi-pt', zorder=3)

# Axes
ax.set_xscale('log')
ax.set_xlim(1, 1000)
ax.set_xlabel('Link Rate (Mbps)', fontsize=12, labelpad=6)
ax.set_ylabel('Normalized Score (higher is better)', fontsize=12, labelpad=6)

# Training point vertical line (after setting xscale so position is correct)
ax.axvline(x=10, color=GRAY, linewidth=0.9, linestyle=':', alpha=0.55,
           zorder=0)

# Place training point annotation at top of plot, offset right
ymin, ymax = ax.get_ylim()
ax.annotate('training\npoint', xy=(10, ymax), xytext=(14, ymax - 0.08 * (ymax - ymin)),
            fontsize=7.5, color='#555555', va='top', ha='left',
            arrowprops=dict(arrowstyle='-', color='#aaaaaa', lw=0.6))

# Grid
ax.grid(True, which='major', linewidth=0.4, alpha=0.3, color='#cccccc')
ax.grid(True, which='minor', linewidth=0.25, alpha=0.15, color='#dddddd')

# Legend
ax.legend(loc='lower left', fontsize=9.5, framealpha=0.92, edgecolor='#cccccc',
          borderpad=0.6, handlelength=2.5)

# Ticks
ax.tick_params(which='major', labelsize=10, length=4)
ax.tick_params(which='minor', length=2)

# Spine styling
for spine in ax.spines.values():
    spine.set_color('#888888')

fig.tight_layout(pad=1.2)

# Save
for ext in ['png', 'pdf']:
    out = FIGURES / f"fig_python_eval.{ext}"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")

plt.close()
