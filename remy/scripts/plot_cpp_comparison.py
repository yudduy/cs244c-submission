#!/usr/bin/env python3
"""C++ simulator comparison figure: Remy trees + PPO + AlphaCC.

Authoritative evaluation — all methods run in Remy's original C++ simulator.

Output: paper/figures/cpp_generalization.{pdf,png}
"""

import csv
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / 'results' / 'dense_cpp'
OUT = REPO / 'paper' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)


def load_csv(path):
    mbps, scores = [], []
    with open(path) as f:
        r = csv.reader(f)
        next(r)  # skip header
        for row in r:
            mbps.append(float(row[1]))
            scores.append(float(row[2]))
    return np.array(mbps), np.array(scores)


# ── Load data ────────────────────────────────────────────────────

remy1x_mbps, remy1x = load_csv(DATA / 'remy-1x.csv')
remy10x_mbps, remy10x = load_csv(DATA / 'remy-10x.csv')
remy20x_mbps, remy20x = load_csv(DATA / 'remy-20x.csv')
alphacc_mbps, alphacc = load_csv(DATA / 'alphacc_evolved.csv')

# PPO: load all brains, interpolate to dense grid
ppo_brains = []
for fp in sorted(DATA.glob('ppo-brain.*.csv')):
    m, s = load_csv(fp)
    interp = np.interp(np.log10(remy1x_mbps), np.log10(m), s)
    ppo_brains.append(interp)

ppo_mean = np.mean(ppo_brains, axis=0)
ppo_std = np.std(ppo_brains, axis=0)

# PPO raw points for scatter overlay
ppo_raw_mbps, ppo_raw_scores = load_csv(sorted(DATA.glob('ppo-brain.*.csv'))[0])


# ── Figure ───────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(5.5, 4))

TICK_MBPS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

# Training point
ax.axvline(10.0, color='#aaa', ls=':', lw=0.8, alpha=0.5, zorder=0)
ax.text(10.0, 0.3, 'trained\nhere', ha='center', fontsize=7, color='#888',
        style='italic')

# Remy 10x training region
ax.axvspan(1.5, 150, alpha=0.04, color='#3498db', zorder=0)

# Remy trees
ax.plot(remy1x_mbps, remy1x, color='#08306B', ls='-', lw=2.5,
        marker='s', markersize=4, markevery=4,
        label='Remy 1x', zorder=5)
ax.plot(remy10x_mbps, remy10x, color='#2171B5', ls='--', lw=1.8,
        markevery=4, label='Remy 10x', zorder=4)
ax.plot(remy20x_mbps, remy20x, color='#6BAED6', ls='-.', lw=1.4,
        markevery=4, label='Remy 20x', zorder=4)

# PPO (interpolated mean + std band, raw points as dots)
ax.plot(remy1x_mbps, ppo_mean, color='#e74c3c', ls='-', lw=2.5,
        marker='o', markersize=4, markevery=4,
        label=f'PPO (mean of {len(ppo_brains)})', zorder=5)
ax.fill_between(remy1x_mbps, ppo_mean - ppo_std, ppo_mean + ppo_std,
                color='#e74c3c', alpha=0.08, zorder=1)

# AlphaCC (C++ port)
ax.plot(alphacc_mbps, alphacc, color='#7B1FA2', ls='-', lw=2.5,
        marker='^', markersize=4, markevery=4,
        label='AlphaCC (source code)', zorder=5)

ax.set_xscale('log')
ax.set_xlabel('Link Rate (Mbps)', fontsize=10)
ax.set_ylabel('Normalized Score  (higher is better)', fontsize=10)
ax.set_xticks(TICK_MBPS)
ax.get_xaxis().set_major_formatter(ScalarFormatter())
ax.set_xlim(remy1x_mbps[0] * 0.9, remy1x_mbps[-1] * 1.1)
ax.set_ylim(-8, 1.0)
ax.grid(True, alpha=0.2, which='both')
ax.legend(loc='lower left', fontsize=8, framealpha=0.95)

plt.tight_layout()
plt.savefig(OUT / 'cpp_generalization.png', dpi=200, bbox_inches='tight')
plt.savefig(OUT / 'cpp_generalization.pdf', bbox_inches='tight')
plt.close()
print(f"Saved cpp_generalization to {OUT}")


# ── Summary table ────────────────────────────────────────────────

key_mbps = [1, 2.4, 6, 9.5, 15, 24, 38, 60, 95]
print(f"\n{'Mbps':>6} | {'Remy1x':>8} | {'Remy10x':>8} | {'Remy20x':>8} | {'AlphaCC':>8} | {'PPO':>8}")
print("-" * 65)

remy1x_wins = ppo_wins = alphacc_wins = 0
for target in key_mbps:
    idx = np.argmin(np.abs(remy1x_mbps - target))
    scores = {
        'Remy1x': remy1x[idx],
        'Remy10x': remy10x[idx],
        'Remy20x': remy20x[idx],
        'AlphaCC': alphacc[idx],
        'PPO': ppo_mean[idx],
    }
    best = max(scores, key=scores.get)
    row = f"{remy1x_mbps[idx]:6.1f}"
    for name in ['Remy1x', 'Remy10x', 'Remy20x', 'AlphaCC', 'PPO']:
        s = scores[name]
        marker = '*' if name == best else ' '
        row += f" | {s:7.2f}{marker}"
    print(row)

print("\nWins by method (single-point trained only: Remy1x, PPO, AlphaCC):")
for target in key_mbps:
    idx = np.argmin(np.abs(remy1x_mbps - target))
    sp = {'Remy1x': remy1x[idx], 'PPO': ppo_mean[idx], 'AlphaCC': alphacc[idx]}
    best = max(sp, key=sp.get)
    if best == 'Remy1x': remy1x_wins += 1
    elif best == 'PPO': ppo_wins += 1
    else: alphacc_wins += 1
print(f"  Remy1x: {remy1x_wins}/9, PPO: {ppo_wins}/9, AlphaCC: {alphacc_wins}/9")
