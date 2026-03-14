#!/usr/bin/env python3
"""Main comparison figure: all CCA methods across 50 log-spaced link rates.

Matches Remy paper's generalization plot style (Fig 2 / Fig 7):
- Log x-axis (link rate in Mbps)
- Y-axis: normalized score (log2(tput/C) - log2(delay/RTT))
- Shaded training region
- All methods: Remy trees (original + Learnability 1000x), PPO, baselines, AlphaCC

Loads:
- results/dense_eval.json (from eval_dense.py)
- results/learnability_eval.json (Learnability paper trees)
- results/ppo-evals/brain-*.csv (PPO data, interpolated)

Output: paper/figures/generalization.{pdf,png}
"""

import sys
import json
import csv
import numpy as np
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

OUT = _REPO_ROOT / 'paper' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)


# ── Load dense evaluation data ────────────────────────────────────

dense_path = _REPO_ROOT / 'results' / 'dense_eval.json'
with open(dense_path) as f:
    dense = json.load(f)

metadata = dense['metadata']
results = dense['results']
link_ppts = np.array(metadata['link_ppts'])
link_mbps = link_ppts * 10  # 1 ppt = 10 Mbps

# ── Load Learnability paper trees ────────────────────────────────
learn_path = _REPO_ROOT / 'results' / 'learnability_eval.json'
if learn_path.exists():
    with open(learn_path) as f:
        learn = json.load(f)
    for method_name, method_data in learn['results'].items():
        key = f'Learn_{method_name}'
        results[key] = method_data
    print(f"Loaded Learnability trees: {list(learn['results'].keys())}")


def get_scores(method_name: str) -> np.ndarray:
    """Extract normalized_score array for a method."""
    return np.array([pt['normalized_score'] for pt in results[method_name]])


# ── Load PPO data ─────────────────────────────────────────────────

ppo_dir = _REPO_ROOT / 'results' / 'ppo-evals'
ppo_brains = {}
for fp in sorted(ppo_dir.glob('brain-*.csv')):
    brain_id = fp.stem
    rows = []
    with open(fp) as f:
        for row in csv.reader(f):
            rows.append([float(x) for x in row])
    ppo_brains[brain_id] = {
        'link_ppts': np.array([r[0] for r in rows]),
        'scores': np.array([r[1] for r in rows]),
    }

for brain_id, data in ppo_brains.items():
    # PPO CSVs already store per-sender average (verified numerically)
    data['normalized'] = data['scores']

ppo_all_interp = []
for brain_id, data in ppo_brains.items():
    interp_scores = np.interp(
        np.log10(link_ppts),
        np.log10(data['link_ppts']),
        data['normalized'],
    )
    ppo_all_interp.append(interp_scores)

ppo_all_interp = np.array(ppo_all_interp)
ppo_mean = np.mean(ppo_all_interp, axis=0)
ppo_std = np.std(ppo_all_interp, axis=0)


# ── Load single-point AlphaCC ─────────────────────────────────────

singlept_path = _REPO_ROOT / 'results' / 'alphacc_singlepoint_dense.json'
singlept_data = None
if singlept_path.exists():
    with open(singlept_path) as f:
        singlept_data = json.load(f)
    singlept_mbps = np.array([p['link_mbps'] for p in singlept_data])
    singlept_scores = np.array([p['normalized_score'] for p in singlept_data])
    print(f"Loaded single-point AlphaCC: {len(singlept_data)} points")


# ── Identify AlphaCC runs (multipoint diverse-seed) ──────────────

primary_alphacc = 'AlphaCC_pacing_r1'
alphacc_systematic = [k for k in results if k.startswith('AlphaCC_')]


# ── Plot (2-panel) ────────────────────────────────────────────────

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

TICK_MBPS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
XLIM = (link_mbps[0] * 0.9, link_mbps[-1] * 1.1)
YLIM = (-8, 1.0)


def format_ax(ax, xlabel=True):
    ax.set_xscale('log')
    if xlabel:
        ax.set_xlabel('Link Rate (Mbps)', fontsize=10)
    ax.set_xticks(TICK_MBPS)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.grid(True, alpha=0.2, which='both')


# ── Panel (a): Three representations head-to-head ────────────────
ax = ax_a
ax.set_title('(a) Three representations', fontsize=10, fontweight='bold')
ax.set_ylabel('Normalized Score  (higher is better)', fontsize=10)

# Training point
ax.axvline(10.0, color='#aaa', ls=':', lw=0.8, alpha=0.5, zorder=0)
ax.text(10.0, 0.6, 'trained\nhere', ha='center', fontsize=7, color='#888',
        style='italic')

# Remy 1x — decision tree
if 'Remy_1x' in results:
    ax.plot(link_mbps, get_scores('Remy_1x'), color='#08306B', ls='-', lw=2.8,
            marker='s', markersize=5, markevery=4,
            label='Remy (decision tree)', zorder=5)

# PPO — neural network
ax.plot(link_mbps, ppo_mean, color='#e74c3c', ls='-', lw=2.8,
        marker='o', markersize=5, markevery=4,
        label='PPO (neural net)', zorder=5)
ax.fill_between(link_mbps, ppo_mean - ppo_std, ppo_mean + ppo_std,
                color='#e74c3c', alpha=0.10, zorder=1)

# AlphaCC single-point — source code
if singlept_data is not None:
    ax.plot(singlept_mbps, singlept_scores, color='#7B1FA2', ls='-', lw=2.8,
            marker='^', markersize=5, markevery=4,
            label='AlphaCC (source code)', zorder=5)

ax.legend(loc='lower left', fontsize=8.5, framealpha=0.95)
format_ax(ax)


# ── Panel (b): Generalization with training breadth ──────────────
ax = ax_b
ax.set_title('(b) Training breadth + baselines', fontsize=10, fontweight='bold')

# Training region shading
ORIG_TRAIN_LO = 2.24
ORIG_TRAIN_HI = 44.7
ax.axvspan(ORIG_TRAIN_LO, ORIG_TRAIN_HI, alpha=0.06, color='#3498db', zorder=0)

# Remy variants
remy_cfg = [
    ('Remy_1x',  'Remy 1x',   '#08306B', '-',  2.2, 's', 4, 5),
    ('Remy_10x', 'Remy 10x',  '#2171B5', '--', 1.6, None, 0, 0),
    ('Remy_20x', 'Remy 20x',  '#6BAED6', '-.', 1.4, None, 0, 0),
]
for key, label, color, ls, lw, marker, ms, me in remy_cfg:
    if key in results:
        kwargs = dict(color=color, ls=ls, lw=lw, label=label, zorder=4)
        if marker:
            kwargs.update(marker=marker, markersize=ms, markevery=me)
        ax.plot(link_mbps, get_scores(key), **kwargs)

# Remy 1000x omitted: no verified .dna tree file in whisker-trees/learnability/
# (only 2x and 100x present; 1000x data in learnability_eval.json is unverified)

# AlphaCC multi-point
if primary_alphacc in results:
    ax.plot(link_mbps, get_scores(primary_alphacc), color='#7B1FA2', ls='--', lw=2.2,
            marker='D', markersize=4, markevery=5,
            label='AlphaCC 6-rate', zorder=5)

# Faint AlphaCC runs
for name in alphacc_systematic:
    if name == primary_alphacc:
        continue
    ax.plot(link_mbps, get_scores(name), color='#d8b4e8', alpha=0.10, lw=0.4, zorder=1)

# Baselines (subdued)
baseline_styles = {
    'Copa':  {'color': '#27ae60', 'ls': '-',  'lw': 1.0, 'alpha': 0.45},
    'AIMD':  {'color': '#95a5a6', 'ls': '--', 'lw': 1.0, 'alpha': 0.45},
    'BBR':   {'color': '#f39c12', 'ls': ':',  'lw': 1.0, 'alpha': 0.45},
}
for name, style in baseline_styles.items():
    if name in results:
        ax.plot(link_mbps, get_scores(name), label=name, zorder=2, **style)

ax.legend(loc='lower left', fontsize=7.5, framealpha=0.95, ncol=2)
format_ax(ax)

plt.tight_layout()
plt.savefig(OUT / 'generalization.png', dpi=200, bbox_inches='tight')
plt.savefig(OUT / 'generalization.pdf', bbox_inches='tight')
plt.close()
print(f"Saved generalization to {OUT}")


# ── Print summary statistics ──────────────────────────────────────

print("\n=== Summary at key link rates ===")
key_mbps = [1, 2, 5, 10, 25, 50, 100, 500, 1000]
header = f"{'Mbps':>6}"
method_names = ['Remy_1x', 'Remy_10x', 'Remy_20x',
                'Learn_Remy_1000x',
                primary_alphacc, 'Copa', 'AIMD', 'BBR']
display = {'Remy_1x': 'Remy1x', 'Remy_10x': 'Remy10x', 'Remy_20x': 'Remy20x',
           'Learn_Remy_1000x': 'R1000x',
           primary_alphacc: 'AlphaCC', 'Copa': 'Copa', 'AIMD': 'AIMD', 'BBR': 'BBR'}
for m in method_names:
    header += f" | {display.get(m, m):>8}"
header += f" | {'PPO':>8}"
print(header)
print("-" * len(header))

for mbps in key_mbps:
    target_ppt = mbps / 10.0
    idx = np.argmin(np.abs(link_ppts - target_ppt))
    row = f"{link_mbps[idx]:6.1f}"
    for m in method_names:
        if m in results:
            score = results[m][idx]['normalized_score']
            row += f" | {score:8.2f}"
        else:
            row += f" | {'N/A':>8}"
    row += f" | {ppo_mean[idx]:8.2f}"
    print(row)

# Best method at each evaluation point
print(f"\n=== Best method at each of {len(link_ppts)} link rates ===")
all_methods = list(results.keys()) + ['PPO_mean']
win_counts = {m: 0 for m in all_methods}
for idx in range(len(link_ppts)):
    best_score = -1e9
    best_name = ''
    for m in results:
        s = results[m][idx]['normalized_score']
        if s > best_score:
            best_score = s
            best_name = m
    if ppo_mean[idx] > best_score:
        best_name = 'PPO_mean'
    win_counts[best_name] = win_counts.get(best_name, 0) + 1

for m, count in sorted(win_counts.items(), key=lambda x: -x[1]):
    if count > 0:
        print(f"  {m}: {count}/{len(link_ppts)} points")
