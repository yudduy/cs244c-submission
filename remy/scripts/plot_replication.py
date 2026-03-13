"""Overlay whisker trees + PPO brains (on=5000 eval data)."""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

LINK_PPT_TO_MBPS = 10.0


def load_csv(filepath):
    rows = []
    with open(filepath) as f:
        for row in csv.reader(f):
            rows.append([float(x) for x in row])
    return np.array(rows)


def main():
    base = Path(__file__).resolve().parent.parent
    eval_dir = base / 'results/eval-csv'
    brain_dir = base / 'results/ppo-brains'
    out_dir = base / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)

    remy_1x = load_csv(eval_dir / 'plot-1x-2src/data/data-cca.179.csv')
    remy_10x = load_csv(eval_dir / 'plot-10x-2src/data/data-cca.36.csv')
    remy_20x = load_csv(eval_dir / 'plot-20x-2src/data/data-cca.19.csv')
    link_mbps = remy_1x[:, 0] * LINK_PPT_TO_MBPS

    brain_csvs = sorted(brain_dir.glob('*/data/data-brain.*.csv'))
    if not brain_csvs:
        print("No brain CSVs found")
        return
    n_points = len(link_mbps)
    brain_scores = []
    for bf in brain_csvs:
        data = load_csv(bf)
        if len(data) != n_points:
            continue  # skip incomplete runs
        brain_scores.append(data[:, 1])
    brain_scores = np.array(brain_scores)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(link_mbps, remy_1x[:, 1], 'b-o', linewidth=2.5, markersize=7,
            label='RemyCC 1x (179 iter)', zorder=6)
    ax.plot(link_mbps, remy_10x[:, 1], 'b--D', linewidth=1.5, markersize=5,
            label='RemyCC 10x (36 iter)', zorder=5, alpha=0.7)
    ax.plot(link_mbps, remy_20x[:, 1], 'b:v', linewidth=1.5, markersize=5,
            label='RemyCC 20x (19 iter)', zorder=5, alpha=0.7)

    ppo_mean = brain_scores.mean(axis=0)
    ppo_lo = brain_scores.min(axis=0)
    ppo_hi = brain_scores.max(axis=0)
    ax.plot(link_mbps, ppo_mean, 'r-s', linewidth=2, markersize=6,
            label=f'PPO (mean of {len(brain_scores)} brains)', zorder=5)
    ax.fill_between(link_mbps, ppo_lo, ppo_hi, color='red', alpha=0.12,
                    label='PPO range')

    ax.axvline(x=15, color='orange', linestyle='--', alpha=0.4, linewidth=1)
    ax.annotate('1x train', xy=(15, -5.5), fontsize=7, ha='center',
                color='orange', alpha=0.7)

    ax.set_xscale('log')
    ax.set_xlabel('Link speed (Mbps)', fontsize=11)
    ax.set_ylabel('Normalized score\n'
                  r'$\log(\mathrm{tput}/C) - \log(\mathrm{delay}/\mathrm{RTT}_{\min})$',
                  fontsize=10)
    ax.set_title('Generalization across link rates (2 senders, RTT=150ms)', fontsize=12)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(1.5, 120)
    ax.set_xticks([2, 5, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        path = out_dir / f'fig_replication.{ext}'
        plt.savefig(path, bbox_inches='tight', dpi=150 if ext == 'png' else None)
        print(f"Saved: {path}")
    plt.close()


if __name__ == '__main__':
    main()
