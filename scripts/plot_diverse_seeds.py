"""
Plot diverse-seed evolution results for the paper.

Generates 2 figures:
1. seed_lineplot: Seed-family performance across link rates
2. seed_boxplot: Box plot of 9-rate means by seed family

Output: paper/figures/seed_lineplot.{pdf,png}, seed_boxplot.{pdf,png}
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pathlib import Path

FAMILY_COLORS = {
    'pacing': '#1f77b4',    # blue
    'constant': '#2ca02c',  # green
    'copa': '#d62728',      # red
    'aimd': '#ff7f0e',      # orange
}

FAMILY_MARKERS = {
    'pacing': 's',
    'constant': 'D',
    'copa': '^',
    'aimd': 'v',
}

FAMILY_RUNS = {
    'pacing': ['pacing_r1', 'pacing_r2', 'pacing_r3'],
    'constant': ['constant_r1', 'constant_r2', 'constant_r3'],
    'copa': ['copa_r1', 'copa_r2', 'copa_r3'],
    'aimd': ['aimd_r1', 'aimd_r2', 'aimd_r3'],
}


def main():
    base = Path(__file__).resolve().parent.parent
    results_dir = base / 'results'
    out_dir = base / 'paper/figures'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load diverse-seed results
    with open(results_dir / 'diverse_seeds/diverse_seed_results.json') as f:
        ds = json.load(f)

    # Load Remy Python-sim scores (keys are in ppt)
    with open(results_dir / 'remy_python_eval.json') as f:
        remy_py = json.load(f)

    # Get link rates from first run
    # NOTE: diverse_seed keys are already in Mbps (NOT ppt)
    first_run = list(ds['runs'].values())[0]
    link_rates = sorted(first_run.keys(), key=float)
    link_mbps = np.array([float(lr) for lr in link_rates])  # already Mbps

    # Remy scores at these rates (remy_py keys are ppt = Mbps / 10)
    remy_scores = []
    for lr in link_rates:
        lr_ppt = float(lr) / 10.0  # convert Mbps to ppt
        best_key = min(remy_py.keys(), key=lambda k: abs(float(k) - lr_ppt))
        remy_scores.append(remy_py[best_key]['mean'])
    remy_scores = np.array(remy_scores)

    # Extract per-run scores
    run_scores = {}
    run_means = {}
    for run_name, scores in ds['runs'].items():
        s = np.array([scores[lr] for lr in link_rates])
        run_scores[run_name] = s
        run_means[run_name] = np.mean(s)

    # Exploratory run removed — evaluated under different simulator version,
    # scores are not comparable. All paper results use current simulator only.
    explr_scores = None

    # ── Figure 1: Seed-family performance across link rates ──
    fig, ax = plt.subplots(figsize=(8, 5))

    # Remy reference
    ax.plot(link_mbps, remy_scores, 'k-o', linewidth=2.5, markersize=6,
            label='Remy 1x', zorder=10)

    # Per-family: median as thick line, individual runs as faint background
    for family, runs in FAMILY_RUNS.items():
        color = FAMILY_COLORS[family]
        marker = FAMILY_MARKERS[family]
        family_scores = np.array([run_scores[r] for r in runs])
        median = np.median(family_scores, axis=0)

        # Individual runs (very faint — just show spread exists)
        for i, run in enumerate(runs):
            ax.plot(link_mbps, run_scores[run], color=color, linewidth=0.5,
                    alpha=0.12, zorder=1)

        # Family median (thick, with distinct markers)
        ax.plot(link_mbps, median, color=color, linewidth=2.8,
                marker=marker, markersize=6,
                label=f'{family.capitalize()} median (n=3)', zorder=7)

    # Exploratory run (dashed gray, not reproduced)
    if explr_scores is not None:
        ax.plot(link_mbps, explr_scores, color='gray', linewidth=1.5,
                linestyle='--', marker='x', markersize=5, alpha=0.6,
                label='Exploratory (not reproduced)', zorder=5)

    ax.set_xscale('log')
    ax.set_xlabel('Link Rate (Mbps)', fontsize=11)
    ax.set_ylabel('Normalized Score  (higher is better)', fontsize=11)
    ax.set_title('Diverse-Seed AlphaCC vs Remy\n(2 senders, RTT=150ms, 6-rate multipoint training)',
                 fontsize=11)
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.25, which='both')
    ax.set_xlim(link_mbps[0] * 0.8, link_mbps[-1] * 1.2)
    ax.set_ylim(-9, 0.5)
    tick_mbps = [2, 5, 10, 20, 50, 100]
    ax.set_xticks(tick_mbps)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(out_dir / f'seed_lineplot.{ext}',
                    dpi=200 if ext == 'png' else None, bbox_inches='tight')
    plt.close()
    print(f"Saved seed_lineplot to {out_dir}")

    # ── Figure 2: Seed comparison box plot ──
    fig, ax = plt.subplots(figsize=(6, 4))

    box_data = []
    box_labels = []
    box_colors = []
    for family in ['pacing', 'constant', 'copa', 'aimd']:
        means = [run_means[r] for r in FAMILY_RUNS[family]]
        box_data.append(means)
        box_labels.append(family.capitalize())
        box_colors.append(FAMILY_COLORS[family])

    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                    widths=0.5, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='white',
                                   markeredgecolor='black', markersize=6))
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    # Individual points
    for i, (family, runs) in enumerate(FAMILY_RUNS.items()):
        means = [run_means[r] for r in runs]
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(means))
        ax.scatter([i + 1 + j for j in jitter], means,
                   color=FAMILY_COLORS[family], s=40, zorder=5, edgecolor='black',
                   linewidth=0.5)

    # Remy reference line
    remy_mean = np.mean(remy_scores)
    ax.axhline(y=remy_mean, color='black', linestyle='--', linewidth=1.5,
               label=f'Remy ({remy_mean:.2f})')

    # Exploratory run reference
    if explr_scores is not None:
        ax.axhline(y=np.mean(explr_scores), color='gray', linestyle=':',
                   linewidth=1, label=f'Exploratory ({np.mean(explr_scores):.2f})')

    ax.set_ylabel('9-Rate Mean Normalized Score', fontsize=11)
    ax.set_title('Seed Family Comparison', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.25, axis='y')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(out_dir / f'seed_boxplot.{ext}',
                    dpi=200 if ext == 'png' else None, bbox_inches='tight')
    plt.close()
    print(f"Saved seed_boxplot to {out_dir}")

    # ── Print summary ──
    print("\n=== Diverse-Seed Results Summary ===")
    print(f"{'Run':>14}  {'Family':>8}  {'9-rate mean':>11}  {'Δ vs Remy':>9}")
    sorted_runs = sorted(run_means.items(), key=lambda x: x[1], reverse=True)
    for run, mean in sorted_runs:
        family = run.rsplit('_', 1)[0]
        delta = mean - remy_mean
        print(f"{run:>14}  {family:>8}  {mean:11.3f}  {delta:+9.3f}")

    print(f"\nRemy 9-rate mean: {remy_mean:.3f}")
    print(f"\nFamily medians:")
    for family in ['pacing', 'constant', 'copa', 'aimd']:
        means = sorted([run_means[r] for r in FAMILY_RUNS[family]])
        print(f"  {family:>8}: median={means[1]:.3f}, range=[{means[0]:.3f}, {means[2]:.3f}]")


if __name__ == '__main__':
    main()
