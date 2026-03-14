"""
Plot evolution trajectory for all 12 diverse-seed runs.

Output: paper/figures/evolution_trajectory.{pdf,png}
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

FAMILY_COLORS = {
    'pacing': '#1f77b4',
    'constant': '#2ca02c',
    'copa': '#d62728',
    'aimd': '#ff7f0e',
}

FAMILY_MARKERS = {
    'pacing': 'o',
    'constant': 's',
    'copa': '^',
    'aimd': 'D',
}

FAMILY_RUNS = {
    'pacing': ['pacing_r1', 'pacing_r2', 'pacing_r3'],
    'constant': ['constant_r1', 'constant_r2', 'constant_r3'],
    'copa': ['copa_r1', 'copa_r2', 'copa_r3'],
    'aimd': ['aimd_r1', 'aimd_r2', 'aimd_r3'],
}

def main():
    base = Path(__file__).resolve().parent.parent
    results_dir = base / 'results' / 'diverse_seeds'
    out_dir = base / 'paper' / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    global_best_by_gen = {}  # gen -> best fitness so far
    all_gens_fits = []  # collect all candidate dots to plot once

    for family, runs in FAMILY_RUNS.items():
        color = FAMILY_COLORS[family]
        marker = FAMILY_MARKERS[family]
        for ri, run_name in enumerate(runs):
            hist_path = results_dir / run_name / 'history.json'
            if not hist_path.exists():
                print(f"  WARNING: {hist_path} not found, skipping")
                continue
            with open(hist_path) as f:
                history = json.load(f)

            # Collect candidate dots (plotted once below)
            for e in history:
                all_gens_fits.append((e['gen'], e['fitness']))

            # Running best per run
            best_so_far = -100
            gen_best = {}
            for e in history:
                if e['fitness'] > best_so_far:
                    best_so_far = e['fitness']
                gen_best[e['gen']] = best_so_far
            sorted_gens = sorted(gen_best.keys())
            best_vals = [gen_best[g] for g in sorted_gens]

            label = f'{family.capitalize()}' if ri == 0 else None
            ax.plot(sorted_gens, best_vals, color=color, linewidth=2.0,
                    alpha=0.85, label=label, zorder=5,
                    marker=marker, markersize=4, markevery=3)

            # Track global best
            for g, v in zip(sorted_gens, best_vals):
                if g not in global_best_by_gen or v > global_best_by_gen[g]:
                    global_best_by_gen[g] = v

    # Plot all candidate dots once (lighter, smaller)
    if all_gens_fits:
        g_arr = [x[0] for x in all_gens_fits]
        f_arr = [x[1] for x in all_gens_fits]
        ax.scatter(g_arr, f_arr, color='#b0b0b0', s=4, alpha=0.15, zorder=1,
                   rasterized=True, edgecolors='none')

    # Global running best
    if global_best_by_gen:
        sorted_gens = sorted(global_best_by_gen.keys())
        global_vals = []
        best = -100
        for g in sorted_gens:
            best = max(best, global_best_by_gen[g])
            global_vals.append(best)
        ax.plot(sorted_gens, global_vals, 'k-', linewidth=3.0,
                marker='*', markersize=7, markevery=2,
                label='Global best', zorder=10)

    ax.set_xlabel('Generation', fontsize=11)
    ax.set_ylabel('Training Fitness (6-rate mean)', fontsize=11)
    ax.set_title('AlphaCC Evolution Trajectory\n(4 seed families × 3 reps = 12 runs)',
                 fontsize=11)
    ax.set_ylim(-8, 0.5)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.95,
              handlelength=2.5, borderpad=0.8)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(out_dir / f'evolution_trajectory.{ext}',
                    dpi=200 if ext == 'png' else None, bbox_inches='tight')
    plt.close()
    print(f"Saved evolution_trajectory to {out_dir}")


if __name__ == '__main__':
    main()
