"""
Standalone AlphaCC figure using dense (50-point) evaluation data.

Shows:
- Remy 1x (black, thick)
- AlphaCC single-point (purple, dashed)
- All 12 multipoint runs (thin by family color) with family median (thick)
- Shaded envelope per family

Output: paper/figures/alphacc_standalone.{pdf,png}
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pathlib import Path

FAMILY_COLORS = {
    'pacing': '#1f77b4',
    'constant': '#2ca02c',
    'copa': '#d62728',
    'aimd': '#ff7f0e',
}

FAMILY_RUNS = {
    'pacing': ['pacing_r1', 'pacing_r2', 'pacing_r3'],
    'constant': ['constant_r1', 'constant_r2', 'constant_r3'],
    'copa': ['copa_r1', 'copa_r2', 'copa_r3'],
    'aimd': ['aimd_r1', 'aimd_r2', 'aimd_r3'],
}

FAMILY_LABELS = {
    'pacing': 'Pacing (rate-based)',
    'constant': 'Constant (window-based)',
    'copa': 'Copa (delay-based)',
    'aimd': 'AIMD (loss-based)',
}


def extract_scores(results_list):
    return np.array([r['normalized_score'] for r in results_list])


def main():
    base = Path(__file__).resolve().parent.parent
    results_dir = base / 'results'
    out_dir = base / 'paper' / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dense eval (50-point)
    with open(results_dir / 'dense_eval.json') as f:
        dense = json.load(f)

    link_ppts = np.array(dense['metadata']['link_ppts'])
    link_mbps = link_ppts * 10

    remy_1x = extract_scores(dense['results']['Remy_1x'])

    # Single-point AlphaCC
    sp_path = results_dir / 'alphacc_singlepoint_dense.json'
    sp_scores = None
    if sp_path.exists():
        with open(sp_path) as f:
            sp_raw = json.load(f)
        sp_scores = extract_scores(sp_raw)

    # Per-run scores from dense eval
    run_scores = {}
    for family, runs in FAMILY_RUNS.items():
        for run_name in runs:
            key = f'AlphaCC_{run_name}'
            if key in dense['results']:
                run_scores[run_name] = extract_scores(dense['results'][key])

    # ── Main figure ──
    fig, ax = plt.subplots(figsize=(10, 6))

    # Remy reference
    ax.plot(link_mbps, remy_1x, 'k-', linewidth=2.5,
            label='Remy 1x (decision tree)', zorder=10)

    # Single-point AlphaCC
    if sp_scores is not None:
        ax.plot(link_mbps, sp_scores, color='#9467bd', linewidth=2,
                linestyle='--',
                label='AlphaCC single-point (10 Mbps)', zorder=9)

    # Per-family: individual runs (thin) + median (thick) + shaded envelope
    for family, runs in FAMILY_RUNS.items():
        color = FAMILY_COLORS[family]
        available = [run_scores[r] for r in runs if r in run_scores]
        if not available:
            continue
        family_arr = np.array(available)
        median = np.median(family_arr, axis=0)
        lo = np.min(family_arr, axis=0)
        hi = np.max(family_arr, axis=0)

        # Shaded range
        ax.fill_between(link_mbps, lo, hi, color=color, alpha=0.08, zorder=2)

        # Individual runs (thin)
        for i, run in enumerate(runs):
            if run in run_scores:
                ax.plot(link_mbps, run_scores[run], color=color, linewidth=0.6,
                        alpha=0.35, zorder=3)

        # Family median (thick)
        ax.plot(link_mbps, median, color=color, linewidth=2.2,
                label=f'{FAMILY_LABELS[family]} median', zorder=7)

    # Training rates annotation (vertical dotted lines)
    train_mbps = [2.37, 5.96, 15.0, 37.73, 59.83, 94.9]
    for i, tr in enumerate(train_mbps):
        ax.axvline(x=tr, color='gray', linewidth=0.4, linestyle=':', alpha=0.4)
    # Label one of them
    ax.annotate('training\nrates', xy=(15, -8.5), fontsize=7, color='gray',
                ha='center', style='italic')

    ax.set_xscale('log')
    ax.set_xlabel('Link Rate (Mbps)', fontsize=12)
    ax.set_ylabel('Normalized Score  (higher is better)', fontsize=12)
    ax.set_title('AlphaCC: LLM-Guided Evolution of Congestion Control\n'
                 '4 seed families × 3 reps = 12 runs, 6-rate multipoint training',
                 fontsize=12)
    ax.legend(loc='lower left', fontsize=8.5, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.2, which='both')
    ax.set_xlim(1, 1000)
    ax.set_ylim(-10, 1)

    tick_mbps = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    ax.set_xticks(tick_mbps)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(out_dir / f'alphacc_standalone.{ext}',
                    dpi=200 if ext == 'png' else None, bbox_inches='tight')
    plt.close()
    print(f"Saved alphacc_standalone to {out_dir}")


if __name__ == '__main__':
    main()
