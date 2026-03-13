"""Plot all Remy trees at on=1000/off=1000 (Keith's canonical params)."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

LINK_PPT_TO_MBPS = 10.0
LINK_PPTS = [0.237, 0.376, 0.596, 0.946, 1.500, 2.379, 3.773, 5.983, 9.490]
LINK_MBPS = [x * LINK_PPT_TO_MBPS for x in LINK_PPTS]


def parse_eval_block(text):
    scores = []
    for line in text.strip().split('\n'):
        if line.startswith('link_ppt') or line.startswith('---') or not line.strip():
            continue
        parts = line.split(',')
        if len(parts) >= 2:
            scores.append(float(parts[1]) / 2)  # per-sender avg
    return scores


def main():
    base = Path(__file__).resolve().parent.parent
    out_dir = base / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_file = base / 'results/calibration/all_trees_on1000.txt'
    if not eval_file.exists():
        print(f"Missing {eval_file}")
        return

    text = eval_file.read_text()
    blocks = text.split('--- ')
    trees = {}
    for block in blocks:
        if not block.strip():
            continue
        name = block.split(' ')[0].strip()
        if name in ('1x', '10x', '20x', '100x'):
            trees[name] = parse_eval_block(block)

    fig, ax = plt.subplots(figsize=(8, 5))

    if '1x' in trees and len(trees['1x']) == 9:
        ax.plot(LINK_MBPS, trees['1x'], 'b-o', linewidth=2.5, markersize=7,
                label='RemyCC 1x (179 iter)', zorder=6)

    if '100x' in trees and len(trees['100x']) == 9:
        ax.plot(LINK_MBPS, trees['100x'], '--*', linewidth=2.5, markersize=9,
                label='RemyCC 100x (published)', zorder=6, color='darkblue')

    if '10x' in trees and len(trees['10x']) == 9:
        ax.plot(LINK_MBPS, trees['10x'], 'c-D', linewidth=1.5, markersize=5,
                label='RemyCC 10x (36 iter, under-trained)', zorder=4, alpha=0.7)

    if '20x' in trees and len(trees['20x']) == 9:
        ax.plot(LINK_MBPS, trees['20x'], 'c:v', linewidth=1.5, markersize=5,
                label='RemyCC 20x (19 iter, under-trained)', zorder=4, alpha=0.5)

    ax.axvline(x=15, color='orange', linestyle='--', alpha=0.3, linewidth=1)
    ax.annotate('1x train', xy=(15, -4.5), fontsize=7, ha='center',
                color='orange', alpha=0.6)

    ax.set_xscale('log')
    ax.set_xlabel('Link speed (Mbps)', fontsize=11)
    ax.set_ylabel('Normalized score (per sender)\n'
                  r'$\frac{1}{N}\sum_i[\log(tp_i/C) - \log(del_i/RTT_{min})]$',
                  fontsize=10)
    ax.set_title('Remy calibration: all trees at on=1000', fontsize=11)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(1.5, 120)
    ax.set_xticks([2, 5, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        path = out_dir / f'fig_calibration.{ext}'
        plt.savefig(path, bbox_inches='tight', dpi=150 if ext == 'png' else None)
        print(f"Saved: {path}")
    plt.close()

    # print table
    print("\n=== Per-sender avg normalized scores (on=1000) ===")
    header = f"{'Mbps':>6}"
    for name in ['1x', '10x', '20x', '100x']:
        if name in trees:
            header += f"  {name:>8}"
    print(header)
    for i, mbps in enumerate(LINK_MBPS):
        row = f"{mbps:6.1f}"
        for name in ['1x', '10x', '20x', '100x']:
            if name in trees and i < len(trees[name]):
                row += f"  {trees[name][i]:8.3f}"
        print(row)


if __name__ == '__main__':
    main()
