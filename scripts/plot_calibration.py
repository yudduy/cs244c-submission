#!/usr/bin/env python3
"""Plot archived C++ tree scores alongside the preserved Python baselines."""

import json
import os
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.ticker import ScalarFormatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = REPO_ROOT / "results" / "calibration" / "all_trees_on1000.txt"
MULTISEED = REPO_ROOT / "results" / "python_sim" / "output_remy_evolve" / "multiseed_eval.json"
PAPER_FIGS = Path(os.environ.get("PAPER_FIGS_DIR", str(REPO_ROOT / "paper" / "figures")))
PAPER_FIGS.mkdir(parents=True, exist_ok=True)


def parse_blocks():
    trees = {}
    for block in CALIBRATION.read_text().split("--- "):
        if not block.strip():
            continue
        name = block.split(" ")[0].strip()
        rows = []
        for line in block.splitlines()[2:]:
            if not line.strip():
                continue
            parts = line.split(",")
            rows.append((float(parts[0]) * 10.0, float(parts[1]) / 2.0))
        trees[name] = rows
    return trees


def main():
    trees = parse_blocks()
    with MULTISEED.open() as fh:
        multiseed = json.load(fh)

    link_keys = sorted(multiseed["Copa"].keys(), key=float)
    link_mbps = np.array([float(key) * 10.0 for key in link_keys])

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    styles = {
        "1x": ("#1f77b4", "o"),
        "10x": ("#4f83cc", "D"),
        "20x": ("#76a9ea", "v"),
        "100x": ("#0f4c81", "*"),
    }
    for tree, rows in trees.items():
        if tree not in styles:
            continue
        color, marker = styles[tree]
        xs = [row[0] for row in rows]
        ys = [row[1] for row in rows]
        ax.plot(xs, ys, color=color, marker=marker, linewidth=2.0, label=f"Remy {tree}")

    for name, color, marker in [("Copa", "#2b8a3e", "D"), ("AIMD", "#6c757d", "^"), ("LLM", "#7c3aed", "*")]:
        means = np.array([multiseed[name][key]["mean"] for key in link_keys])
        ax.plot(link_mbps, means, color=color, marker=marker, linewidth=1.8, label=name)

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks([2.4, 5, 10, 20, 50, 95])
    ax.set_xlabel("Link speed (Mbps)")
    ax.set_ylabel("Normalized score per sender")
    ax.set_title("Archived C++ Trees vs. Preserved Python Baselines")
    ax.grid(True, alpha=0.25)
    ax.legend(framealpha=0.95, fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(PAPER_FIGS / "fig_calibration.pdf", bbox_inches="tight")
    plt.savefig(PAPER_FIGS / "fig_calibration.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Wrote {PAPER_FIGS / 'fig_calibration.pdf'}")


if __name__ == "__main__":
    main()
