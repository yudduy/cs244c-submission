#!/usr/bin/env python3
"""Generate the paper figures from archived result files."""

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.ticker import ScalarFormatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PY = REPO_ROOT / "results" / "python_sim" / "output_remy_evolve"
PAPER_FIGS = Path(os.environ.get("PAPER_FIGS_DIR", str(REPO_ROOT / "paper" / "figures")))
PAPER_FIGS.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path):
    with path.open() as fh:
        return json.load(fh)


def _sorted_keys(dct):
    return sorted(dct.keys(), key=float)


def main():
    multiseed = _load_json(RESULTS_PY / "multiseed_eval.json")
    rtt_sweep = _load_json(RESULTS_PY / "rtt_sweep.json")
    history = _load_json(RESULTS_PY / "history.json")

    link_keys = _sorted_keys(multiseed["Remy"])
    link_mbps = [float(key) * 10.0 for key in link_keys]

    policies = {
        "Remy": {"label": "Remy 1x tree", "color": "#1f77b4", "marker": "o", "lw": 2.5, "z": 5},
        "PPO": {"label": "PPO", "color": "#d64d4d", "marker": "s", "lw": 2.0, "z": 4},
        "Copa": {"label": "Copa", "color": "#2b8a3e", "marker": "D", "lw": 1.8, "z": 3},
        "AIMD": {"label": "AIMD", "color": "#6c757d", "marker": "^", "lw": 1.6, "z": 3},
        "LLM": {"label": "AlphaCC", "color": "#7c3aed", "marker": "*", "lw": 2.2, "z": 6},
    }

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for name, style in policies.items():
        means = np.array([multiseed[name][key]["mean"] for key in link_keys])
        stds = np.array([multiseed[name][key].get("std", 0.0) for key in link_keys])
        ax.plot(
            link_mbps,
            means,
            color=style["color"],
            marker=style["marker"],
            linewidth=style["lw"],
            markersize=9 if name == "LLM" else 6,
            label=style["label"],
            zorder=style["z"],
        )
        if np.any(stds > 0):
            ax.fill_between(link_mbps, means - stds, means + stds, color=style["color"], alpha=0.12, zorder=1)

    ax.axvline(x=10, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(10, -0.15, "train", color="gray", ha="center", va="bottom", fontsize=9)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks([2.4, 5, 10, 20, 50, 95])
    ax.set_xlabel("Link speed (Mbps)")
    ax.set_ylabel("Normalized score")
    ax.set_title("Generalization Across Link Rates")
    ax.set_ylim(-5.2, 0.4)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95, ncol=2)
    plt.tight_layout()
    plt.savefig(PAPER_FIGS / "fig1_generalization_v2.pdf", bbox_inches="tight")
    plt.savefig(PAPER_FIGS / "fig1_generalization_v2.png", dpi=180, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    rtts = [50, 150, 300]
    rtt_link_keys = ["0.237", "0.946", "5.983"]
    rtt_labels = ["2.4 Mbps", "9.5 Mbps", "59.8 Mbps"]
    bar_defs = [
        ("Copa", "Copa", "#2b8a3e"),
        ("AIMD", "AIMD", "#6c757d"),
        ("BBR", "BBR", "#d97706"),
        ("LLM_evolved", "AlphaCC", "#7c3aed"),
    ]
    for idx, rtt in enumerate(rtts):
        ax = axes[idx]
        x = np.arange(len(rtt_link_keys))
        width = 0.2
        for bar_idx, (key, label, color) in enumerate(bar_defs):
            vals = [rtt_sweep[key][f"{rtt}_{link}"]["normalized"] for link in rtt_link_keys]
            ax.bar(x + (bar_idx - 1.5) * width, vals, width=width, color=color, label=label if idx == 0 else "")
        ax.set_xticks(x)
        ax.set_xticklabels(rtt_labels)
        ax.set_title(f"RTT = {rtt} ms")
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_ylim(-5.4, 0.2)
        if idx == 0:
            ax.set_ylabel("Normalized score")
    axes[0].legend(loc="lower left", fontsize=9, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(PAPER_FIGS / "fig2_rtt_sweep.pdf", bbox_inches="tight")
    plt.savefig(PAPER_FIGS / "fig2_rtt_sweep.png", dpi=180, bbox_inches="tight")
    plt.close()

    scatter_src_pdf = RESULTS_PY / "fig3_scatter_v2.pdf"
    scatter_src_png = RESULTS_PY / "fig3_scatter_v2.png"
    if scatter_src_pdf.exists():
        shutil.copyfile(scatter_src_pdf, PAPER_FIGS / "fig3_scatter_v2.pdf")
    if scatter_src_png.exists():
        shutil.copyfile(scatter_src_png, PAPER_FIGS / "fig3_scatter_v2.png")

    per_gen = defaultdict(list)
    for item in history:
        per_gen[item["gen"]].append(item["fitness"])
    generations = sorted(per_gen.keys())
    best_per_gen = [max(per_gen[g]) for g in generations]
    running_best = []
    best_so_far = -float("inf")
    for score in best_per_gen:
        best_so_far = max(best_so_far, score)
        running_best.append(best_so_far)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for gen in generations:
        ax.scatter([gen] * len(per_gen[gen]), per_gen[gen], color="0.65", s=18, alpha=0.45, zorder=1)
    ax.plot(generations, best_per_gen, color="#7c3aed", marker="o", linewidth=2.2, label="Best per generation", zorder=3)
    ax.plot(generations, running_best, color="#2b8a3e", linestyle="--", linewidth=2.2, label="Running best", zorder=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Normalized score")
    ax.set_title("AlphaCC Evolution Trajectory")
    ax.grid(True, alpha=0.25)
    ax.legend(framealpha=0.95)
    plt.tight_layout()
    plt.savefig(PAPER_FIGS / "fig4_evolution_v2.pdf", bbox_inches="tight")
    plt.savefig(PAPER_FIGS / "fig4_evolution_v2.png", dpi=180, bbox_inches="tight")
    plt.close()

    print(f"Wrote figures to {PAPER_FIGS}")


if __name__ == "__main__":
    main()
