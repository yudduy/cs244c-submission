#!/usr/bin/env python3
"""Plot the strongest preserved multi-point AlphaCC run against the single-point run."""

import json
import os
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.ticker import ScalarFormatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PY = REPO_ROOT / "results" / "python_sim"
PAPER_FIGS = Path(os.environ.get("PAPER_FIGS_DIR", str(REPO_ROOT / "paper" / "figures")))
PAPER_FIGS.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path):
    with path.open() as fh:
        return json.load(fh)


def _find_key(dct, target):
    return min(dct.keys(), key=lambda key: abs(float(key) - float(target)))


def main():
    multiseed = _load_json(RESULTS_PY / "output_remy_evolve" / "multiseed_eval.json")
    single = multiseed["LLM"]
    best_multipoint = _load_json(RESULTS_PY / "output_remy_evolve_multipoint" / "generalization.json")

    link_keys = sorted(multiseed["Remy"].keys(), key=float)
    link_mbps = np.array([float(key) * 10.0 for key in link_keys])
    remy = np.array([multiseed["Remy"][key]["mean"] for key in link_keys])
    ppo = np.array([multiseed["PPO"][key]["mean"] for key in link_keys])
    copa = np.array([multiseed["Copa"][key]["mean"] for key in link_keys])
    aimd = np.array([multiseed["AIMD"][key]["mean"] for key in link_keys])
    alpha_single = np.array([single[key]["mean"] for key in link_keys])
    alpha_multi = np.array([best_multipoint[_find_key(best_multipoint, key)]["normalized"] for key in link_keys])

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.plot(link_mbps, remy, color="#1f77b4", marker="o", linewidth=2.5, label="Remy 1x tree", zorder=6)
    ax.plot(link_mbps, ppo, color="#d64d4d", marker="s", linewidth=1.8, label="PPO", zorder=5)
    ax.plot(link_mbps, copa, color="#2b8a3e", marker="D", linewidth=1.6, label="Copa", zorder=3)
    ax.plot(link_mbps, aimd, color="#6c757d", marker="^", linewidth=1.6, label="AIMD", zorder=3)
    ax.plot(link_mbps, alpha_single, color="#7c3aed", marker="*", linewidth=1.8, linestyle="--", markersize=9, label="AlphaCC single-point", zorder=4)
    ax.plot(link_mbps, alpha_multi, color="#5b21b6", marker="P", linewidth=2.4, markersize=7, label="AlphaCC multipoint (best preserved run)", zorder=7)

    ax.axvline(x=10, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(10, -0.2, "train", color="gray", ha="center", va="bottom", fontsize=9)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks([2.4, 5, 10, 20, 50, 95])
    ax.set_xlabel("Link speed (Mbps)")
    ax.set_ylabel("Normalized score")
    ax.set_title("Single-point vs. Strongest Preserved Multi-point AlphaCC Run")
    ax.set_ylim(-5.2, 0.4)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95, ncol=2)
    plt.tight_layout()
    plt.savefig(PAPER_FIGS / "fig5_multipoint.pdf", bbox_inches="tight")
    plt.savefig(PAPER_FIGS / "fig5_multipoint.png", dpi=180, bbox_inches="tight")
    plt.close()

    print(f"Wrote {PAPER_FIGS / 'fig5_multipoint.pdf'}")


if __name__ == "__main__":
    main()
