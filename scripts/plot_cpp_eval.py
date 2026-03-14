#!/usr/bin/env python3
"""Figure 1: C++ Simulator Evaluation — Remy trees + PPO brains."""

import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "paper/figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Parse Remy data
# ---------------------------------------------------------------------------
remy_file = ROOT / "results/calibration/all_trees_on1000.txt"
remy_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

current_label = None
link_ppts: list[float] = []
scores: list[float] = []

for line in remy_file.read_text().splitlines():
    line = line.strip()
    if line.startswith("---"):
        # Save previous block; divide scores by 2 (file stores aggregate sum over 2 senders)
        if current_label and link_ppts:
            remy_data[current_label] = (np.array(link_ppts), np.array(scores) / 2.0)
        # Extract label: e.g. "1x", "100x"
        current_label = line.split("(")[0].replace("-", "").strip()
        link_ppts, scores = [], []
    elif line.startswith("link_ppt") or not line:
        continue
    else:
        parts = line.split(",")
        link_ppts.append(float(parts[0]))
        scores.append(float(parts[1]))

if current_label and link_ppts:
    remy_data[current_label] = (np.array(link_ppts), np.array(scores) / 2.0)

# ---------------------------------------------------------------------------
# 2. Parse PPO brain CSVs
# ---------------------------------------------------------------------------
ppo_dir = ROOT / "results/ppo-evals"
brain_csvs = sorted(ppo_dir.glob("brain-*.csv"))

all_ppo: list[tuple[np.ndarray, np.ndarray]] = []
for csv_path in brain_csvs:
    rows = [l.strip().split(",") for l in csv_path.read_text().splitlines() if l.strip()]
    lppt = np.array([float(r[0]) for r in rows])
    # PPO CSVs already store per-sender average (verified numerically)
    score = np.array([float(r[1]) for r in rows])
    all_ppo.append((lppt, score))

# Compute mean ± std across brains (aligned on same link_ppt grid)
ppo_link = all_ppo[0][0]  # same grid for all brains
ppo_matrix = np.stack([s for _, s in all_ppo])  # (6, 9)
ppo_mean = ppo_matrix.mean(axis=0)
ppo_std = ppo_matrix.std(axis=0)

# ---------------------------------------------------------------------------
# 3. Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

# Convert link_ppt to Mbps: link_ppt * 10
def to_mbps(lppt: np.ndarray) -> np.ndarray:
    return lppt * 10.0

# Remy 1x — prominent solid blue
lppt, sc = remy_data["1x"]
ax.plot(to_mbps(lppt), sc, color="#2166ac", linewidth=2.2, marker="o",
        markersize=5, label="Remy 1× (179 rules)", zorder=5)

# Remy 100x — dashed dark blue
lppt, sc = remy_data["100x"]
ax.plot(to_mbps(lppt), sc, color="#4393c3", linewidth=1.8, linestyle="--",
        marker="s", markersize=4, label="Remy 100× (Learnability)", zorder=4)

# PPO — red with shaded band
ax.plot(to_mbps(ppo_link), ppo_mean, color="#d6604d", linewidth=2.0,
        marker="D", markersize=4, label="PPO (6 seeds, ±1σ)", zorder=6)
ax.fill_between(to_mbps(ppo_link), ppo_mean - ppo_std, ppo_mean + ppo_std,
                color="#d6604d", alpha=0.18, zorder=2)

# Training point vertical line
ax.axvline(x=10.0, color="#888888", linewidth=1.0, linestyle=":", alpha=0.6,
           zorder=1, label="_nolegend_")

# Axes
ax.set_xscale("log")
ax.set_xlabel("Link Rate (Mbps)", fontsize=11)
ax.set_ylabel("Normalized Score (per sender)", fontsize=11)

# X-ticks at actual link rates
mbps_ticks = to_mbps(ppo_link)
ax.set_xticks(mbps_ticks)
ax.set_xticklabels([f"{v:.1f}" for v in mbps_ticks], fontsize=9)
ax.tick_params(axis="y", labelsize=9)

# Grid
ax.grid(True, which="major", linewidth=0.4, alpha=0.5)
ax.set_axisbelow(True)

# Training point annotation — place at bottom right of vertical line
ylo, yhi = ax.get_ylim()
ax.text(11.0, ylo + 0.05 * (yhi - ylo), "training\npoint", fontsize=7.5,
        color="#777777", va="bottom", ha="left", style="italic")

# Legend
ax.legend(fontsize=9.5, loc="upper right", framealpha=0.9, edgecolor="#cccccc")

fig.tight_layout()

# Save
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_cpp_eval.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved {OUT / f'fig_cpp_eval.{ext}'}")

plt.close(fig)
