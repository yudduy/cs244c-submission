#!/usr/bin/env python3
"""Dense 50-point C++ evaluation across 1-1000 Mbps for the paper.

Evaluates:
  1. AlphaCC (evolved-runner) -- hardcoded policy, no input file
  2. Remy whisker trees (sender-runner) -- 1x, 10x, 20x range trees
  3. PPO brains -- loaded from existing 9-point CSVs (no LibTorch locally)

All evaluations use: rtt=150, on=5000, off=5000, nsrc=2, 50 log-spaced link rates.
Output: per-sender normalized_score (C++ outputs sum; we divide by nsrc).

Convention: link_ppt = Mbps / 10 (1 ppt = 10 Mbps).
  Range: 0.1 ppt (1 Mbps) to 100 ppt (1000 Mbps).
"""

import os
import sys
import csv
import re
import subprocess
import time
import math
import argparse
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
EVOLVED_RUNNER = SRC / "evolved-runner"
SENDER_RUNNER = SRC / "sender-runner"
WHISKER_DIR = ROOT / "results" / "whisker-trees"
PPO_CSV_DIR = ROOT / "results" / "eval-csv" / "combined-plot" / "data"
OUTPUT_DIR = ROOT / "results" / "dense_cpp"

# Evaluation parameters
NUM_SENDERS = 2
RTT = 150
ON = 5000
OFF = 5000
NUM_POINTS = 50
LINK_PPT_MIN = 0.1    # 1 Mbps
LINK_PPT_MAX = 100.0  # 1000 Mbps

# Whisker trees to evaluate (dir_name, tree_file, label)
WHISKER_TREES = [
    ("1x-2src", "cca.179", "remy-1x"),
    ("10x-2src", "cca.36", "remy-10x"),
    ("20x-2src", "cca.19", "remy-20x"),
]

NORM_SCORE_RE = re.compile(r"^normalized_score = (-?\d+(?:\.\d+)?)$", re.MULTILINE)
PPT_TO_MBPS = 10


def log_spaced_link_rates(n=NUM_POINTS, lo=LINK_PPT_MIN, hi=LINK_PPT_MAX):
    """Generate n log-spaced link rates in packets-per-ms."""
    return [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]


def run_evolved(link_ppt):
    """Run evolved-runner at a single link rate. Returns per-sender norm score."""
    cmd = [
        str(EVOLVED_RUNNER),
        f"link={link_ppt}",
        f"rtt={RTT}",
        f"on={ON}",
        f"off={OFF}",
        f"nsrc={NUM_SENDERS}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    m = NORM_SCORE_RE.search(result.stdout)
    if not m:
        print(f"  WARNING: no normalized_score in output for link={link_ppt}", file=sys.stderr)
        print(f"  stdout: {result.stdout[:500]}", file=sys.stderr)
        print(f"  stderr: {result.stderr[:500]}", file=sys.stderr)
        return None
    return float(m.group(1)) / NUM_SENDERS


def run_whisker(tree_path, link_ppt):
    """Run sender-runner with a whisker tree at a single link rate. Returns per-sender norm score."""
    cmd = [
        str(SENDER_RUNNER),
        f"if={tree_path}",
        f"link={link_ppt}",
        f"rtt={RTT}",
        f"on={ON}",
        f"off={OFF}",
        f"nsrc={NUM_SENDERS}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    m = NORM_SCORE_RE.search(result.stdout)
    if not m:
        print(f"  WARNING: no normalized_score for {tree_path} link={link_ppt}", file=sys.stderr)
        return None
    return float(m.group(1)) / NUM_SENDERS


def eval_single(name, run_fn, link_rates, output_path):
    """Evaluate a single CCA across all link rates and save CSV."""
    print(f"\n=== Evaluating {name} ({len(link_rates)} points) ===")
    results = []
    t0 = time.time()

    for i, ppt in enumerate(link_rates):
        mbps = ppt * PPT_TO_MBPS
        score = run_fn(ppt)
        results.append((ppt, score))
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(link_rates) - i - 1)
        status = f"  [{i+1}/{len(link_rates)}] link={ppt:.4f} ppt ({mbps:.1f} Mbps) => score={score}  (ETA {eta:.0f}s)"
        print(status, end="\r" if i < len(link_rates) - 1 else "\n")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Write CSV: link_ppt, link_mbps, normalized_score_per_sender
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["link_ppt", "link_mbps", "normalized_score_per_sender"])
        for ppt, score in results:
            if score is not None:
                w.writerow([f"{ppt:.10f}", f"{ppt * PPT_TO_MBPS:.4f}", f"{score:.6f}"])

    print(f"  Saved: {output_path}")
    return results


def load_ppo_csvs():
    """Load existing 9-point PPO brain CSVs."""
    ppo_data = {}
    for f in sorted(PPO_CSV_DIR.glob("data-brain.*.csv")):
        label = f.stem.replace("data-", "ppo-")  # e.g. ppo-brain.798
        rows = []
        with open(f) as fh:
            reader = csv.reader(fh)
            for row in reader:
                # Format: link_ppt, norm_score, tp1, del1, tp2, del2, ...
                link_ppt = float(row[0])
                norm_score = float(row[1])
                rows.append((link_ppt, norm_score))
        ppo_data[label] = rows
        print(f"  Loaded {label}: {len(rows)} points from {f.name}")
    return ppo_data


def main():
    parser = argparse.ArgumentParser(description="Dense C++ evaluation for paper figures")
    parser.add_argument("--points", type=int, default=NUM_POINTS, help="Number of evaluation points")
    parser.add_argument("--evolved-only", action="store_true", help="Only evaluate evolved-runner")
    parser.add_argument("--whisker-only", action="store_true", help="Only evaluate whisker trees")
    parser.add_argument("--skip-ppo", action="store_true", help="Skip PPO CSV loading")
    args = parser.parse_args()

    link_rates = log_spaced_link_rates(n=args.points)

    print(f"Link rates: {args.points} points, {link_rates[0]:.4f} to {link_rates[-1]:.4f} ppt "
          f"({link_rates[0]*PPT_TO_MBPS:.1f} to {link_rates[-1]*PPT_TO_MBPS:.1f} Mbps)")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. AlphaCC (evolved-runner)
    if not args.whisker_only:
        if not EVOLVED_RUNNER.exists():
            print(f"ERROR: evolved-runner not found at {EVOLVED_RUNNER}", file=sys.stderr)
            print("  Run: ./build_evolved.sh", file=sys.stderr)
        else:
            eval_single(
                "AlphaCC (evolved)",
                run_evolved,
                link_rates,
                OUTPUT_DIR / "alphacc_evolved.csv",
            )

    # 2. Remy whisker trees
    if not args.evolved_only:
        if not SENDER_RUNNER.exists():
            print(f"ERROR: sender-runner not found at {SENDER_RUNNER}", file=sys.stderr)
            print("  Run: ./build_sender_runner.sh", file=sys.stderr)
        else:
            for dir_name, tree_file, label in WHISKER_TREES:
                tree_path = WHISKER_DIR / dir_name / tree_file
                if not tree_path.exists():
                    print(f"  WARNING: tree not found: {tree_path}", file=sys.stderr)
                    continue
                eval_single(
                    f"Remy {label} ({tree_file})",
                    lambda ppt, tp=str(tree_path): run_whisker(tp, ppt),
                    link_rates,
                    OUTPUT_DIR / f"{label}.csv",
                )

    # 3. PPO brains (existing CSVs)
    if not args.skip_ppo and not args.evolved_only and not args.whisker_only:
        print(f"\n=== Loading PPO brain CSVs from {PPO_CSV_DIR} ===")
        ppo_data = load_ppo_csvs()
        for label, rows in ppo_data.items():
            out_path = OUTPUT_DIR / f"{label}.csv"
            with open(out_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["link_ppt", "link_mbps", "normalized_score_per_sender"])
                for ppt, score in rows:
                    w.writerow([f"{ppt:.10f}", f"{ppt * PPT_TO_MBPS:.4f}", f"{score:.6f}"])
            print(f"  Saved: {out_path}")

    print(f"\n=== All results in {OUTPUT_DIR} ===")
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
