"""Cross-validate Python simulator against C++ sender-runner.

Parses a .dna whisker tree, runs it in the Python sim, and compares
against C++ sender-runner output at the same link rates and parameters.

This is the key calibration: if the same tree produces the same scores
in both sims, the Python sim is a faithful replica of Remy's network model.
"""

import csv
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'protobufs'))
from dna_pb2 import WhiskerTree


# ── Whisker tree interpreter ──────────────────────────────────────

class WhiskerTreeInterpreter:
    """Parse a .dna protobuf and do tree lookups matching C++ WhiskerTree."""

    def __init__(self, filepath):
        self.tree = WhiskerTree()
        with open(filepath, 'rb') as f:
            self.tree.ParseFromString(f.read())
        self.leaves = []
        self._extract_leaves(self.tree)

    def _extract_leaves(self, node):
        if node.HasField('leaf'):
            self.leaves.append(node.leaf)
            return
        for child in node.children:
            self._extract_leaves(child)

    def lookup(self, send_ewma, rec_ewma, rtt_ratio, slow_rec_ewma):
        """Find the matching leaf for given memory values.

        Walks the tree checking MemoryRange bounds, same as C++ WhiskerTree::use_whisker().
        """
        return self._walk(self.tree, send_ewma, rec_ewma, rtt_ratio, slow_rec_ewma)

    def _walk(self, node, se, re, rr, sre):
        if node.HasField('leaf'):
            return (node.leaf.window_increment,
                    node.leaf.window_multiple,
                    node.leaf.intersend)
        for child in node.children:
            d = child.domain if child.HasField('domain') else (child.leaf.domain if child.HasField('leaf') else None)
            if d is None:
                continue
            if self._in_range(d, se, re, rr, sre):
                result = self._walk(child, se, re, rr, sre)
                if result is not None:
                    return result
        # fallback: return first leaf if no match (shouldn't happen for valid inputs)
        if self.leaves:
            l = self.leaves[0]
            return (l.window_increment, l.window_multiple, l.intersend)
        return (0, 1.0, 0.0)

    def _in_range(self, domain, se, re, rr, sre):
        lo = domain.lower
        hi = domain.upper
        if se < lo.rec_send_ewma or se >= hi.rec_send_ewma:
            return False
        if re < lo.rec_rec_ewma or re >= hi.rec_rec_ewma:
            return False
        if rr < lo.rtt_ratio or rr >= hi.rtt_ratio:
            return False
        if sre < lo.slow_rec_rec_ewma or sre >= hi.slow_rec_rec_ewma:
            return False
        return True


# ── Python sim with whisker tree ──────────────────────────────────

# Inline the sim here to keep this script self-contained.
# We import from remy_eval if available, otherwise define minimal version.

def make_tree_policy(tree_interp):
    """Create a Remy-compatible policy function from a parsed whisker tree."""
    try:
        # Try importing the actual RemyAction/RemyMemory
        parent = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(parent))
        from alphacc.remy_eval import RemyAction
    except ImportError:
        from collections import namedtuple
        RemyAction = namedtuple('RemyAction', ['window_increment', 'window_multiple', 'intersend'])

    def policy(memory):
        inc, mult, intersend = tree_interp.lookup(
            memory.send_ewma, memory.rec_ewma,
            memory.rtt_ratio, memory.slow_rec_ewma,
        )
        return RemyAction(window_increment=inc, window_multiple=mult, intersend=intersend)

    return policy


# ── C++ sender-runner evaluation ──────────────────────────────────

def run_cpp_sender_runner(dna_path, link_ppt, rtt=150, nsrc=2, on=1000, off=1000,
                          sender_runner='./src/sender-runner'):
    """Run C++ sender-runner and parse output for normalized score."""
    cmd = [
        sender_runner,
        f'if={dna_path}',
        f'link={link_ppt}',
        f'rtt={rtt}',
        f'nsrc={nsrc}',
        f'on={on}',
        f'off={off}',
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        # Parse the output for throughput and delay
        for line in result.stdout.split('\n'):
            if 'score' in line.lower() or 'throughput' in line.lower():
                pass  # TODO: parse actual output format
        return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return None


# ── Main calibration ──────────────────────────────────────────────

LINK_PPTS = [0.237, 0.376, 0.596, 0.946, 1.500, 2.379, 3.773, 5.983, 9.490]
LINK_MBPS = [x * 10.0 for x in LINK_PPTS]


def run_python_sim_calibration(tree_path, link_ppts=None, rtt_ms=150.0,
                                nsrc=2, on_ms=1000.0, off_ms=1000.0,
                                duration_ms=100_000.0, seed=42):
    """Run a whisker tree in the Python sim at multiple link rates."""
    parent = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(parent))
    from alphacc.remy_eval import run_remy_sim

    if link_ppts is None:
        link_ppts = LINK_PPTS

    interp = WhiskerTreeInterpreter(tree_path)
    policy = make_tree_policy(interp)
    print(f"Loaded tree: {tree_path} ({len(interp.leaves)} leaves)")

    results = []
    for i, lp in enumerate(link_ppts):
        r = run_remy_sim(
            policies=[policy],
            link_ppt=lp,
            rtt_ms=rtt_ms,
            num_senders=nsrc,
            duration_ms=duration_ms,
            mean_on_ms=on_ms,
            mean_off_ms=off_ms,
            seed=seed + i,
        )
        results.append(r)
        mbps = lp * 10.0
        print(f"  {mbps:6.1f} Mbps: norm={r['normalized_score']:.3f} "
              f"tput={r['throughput_ppt']:.4f} delay={r['avg_delay_ms']:.1f}ms")

    return results


def load_cpp_results(csv_path):
    """Load C++ sender-runner CSV results."""
    rows = []
    with open(csv_path) as f:
        for row in csv.reader(f):
            rows.append([float(x) for x in row])
    return rows


def main():
    base = Path(__file__).resolve().parent.parent
    out_dir = base / 'results' / 'calibration'
    out_dir.mkdir(parents=True, exist_ok=True)

    trees = {
        '1x': (base / 'results/whisker-trees/1x-2src/cca.179', 179),
        '10x': (base / 'results/whisker-trees/10x-2src/cca.36', 36),
        '20x': (base / 'results/whisker-trees/20x-2src/cca.19', 19),
    }

    # Also load the 100x reference tree if available
    ref_100x = base / 'tests/RemyCC-2014-100x.dna'
    if ref_100x.exists():
        trees['100x'] = (ref_100x, 'published')

    # C++ results (from results/calibration/all_trees_on1000.txt)
    cpp_file = out_dir / 'all_trees_on1000.txt'
    cpp_scores = {}
    if cpp_file.exists():
        text = cpp_file.read_text()
        for block in text.split('--- '):
            if not block.strip():
                continue
            name = block.split(' ')[0].strip()
            scores = []
            for line in block.strip().split('\n'):
                if line.startswith('link_ppt') or line.startswith(name) or not line.strip():
                    continue
                if line.startswith('---'):
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    scores.append(float(parts[1]))
                    # These are 2-sender totals; per-sender = /2
            cpp_scores[name] = scores

    print("=" * 72)
    print("CALIBRATION: Python sim vs C++ sender-runner")
    print("Parameters: on=1000, off=1000, RTT=150ms, nsrc=2, duration=100s")
    print("=" * 72)

    all_results = {}

    for name, (tree_path, iters) in trees.items():
        if not tree_path.exists():
            print(f"\n{name}: tree not found at {tree_path}")
            continue

        print(f"\n--- {name} tree ({iters} iterations) ---")
        py_results = run_python_sim_calibration(
            tree_path, duration_ms=100_000.0, seed=42,
        )

        py_scores = [r['normalized_score'] for r in py_results]
        all_results[name] = {'python': py_scores}

        if name in cpp_scores and len(cpp_scores[name]) == len(py_scores):
            cpp = cpp_scores[name]
            all_results[name]['cpp'] = cpp

            print(f"\n  {'Mbps':>6}  {'C++ score':>10}  {'Py score':>10}  {'Delta':>8}  {'Error':>8}")
            print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")
            deltas = []
            for j, mbps in enumerate(LINK_MBPS):
                delta = py_scores[j] - cpp[j]
                pct = (delta / abs(cpp[j]) * 100) if cpp[j] != 0 else 0
                deltas.append(abs(pct))
                print(f"  {mbps:6.1f}  {cpp[j]:10.3f}  {py_scores[j]:10.3f}  "
                      f"{delta:+8.3f}  {pct:+7.1f}%")

            mean_err = sum(deltas) / len(deltas)
            max_err = max(deltas)
            print(f"\n  Mean |error|: {mean_err:.1f}%  Max |error|: {max_err:.1f}%")
        else:
            print(f"  (no C++ comparison data for {name})")

    # Write calibration summary
    summary_path = out_dir / 'python_vs_cpp.csv'
    with open(summary_path, 'w') as f:
        f.write("tree,link_mbps,cpp_score,python_score,delta,error_pct\n")
        for name, data in all_results.items():
            py = data['python']
            cpp = data.get('cpp', [None] * len(py))
            for j, mbps in enumerate(LINK_MBPS):
                c = cpp[j] if cpp[j] is not None else ''
                p = py[j]
                if c != '':
                    d = p - c
                    e = (d / abs(c) * 100) if c != 0 else 0
                    f.write(f"{name},{mbps:.1f},{c:.4f},{p:.4f},{d:.4f},{e:.1f}\n")
                else:
                    f.write(f"{name},{mbps:.1f},,{p:.4f},,\n")
    print(f"\nSaved: {summary_path}")


if __name__ == '__main__':
    main()
