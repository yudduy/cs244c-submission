"""Load Remy WhiskerTree protobufs and evaluate them as Python policies.

This allows running Remy's C++ decision trees through the Python simulator
for apples-to-apples comparison with LLM-evolved policies.
"""

from . import dna_pb2
from .remy_eval import RemyAction, RemyMemory


def load_whisker_tree(path: str) -> dna_pb2.WhiskerTree:
    """Load a WhiskerTree protobuf from disk."""
    tree = dna_pb2.WhiskerTree()
    with open(path, 'rb') as f:
        tree.ParseFromString(f.read())
    return tree


def _get_memory_field(memory: RemyMemory, axis: int) -> float:
    """Get memory field by axis index (matches C++ Memory::field ordering).

    Axis enum: SEND_EWMA=0, REC_EWMA=1, RTT_RATIO=2, SLOW_REC_EWMA=3
    """
    if axis == 0:
        return memory.send_ewma
    elif axis == 1:
        return memory.rec_ewma
    elif axis == 2:
        return memory.rtt_ratio
    elif axis == 3:
        return memory.slow_rec_ewma
    return 0.0


def _get_proto_field(proto_memory, axis: int) -> float:
    """Get protobuf Memory field by axis index."""
    if axis == 0:
        return proto_memory.rec_send_ewma
    elif axis == 1:
        return proto_memory.rec_rec_ewma
    elif axis == 2:
        return proto_memory.rtt_ratio
    elif axis == 3:
        return proto_memory.slow_rec_rec_ewma
    return 0.0


def _memory_in_range(memory: RemyMemory, domain) -> bool:
    """Check if memory falls within a MemoryRange domain.

    Matches C++ MemoryRange::contains exactly:
    - Only checks active_axis fields
    - Half-open interval: lower <= value < upper
    """
    if not domain.HasField('lower') or not domain.HasField('upper'):
        return True

    for axis in domain.active_axis:
        val = _get_memory_field(memory, axis)
        lo = _get_proto_field(domain.lower, axis)
        hi = _get_proto_field(domain.upper, axis)
        if not (val >= lo and val < hi):
            return False

    return True


def _lookup_whisker(tree, memory: RemyMemory):
    """Traverse the WhiskerTree to find the matching Whisker for a memory state."""
    # If this node is a leaf, return it
    if tree.HasField('leaf'):
        return tree.leaf

    # Otherwise, find the matching child
    for child in tree.children:
        if child.HasField('domain') and _memory_in_range(memory, child.domain):
            return _lookup_whisker(child, memory)

    # Fallback: if no child matches (shouldn't happen with well-formed trees),
    # try the first child or return a default
    if tree.children:
        return _lookup_whisker(tree.children[0], memory)

    # Absolute fallback
    return None


def make_whisker_policy(tree_path: str):
    """Create a Python policy function from a WhiskerTree protobuf.

    Returns a callable with signature (RemyMemory) -> RemyAction,
    compatible with run_remy_sim().
    """
    tree = load_whisker_tree(tree_path)

    def whisker_policy(memory: RemyMemory) -> RemyAction:
        whisker = _lookup_whisker(tree, memory)
        if whisker is None:
            return RemyAction(window_increment=1, window_multiple=1.0, intersend=0.0)
        return RemyAction(
            window_increment=whisker.window_increment if whisker.HasField('window_increment') else 0,
            window_multiple=whisker.window_multiple if whisker.HasField('window_multiple') else 1.0,
            intersend=whisker.intersend if whisker.HasField('intersend') else 0.0,
        )

    whisker_policy.__name__ = f"whisker_{tree_path.split('/')[-1]}"
    return whisker_policy


if __name__ == '__main__':
    import sys
    import glob
    from pathlib import Path

    # Quick test: load and dump tree stats
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        repo_root = Path(__file__).resolve().parents[1]
        paths = sorted(glob.glob(str(repo_root / 'remy' / 'tests' / 'RemyCC-*.dna')))
        if not paths:
            print("No tree files found")
            sys.exit(1)
        path = paths[-1]

    tree = load_whisker_tree(path)
    print(f"Loaded tree from {path}")
    print(f"  Has domain: {tree.HasField('domain')}")
    print(f"  Num children: {len(tree.children)}")
    print(f"  Has leaf: {tree.HasField('leaf')}")

    # Count leaves
    def count_leaves(t):
        if t.HasField('leaf'):
            return 1
        return sum(count_leaves(c) for c in t.children)

    print(f"  Total leaves (whiskers): {count_leaves(tree)}")
