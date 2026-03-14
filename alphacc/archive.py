# Population archive + parent selection — ported from DGM's choose_selfimproves().
import json
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Individual:
    id: str                                    # timestamp-based
    parent_id: Optional[str]                   # lineage
    cca_code: str                              # Z3 CCA function source
    fitness: float                             # 0-1 composite score
    properties_proven: List[str] = field(default_factory=list)
    counterexamples: List[str] = field(default_factory=list)
    children_count: int = 0
    generation: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "cca_code": self.cca_code,
            "fitness": self.fitness,
            "properties_proven": self.properties_proven,
            "counterexamples": self.counterexamples,
            "children_count": self.children_count,
            "generation": self.generation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Individual":
        return cls(
            id=d["id"],
            parent_id=d.get("parent_id"),
            cca_code=d["cca_code"],
            fitness=d["fitness"],
            properties_proven=d.get("properties_proven", []),
            counterexamples=d.get("counterexamples", []),
            children_count=d.get("children_count", 0),
            generation=d.get("generation", 0),
        )


def make_id() -> str:
    """Generate a timestamp-based unique ID."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


class Archive:
    """Population archive with selection and persistence.

    Ported from DGM's choose_selfimproves() in DGM_outer.py.
    Uses keep-all strategy (no pruning) + JSONL append for crash recovery.
    """

    def __init__(self):
        self.individuals: Dict[str, Individual] = {}

    def add(self, ind: Individual):
        self.individuals[ind.id] = ind
        # Update parent's children count
        if ind.parent_id and ind.parent_id in self.individuals:
            self.individuals[ind.parent_id].children_count += 1

    def add_batch(self, inds: List[Individual]):
        for ind in inds:
            self.add(ind)

    def best(self) -> Optional[Individual]:
        if not self.individuals:
            return None
        return max(self.individuals.values(), key=lambda x: x.fitness)

    def select_parent(self, method: str = "score_child_prop") -> Individual:
        """Select a parent for mutation.

        Ported from DGM's choose_selfimproves() — score_child_prop method.
        sigmoid: 1 / (1 + exp(-10 * (score - 0.5)))
        child_penalty: 1 / (1 + children_count)
        probability: sigmoid * child_penalty / sum
        """
        candidates = list(self.individuals.values())
        assert candidates, "Archive is empty — cannot select parent"

        if method == "score_child_prop":
            # Sigmoid transform on fitness
            scores = [1 / (1 + math.exp(-10 * (ind.fitness - 0.5)))
                      for ind in candidates]
            # Child penalty: prefer parents with fewer children
            child_pen = [1 / (1 + ind.children_count)
                         for ind in candidates]
            # Combined weights
            weights = [s * cp for s, cp in zip(scores, child_pen)]
            total = sum(weights)
            if total == 0:
                return random.choice(candidates)
            probs = [w / total for w in weights]
            return random.choices(candidates, probs, k=1)[0]

        elif method == "score_prop":
            scores = [1 / (1 + math.exp(-10 * (ind.fitness - 0.5)))
                      for ind in candidates]
            total = sum(scores)
            if total == 0:
                return random.choice(candidates)
            probs = [s / total for s in scores]
            return random.choices(candidates, probs, k=1)[0]

        elif method == "best":
            return max(candidates, key=lambda x: x.fitness)

        else:  # random
            return random.choice(candidates)

    def __len__(self) -> int:
        return len(self.individuals)

    # ── Persistence (JSONL append, ported from DGM pattern) ──────────

    def save_generation(self, gen: int, output_dir: str):
        """Append current generation state to JSONL log."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Append all individuals from this generation
        log_file = output_path / "archive.jsonl"
        with open(log_file, "a") as f:
            for ind in self.individuals.values():
                if ind.generation == gen:
                    record = {"gen": gen, **ind.to_dict()}
                    f.write(json.dumps(record) + "\n")

        # Also save full snapshot
        snapshot_file = output_path / f"archive_gen{gen:03d}.json"
        snapshot = {
            "generation": gen,
            "population_size": len(self.individuals),
            "best_fitness": self.best().fitness if self.best() else 0,
            "individuals": [ind.to_dict() for ind in self.individuals.values()],
        }
        with open(snapshot_file, "w") as f:
            json.dump(snapshot, f, indent=2)

    def save_best_cca(self, output_dir: str):
        """Save the best CCA code to a standalone file."""
        best = self.best()
        if best is None:
            return
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "best_cca.py", "w") as f:
            f.write(f"# Best CCA — fitness: {best.fitness:.3f}\n")
            f.write(f"# Properties proven: {best.properties_proven}\n")
            f.write(f"# ID: {best.id}, parent: {best.parent_id}\n\n")
            f.write(best.cca_code)

    @classmethod
    def load_from_jsonl(cls, jsonl_path: str) -> "Archive":
        """Reconstruct archive from JSONL log (crash recovery)."""
        archive = cls()
        if not os.path.exists(jsonl_path):
            return archive

        seen = {}
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                ind = Individual.from_dict(record)
                if ind.id in seen:
                    continue  # keep first occurrence
                seen[ind.id] = ind

        # Re-add in order to rebuild children counts
        for ind in seen.values():
            ind.children_count = 0  # will be recomputed
        for ind in seen.values():
            archive.individuals[ind.id] = ind
        for ind in seen.values():
            if ind.parent_id and ind.parent_id in archive.individuals:
                archive.individuals[ind.parent_id].children_count += 1

        return archive
