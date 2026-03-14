from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AlphaCCConfig:
    # Evolution
    population_size: int = 20
    generations: int = 50
    children_per_gen: int = 4
    selection_method: str = "score_child_prop"

    # LLM
    diagnose_model: str = "o3-mini"
    mutate_model: str = "gpt-4o"
    temperature: float = 0.7

    # Seed
    seed: str = "aimd_simple"

    # CCAC
    ccac_timeout: int = 30
    ccac_timesteps: int = 10
    ccac_flows: int = 1
    ccac_buf_min: float = 1.0

    # Paths
    ccac_dir: str = str(Path(__file__).parent.parent / "ccac")
    output_dir: str = str(Path(__file__).parent.parent / "output_alphacc")
