from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
PY_RESULTS_DIR = RESULTS_DIR / "python_sim"
CPP_RESULTS_DIR = RESULTS_DIR / "cpp_sim" / "dense_cpp"
PAPER_FIGURES_DIR = REPO_ROOT / "paper" / "figures"
