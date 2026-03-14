# CS244C Submission Artifact

This repository contains both codepaths used in the CS244C project:

- `remy/`: the C++ Remy codebase plus the PPO neural-policy trainer and evaluator.
- `alphacc/`: Stack for LLM-guided mutation of Python congestion-control policies.

`alphacc` running requires OpenEvolve (https://github.com/algorithmicsuperintelligence/openevolve) copied over and integrating. The retained AlphaCC implementation is:

- `alphacc/evolve_remy.py`: mutation loop, seed policies, prompt wiring, and run orchestration.
- `alphacc/remy_eval.py`: Remy-compatible event-driven simulator and baseline policies.
- `alphacc/whisker_loader.py`: loader for Remy `.dna` whisker trees so they can be evaluated in the Python simulator.
- `alphacc/llm_client.py`: OpenAI client wrapper used by the evolution loop.
- `alphacc/dna_pb2.py`: protobuf bindings for Remy whisker trees.

The rest of the repo is supporting material for the evaluated comparison:

- `results/`: raw and derived artifacts for Remy, PPO, and AlphaCC.
- `scripts/`: plotting and evaluation helpers for the AlphaCC/Remy experiments.
- `paper/`: paper source and generated figures.

## Running Remy / PPO

Build and use the C++ path from `remy/`:

```bash
cd remy
./autogen.sh
./configure
make CXXFLAGS="-Wno-error=deprecated-copy" -j$(nproc)
```

Example commands:

```bash
# train a whisker tree
./src/remy cf=configs/link-1x.cfg of=output/cca

# train a PPO brain
./src/rattrainer cf=configs/link-1x.cfg

# evaluate with sender-runner
./src/sender-runner if=tests/RemyCC-2014-100x.dna link=1.5 rtt=150 nsrc=2 on=1000 off=1000
```

## Running AlphaCC

Install the Python dependencies first:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Single-point training:

```bash
python3 -m alphacc.evolve_remy --generations 15 --population 5 --train-mult 1.0 --duration 30000 --output results/python_sim/output_remy_evolve
```

Multipoint training:

```bash
python3 -m alphacc.evolve_remy --multipoint --generations 30 --population 5 --duration 30000 --output results/python_sim/output_remy_evolve_multipoint
```

Baseline-only evaluation:

```bash
python3 -m alphacc.evolve_remy --baselines-only --duration 30000 --output results/python_sim/output_remy_evolve
```

Paper figures:

```bash
python3 scripts/plot_paper_v2.py
python3 scripts/plot_multipoint.py
```

These scripts read from the preserved artifacts under `results/` and write the
presentation figures into `paper/figures/`.

## Raw Artifact Locations

- Raw dense `sender-runner`-style CSVs for the C++ comparison live in `results/cpp_sim/dense_cpp/`.
- This includes `remy-1x.csv`, `remy-10x.csv`, `remy-20x.csv`, `ppo-brain.*.csv`, and `alphacc_evolved.csv`.
- AlphaCC Python evolution outputs live under `results/python_sim/`.

## Notes

- `remy/` is the codepath for Remy trees and PPO.
- `alphacc/` is the codepath for AlphaCC.
- `results/cpp_sim/` contains preserved raw C++ evaluation outputs for Remy, PPO, and the AlphaCC export.
- `results/python_sim/` contains preserved AlphaCC runs and auxiliary Python-simulator outputs.
- Cross-method comparison is qualitative: Remy/PPO artifacts come from C++, while AlphaCC was evolved in the Python reimplementation.
