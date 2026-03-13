# C++ Comparison Artifacts

This directory contains the preserved raw CSV outputs used to compare:

- Remy whisker trees
- PPO brains trained in the Remy C++ simulator
- The exported AlphaCC policy evaluated through the C++ path

The dense comparison files are in `dense_cpp/`:

- `remy-1x.csv`
- `remy-10x.csv`
- `remy-20x.csv`
- `ppo-brain.248.csv`
- `ppo-brain.383.csv`
- `ppo-brain.751.csv`
- `ppo-brain.793.csv`
- `ppo-brain.798.csv`
- `ppo-brain.837.csv`
- `alphacc_evolved.csv`

These are the files to use when regenerating link-range comparison plots with
updated PPO or Remy outputs.
