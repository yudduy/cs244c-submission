# Remy + PPO

**CS 244C Winter 2026** — Armaan Abraham, Duy Nguyen, Andrew Grant

Fork of [Remy](http://web.mit.edu/remy) (Winstein & Balakrishnan, SIGCOMM 2013) with a PPO neural network trainer (`rattrainer`) as an alternative to whisker-tree optimization.

## Repo layout

```
src/                    # C++ source — remy, rattrainer, sender-runner
configs/                # Training configs (link-1x.cfg, link-10x.cfg, link-20x.cfg)
tests/                  # Keith's published .dna files + verification tests
results/
  whisker-trees/        # Our trained trees: 1x (179 iter), 10x (36 iter), 20x (19 iter)
  ppo-brains/           # PPO brain evals (brain-279 is the best)
  eval-csv/             # Raw sender-runner CSVs (on=5000/off=5000)
  calibration/          # Trees re-evaluated at on=1000 (Keith's canonical params)
figures/                # Generated plots
scripts/                # Plotting + eval scripts
slurm/                  # IRIS cluster job scripts
```

## Calibration

We verified our sender-runner against Keith's published 100x tree (`tests/RemyCC-2014-100x.dna`):

| Link (Mbps) | Expected Tput | Ours | Error |
|-------------|--------------|------|-------|
| 2.21 | 1.80 | 1.78 | -1.3% |
| 9.93 | 6.00 | 6.03 | +0.4% |
| 29.41 | 18.00 | 18.34 | +1.9% |
| 185.04 | 91.10 | 91.85 | +0.8% |

All within 2.5% (tolerance: 5%). The 10x and 20x trees are under-trained (36 and 19 iterations respectively) — the 20x tree is actually worse than the 1x at low link rates.

Note: most eval CSVs in this repo use `on=5000/off=5000`. The `results/calibration/` data uses `on=1000/off=1000` to match Keith's verification tests.

## Build

```bash
sudo apt-get install -y g++ protobuf-compiler libprotobuf-dev libboost-all-dev autoconf automake libtool
./autogen.sh && ./configure && make CXXFLAGS="-Wno-error=deprecated-copy" -j$(nproc)
```

For PPO you need LibTorch 2.10.0 — it bundles protobuf 3.13.0, so your system protobuf needs to match. Easiest way is `conda create -n remy libprotobuf=3.13.0`, then `PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH ./configure`.

## Usage

```bash
# train a whisker tree
./src/remy cf=configs/link-1x.cfg of=output/cca

# train a PPO brain
./src/rattrainer cf=configs/link-1x.cfg

# evaluate
./src/sender-runner if=tests/RemyCC-2014-100x.dna link=1.5 rtt=150 nsrc=2 on=1000 off=1000

# plots
python3 scripts/plot_calibration.py
python3 scripts/plot_replication.py
```

## References

- Winstein & Balakrishnan, [TCP ex Machina](http://web.mit.edu/remy), SIGCOMM 2013
- Winstein, Sivaraman & Balakrishnan, Stochastic Forecasts Achieve High Throughput and Low Delay over Cellular Networks, NSDI 2013

Thanks to Keith Winstein for the original codebase.
