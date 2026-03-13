# Paper Build

Build from the `paper/` directory:

```bash
make
```

The source lives in `paper.tex` and `body.tex`. Generated figures used by the
class presentation are written to `paper/figures/` by:

```bash
python3 ../scripts/plot_paper_v2.py
python3 ../scripts/plot_multipoint.py
```
