# Manuscript build

`main.tex` is the arXiv source and `main.pdf` is the compiled manuscript.

From `paper/`:

```bash
tectonic main.tex
```

The paper consumes four PDF figures in `figures/` and
`tables/table1_summary.tex`. Regenerate those artifacts from the released CSV:

```bash
cd ..
python -m pip install -e ".[dev]"
python run.py summarize
python run.py validate
```

Rerun the paired benchmark first when a numerical method changes:

```bash
python run.py experiment --config experiments/configs/maxcut_small.yaml
python run.py summarize
python run.py validate
```

Every reported result is backed by row-level evidence in
`experiments/results/maxcut_small.csv`; exact aggregates live in
`experiments/results/summary.json`.
