# Manuscript build

`main.tex` is the arXiv source and `main.pdf` is the compiled manuscript.

From `paper/`:

```bash
tectonic main.tex
```

The paper consumes four PDF figures in `figures/` and
`tables/table1_summary.tex`. Regenerate those artifacts from the released CSV:

```bash
cd ../code
python experiments/summarize_results.py
```

Rerun the paired benchmark first when a numerical method changes:

```bash
python experiments/run_experiment.py \
  --config experiments/configs/maxcut_small.yaml
python experiments/summarize_results.py
```

Every reported result is backed by row-level evidence in
`../code/results/maxcut_small.csv`; exact aggregates live in
`../code/results/summary.json`.
