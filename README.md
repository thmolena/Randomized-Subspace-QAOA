# Randomized Subspace QAOA

Certified active-subspace optimization for multi-angle QAOA.

RSQ optimizes ma-QAOA in a matrix-free subspace of the per-edge observable
Jacobian. It supports adaptive Frobenius or spectral residual certificates,
certificate-gated refresh, augmented-and-recompressed basis recycling, and a
forward-only path whose build and refresh require zero VJPs.

## Released result

The committed CPU grid contains 36 paired MaxCut instances: three graph
families, `n in {8, 10}`, `p in {1, 2}`, and three seeds. At depth one,
two-sided RSQ (`tol=0.01`) reduces the optimized dimension by 41.5% and changes
approximation ratio by -0.21 +/- 0.89 percentage points (mean +/- 2SE) relative
to full ma-QAOA. At depth two, it reduces the dimension by 70.8% with a
-2.86 +/- 1.43 point change. Operator counters show a setup overhead on these
small simulators, so the supported claim is parameter compression, not a
quantum-query advantage.

See [paper/main.pdf](paper/main.pdf) for the complete protocol and limitations.

## Install

The canonical package lives under `code/`:

```bash
python -m pip install ./code
```

For tests and experiment tooling:

```bash
python -m pip install -e 'code[dev,experiments]'
python -m pytest code/tests -q
```

The package name is `rsqaoa`. After this repository is public, Git users can
also install directly from the `code/` subdirectory:

```bash
python -m pip install \
  'git+https://github.com/thmolena/Randomized-Subspace-QAOA.git#subdirectory=code'
```

`pip install rsqaoa` by name alone will work only after the distribution is
published to a Python package index; repository readiness does not publish it.

## Quickstart

```python
from rsqaoa import MaxCutProblem, graphs, optimize_rsq

edges = graphs.random_regular(n=10, degree=3, seed=0)
problem = MaxCutProblem(n=10, edges=edges, p=1)
result = optimize_rsq(problem, tol=1e-2, steps=100, seed=0)

print(result.cut, result.final_rank, result.refreshes, result.counts)
```

## Reproduce the manuscript

```bash
cd code
python experiments/run_experiment.py \
  --config experiments/configs/maxcut_small.yaml
python experiments/summarize_results.py
```

This regenerates `code/results/maxcut_small.csv`, `code/results/summary.json`,
the four files under `paper/figures/`, and `paper/tables/table1_summary.tex`.

## Repository map

- `code/src/rsqaoa/`: simulator, sensitivity operator, randomized QB, optimizer,
  baselines, graph generators, and CLI.
- `code/tests/`: numerical and packaging tests.
- `code/experiments/`: paired experiment and deterministic summarizer.
- `code/results/`: released row-level data and aggregates.
- `paper/`: arXiv manuscript, bibliography, figures, and table.
- `index.html`: GitHub Pages project page.

MIT licensed. Independent replication and issue reports are welcome.
