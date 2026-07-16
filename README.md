# Randomized Subspace QAOA

[![Package CI](https://github.com/thmolena/Randomized-Subspace-QAOA/actions/workflows/ci.yml/badge.svg)](https://github.com/thmolena/Randomized-Subspace-QAOA/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

Residual-controlled active-subspace optimization for multi-angle QAOA.

Multi-angle QAOA uses `d = p(|E| + |V|)` parameters. RSQ treats the Jacobian of
the per-edge observable vector as a matrix-free operator, builds a low-dimensional
parameter-space basis with randomized QB, optimizes only inside that basis, and
refreshes it when a randomized residual diagnostic detects drift. A separate
forward-only path uses finite-difference Jacobian-vector products and zero
vector-Jacobian products.

The released result is parameter compression, not quantum-query advantage. The
grid contains 36 paired exact-statevector MaxCut runs on 14 unique graph
topologies (28 topology-depth clusters). At tolerance `0.01`, two-sided RSQ
reduces optimized dimension by 39.1% at depth one with a paired approximation-
ratio change of `+0.48 +/- 0.94` percentage points (mean `+/- 2SE` across 14
cluster means). At depth two, the reduction is 69.6% with a
`-4.77 +/- 2.47` point change. Forward-only rank-8 RSQ compresses by 64.7% and
82.3%, with changes of `-6.39 +/- 1.98` and `-14.73 +/- 2.65` points. All 324
of 324 scheduled residual checks triggered rebuilding. Instrumented setup cost
exceeds full-space optimization on these small simulators.

- [Manuscript](paper/main.pdf) · [LaTeX source](paper/main.tex)
- [Project page](https://thmolena.github.io/Randomized-Subspace-QAOA/)
- [Installable package](rsqaoa/) · [Released rows](experiments/results/maxcut_small.csv)

## Claims and verification paths

| Supported statement | Evidence | Reproduction path | Boundary |
| --- | --- | --- | --- |
| The scalar weighted-cut gradient lies in the observable Jacobian row space | Chain-rule proposition | `paper/main.tex`, Proposition 1; `test_dense_jacobian_consistency` | Local and observable-dependent |
| A true residual controls discarded first-order change | Conditional Taylor bound | `paper/main.tex`, Theorem 1 | No global convergence or approximation-ratio guarantee |
| Finite probes estimate the Frobenius residual ratio | Moment identity and Chebyshev envelope | `paper/main.tex`, Propositions 2--3; `residual_ratio_confidence` | Released 12-probe envelope is not run-wise informative at 95% confidence |
| Depth-one compression has a small measured change | 18 paired runs aggregated into 14 topology clusters | [`summary.json`](experiments/results/summary.json); Fig. 1 | Not a formal equivalence trial |
| Depth-two compression exposes a failure boundary | 18 paired runs aggregated into 14 topology clusters | [`summary.json`](experiments/results/summary.json); Figs. 1--2 | `n <= 10`, tested families and optimizer only |
| The method is not query-advantaged in the released regime | Instrumented forward/JVP/VJP counts | Fig. 4 and [released rows](experiments/results/maxcut_small.csv) | Counters are separate, not additive shot costs |

## Install

Install the canonical package from the repository root:

```bash
python -m pip install .
```

From the public Git repository, without keeping a checkout:

```bash
python -m pip install \
  "rsqaoa @ git+https://github.com/thmolena/Randomized-Subspace-QAOA.git"
```

For a reproducible downstream environment, replace the default branch in that
URL with a release tag or commit SHA. For development in a checkout:

```bash
python -m pip install -e ".[dev,release]"
python -m pytest tests -q
python -m build
python -m twine check dist/*
```

The distribution and import names are both `rsqaoa`. The repository does not
claim that a Python Package Index release exists: bare `pip install rsqaoa`
becomes valid only after a separately authenticated package-index publication.
The wheel contains the library and `rsqaoa-experiment` command. The paper,
released rows, and full experiment grid remain repository artifacts rather than
wheel contents.

## Quickstart

```python
from rsqaoa import MaxCutProblem, graphs, optimize_rsq

edges = graphs.random_regular(n=10, degree=3, seed=0)
problem = MaxCutProblem(n=10, edges=edges, p=2)
result = optimize_rsq(
    problem,
    tol=1e-2,
    maxrank=15,
    steps=100,
    refresh_every=25,
    eps_refresh=5e-2,
    indicator="fro",
    residual_probes=12,
    recycle=True,
    step_cap=None,
    seed=0,
)

print(result.cut, result.best_cut, result.final_rank)
print(list(zip(result.residual_steps, result.residual_history)))
print(result.refresh_steps, result.counts)
```

For a configurable one-instance comparison against full and symmetry-reduced
ma-QAOA:

```bash
rsqaoa-experiment --family regular --n 10 --p 2 \
  --tol 0.01 --maxrank 15 --refresh-every 25 \
  --residual-probes 12 --steps 100 --json
```

`rsqaoa-experiment --help` exposes the indicator, block size, finite-difference
step, learning rate, refresh threshold, rank cap, recycling switch, step cap,
and forward-only path. `rsqaoa-experiment --version` reports the installed
release.

## What the residual statement means

For Gaussian probes, `residual_ratio_confidence(s, delta)` reports the
Chebyshev/union-bound envelope proved in the manuscript. With `s=12` and
`delta=0.05`, the elementary envelope is non-informative; the observed ratio is
a drift diagnostic. Increasing the probe count trades operator applications for
a finite confidence envelope. The conditional objective-loss theorem assumes
the true operator residual, not merely a favorable finite-probe draw.

## Connect another simulator or hardware service

The randomized core is not tied to the bundled statevector backend. Wrap custom
Jacobian-vector and vector-Jacobian callbacks with the explicit CPU-float64
operator contract:

```python
import torch
from rsqaoa import MatrixFreeSensitivity, active_subspace

J = torch.randn(12, 30, dtype=torch.float64)  # replace with service callbacks
operator = MatrixFreeSensitivity(
    d=30,
    m=12,
    jvp=lambda v: J @ v,
    vjp=lambda u: J.T @ u,
    forward_evals_per_jvp=2,
    forward_evals_per_vjp=1,
)
factor = active_subspace(
    operator,
    tol=1e-3,
    indicator="fro",
)
Q, B = factor.Q, factor.B
```

For a service exposing only forward directional actions, omit `vjp` and call
`active_subspace_adjoint_free`. The adapter checks callback shapes and finite
values while counting declared forward costs. This interface does not turn the
released QAOA benchmark into evidence for an unrelated backend or application.

## Reproduce the manuscript

Full-paper reproduction requires a Git checkout because the experiment grid,
released rows, paper source, and figure destinations are intentionally not
embedded in the installable wheel:

```bash
python -m pip install -e ".[dev]"
python run.py experiment --config experiments/configs/maxcut_small.yaml
python run.py summarize
python run.py validate
```

The exact environment used for the committed grid is recorded in every CSV row
and in [`reproduction-environment.yml`](experiments/reproduction-environment.yml).
For execution environments with short process limits, run four deterministic
shards with `--shard-count 4 --shard-index 0..3`, then combine them with
`experiments/merge_shards.py`; the merger enforces full-grid completeness before
writing a release CSV.

The experiment runner writes 216 tidy rows with separate graph, initialization,
and sketch seeds; graph hashes and edge lists; optimization trajectories;
residual, refresh, and build histories; environment versions; and source hashes.
The deterministic summarizer reads that CSV and regenerates
`results/summary.json`, four paper figures, and the LaTeX result table. The
validator checks the complete paired grid, package version, implementation,
runner and summarizer fingerprints, deterministic-execution record, source-CSV
hash, manuscript headline values, refresh totals, and every generated artifact
hash. Because 324 of 324 scheduled checks triggered, these data evaluate a
periodically rebuilt setting and do not demonstrate saved rebuilds.

## Repository map

```text
rsqaoa/                canonical installable package
tests/                 numerical, validation, API, CLI, and backend tests
experiments/           paired grid runner, summarizer, and provenance validator
experiments/results/   released row-level data and aggregates
run.py                 unified demo, experiment, and validation dispatcher
paper/main.tex         article source
paper/main.pdf         compiled article
paper/figures/         generated figures
paper/tables/          generated LaTeX table
index.html             static project page
.github/workflows/     package CI and GitHub Pages deployment
```

The canonical source tree is the root-level `rsqaoa/` package; there is no
second package implementation or nested build configuration to keep
synchronized. Future package-index artifacts should be built from the
repository root.

## Scope and extension points

- The statevector backend is exponentially expensive and intended for small
  validation studies. `MaxCutProblem` accepts weights, but the released grid is
  unweighted.
- `indicator="spec"` gives a worst-direction power-iteration diagnostic;
  `indicator="fro"` gives an average-sensitivity diagnostic.
- Every optimization result exposes final and best-observed iterates, residual
  check values and steps, refresh steps, ranks, and separated operator counts.
- `adjoint_free=True` removes all VJPs, including refresh, at the measured cost
  of stronger quality loss in the released benchmark.
- At the primary tolerance, two-sided RSQ averages 749/134/481 counted
  forward/JVP/VJP actions at depth one and 867/155/556 at depth two. The
  forward-only path averages 488--489 forward actions and 143--144 JVPs with
  zero VJPs. These overlapping categories must not be summed as shot counts.
- The current grid triggered 108/108 checks for each two-sided tolerance and
  108/108 forward-only checks; it does not establish savings from the gate.
- For the released fixed objective, one VJP with known edge weights returns the
  scalar gradient directly. A reusable observable subspace could be amortized
  across weights, objectives, or related instances, but that use case is untested.
- Full-space SPSA also uses two objective evaluations per step. Because the
  release has no full-space SPSA baseline, the forward-only path demonstrates
  zero-VJP execution and compression, not an objective-query advantage.
- A credible scaling claim requires noisy-device results, larger instances,
  amortized basis construction, and matched physical shot accounting.

## Citation and license

```bibtex
@article{huynh2026rsqaoa,
  title  = {Randomized Subspace QAOA: Residual-Controlled
            Active-Subspace Optimization for Multi-Angle QAOA},
  author = {Huynh, Molena},
  year   = {2026}
}
```

MIT licensed. Contributions should add a focused regression test and must not
convert parameter compression into an unsupported query-advantage claim.
