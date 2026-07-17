# Randomized Subspace QAOA

[![Package CI](https://github.com/thmolena/Randomized-Subspace-QAOA/actions/workflows/ci.yml/badge.svg)](https://github.com/thmolena/Randomized-Subspace-QAOA/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

Task-distribution-weighted observable tangent spaces for repeated multi-angle
QAOA, with matched optimizer controls and physical measurement accounting.

For a vector of edge observables `F(theta)`, task weights `w`, Jacobian `J`, and
task second moment `M = E[ww^T] = LL^T`, the leading left singular vectors of
`J^T L` minimize expected discarded local gradient energy at a fixed anchor.
The package constructs this representation matrix-free, separates basis-training
weights from evaluation trajectories, and audits no-refresh, fixed, gated,
per-task, random-refresh, random-basis, and full-space SPSA controls.

The result is presently a falsified operational hypothesis, not an efficiency
claim. Across 16 exact-statevector topology-depth cells from eight graphs,
residual-gated reuse changes mean approximation ratio by `-1.57 +/- 1.04`
percentage points and uses `1.218x` the forward circuit points of full SPSA.
It therefore fails the declared development criterion. In a separate hybrid
256-shot objective pilot, per-task refresh changes the mean by
`+0.61 +/- 1.57` points, but subspace construction still uses exact simulator
observables and VJPs. Neither result supports hardware, query, wall-time, or
quantum advantage.

- [Manuscript](paper/main.pdf) · [LaTeX source](paper/main.tex)
- [Project page](https://thmolena.github.io/Randomized-Subspace-QAOA/)
- [Installable package](rsqaoa/) · [Exact rows](experiments/results/amortized_development.csv) · [Hybrid shot rows](experiments/results/amortized_shot_development.csv)

The two development protocols retain their complete 19-file frozen source
closure under `experiments/protocol/frozen_source/`. The present checkout adds
later physical-accounting code and is therefore intentionally not
byte-compatible with those historical executions. Validation audits the
preserved bytes, protocol hashes, rows, and summaries; an incompatible
`rsqaoa-reproduce --rerun` aborts before writing rather than silently
refreezing the record. The standalone package's `replay` command verifies the
historical evidence, while `full` creates a separately frozen seeded run.

## Claims and verification paths

| Supported statement | Evidence | Reproduction path | Boundary |
| --- | --- | --- | --- |
| The leading subspace of `J^T L` is locally optimal for a declared task second moment | Trace identity and Eckart-Young-Ky Fan argument | Task-weighted optimality theorem in `paper/main.tex`; `test_task_weighted_qb_matches_dense_operator` | Local; no optimizer convergence claim |
| Training and evaluation weights are disjoint | Independent recorded seeds and hashes | Frozen protocols and every result row | Same synthetic weight model is shared |
| Residual-gated reuse fails the exact development gate | Matched full/reduced SPSA over 16 configured cells | [`amortized_development_summary.json`](experiments/results/amortized_development_summary.json) | Development data; eight unique topologies |
| The finite-shot signal is unresolved and hybrid | Three nested measurement repeats on eight graphs | [`amortized_shot_development_summary.json`](experiments/results/amortized_shot_development_summary.json) | Basis construction and final scoring remain exact |
| One bitstring batch can evaluate all task weights at one circuit point | Shared-observable covariance identity and tested API | `shared_task_objectives`; `test_physical_accounting.py` | Cannot merge task-specific parameter trajectories |
| A prospective confirmatory design is specified before execution | Design-only 40-topology plan and validator | `python experiments/validate_nmi_design.py` | Not registered, not code-complete, and not executable |

### Earlier fixed-objective benchmark

The repository retains the original fixed-objective development grid for
auditability: 36 paired runs on 14 unique topologies, with 39.1% and 69.6%
dimension reduction at depths one and two and paired changes of
`+0.48 +/- 0.94` and `-4.77 +/- 2.47` points. The forward-only reductions were
64.7% and 82.3%. All 324 of 324 scheduled checks triggered rebuilding. These
numbers are historical development evidence and are not the primary evidence
for the task-weighted manuscript.

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
wheel contents; they are included in the source distribution for an archived
reproduction bundle.

## Quickstart

For repeated tasks, estimate all task objectives from one bitstring batch at
the same circuit point:

```python
import torch
from rsqaoa import MaxCutProblem, graphs
from rsqaoa.amortized import ShotEvaluator, shared_task_objectives

problem = MaxCutProblem(6, graphs.ring(6), p=1)
theta = problem.random_theta(generator=torch.Generator().manual_seed(0))
weights = torch.stack([
    torch.ones(problem.m),
    torch.linspace(0.5, 1.5, problem.m),
])
evaluator = ShotEvaluator(problem, shots=1024, seed=1)
batch = shared_task_objectives(evaluator, theta, weights)

print(batch.values)
print(batch.circuit_points, batch.shots)  # 1, 1024
```

This reuse is valid only because every scalarization is evaluated at the same
`theta`. It cannot merge bitstring batches after task optimizers follow
different parameter trajectories.

The original one-objective RSQ API remains available:

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
python experiments/analyze_amortized.py
python experiments/make_amortized_paper_assets.py
python experiments/validate_nmi_design.py
python run.py validate
```

The design validator reports `execution_ready: false`. The 40-topology
confirmatory plan has not been externally registered, its full runner and
coordinate-gradient controls are incomplete, and no hardware stage has been
run. `python experiments/validate_nmi_design.py --require-executable` therefore
fails intentionally rather than generating outcomes under a retrospective
protocol.

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

## Current scope and extension points

- The task-weighted theorem is local and conditional on a specified weight
  second moment. It does not imply global optimization convergence.
- The exact repeated-task grid uses eight unique graph topologies at `n <= 10`
  and depths one and two. Its development gate fails.
- The 256-shot pilot samples scalar objectives only; basis construction and
  final scoring remain exact and simulator-assisted.
- `shared_task_objectives` correctly reuses one bitstring batch across weights
  at an identical parameter vector. It does not merge diverged task trajectories.
- The confirmatory design is not registered, code-complete, or executable.
  Hardware evidence, larger instances, full-rank/OOD task streams, matched
  coordinate-gradient controls, and calibrated gating remain future work.

### Earlier fixed-objective package extension points

- The statevector backend is exponentially expensive and intended for small
  validation studies. The earlier fixed-objective grid is unweighted.
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
  scalar gradient directly. The newer repeated-task pilot tests amortization,
  but does not establish an operational advantage.
- Full-space SPSA also uses two objective evaluations per step. Because the
  release has no full-space SPSA baseline, the forward-only path demonstrates
  zero-VJP execution and compression, not an objective-query advantage.
- A credible scaling claim requires noisy-device results, larger instances,
  amortized basis construction, and matched physical shot accounting.

## Citation and license

```bibtex
@article{huynh2026rsqaoa,
  title  = {Task-distribution-weighted observable tangent spaces
            for repeated multi-angle QAOA: A matched resource audit},
  author = {Huynh, Molena},
  year   = {2026}
}
```

MIT licensed. Contributions should add a focused regression test and must not
convert parameter compression into an unsupported query-advantage claim.
