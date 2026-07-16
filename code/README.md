# `rsqaoa`

Installable reference implementation of Randomized Subspace QAOA: adaptive,
matrix-free active-subspace optimization for multi-angle QAOA with residual-
gated basis refresh and basis recycling, plus a distinct fixed-rank
forward-only path.

## Install and verify

From this `code/` directory, choose a runtime install:

```bash
python -m pip install .
rsqaoa-experiment --version
```

or an editable development install:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q
```

Core dependencies are NumPy, PyTorch, and NetworkX. PennyLane is an optional
independent simulator cross-check. CI spans every supported minor version from
Python 3.9 through 3.13, exercises PennyLane on Python 3.12, validates root
installation, and installs the canonical built wheel in a clean working
directory before running the real console command.

Install directly from Git without keeping a checkout:

```bash
python -m pip install \
  "rsqaoa @ git+https://github.com/thmolena/Randomized-Subspace-QAOA.git#subdirectory=code"
```

The project does not claim that `pip install rsqaoa` by name is available yet;
that command requires a separate package-index release. Built distributions
contain the reusable library, metadata, license, and `rsqaoa-experiment`
entrypoint. Repository-level experiments, released results, and paper assets are
kept outside the wheel and source distribution.

## Quickstart

```python
from rsqaoa import MaxCutProblem, graphs, optimize_rsq

problem = MaxCutProblem(
    n=8,
    edges=graphs.random_regular(8, degree=3, seed=0),
    p=1,
)
result = optimize_rsq(problem, tol=1e-2, steps=100, seed=0)
print(result.cut, result.best_cut, result.final_rank)
print(result.residual_steps, result.residual_history, result.refresh_steps)
```

```bash
rsqaoa-experiment --family regular --n 8 --p 1 --steps 100 --json
```

## Module contract

| Module | Responsibility |
| --- | --- |
| `circuits` | Differentiable complex-128 ma-QAOA statevector and observables |
| `operator` | Counted QAOA and backend-neutral JVP/VJP sensitivity actions |
| `randqb` | Adaptive QB, residual indicators, recycling, confidence utility |
| `subspace_opt` | Residual-gated reduced optimization |
| `baselines` | Full, symmetry-reduced, and dense-SVD reference methods |
| `graphs` | Graph generators, weights, and exact small-instance references |
| `backends_pennylane` | Optional independent backend |

`randqb(matvec, rmatvec, dout, din, ...)` is a backend-neutral randomized
linear-algebra primitive. `RSQResult` reports final and best-observed iterates,
the residual-check trajectory, refresh steps, rank history, and separated
operator counts so callers can audit every adaptive decision.

`MatrixFreeSensitivity(d, m, jvp, vjp)` wraps checked CPU-float64 callbacks for
another simulator or service. The transpose callback can be omitted with
`active_subspace_adjoint_free`; declared forward costs remain visible in the
operator ledger.

## Finite-probe interpretation

`residual_ratio_confidence` exposes the conservative finite-probe envelope
proved in the manuscript. The default 12-probe diagnostic does not supply an
informative 95% run-wise envelope; it is used as a refresh trigger. The local
objective-loss theorem is conditional on the true residual.

## Checkout-only experiments

These commands require a Git checkout. Install the repository-only orchestration
dependencies first; they are not needed by library users:

```bash
python -m pip install -e ".[dev]"
python experiments/run_experiment.py \
  --config experiments/configs/maxcut_small.yaml
python experiments/summarize_results.py
python experiments/validate_release.py
```

`experiments/reproduction-environment.yml` pins the environment used for the
committed evidence. Long runs may be split with `--shard-count` and
`--shard-index`; `experiments/merge_shards.py` validates and combines them.

The released CSV contains 216 method rows over 36 paired runs, 14 unique graph
topologies, and 28 topology--depth clusters. Separate graph, initialization,
and sketch seeds plus graph hashes prevent repeated ring restarts from being
misreported as independent topologies. The summarizer is the only path from
those rows to paper aggregates. The validator checks design completeness,
source and driver versions, CSV and summary hashes, manuscript claims, figures,
and table provenance.

The bundled statevector study is small-scale software evidence. The callback
interfaces do not, by themselves, establish device performance, hardware
compatibility, industrial scaling, or a physical-query advantage.

MIT licensed. See the repository-level README for the claim ledger and limits.
