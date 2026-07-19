# Randomized Subspace QAOA self-contained code package

This directory is an independently buildable `src`-layout distribution. It
contains a byte-identical mirror of the canonical root `rsqaoa/` package, every
committed experiment driver and frozen configuration, row-level and summary
data, generated figures and tables, and a single command that distinguishes
immutable artifact replay from a new seeded execution.

## Fresh-clone Mac CPU install

Python 3.9 through 3.13 is supported. From a fresh clone:

```bash
cd code
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[test,release]"
```

For the exact universal resolver state in `uv.lock`:

```bash
uv sync --locked --extra test --extra release
source .venv/bin/activate
```

The lock records exact versions, hashes, Python markers, and platform wheels
across the declared interpreter range. PyTorch wheel availability still
depends on Python and CPU architecture.

## Verify or replay the committed evidence

This mode hashes every bundled byte and validates row counts, source hashes,
protocol hashes, and the generated-asset manifest. It performs no simulation,
optimization, summarization, or plotting:

```bash
rsqaoa-reproduce-all replay
python tools/check_source_sync.py
python -m pytest
rsqaoa-experiment --help
rsqaoa-experiment --family ring --n 6 --p 1 --steps 10 --json
```

Materialize an exact byte-for-byte copy at the original repository-relative
paths:

```bash
rsqaoa-reproduce-all replay --output replayed-evidence
```

## Run all seeded studies again

Full mode reruns:

1. the fixed-objective `maxcut_small` grid;
2. the exact task-stream amortized development study;
3. the finite-shot hybrid development study;
4. both analyzers and all committed figure/table generators; and
5. the design-only confirmatory-plan validator.

```bash
rsqaoa-reproduce-all full --output full-rerun
```

Inspect the complete command plan without executing:

```bash
rsqaoa-reproduce-all full --output full-rerun --dry-run
```

A smaller one-job smoke execution is available:

```bash
rsqaoa-reproduce-all full --output quick-rerun --quick
```

Replay and rerun are intentionally distinct:

- `replay` establishes byte/hash identity for the released CSV, JSON, YAML,
  protocol, figure, and table files.
- `full` creates a new seeded execution. Deterministic PyTorch settings and
  seeds are retained, but BLAS, PyTorch, compiler, and platform changes can
  produce floating-point drift. Cross-platform bitwise identity is not claimed.

The confirmatory design reports `execution_ready: false`; the command validates
that status and does not fabricate an unregistered confirmatory experiment.

## Complete manuscript release gate

From the repository root, one command recomputes every feasible summary,
regenerates all 13 figures and five numbered tables, validates evidence and
source synchronization, builds the REVTeX manuscript and deterministic source
archive, and recompiles the extracted archive:

```bash
python run.py release
```

The command fails immediately if a locked evidence byte, packaged mirror,
reported value, manuscript input, citation, or archive member has drifted.

## Python API

The public API is importable independently of the command line:

```python
from rsqaoa import MaxCutProblem, optimize_rsq

problem = MaxCutProblem(n=6, edges=[(0, 1), (1, 2), (2, 3),
                                   (3, 4), (4, 5), (0, 5)], p=1)
result = optimize_rsq(problem, steps=10, maxrank=4, seed=7)
print(result.best_cut, result.best_theta)
```

## Build and wheel-install checks

```bash
python -m build .
python -m twine check dist/*
python -m venv /tmp/rsqaoa-wheel-venv
source /tmp/rsqaoa-wheel-venv/bin/activate
python -m pip install dist/rsqaoa-0.3.0-py3-none-any.whl
rsqaoa-reproduce-all replay
rsqaoa-experiment --family ring --n 6 --p 1 --steps 10 --json
```

The canonical implementation remains the repository-root `rsqaoa/` directory.
`tools/check_source_sync.py` rejects any missing, added, or byte-different file
and checks `source_manifest.json`, preventing silent divergence of the package
mirror.
