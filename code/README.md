# `rsqaoa` — installable package

This folder is the canonical, fact-checkable, pip-installable package.

```bash
pip install .            # core: numpy, torch, networkx
pip install '.[pennylane]'   # lightning.qubit cross-check backend
pip install '.[experiments]' # lightning + pandas + matplotlib grid runner
pip install -e '.[dev]'      # editable + pytest
pytest                       # run the correctness suite
```

After the GitHub repository is public, direct installation from this folder is:

```bash
pip install 'git+https://github.com/thmolena/Randomized-Subspace-QAOA.git#subdirectory=code'
```

Installation by the bare name `pip install rsqaoa` additionally requires a
separate release to a Python package index.

## Modules (`src/rsqaoa/`)

| File | Contents |
|------|----------|
| `circuits.py` | Pure-torch ma-QAOA state-vector simulator (differentiable). |
| `operator.py` | Matrix-free `J`: `jvp` (finite-difference, adjoint-free) and `vjp` (reverse-mode), with call counting. |
| `randqb.py` | Adaptive randomized QB; Frobenius **and** spectral residual indicators; rank pruning; subspace recycling; two-sided and adjoint-free active-subspace builders; refresh certificate. |
| `subspace_opt.py` | The RSQ optimizer: recycled certificate-gated refresh + trust-region step. |
| `baselines.py` | Full ma-QAOA, fixed-rank truncated-SVD subspace, symmetry-reduced ma-QAOA. |
| `graphs.py` | Graph families + exact small-instance MaxCut. |
| `backends_pennylane.py` | Optional `lightning.qubit` cross-check / finite-shot sampler. |

## Tests (`tests/`)

`test_circuits` (simulator vs. dense Kronecker reference), `test_operator`
(adjoint identity, FD vs. autograd JVP), `test_randqb` (rank vs. tolerance),
`test_extensions` (spectral indicator vs. dense; recycling), `test_subspace_opt`
(optimizer + baselines), and an optional `test_pennylane_backend`.

## Experiments (`experiments/`)

`run_experiment.py` writes tidy per-instance CSVs; `lightning_module.py` is an
optional PyTorch Lightning wrapper used purely for orchestration (seeding,
logging, checkpoints) — it is **not** part of the numerical method.

The released grid is `results/maxcut_small.csv` (216 rows over 36 paired
instances). Regenerate the figures, LaTeX table, and `results/summary.json` with:

```bash
python experiments/summarize_results.py
```
