"""Minimal CLI: run one RSQ-vs-baselines comparison on a small MaxCut instance
and print a table. For the full experiment grid see ``experiments/``.

    rsqaoa-experiment --n 10 --p 2 --family regular --tol 1e-2
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from .circuits import MaxCutProblem
from . import graphs
from .subspace_opt import optimize_rsq
from .baselines import full_maqaoa, symmetry_reduced


def _make_edges(family: str, n: int, seed: int):
    if family == "regular":
        return graphs.random_regular(n, degree=3, seed=seed)
    if family == "er":
        return graphs.erdos_renyi(n, prob=0.5, seed=seed)
    if family == "ring":
        return graphs.ring(n)
    raise ValueError(family)


def main(argv=None):
    ap = argparse.ArgumentParser(description="RSQ vs baselines on a MaxCut instance.")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--family", choices=["regular", "er", "ring"], default="regular")
    ap.add_argument("--tol", type=float, default=1e-2)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    edges = _make_edges(args.family, args.n, args.seed)
    problem = MaxCutProblem(n=args.n, edges=edges, p=args.p)
    best = graphs.brute_force_maxcut(args.n, edges) if args.n <= 22 else None

    gen = torch.Generator().manual_seed(args.seed)
    theta0 = problem.random_theta(generator=gen)

    rsq = optimize_rsq(problem, theta0=theta0.clone(), tol=args.tol,
                       steps=args.steps, seed=args.seed)
    full = full_maqaoa(problem, theta0=theta0.clone(), steps=args.steps, seed=args.seed)
    sym = symmetry_reduced(problem, steps=args.steps, seed=args.seed)

    def ar(c):
        return f"{c/best:.4f}" if best else "n/a"

    print(f"\nMaxCut  n={args.n}  p={args.p}  family={args.family}  d={problem.dim}  "
          f"|E|={problem.m}  tol={args.tol}")
    print("-" * 68)
    print(f"{'method':<22}{'cut':>10}{'approx':>10}{'params':>9}{'rank':>7}")
    print(f"{'RSQ (adaptive)':<22}{rsq.cut:>10.4f}{ar(rsq.cut):>10}"
          f"{rsq.final_rank:>9}{rsq.final_rank:>7}")
    print(f"{'full ma-QAOA':<22}{full.cut:>10.4f}{ar(full.cut):>10}"
          f"{full.n_params_opt:>9}{'-':>7}")
    print(f"{'symmetry-reduced':<22}{sym.cut:>10.4f}{ar(sym.cut):>10}"
          f"{sym.n_params_opt:>9}{'-':>7}")
    print("-" * 68)
    print(f"RSQ operator budget: {rsq.counts}  refreshes={rsq.refreshes}\n")
    print("Note: results are instance/seed specific -- run the experiment grid "
          "for statistics.")


if __name__ == "__main__":
    main()
