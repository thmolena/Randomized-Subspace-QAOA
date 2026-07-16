"""Experiment grid runner: RSQ vs. baselines over graph families, sizes, depths,
tolerances, and seeds. Writes one tidy CSV row per (method, instance, seed).

This script is the primary reproduction path. It has no hard Lightning
dependency; if `lightning` is installed it is used only for `seed_everything`
(deterministic seeding). Figures in the manuscript are built from the CSV.

Usage:
    python experiments/run_experiment.py --config experiments/configs/maxcut_small.yaml
    python experiments/run_experiment.py --n 8 10 12 --p 1 2 --seeds 5 --out results/run.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import time
from typing import List

import torch

from rsqaoa.circuits import MaxCutProblem
from rsqaoa import graphs
from rsqaoa.subspace_opt import optimize_rsq
from rsqaoa.baselines import full_maqaoa, symmetry_reduced, fixed_rank_subspace


def _seed_everything(seed: int):
    # The numerical path is plain PyTorch.  Importing Lightning merely for its
    # seeding helper can load a second OpenMP runtime on macOS (for example when
    # both conda PyTorch and a pip scientific stack are installed), aborting an
    # otherwise valid CPU run.  Seed PyTorch directly and keep orchestration out
    # of the reproducibility-critical path.
    torch.manual_seed(seed)


def make_edges(family: str, n: int, seed: int):
    if family == "regular":
        return graphs.random_regular(n, degree=3, seed=seed)
    if family == "er":
        return graphs.erdos_renyi(n, prob=0.5, seed=seed)
    if family == "ring":
        return graphs.ring(n)
    raise ValueError(family)


FIELDS = ["method", "family", "n", "p", "seed", "tol", "d", "m",
          "cut", "approx_ratio", "params_opt", "rank", "refreshes",
          "forward_F", "jvp", "vjp", "wall_s"]


def _row(method, family, n, p, seed, tol, problem, best, cut, params, rank,
         refreshes, counts, wall):
    return {
        "method": method, "family": family, "n": n, "p": p, "seed": seed,
        "tol": tol, "d": problem.dim, "m": problem.m, "cut": round(cut, 6),
        "approx_ratio": round(cut / best, 6) if best else "",
        "params_opt": params, "rank": rank, "refreshes": refreshes,
        "forward_F": counts.get("forward_F", 0), "jvp": counts.get("jvp", 0),
        "vjp": counts.get("vjp", 0), "wall_s": round(wall, 4),
    }


def run(families: List[str], ns: List[int], ps: List[int], tols: List[float],
        seeds: int, steps: int, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rows = []
    for family, n, p, seed in itertools.product(families, ns, ps, range(seeds)):
        _seed_everything(seed)
        edges = make_edges(family, n, seed)
        problem = MaxCutProblem(n=n, edges=edges, p=p)
        best = graphs.brute_force_maxcut(n, edges) if n <= 22 else None
        theta0 = problem.random_theta(generator=torch.Generator().manual_seed(seed))

        # full ma-QAOA
        t = time.time()
        full = full_maqaoa(problem, theta0=theta0.clone(), steps=steps, seed=seed)
        rows.append(_row("full_maqaoa", family, n, p, seed, "", problem, best,
                         full.cut, full.n_params_opt, "", 0, full.counts, time.time() - t))

        # symmetry-reduced
        t = time.time()
        sym = symmetry_reduced(problem, steps=steps, seed=seed)
        rows.append(_row("symmetry", family, n, p, seed, "", problem, best,
                         sym.cut, sym.n_params_opt, "", 0, sym.counts, time.time() - t))

        # Dense-SVD oracle used only on these small validation instances.  Its
        # Jacobian-formation setup cost is deliberately not folded into the
        # operator counters, so tables label it as an oracle rather than a
        # deployable matrix-free baseline.
        fixed_rank = min(8, problem.m, problem.dim)
        t = time.time()
        fixed = fixed_rank_subspace(problem, rank=fixed_rank,
                                    theta0=theta0.clone(), steps=steps, seed=seed)
        rows.append(_row("fixed_rank_oracle", family, n, p, seed, "", problem,
                         best, fixed.cut, fixed.n_params_opt, fixed_rank, 0,
                         fixed.counts, time.time() - t))

        # RSQ (adaptive) at each tolerance
        for tol in tols:
            t = time.time()
            rsq = optimize_rsq(problem, theta0=theta0.clone(), tol=tol, steps=steps, seed=seed)
            rows.append(_row("rsq", family, n, p, seed, tol, problem, best,
                             rsq.cut, rsq.final_rank, rsq.final_rank, rsq.refreshes,
                             rsq.counts, time.time() - t))

        # RSQ adjoint-free
        t = time.time()
        af = optimize_rsq(problem, theta0=theta0.clone(), steps=steps,
                          adjoint_free=True, af_rank=min(8, problem.dim), seed=seed)
        rows.append(_row("rsq_adjoint_free", family, n, p, seed, "", problem, best,
                         af.cut, af.final_rank, af.final_rank, af.refreshes,
                         af.counts, time.time() - t))

        print(f"done: {family} n={n} p={p} seed={seed}")

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--families", nargs="+", default=["regular", "er", "ring"])
    ap.add_argument("--n", nargs="+", type=int, default=[8, 10, 12])
    ap.add_argument("--p", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--tols", nargs="+", type=float, default=[1e-1, 1e-2, 1e-3])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", type=str, default="results/run.csv")
    args = ap.parse_args(argv)

    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        args.families = cfg.get("families", args.families)
        args.n = cfg.get("n", args.n)
        args.p = cfg.get("p", args.p)
        args.tols = cfg.get("tols", args.tols)
        args.seeds = cfg.get("seeds", args.seeds)
        args.steps = cfg.get("steps", args.steps)
        args.out = cfg.get("out", args.out)

    run(args.families, args.n, args.p, args.tols, args.seeds, args.steps, args.out)


if __name__ == "__main__":
    main()
