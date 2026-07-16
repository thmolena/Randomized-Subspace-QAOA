"""Experiment grid runner: RSQ vs. baselines over graph families, sizes, depths,
tolerances, and random restarts.

Graph generation, initialization, and randomized linear algebra use separate
recorded seeds.  Every row also carries a canonical edge list and graph hash,
so downstream summaries can distinguish paired runs from independent graph
topologies instead of treating repeated ring restarts as new graphs.

This script is the primary reproduction path. It uses deterministic PyTorch
algorithms, one intra-op and one inter-op thread, and separately recorded NumPy
and PyTorch seeds. Figures in the manuscript are built from the CSV.

Usage:
    python experiments/run_experiment.py --config experiments/configs/maxcut_small.yaml
    python experiments/run_experiment.py --n 8 10 12 --p 1 2 --seeds 5 --out experiments/results/run.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import networkx as nx
import torch

from rsqaoa.circuits import MaxCutProblem
from rsqaoa import __version__ as rsqaoa_version, graphs
from rsqaoa.subspace_opt import optimize_rsq
from rsqaoa.baselines import full_maqaoa, symmetry_reduced, fixed_rank_subspace


def _seed_everything(seed: int):
    # The numerical path is plain PyTorch.  Importing Lightning merely for its
    # seeding helper can load a second OpenMP runtime on macOS (for example when
    # both conda PyTorch and a pip scientific stack are installed), aborting an
    # otherwise valid CPU run.  Seed PyTorch directly and keep orchestration out
    # of the reproducibility-critical path.
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_edges(family: str, n: int, seed: int):
    if family == "regular":
        return graphs.random_regular(n, degree=3, seed=seed)
    if family == "er":
        return graphs.erdos_renyi(n, prob=0.5, seed=seed)
    if family == "ring":
        return graphs.ring(n)
    raise ValueError(family)


def canonical_edge_list(edges):
    """Return a stable JSON-compatible representation of an unweighted graph."""
    return sorted([min(int(u), int(v)), max(int(u), int(v))] for u, v in edges)


def graph_fingerprint(n, edges):
    encoded = json.dumps(canonical_edge_list(edges), separators=(",", ":"))
    payload = json.dumps(
        {"edges": json.loads(encoded), "n": int(n)},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], encoded


def source_tree_fingerprint():
    root = Path(__file__).resolve().parents[1] / "rsqaoa"
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def runner_fingerprint():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


FIELDS = ["method", "family", "n", "p", "seed", "graph_seed", "init_seed",
          "sketch_seed", "graph_id", "edge_list", "tol", "d", "m",
          "cut", "approx_ratio", "params_opt", "rank", "refreshes",
          "forward_F", "jvp", "vjp", "dense_jacobian", "wall_s",
          "best_cut", "best_approx_ratio", "best_step", "subspace_builds",
          "residual_steps", "residual_history", "refresh_steps",
          "build_history", "objective_history", "theta_final",
          "steps", "learning_rate", "refresh_every", "eps_refresh", "block",
          "residual_probes", "fd_eps", "spsa_eps", "af_rank", "indicator",
          "graph_seed_offset", "init_seed_offset", "sketch_seed_offset",
          "experiment_schema", "rsqaoa_version", "implementation_sha256",
          "runner_sha256", "experiment_config_sha256",
          "torch_deterministic", "torch_num_threads",
          "torch_num_interop_threads",
          "design_families", "design_n", "design_p", "design_tols",
          "design_restarts",
          "python_version", "numpy_version", "torch_version",
          "networkx_version", "platform"]


def _row(method, family, n, p, seed, graph_seed, init_seed, sketch_seed,
         graph_id, edge_list, tol, problem, best, cut, params, rank,
         refreshes, counts, wall, protocol, result):
    if hasattr(result, "cut_history"):
        objective_history = [float(value) for value in result.cut_history]
        best_cut = float(result.best_cut)
        best_step = int(result.best_step)
        subspace_builds = int(result.subspace_builds)
        residual_steps = [int(value) for value in result.residual_steps]
        residual_history = [float(value) for value in result.residual_history]
        refresh_steps = [int(value) for value in result.refresh_steps]
        build_history = result.build_history
    else:
        objective_history = [-float(value) for value in result.history]
        candidates = objective_history + [float(cut)]
        best_step = int(max(range(len(candidates)), key=candidates.__getitem__))
        best_cut = float(candidates[best_step])
        subspace_builds = 0
        residual_steps = []
        residual_history = []
        refresh_steps = []
        build_history = []
    row = {
        "method": method, "family": family, "n": n, "p": p, "seed": seed,
        "graph_seed": graph_seed, "init_seed": init_seed,
        "sketch_seed": sketch_seed, "graph_id": graph_id,
        "edge_list": edge_list,
        "tol": tol, "d": problem.dim, "m": problem.m, "cut": round(cut, 6),
        "approx_ratio": round(cut / best, 6) if best else "",
        "params_opt": params, "rank": rank, "refreshes": refreshes,
        "forward_F": counts.get("forward_F", 0), "jvp": counts.get("jvp", 0),
        "vjp": counts.get("vjp", 0),
        "dense_jacobian": counts.get("dense_jacobian", 0),
        "wall_s": round(wall, 4),
        "best_cut": round(best_cut, 6),
        "best_approx_ratio": round(best_cut / best, 6) if best else "",
        "best_step": best_step,
        "subspace_builds": subspace_builds,
        "residual_steps": json.dumps(residual_steps, separators=(",", ":")),
        "residual_history": json.dumps(residual_history, separators=(",", ":")),
        "refresh_steps": json.dumps(refresh_steps, separators=(",", ":")),
        "build_history": json.dumps(build_history, separators=(",", ":")),
        "objective_history": json.dumps(
            objective_history, separators=(",", ":")),
        "theta_final": json.dumps(
            [float(value) for value in result.theta.detach().cpu()],
            separators=(",", ":"),
        ),
    }
    row.update(protocol)
    return row


def run(families: List[str], ns: List[int], ps: List[int], tols: List[float],
        seeds: int, steps: int, out_path: str, learning_rate: float = 0.05,
        refresh_every: int = 25, eps_refresh: float = 5e-2, block: int = 4,
        residual_probes: int = 12, fd_eps: float = 1e-4,
        spsa_eps: float = 1e-4, af_rank: int = 8,
        indicator: str = "fro", graph_seed_offset: int = 0,
        init_seed_offset: int = 100_000,
        sketch_seed_offset: int = 200_000, shard_index: int = 0,
        shard_count: int = 1):
    if int(shard_count) != shard_count or shard_count < 1:
        raise ValueError("shard_count must be a positive integer")
    if int(shard_index) != shard_index or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must lie in [0, shard_count)")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting the inter-op pool only before parallel work.
        # Record the already-fixed value when an embedding process initialized it.
        pass
    rows = []
    effective_config = {
        "families": list(families), "n": list(ns), "p": list(ps),
        "tols": list(tols), "seeds": seeds, "steps": steps,
        "learning_rate": learning_rate, "refresh_every": refresh_every,
        "eps_refresh": eps_refresh, "block": block,
        "residual_probes": residual_probes, "fd_eps": fd_eps,
        "spsa_eps": spsa_eps, "af_rank": af_rank, "indicator": indicator,
        "graph_seed_offset": graph_seed_offset,
        "init_seed_offset": init_seed_offset,
        "sketch_seed_offset": sketch_seed_offset,
    }
    config_sha256 = hashlib.sha256(json.dumps(
        effective_config, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    protocol = {
        "steps": steps,
        "learning_rate": learning_rate,
        "refresh_every": refresh_every,
        "eps_refresh": eps_refresh,
        "block": block,
        "residual_probes": residual_probes,
        "fd_eps": fd_eps,
        "spsa_eps": spsa_eps,
        "af_rank": af_rank,
        "indicator": indicator,
        "graph_seed_offset": graph_seed_offset,
        "init_seed_offset": init_seed_offset,
        "sketch_seed_offset": sketch_seed_offset,
        "experiment_schema": 3,
        "rsqaoa_version": rsqaoa_version,
        "implementation_sha256": source_tree_fingerprint(),
        "runner_sha256": runner_fingerprint(),
        "experiment_config_sha256": config_sha256,
        "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "design_families": json.dumps(list(families), separators=(",", ":")),
        "design_n": json.dumps(list(ns), separators=(",", ":")),
        "design_p": json.dumps(list(ps), separators=(",", ":")),
        "design_tols": json.dumps(list(tols), separators=(",", ":")),
        "design_restarts": seeds,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "networkx_version": nx.__version__,
        "platform": platform.platform(),
    }
    jobs = list(itertools.product(families, ns, ps, range(seeds)))
    jobs = [job for index, job in enumerate(jobs)
            if index % shard_count == shard_index]
    for family, n, p, seed in jobs:
        graph_seed = graph_seed_offset + seed
        init_seed = init_seed_offset + seed
        sketch_seed = sketch_seed_offset + seed
        _seed_everything(init_seed)
        edges = make_edges(family, n, graph_seed)
        graph_id, edge_list = graph_fingerprint(n, edges)
        problem = MaxCutProblem(n=n, edges=edges, p=p)
        best = graphs.brute_force_maxcut(n, edges) if n <= 22 else None
        theta0 = problem.random_theta(
            generator=torch.Generator().manual_seed(init_seed)
        )

        # full ma-QAOA
        t = time.time()
        full = full_maqaoa(
            problem, theta0=theta0.clone(), steps=steps,
            lr=learning_rate, seed=init_seed
        )
        rows.append(_row("full_maqaoa", family, n, p, seed,
                         graph_seed, init_seed, sketch_seed, graph_id, edge_list,
                         "", problem, best,
                         full.cut, full.n_params_opt, "", 0, full.counts,
                         time.time() - t, protocol, full))

        # symmetry-reduced
        t = time.time()
        sym = symmetry_reduced(
            problem, theta0=theta0.clone(), steps=steps,
            lr=learning_rate, seed=init_seed
        )
        rows.append(_row("symmetry", family, n, p, seed,
                         graph_seed, init_seed, sketch_seed, graph_id, edge_list,
                         "", problem, best,
                         sym.cut, sym.n_params_opt, "", 0, sym.counts,
                         time.time() - t, protocol, sym))

        # Dense-SVD oracle used only on these small validation instances.  Its
        # Jacobian materialization is recorded separately because it is not
        # comparable to a single matrix-free JVP or VJP action.
        fixed_rank = min(8, problem.m, problem.dim)
        t = time.time()
        fixed = fixed_rank_subspace(problem, rank=fixed_rank,
                                    theta0=theta0.clone(), steps=steps,
                                    lr=learning_rate, seed=init_seed)
        rows.append(_row("fixed_rank_oracle", family, n, p, seed,
                         graph_seed, init_seed, sketch_seed, graph_id, edge_list,
                         "", problem,
                         best, fixed.cut, fixed.n_params_opt, fixed_rank, 0,
                         fixed.counts, time.time() - t, protocol, fixed))

        # RSQ (adaptive) at each tolerance
        for tol in tols:
            t = time.time()
            rsq = optimize_rsq(
                problem, theta0=theta0.clone(), tol=tol, steps=steps,
                inner_lr=learning_rate, refresh_every=refresh_every,
                eps_refresh=eps_refresh, block=block, indicator=indicator,
                residual_probes=residual_probes, fd_eps=fd_eps,
                spsa_eps=spsa_eps, seed=sketch_seed,
            )
            rows.append(_row("rsq", family, n, p, seed,
                             graph_seed, init_seed, sketch_seed,
                             graph_id, edge_list, tol, problem, best,
                             rsq.cut, rsq.final_rank, rsq.final_rank, rsq.refreshes,
                             rsq.counts, time.time() - t, protocol, rsq))

        # RSQ adjoint-free
        t = time.time()
        af = optimize_rsq(
            problem, theta0=theta0.clone(), steps=steps,
            inner_lr=learning_rate, refresh_every=refresh_every,
            eps_refresh=eps_refresh, block=block, indicator=indicator,
            adjoint_free=True, af_rank=min(af_rank, problem.dim),
            residual_probes=residual_probes, fd_eps=fd_eps,
            spsa_eps=spsa_eps, seed=sketch_seed,
        )
        rows.append(_row("rsq_adjoint_free", family, n, p, seed,
                         graph_seed, init_seed, sketch_seed, graph_id, edge_list,
                         "", problem, best,
                         af.cut, af.final_rank, af.final_rank, af.refreshes,
                         af.counts, time.time() - t, protocol, af))

        print(f"done: {family} n={n} p={p} seed={seed}", flush=True)

    destination = Path(out_path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    os.replace(temporary, destination)
    print(
        f"\nwrote {len(rows)} rows to {out_path} "
        f"(shard {shard_index + 1}/{shard_count})",
        flush=True,
    )


def main(argv=None):
    raw_args = list(sys.argv[1:] if argv is None else argv)
    provided_flags = {token.split("=", 1)[0]
                      for token in raw_args if token.startswith("--")}
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--families", nargs="+", default=["regular", "er", "ring"])
    ap.add_argument("--n", nargs="+", type=int, default=[8, 10, 12])
    ap.add_argument("--p", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--tols", nargs="+", type=float, default=[1e-1, 1e-2, 1e-3])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--refresh-every", type=int, default=25)
    ap.add_argument("--eps-refresh", type=float, default=5e-2)
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--residual-probes", type=int, default=12)
    ap.add_argument("--fd-eps", type=float, default=1e-4)
    ap.add_argument("--spsa-eps", type=float, default=1e-4)
    ap.add_argument("--af-rank", type=int, default=8)
    ap.add_argument("--indicator", choices=["fro", "spec"], default="fro")
    ap.add_argument("--graph-seed-offset", type=int, default=0)
    ap.add_argument("--init-seed-offset", type=int, default=100_000)
    ap.add_argument("--sketch-seed-offset", type=int, default=200_000)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--out", type=str, default="experiments/results/run.csv")
    args = ap.parse_args(argv)

    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        def take_config(attribute, key, flag):
            if flag not in provided_flags and key in cfg:
                setattr(args, attribute, cfg[key])

        for attribute, key, flag in [
            ("families", "families", "--families"),
            ("n", "n", "--n"), ("p", "p", "--p"),
            ("tols", "tols", "--tols"), ("seeds", "seeds", "--seeds"),
            ("steps", "steps", "--steps"),
            ("learning_rate", "learning_rate", "--learning-rate"),
            ("refresh_every", "refresh_every", "--refresh-every"),
            ("eps_refresh", "eps_refresh", "--eps-refresh"),
            ("block", "block", "--block"),
            ("residual_probes", "residual_probes", "--residual-probes"),
            ("fd_eps", "fd_eps", "--fd-eps"),
            ("spsa_eps", "spsa_eps", "--spsa-eps"),
            ("af_rank", "af_rank", "--af-rank"),
            ("indicator", "indicator", "--indicator"),
            ("graph_seed_offset", "graph_seed_offset", "--graph-seed-offset"),
            ("init_seed_offset", "init_seed_offset", "--init-seed-offset"),
            ("sketch_seed_offset", "sketch_seed_offset", "--sketch-seed-offset"),
            ("out", "out", "--out"),
        ]:
            take_config(attribute, key, flag)

    run(
        args.families, args.n, args.p, args.tols, args.seeds, args.steps,
        args.out, learning_rate=args.learning_rate,
        refresh_every=args.refresh_every, eps_refresh=args.eps_refresh,
        block=args.block, residual_probes=args.residual_probes,
        fd_eps=args.fd_eps, spsa_eps=args.spsa_eps, af_rank=args.af_rank,
        indicator=args.indicator, graph_seed_offset=args.graph_seed_offset,
        init_seed_offset=args.init_seed_offset,
        sketch_seed_offset=args.sketch_seed_offset,
        shard_index=args.shard_index, shard_count=args.shard_count,
    )


if __name__ == "__main__":
    main()
