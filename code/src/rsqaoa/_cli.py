"""Configurable RSQ comparison on one generated MaxCut instance."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np  # import before torch to avoid mixed macOS OpenMP runtimes
import torch

from . import graphs
from . import __version__
from .baselines import full_maqaoa, symmetry_reduced
from .circuits import MaxCutProblem
from .randqb import residual_ratio_confidence
from .subspace_opt import optimize_rsq


def _make_edges(family: str, n: int, seed: int):
    if family == "regular":
        return graphs.random_regular(n, degree=3, seed=seed)
    if family == "er":
        return graphs.erdos_renyi(n, prob=0.5, seed=seed)
    if family == "ring":
        return graphs.ring(n)
    raise ValueError(family)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="RSQ and matched baselines on one exact-statevector MaxCut instance."
    )
    parser.add_argument("--version", action="version", version=f"rsqaoa {__version__}")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--family", choices=["regular", "er", "ring"],
                        default="regular")
    parser.add_argument("--tol", type=float, default=1e-2)
    parser.add_argument("--maxrank", type=int, default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--inner-lr", type=float, default=0.05)
    parser.add_argument("--refresh-every", type=int, default=25)
    parser.add_argument("--eps-refresh", type=float, default=5e-2)
    parser.add_argument("--block", type=int, default=4)
    parser.add_argument("--indicator", choices=["fro", "spec"], default="fro")
    parser.add_argument("--no-recycle", action="store_true")
    parser.add_argument("--step-cap", type=float, default=None)
    parser.add_argument("--adjoint-free", action="store_true")
    parser.add_argument("--af-rank", type=int, default=8)
    parser.add_argument("--residual-probes", type=int, default=12)
    parser.add_argument("--fd-eps", type=float, default=1e-4)
    parser.add_argument("--spsa-eps", type=float, default=1e-4)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    if args.n < 4 or args.p < 1 or args.steps < 1 or args.residual_probes < 1:
        parser.error("n >= 4, p >= 1, steps >= 1, and residual-probes >= 1 are required")
    if not 0.0 < args.tol < 1.0:
        parser.error("tol must lie in (0, 1)")
    if args.maxrank is not None and args.maxrank < 1:
        parser.error("maxrank must be positive when supplied")
    if args.inner_lr <= 0 or args.fd_eps <= 0 or args.spsa_eps <= 0:
        parser.error("inner-lr, fd-eps, and spsa-eps must be positive")
    if args.refresh_every < 0 or args.eps_refresh < 0 or args.block < 1:
        parser.error(
            "refresh-every and eps-refresh must be nonnegative; block must be positive"
        )
    if args.step_cap is not None and args.step_cap <= 0:
        parser.error("step-cap must be positive when supplied")
    if args.af_rank < 1:
        parser.error("af-rank must be positive")
    if args.adjoint_free and args.maxrank is not None:
        parser.error("use af-rank, not maxrank, with adjoint-free mode")
    if args.adjoint_free and args.indicator != "fro":
        parser.error("adjoint-free mode requires the Frobenius indicator")
    if not 0.0 < args.confidence < 1.0:
        parser.error("confidence must lie in (0, 1)")
    if args.family == "regular" and (args.n <= 3 or (3 * args.n) % 2):
        parser.error("3-regular graphs require n > 3 and an even n")
    np.random.seed(args.seed)

    edges = _make_edges(args.family, args.n, args.seed)
    if not edges:
        parser.error("the generated graph has no edges; choose another seed")
    problem = MaxCutProblem(n=args.n, edges=edges, p=args.p)
    best = graphs.brute_force_maxcut(args.n, edges) if args.n <= 22 else None
    generator = torch.Generator().manual_seed(args.seed)
    theta0 = problem.random_theta(generator=generator)

    rsq = optimize_rsq(
        problem,
        theta0=theta0.clone(),
        tol=args.tol,
        maxrank=args.maxrank,
        steps=args.steps,
        inner_lr=args.inner_lr,
        refresh_every=args.refresh_every,
        eps_refresh=args.eps_refresh,
        block=args.block,
        indicator=args.indicator,
        recycle=not args.no_recycle,
        step_cap=args.step_cap,
        adjoint_free=args.adjoint_free,
        af_rank=args.af_rank,
        residual_probes=args.residual_probes,
        fd_eps=args.fd_eps,
        spsa_eps=args.spsa_eps,
        seed=args.seed,
    )
    full = full_maqaoa(problem, theta0=theta0.clone(), steps=args.steps,
                       seed=args.seed)
    symmetry = symmetry_reduced(problem, theta0=theta0.clone(), steps=args.steps,
                                seed=args.seed)
    envelope = (residual_ratio_confidence(
        args.residual_probes, failure_probability=1.0 - args.confidence)
        if rsq.indicator in {"fro", "fro-forward"} else None)

    def row(name, result, rank=None):
        return {
            "method": name,
            "cut": result.cut,
            "approximation_ratio": result.cut / best if best else None,
            "parameters_optimized": rank if rank is not None else result.n_params_opt,
        }

    rows = [
        row("forward_only_rsq" if args.adjoint_free else "rsq",
            rsq, rsq.final_rank),
        row("full_maqaoa", full),
        row("symmetry", symmetry),
    ]
    envelope_payload = None if envelope is None else dict(envelope.__dict__)
    if (envelope_payload is not None
            and not math.isfinite(envelope_payload["upper_multiplier"])):
        envelope_payload["upper_multiplier"] = None
    payload = {
        "problem": {"family": args.family, "n": args.n, "p": args.p,
                    "edges": problem.m, "ambient_dimension": problem.dim,
                    "seed": args.seed},
        "rsq": {
            "refreshes": rsq.refreshes,
            "refresh_steps": rsq.refresh_steps,
            "subspace_builds": rsq.subspace_builds,
            "build_history": rsq.build_history,
            "operator_counts": rsq.counts,
            "indicator": rsq.indicator,
            "final_cut": rsq.cut,
            "best_cut": rsq.best_cut,
            "best_step": rsq.best_step,
            "residual_checks": [
                {"step": step, "estimate": value}
                for step, value in zip(rsq.residual_steps, rsq.residual_history)
            ],
            "probe_confidence_envelope": envelope_payload,
        },
        "results": rows,
    }
    if args.as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(
        f"RSQ comparison family={args.family} n={args.n} p={args.p} "
        f"d={problem.dim} |E|={problem.m}"
    )
    print(f"{'method':<22}{'cut':>11}{'ratio':>11}{'parameters':>13}")
    for item in rows:
        ratio = "n/a" if item["approximation_ratio"] is None else f"{item['approximation_ratio']:.4f}"
        print(f"{item['method']:<22}{item['cut']:>11.4f}{ratio:>11}{item['parameters_optimized']:>13}")
    print(f"operator_counts={rsq.counts} refreshes={rsq.refreshes}")
    if envelope is not None and not envelope.informative:
        print(
            "finite-probe note: the elementary run-wise confidence envelope is "
            "non-informative at this probe count; use the observed ratio as a diagnostic"
        )


if __name__ == "__main__":
    main()
