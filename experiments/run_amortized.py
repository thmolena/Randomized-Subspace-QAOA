"""Run the development or frozen confirmatory amortized-RSQ protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
from pathlib import Path

import numpy as np  # import before torch on macOS to avoid mixed OpenMP loading
import networkx as nx
import torch
import yaml

from rsqaoa import graphs
from rsqaoa.circuits import MaxCutProblem
from rsqaoa.amortized import (SPSAConfig, make_low_rank_weight_model,
                              sample_low_rank_drift_stream,
                              optimize_amortized_stream,
                              optimize_full_stream, validate_protocol)


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / "experiments/configs/amortized_development.yaml"
DEFAULT_PROTOCOL = REPOSITORY / "experiments/protocol/amortized_development.json"


def _edges(family: str, n: int, seed: int):
    if family == "regular":
        return graphs.random_regular(n, degree=3, seed=seed)
    if family == "er":
        return graphs.erdos_renyi(n, prob=0.5, seed=seed)
    if family == "ring":
        return graphs.ring(n)
    raise ValueError(f"unknown graph family {family!r}")


def _graph_id(n: int, edges) -> str:
    normalized = sorted(tuple(sorted((int(i), int(j)))) for i, j in edges)
    payload = json.dumps({"n": n, "edges": normalized}, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _weight_hash(weights: torch.Tensor) -> str:
    values = weights.detach().cpu().numpy().astype("<f8", copy=False)
    return hashlib.sha256(values.tobytes()).hexdigest()[:16]


RESOURCE_NAMES = [
    "objective_evaluations", "observable_evaluations", "shots",
    "sensitivity_jvps", "simulator_vjps", "subspace_builds",
    "residual_checks", "refreshes", "forward_circuit_evaluations",
]


FIELDS = [
    "method", "family", "n", "p", "graph_seed", "graph_id",
    "task_index", "weight_hash", "basis_bank_hash", "stream_mode",
    "weight_model_seed", "basis_stream_seed", "stream_seed",
    "measurement_repeat", "measurement_seed",
    "latent_rank", "drift_scale", "ambient_dimension",
    "optimized_dimension", "basis_rank", "exact_cut", "exact_optimum",
    "approximation_ratio", "reported_best_cut", "final_cut", "residual",
    "refreshed", "build_status_json", "build_stop_reason",
    "build_converged", "build_relative_residual",
    "protocol_sha256", "config_sha256",
    "implementation_sha256", "python_version", "numpy_version",
    "torch_version", "networkx_version", "platform",
] + [f"task_{name}" for name in RESOURCE_NAMES] + [
    f"cumulative_{name}" for name in RESOURCE_NAMES
]


def run(config: dict, protocol: dict, *, output: Path,
        shard_index: int = 0, shard_count: int = 1,
        limit_jobs: int | None = None) -> None:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard index/count")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    design = config["design"]
    subspace = config["subspace"]
    refresh = config["refresh"]
    evaluation = config["evaluation"]
    spsa = SPSAConfig(**config["spsa"])
    jobs = list(itertools.product(
        design["families"], design["n"], design["p"],
        design["graph_seeds"],
    ))
    jobs = [job for index, job in enumerate(jobs)
            if index % shard_count == shard_index]
    if limit_jobs is not None:
        jobs = jobs[:limit_jobs]
    rows = []
    for job_index, (family, n, p, graph_seed) in enumerate(jobs):
        edges = _edges(family, int(n), int(graph_seed))
        if not edges:
            raise ValueError(f"empty graph for {family} n={n} seed={graph_seed}")
        graph_id = _graph_id(int(n), edges)
        problem = MaxCutProblem(n=int(n), edges=edges, p=int(p))
        seed_suffix = (
            1000 * int(graph_seed)
            + 100 * int(n) + 10 * int(p)
            + sum(ord(char) for char in family)
        )
        model_seed = int(design["weight_model_seed_offset"]) + seed_suffix
        basis_stream_seed = (
            int(design["basis_stream_seed_offset"]) + seed_suffix
        )
        stream_seed = (
            int(design["evaluation_stream_seed_offset"]) + seed_suffix
        )
        weight_model = make_low_rank_weight_model(
            problem.m, latent_rank=int(design["latent_rank"]),
            seed=model_seed,
        )
        basis_stream = sample_low_rank_drift_stream(
            weight_model, problem.m,
            n_tasks=int(design["basis_n_tasks"]),
            drift_scale=float(design["drift_scale"]),
            persistence=float(design["persistence"]),
            seed=basis_stream_seed,
        )
        stream = sample_low_rank_drift_stream(
            weight_model, problem.m, n_tasks=int(design["n_tasks"]),
            drift_scale=float(design["drift_scale"]),
            persistence=float(design["persistence"]), seed=stream_seed,
        )
        basis_bank_hash = _weight_hash(basis_stream.weights)
        init_seed = int(design["init_seed_offset"]) + 1000 * job_index
        theta0 = problem.random_theta(
            generator=torch.Generator().manual_seed(init_seed)
        )
        optimizer_seed = int(design["optimizer_seed_offset"]) + 1000 * job_index
        measurement_seeds = evaluation.get("measurement_seeds", [0])
        for measurement_repeat, measurement_offset in enumerate(
                measurement_seeds):
            measurement_seed = (
                600_000 + 1000 * job_index + int(measurement_offset)
            )
            for method in config["methods"]:
                if method == "full_spsa":
                    result = optimize_full_stream(
                        problem, stream, theta0.clone(), spsa=spsa,
                        seed=optimizer_seed, measurement_seed=measurement_seed,
                        shots=evaluation["shots"],
                        readout_error=float(evaluation["readout_error"]),
                    )
                elif method.startswith("amortized_"):
                    mode = method.removeprefix("amortized_")
                    result = optimize_amortized_stream(
                        problem, stream, theta0.clone(), mode=mode,
                        rank=int(subspace["rank"]),
                        subspace_tol=float(subspace["tolerance"]),
                        subspace_block=int(subspace["block"]),
                        subspace_norm_probes=int(subspace["norm_probes"]),
                        subspace_residual_probes=int(
                            subspace["residual_probes"]),
                        refresh_every_tasks=int(refresh["every_tasks"]),
                        refresh_threshold=float(refresh["threshold"]),
                        gate_probes=int(refresh["gate_probes"]),
                        gate_fd_eps=float(refresh["gate_fd_eps"]),
                        random_refresh_probability=float(
                            refresh["random_probability"]),
                        spsa=spsa, seed=optimizer_seed,
                        measurement_seed=measurement_seed,
                        shots=evaluation["shots"],
                        readout_error=float(evaluation["readout_error"]),
                        basis_weights=basis_stream.weights,
                    )
                else:
                    raise ValueError(f"unknown method {method!r}")

                cumulative = result.ledger.__class__()
                builds_by_task = {
                    int(status["task"]): status
                    for status in result.build_status
                }
                for record, weights in zip(result.records, stream.weights):
                    cumulative.add(record.ledger)
                    numeric_weights = weights.detach().cpu().numpy()
                    optimum = graphs.brute_force_maxcut(
                        problem.n, problem.edges, numeric_weights
                    )
                    task_resources = record.ledger.as_dict()
                    cumulative_resources = cumulative.as_dict()
                    build_status = builds_by_task.get(record.task_index)
                    row = {
                        "method": method, "family": family, "n": int(n),
                        "p": int(p), "graph_seed": int(graph_seed),
                        "graph_id": graph_id, "task_index": record.task_index,
                        "weight_hash": _weight_hash(weights),
                        "basis_bank_hash": basis_bank_hash,
                        "stream_mode": stream.mode,
                        "weight_model_seed": weight_model.seed,
                        "basis_stream_seed": basis_stream.seed,
                        "stream_seed": stream.seed,
                        "measurement_repeat": measurement_repeat,
                        "measurement_seed": measurement_seed,
                        "latent_rank": stream.latent_rank,
                        "drift_scale": stream.drift_scale,
                        "ambient_dimension": problem.dim,
                        "optimized_dimension": record.optimized_dimension,
                        "basis_rank": record.basis_rank,
                        "exact_cut": f"{record.exact_cut:.12g}",
                        "exact_optimum": f"{optimum:.12g}",
                        "approximation_ratio": (
                            f"{record.exact_cut / optimum:.12g}"),
                        "reported_best_cut": f"{record.reported_best_cut:.12g}",
                        "final_cut": f"{record.final_cut:.12g}",
                        "residual": "" if record.residual is None
                        else f"{record.residual:.12g}",
                        "refreshed": record.refreshed,
                        "build_status_json": (
                            "[]" if build_status is None else json.dumps(
                                [build_status], sort_keys=True,
                                separators=(",", ":"),
                            )
                        ),
                        "build_stop_reason": (
                            "" if build_status is None
                            else build_status["stop_reason"]
                        ),
                        "build_converged": (
                            "" if build_status is None
                            else build_status["converged"]
                        ),
                        "build_relative_residual": (
                            "" if build_status is None
                            or build_status["relative_residual"] is None
                            else f"{build_status['relative_residual']:.12g}"
                        ),
                        "protocol_sha256": protocol["protocol_sha256"],
                        "config_sha256": protocol["config_sha256"],
                        "implementation_sha256": (
                            protocol["implementation_sha256"]),
                        "python_version": platform.python_version(),
                        "numpy_version": np.__version__,
                        "torch_version": torch.__version__,
                        "networkx_version": nx.__version__,
                        "platform": platform.platform(),
                    }
                    row.update({f"task_{name}": task_resources[name]
                                for name in RESOURCE_NAMES})
                    row.update({
                        f"cumulative_{name}": cumulative_resources[name]
                        for name in RESOURCE_NAMES
                    })
                    rows.append(row)
        print(
            f"done {family} n={n} p={p} seed={graph_seed} "
            f"({job_index + 1}/{len(jobs)})", flush=True
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)
    print(f"wrote {len(rows)} rows to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output", default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit-jobs", type=int, default=None)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    protocol = json.loads(Path(args.protocol).read_text())
    validate_protocol(protocol, config_path, REPOSITORY)
    output = Path(args.output or config["output"])
    if not output.is_absolute():
        output = REPOSITORY / output
    run(
        config, protocol, output=output,
        shard_index=args.shard_index, shard_count=args.shard_count,
        limit_jobs=args.limit_jobs,
    )


if __name__ == "__main__":
    main()
