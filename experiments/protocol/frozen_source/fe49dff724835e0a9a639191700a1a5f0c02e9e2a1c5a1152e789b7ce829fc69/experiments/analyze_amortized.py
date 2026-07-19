"""Analyze amortized-RSQ rows at the topology/depth sampling-unit level."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPOSITORY / "experiments/results/amortized_development.csv"
DEFAULT_JSON = REPOSITORY / "experiments/results/amortized_development_summary.json"


def _mean_2se(values):
    values = list(values)
    if not values:
        return {"mean": None, "sd": None, "two_se": None, "n": 0}
    sd = stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": mean(values), "sd": sd,
        "two_se": 2.0 * sd / math.sqrt(len(values)), "n": len(values),
    }


def analyze(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("no amortized experiment rows")
    methods = sorted({row["method"] for row in rows})
    units = defaultdict(list)
    for row in rows:
        key = (row["family"], int(row["n"]), int(row["p"]),
               int(row["graph_seed"]), row["graph_id"],
               int(row.get("measurement_repeat", 0)), row["method"])
        units[key].append(row)
    expected_tasks = max(int(row["task_index"]) for row in rows) + 1
    for key, group in units.items():
        tasks = sorted(int(row["task_index"]) for row in group)
        if tasks != list(range(expected_tasks)):
            raise ValueError(f"incomplete task stream for {key}: {tasks}")

    trial_summaries = defaultdict(list)
    for key, group in units.items():
        method = key[-1]
        group = sorted(group, key=lambda row: int(row["task_index"]))
        summary = {
            "unit": key[:-1],
            "mean_ratio": mean(float(row["approximation_ratio"]) for row in group),
            "final_ratio": float(group[-1]["approximation_ratio"]),
            "forward_circuit_evaluations": sum(
                int(row["task_forward_circuit_evaluations"]) for row in group),
            "objective_evaluations": sum(
                int(row["task_objective_evaluations"]) for row in group),
            "observable_evaluations": sum(
                int(row["task_observable_evaluations"]) for row in group),
            "shots": sum(int(row["task_shots"]) for row in group),
            "simulator_vjps": sum(
                int(row["task_simulator_vjps"]) for row in group),
            "subspace_builds": sum(
                int(row["task_subspace_builds"]) for row in group),
            "refreshes": sum(int(row["task_refreshes"]) for row in group),
            "residual_checks": sum(
                int(row["task_residual_checks"]) for row in group),
            "optimized_dimension": mean(
                int(row["optimized_dimension"]) for row in group),
        }
        # Measurement repetitions are nested within a topology/depth unit;
        # they are not independent graph samples. Average them before the
        # across-topology uncertainty calculation.
        topology_unit = key[:-2]
        trial_summaries[(topology_unit, method)].append(summary)

    metric_names = [
        "mean_ratio", "final_ratio", "forward_circuit_evaluations",
        "objective_evaluations", "observable_evaluations", "shots",
        "simulator_vjps", "subspace_builds", "refreshes", "residual_checks",
        "optimized_dimension",
    ]
    per_method = defaultdict(list)
    per_unit_ratio = {}
    repeats_per_unit = []
    for (topology_unit, method), trials in trial_summaries.items():
        clustered = {"unit": topology_unit}
        for metric in metric_names:
            clustered[metric] = mean(item[metric] for item in trials)
        per_method[method].append(clustered)
        per_unit_ratio[(topology_unit, method)] = clustered["mean_ratio"]
        repeats_per_unit.append(len(trials))

    aggregate = {}
    for method in methods:
        values = per_method[method]
        aggregate[method] = {
            "mean_ratio": _mean_2se(item["mean_ratio"] for item in values),
            "final_ratio": _mean_2se(item["final_ratio"] for item in values),
            "forward_circuit_evaluations": _mean_2se(
                item["forward_circuit_evaluations"] for item in values),
            "objective_evaluations": _mean_2se(
                item["objective_evaluations"] for item in values),
            "observable_evaluations": _mean_2se(
                item["observable_evaluations"] for item in values),
            "shots": _mean_2se(item["shots"] for item in values),
            "simulator_vjps": _mean_2se(
                item["simulator_vjps"] for item in values),
            "subspace_builds": _mean_2se(
                item["subspace_builds"] for item in values),
            "refreshes": _mean_2se(item["refreshes"] for item in values),
            "residual_checks": _mean_2se(
                item["residual_checks"] for item in values),
            "optimized_dimension": _mean_2se(
                item["optimized_dimension"] for item in values),
        }

    comparisons = {}
    units_without_method = sorted({key[0] for key in per_unit_ratio})
    for method in methods:
        if method == "full_spsa":
            continue
        differences = [
            per_unit_ratio[(unit, method)]
            - per_unit_ratio[(unit, "full_spsa")]
            for unit in units_without_method
            if (unit, method) in per_unit_ratio
            and (unit, "full_spsa") in per_unit_ratio
        ]
        comparisons[f"{method}_minus_full_spsa"] = _mean_2se(differences)

    proposed = "amortized_gated"
    random_control = "amortized_random_basis"
    if proposed in aggregate and random_control in aggregate:
        quality_delta = comparisons[
            f"{proposed}_minus_full_spsa"]["mean"]
        geometry_delta = mean(
            per_unit_ratio[(unit, proposed)]
            - per_unit_ratio[(unit, random_control)]
            for unit in units_without_method
        )
        cost_ratio = (
            aggregate[proposed]["forward_circuit_evaluations"]["mean"]
            / aggregate["full_spsa"]["forward_circuit_evaluations"]["mean"]
        )
        thesis_gate = {
            "quality_delta_vs_full": quality_delta,
            "quality_delta_vs_random_basis": geometry_delta,
            "forward_cost_ratio_vs_full": cost_ratio,
            "criterion": (
                "development-only: no worse than 1 point versus full SPSA, "
                "better than a random basis with the same maximum rank cap, "
                "and <=1.25x forward cost"
            ),
            "survives_first_pilot": bool(
                quality_delta >= -0.01 and geometry_delta > 0.0
                and cost_ratio <= 1.25
            ),
        }
    else:
        thesis_gate = None

    return {
        "analysis_schema": 1,
        "development_only": True,
        "n_rows": len(rows),
        "n_sampling_units": len(units_without_method),
        "measurement_repeats_per_method_unit": sorted(set(repeats_per_unit)),
        "n_tasks_per_unit": expected_tasks,
        "methods": methods,
        "aggregate": aggregate,
        "paired_comparisons": comparisons,
        "thesis_gate": thesis_gate,
        "protocol_sha256": sorted({row["protocol_sha256"] for row in rows}),
        "implementation_sha256": sorted(
            {row["implementation_sha256"] for row in rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--output", default=str(DEFAULT_JSON))
    args = parser.parse_args()
    with Path(args.csv).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = analyze(rows)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["thesis_gate"], indent=2, sort_keys=True))
    print(destination)


if __name__ == "__main__":
    main()
