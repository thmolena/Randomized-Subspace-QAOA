"""Validate the RSQ confirmatory design and refuse premature execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    REPOSITORY / "experiments/configs/nmi_confirmatory_design.yaml"
)


def _canonical_hash(protocol: dict) -> str:
    payload = dict(protocol)
    payload.pop("protocol_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate(protocol: dict, *, check_hash: bool = True) -> dict:
    if protocol.get("protocol_schema") != 2:
        raise ValueError("confirmatory protocol_schema must be 2")
    if protocol.get("status") != "design_only_not_registered_not_run":
        raise ValueError("confirmatory plan must remain design-only and unrun")
    registration = protocol["registration"]
    execution = protocol["execution_status"]
    if registration.get("required_before_execution") is not True:
        raise ValueError("external registration must be required")
    if execution.get("code_complete") is not False:
        raise ValueError("code cannot be marked complete before implementation")
    if execution.get("execution_permitted") is not False:
        raise ValueError("unregistered incomplete design cannot permit execution")
    boundary = protocol["confirmatory_boundary"]
    required_true = [
        "training_weights_disjoint_from_evaluation_weights",
        "confirmatory_graph_seeds_absent_from_development",
        "hyperparameters_locked_from_development",
    ]
    if not all(boundary.get(name) is True for name in required_true):
        raise ValueError("confirmatory leakage boundary is incomplete")
    if boundary.get("outcomes_generated_or_inspected") is not False:
        raise ValueError("design-only plan cannot contain inspected outcomes")
    if boundary.get("outcome_dependent_exclusions_allowed") is not False:
        raise ValueError("outcome-dependent exclusions must be prohibited")

    design = protocol["primary_design"]
    calculated_units = (
        len(design["families"]) * len(design["n"])
        * len(design["graph_seeds"])
    )
    if calculated_units != design["independent_topology_units"]:
        raise ValueError("independent topology-unit count is inconsistent")
    if calculated_units < 35:
        raise ValueError("primary design is smaller than the development power target")
    if set(design["graph_seeds"]) & {41, 42}:
        raise ValueError("development graph seeds leaked into confirmation")
    if design["evaluation_tasks"] not in protocol["secondary_design"]["task_counts"]:
        raise ValueError("primary task count is absent from the K sweep")

    required_methods = {
        "full_spsa", "amortized_gated", "amortized_per_task",
        "amortized_fixed", "amortized_random_basis",
        "mean_weight_rank1_basis", "unweighted_observable_jacobian_basis",
        "dense_svd_oracle", "symmetry_tied_maqaoa",
        "empirical_gradient_bank_active_subspace",
        "full_coordinate_fd", "task_weighted_coordinate_fd",
    }
    if set(protocol["methods"]["required"]) != required_methods:
        raise ValueError("required matched-control set changed")
    accounting = protocol["resource_accounting"]
    if not accounting["all_edge_observables_share_one_bitstring_batch"]:
        raise ValueError("commuting-observable measurement reuse is not enabled")
    if accounting["simulator_vjp_to_physical_query_conversion"] != "prohibited":
        raise ValueError("simulator VJPs must not be converted to physical queries")
    if not accounting["distinct_parameter_vectors_are_never_merged"]:
        raise ValueError("measurement reuse cannot cross parameter vectors")
    required_regimes = {
        "low_rank_drift", "changepoint", "iid_full_rank",
        "loading_shift_ood",
    }
    if set(protocol["secondary_design"]["stream_regimes"]) != required_regimes:
        raise ValueError("full-rank, changepoint, or OOD stress regime changed")

    outcomes = protocol["outcomes"]
    if outcomes["primary_estimand"] != "paired_difference_in_circuit_points_to_target":
        raise ValueError("primary target-hitting estimand changed")
    if not 0 < outcomes["quality_equivalence_margin"] < 0.1:
        raise ValueError("invalid quality equivalence margin")
    if not 0 < outcomes["operational_saving_threshold"] < 1:
        raise ValueError("invalid operational saving threshold")
    for metric in (
        "retained_gradient_energy", "principal_angle_drift",
        "optimization_gain_conditional_on_representation_metrics",
    ):
        if outcomes.get(metric) != "required":
            raise ValueError(f"representation diagnostic {metric} is required")
    if not outcomes["gate_event"].startswith("oracle_per_task_refresh"):
        raise ValueError("gate calibration target must use oracle refresh benefit")
    if protocol["hardware_stage"]["status"] != "not_executed":
        raise ValueError("hardware evidence cannot be marked complete without data")
    guardrails = protocol["publication_claim_guardrails"]
    if any([
        guardrails["quantum_advantage_claim_allowed"],
        guardrails["hardware_efficiency_claim_allowed_before_hardware_stage"],
        guardrails["query_advantage_claim_allowed_from_parameter_compression_alone"],
    ]):
        raise ValueError("publication guardrails permit an unsupported claim")

    expected_hash = _canonical_hash(protocol)
    if check_hash and protocol.get("protocol_sha256") != expected_hash:
        raise ValueError("protocol_sha256 does not match the design hash")
    return {
        "protocol_sha256": expected_hash,
        "independent_topology_units": calculated_units,
        "primary_tasks_per_unit": design["evaluation_tasks"],
        "task_count_sweep": protocol["secondary_design"]["task_counts"],
        "hardware_stage": protocol["hardware_stage"]["status"],
        "outcomes_generated_or_inspected": boundary[
            "outcomes_generated_or_inspected"
        ],
        "externally_registered": registration["external_receipt"] is not None,
        "code_complete": execution["code_complete"],
        "execution_ready": bool(
            registration["external_receipt"]
            and execution["code_complete"]
            and execution["execution_permitted"]
        ),
    }


def require_execution_ready(protocol: dict) -> None:
    """Reject outcome generation until registration and code gates are met."""
    report = validate(protocol)
    if not report["execution_ready"]:
        missing = protocol["execution_status"]["missing_components"]
        raise RuntimeError(
            "confirmatory execution refused: the design is not externally "
            "registered and implementation is incomplete; missing components: "
            + ", ".join(missing)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument(
        "--print-hash", action="store_true",
        help="print the canonical hash while allowing a pending hash field",
    )
    parser.add_argument(
        "--require-executable", action="store_true",
        help="fail unless registration and implementation permit execution",
    )
    args = parser.parse_args()
    protocol = yaml.safe_load(Path(args.protocol).read_text())
    report = validate(protocol, check_hash=not args.print_hash)
    if args.require_executable:
        require_execution_ready(protocol)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
