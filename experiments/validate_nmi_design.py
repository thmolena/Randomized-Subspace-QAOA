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
LIFECYCLE_STATUSES = {
    "design_only_incomplete_unregistered",
    "implementation_complete_unregistered",
    "registered_execution_locked",
    "registered_execution_ready",
}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_hash(protocol: dict) -> str:
    payload = dict(protocol)
    payload.pop("protocol_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate(protocol: dict, *, check_hash: bool = True) -> dict:
    """Validate schema, leakage guards, accounting, and execution locks."""
    if protocol.get("protocol_schema") != 3:
        raise ValueError("confirmatory protocol_schema must be 3")
    if protocol.get("status") not in LIFECYCLE_STATUSES:
        raise ValueError("unknown confirmatory lifecycle status")
    expected_hash = _canonical_hash(protocol)
    if check_hash and protocol.get("protocol_sha256") != expected_hash:
        raise ValueError("protocol_sha256 does not match the design hash")
    registration = protocol["registration"]
    execution = protocol["execution_status"]
    if registration.get("required_before_execution") is not True:
        raise ValueError("external registration must be required")
    registration_values = [
        registration.get(field)
        for field in (
            "external_receipt",
            "registry",
            "registered_artifact_sha256",
        )
    ]
    if any(value is not None for value in registration_values) and not all(
        value is not None for value in registration_values
    ):
        raise ValueError("registration metadata must be wholly absent or complete")
    registered = all(value is not None for value in registration_values)
    if registered and not _is_sha256(registration["registered_artifact_sha256"]):
        raise ValueError("registered artifact SHA-256 is malformed")

    required_components = set(execution["required_components"])
    implemented_components = set(execution["implemented_components"])
    missing_components = set(execution["missing_components"])
    if implemented_components & missing_components:
        raise ValueError("implemented and missing components overlap")
    if implemented_components | missing_components != required_components:
        raise ValueError("implementation-status lists do not partition requirements")
    code_complete = not missing_components
    if execution.get("code_complete") is not code_complete:
        raise ValueError("code_complete does not match the component partition")
    execution_permitted = execution.get("execution_permitted")
    if not isinstance(execution_permitted, bool):
        raise ValueError("execution_permitted must be Boolean")
    if execution_permitted and not (registered and code_complete):
        raise ValueError(
            "execution requires complete code and external registration"
        )
    if execution_permitted:
        expected_status = "registered_execution_ready"
    elif registered:
        expected_status = "registered_execution_locked"
    elif code_complete:
        expected_status = "implementation_complete_unregistered"
    else:
        expected_status = "design_only_incomplete_unregistered"
    if protocol["status"] != expected_status:
        raise ValueError(
            f"lifecycle status must be {expected_status!r} for these gates"
        )

    boundary = protocol["confirmatory_boundary"]
    if not boundary["training_weights_disjoint_from_evaluation_weights"]:
        raise ValueError("training and evaluation weights must be disjoint")
    if not boundary["confirmatory_graph_seed_ids_absent_from_development"]:
        raise ValueError("confirmatory graph seed IDs must be held out")
    if boundary["outcome_dependent_exclusions_allowed"]:
        raise ValueError("outcome-dependent exclusions must be prohibited")
    if boundary["outcomes_generated_or_inspected"]:
        raise ValueError("pre-execution protocol cannot contain inspected outcomes")
    if boundary["inspected_outcome_count"] != 0:
        raise ValueError("inspected_outcome_count must remain zero")
    if boundary["outcome_artifacts"]:
        raise ValueError("unrun design cannot list outcome artifacts")

    locks = protocol["parameter_locks"]
    existing = locks["existing_implemented_methods"]
    if existing["status"] != "locked_to_frozen_development_configuration":
        raise ValueError("implemented methods are not locked")
    if existing["confirmatory_retuning_allowed"]:
        raise ValueError("confirmatory retuning must be prohibited")
    for group in ("new_controls", "hardware_choices"):
        if locks[group]["confirmatory_outcomes_allowed_for_tuning"]:
            raise ValueError(
                f"{group} permit tuning on confirmatory outcomes"
            )
        receipt = locks[group]["freeze_receipt"]
        configuration_sha256 = locks[group]["configuration_sha256"]
        if (receipt is None) != (configuration_sha256 is None):
            raise ValueError(
                f"{group} freeze receipt and configuration hash must co-occur"
            )
        if configuration_sha256 is not None and not _is_sha256(
            configuration_sha256
        ):
            raise ValueError(f"{group} configuration SHA-256 is malformed")
        if registered and receipt is None:
            raise ValueError(f"{group} must be frozen before registration")

    design = protocol["primary_design"]
    calculated_units = (
        len(design["families"])
        * len(design["n"])
        * len(design["graph_seed_ids"])
    )
    if calculated_units != design["independent_topology_units"]:
        raise ValueError("independent topology-unit count is inconsistent")
    if set(design["graph_seed_ids"]) & {41, 42}:
        raise ValueError("development graph seed IDs leaked into confirmation")
    if design["evaluation_tasks"] not in protocol["secondary_design"]["task_counts"]:
        raise ValueError("primary task count is absent from the K sweep")
    expected_sample_size_status = (
        "frozen_after_censored_primary_power_analysis"
        if registered
        else "provisional_feasibility_grid_not_powered_for_censored_primary"
    )
    if design["sample_size_status"] != expected_sample_size_status:
        raise ValueError("sample-size status is inconsistent with registration")

    seeds = protocol["rng_and_seed_schedule"]
    if seeds["collision_policy"] != "abort_before_execution":
        raise ValueError("seed collisions must abort execution")
    if code_complete:
        if seeds["status"] != "implemented_and_frozen":
            raise ValueError("complete code requires a frozen seed schedule")
        if not _is_sha256(seeds["seed_manifest_sha256"]):
            raise ValueError("complete code requires a seed-manifest SHA-256")
    else:
        if seeds["status"] != "specified_not_implemented":
            raise ValueError("incomplete code must mark the seed schedule unimplemented")
        if seeds["seed_manifest_sha256"] is not None:
            raise ValueError("unimplemented seed schedule cannot have a manifest")

    power = protocol["sample_size_and_power"]
    power_analysis = power["primary_endpoint_power_analysis"]
    if not power_analysis["required_before_external_registration"]:
        raise ValueError("power analysis must precede external registration")
    if registered:
        if power_analysis["status"] != "performed_and_frozen":
            raise ValueError("registered design requires a frozen power analysis")
        if (
            power_analysis["frozen_independent_topology_units"]
            != calculated_units
        ):
            raise ValueError("frozen powered topology count differs from design")
    else:
        if power["supports_power_claim_for_primary_endpoint"]:
            raise ValueError("unregistered grid cannot support a power claim")
        if power_analysis["status"] != "not_performed":
            raise ValueError("unregistered primary power analysis must be pending")
        if power_analysis["frozen_independent_topology_units"] is not None:
            raise ValueError("topology count cannot be frozen before power analysis")

    required_methods = {
        "full_spsa",
        "amortized_gated",
        "amortized_per_task",
        "amortized_fixed",
        "amortized_random_basis",
        "mean_weight_rank1_basis",
        "unweighted_observable_jacobian_basis",
        "dense_svd_oracle",
        "symmetry_tied_maqaoa",
        "empirical_gradient_bank_active_subspace",
        "full_coordinate_fd",
        "task_weighted_coordinate_fd",
    }
    if set(protocol["methods"]["required"]) != required_methods:
        raise ValueError("required matched-control set changed")
    comparison = protocol["methods"]["primary_comparison"]
    if comparison["experimental"] != "task_weighted_coordinate_fd":
        raise ValueError("primary experimental method changed")
    if comparison["comparator"] != "full_coordinate_fd":
        raise ValueError("primary comparator changed")

    random_control = protocol["random_refresh_control"]
    source_rate = (
        random_control["source_refresh_events"]
        / random_control["source_refresh_opportunities"]
    )
    if abs(source_rate - random_control["source_probability"]) > 1e-15:
        raise ValueError("random-refresh source probability is inconsistent")
    if random_control["primary_refreshes_per_stratum"] > (
        random_control["primary_slots_per_stratum"]
    ):
        raise ValueError("random-refresh schedule exceeds available slots")
    if random_control["confirmatory_gated_events_used_to_set_schedule"]:
        raise ValueError("random-refresh control leaks confirmatory events")
    if not random_control["schedule_manifest_required_before_execution"]:
        raise ValueError("random-refresh schedule must be precommitted")

    oracle = protocol["oracle_gate_labels"]
    if oracle["labels_or_branch_outcomes_may_change_primary_trajectories"]:
        raise ValueError("oracle branches cannot alter primary trajectories")
    if oracle["labels_or_branch_outcomes_may_tune_gate_on_same_topology"]:
        raise ValueError("oracle labels cannot tune the same topology")
    if oracle["oracle_branch_resources"] != (
        "separate_audit_ledger_excluded_from_deployed_method_cost"
    ):
        raise ValueError("oracle branch resources are not separated")

    accounting = protocol["resource_accounting"]
    if not accounting["all_edge_observables_share_one_bitstring_batch"]:
        raise ValueError("commuting-observable measurement reuse is disabled")
    if accounting["simulator_vjp_to_physical_query_conversion"] != "prohibited":
        raise ValueError("simulator VJPs must not become physical queries")
    if not accounting["distinct_parameter_vectors_are_never_merged"]:
        raise ValueError("measurement reuse cannot cross parameter vectors")
    if not accounting["right_censor_unreached_targets"]:
        raise ValueError("unreached targets must be right censored")
    if accounting["overhead_attribution"]["prorating_or_omitting_overhead"] != (
        "prohibited"
    ):
        raise ValueError("basis and gate overhead cannot be omitted")

    required_regimes = {
        "low_rank_drift",
        "changepoint",
        "iid_full_rank",
        "loading_shift_ood",
    }
    if set(protocol["secondary_design"]["stream_regimes"]) != required_regimes:
        raise ValueError("full-rank, changepoint, or OOD regime changed")

    outcomes = protocol["outcomes"]
    primary = outcomes["primary_estimand"]
    if primary["name"] != (
        "topology_paired_difference_in_restricted_mean_circuit_points_to_target"
    ):
        raise ValueError("primary restricted-mean estimand changed")
    if primary["target_approximation_ratio"] != 0.80:
        raise ValueError("primary target approximation ratio changed")
    if primary["non_hit_treatment"] != (
        "assign_fixed_horizon_and_include_in_restricted_mean"
    ):
        raise ValueError("primary non-hit treatment changed")
    if primary["sampling_unit"] != "graph_topology":
        raise ValueError("primary sampling unit must be graph topology")
    if primary["uncertainty"] != (
        "family_size_stratified_topology_cluster_bootstrap"
    ):
        raise ValueError("primary uncertainty procedure changed")
    if not 0 < outcomes["quality_equivalence_margin"] < 0.1:
        raise ValueError("invalid quality equivalence margin")
    if outcomes["primary_operational_effect"]["success_threshold"] != 0.80:
        raise ValueError("twenty-percent operational threshold changed")
    for metric in (
        "retained_gradient_energy",
        "principal_angle_drift",
        "optimization_gain_conditional_on_representation_metrics",
    ):
        if outcomes.get(metric) != "required":
            raise ValueError(f"representation diagnostic {metric} is required")

    analysis = protocol["analysis"]
    if analysis["sampling_unit"] != "graph_topology":
        raise ValueError("analysis sampling unit changed")
    if not analysis["cluster_tasks_depths_and_measurement_repeats_within_topology"]:
        raise ValueError("nested observations must be clustered by topology")
    if (
        analysis["primary_endpoint_power_or_precision_claim_currently_allowed"]
        and not (
            registered
            and power["supports_power_claim_for_primary_endpoint"]
        )
    ):
        raise ValueError("primary power claim is not supported by a registered design")

    success = protocol["success_rule"]
    expected_success_status = (
        "registered_and_frozen"
        if registered
        else "provisional_until_primary_endpoint_power_analysis_is_frozen"
    )
    if success["status"] != expected_success_status:
        raise ValueError("success-rule status is inconsistent with registration")
    if not success["all_conditions_required"]:
        raise ValueError("success rule must remain conjunctive")
    if protocol["hardware_stage"]["status"] != "not_executed":
        raise ValueError("hardware evidence cannot be marked complete")

    guardrails = protocol["publication_claim_guardrails"]
    if any(
        (
            guardrails["quantum_advantage_claim_allowed"],
            guardrails["hardware_efficiency_claim_allowed_before_hardware_stage"],
            guardrails["query_advantage_claim_allowed_from_parameter_compression_alone"],
        )
    ):
        raise ValueError("publication guardrails permit an unsupported claim")
    if not guardrails["negative_or_null_result_must_be_reported"]:
        raise ValueError("negative-result reporting guardrail was removed")

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
        "code_complete": code_complete,
        "execution_ready": bool(
            registered and code_complete and execution_permitted
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
        "--print-hash",
        action="store_true",
        help="print the canonical hash while allowing a pending hash field",
    )
    parser.add_argument(
        "--require-executable",
        action="store_true",
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
