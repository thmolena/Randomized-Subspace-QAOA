"""The confirmatory design is hash-stable and cannot be run prematurely."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from experiments.validate_nmi_design import (
    _canonical_hash,
    require_execution_ready,
    validate,
)


DESIGN = (
    Path(__file__).resolve().parents[1]
    / "experiments/configs/nmi_confirmatory_design.yaml"
)


def _design():
    return yaml.safe_load(DESIGN.read_text())


def test_design_is_valid_but_explicitly_not_execution_ready():
    report = validate(_design())
    assert report["independent_topology_units"] == 40
    assert report["execution_ready"] is False
    assert report["externally_registered"] is False
    assert report["code_complete"] is False


def test_design_tampering_and_premature_execution_are_rejected():
    changed = deepcopy(_design())
    changed["outcomes"]["primary_estimand"][
        "target_approximation_ratio"
    ] = 0.81
    with pytest.raises(ValueError, match="protocol_sha256"):
        validate(changed)
    with pytest.raises(RuntimeError, match="execution refused"):
        require_execution_ready(_design())


def test_registered_execution_ready_lifecycle_is_representable():
    future = _design()
    future["status"] = "registered_execution_ready"
    future["registration"].update({
        "external_receipt": "registry:example:1",
        "registry": "example-registry",
        "registered_artifact_sha256": "1" * 64,
    })
    execution = future["execution_status"]
    execution["implemented_components"] = list(execution["required_components"])
    execution["missing_components"] = []
    execution["code_complete"] = True
    execution["execution_permitted"] = True
    for group in ("new_controls", "hardware_choices"):
        future["parameter_locks"][group]["freeze_receipt"] = "freeze:example:1"
        future["parameter_locks"][group]["configuration_sha256"] = "2" * 64
    future["primary_design"]["sample_size_status"] = (
        "frozen_after_censored_primary_power_analysis"
    )
    seeds = future["rng_and_seed_schedule"]
    seeds["status"] = "implemented_and_frozen"
    seeds["seed_manifest_sha256"] = "3" * 64
    power = future["sample_size_and_power"]["primary_endpoint_power_analysis"]
    power["status"] = "performed_and_frozen"
    power["frozen_independent_topology_units"] = 40
    future["success_rule"]["status"] = "registered_and_frozen"
    future["protocol_sha256"] = _canonical_hash(future)

    report = validate(future)
    assert report["externally_registered"] is True
    assert report["code_complete"] is True
    assert report["execution_ready"] is True
