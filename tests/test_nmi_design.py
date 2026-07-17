"""The confirmatory design is hash-stable and cannot be run prematurely."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from experiments.validate_nmi_design import (
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
    changed["outcomes"]["target_approximation_ratio"] = 0.81
    with pytest.raises(ValueError, match="protocol_sha256"):
        validate(changed)
    with pytest.raises(RuntimeError, match="execution refused"):
        require_execution_ready(_design())
