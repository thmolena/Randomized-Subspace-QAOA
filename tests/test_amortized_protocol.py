"""Protocol hashing is deterministic and detects source/config drift."""

from pathlib import Path

import pytest

from rsqaoa.amortized.protocol import build_protocol, validate_protocol


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY / "experiments/configs/amortized_development.yaml"


def test_protocol_hash_is_deterministic_and_validates():
    first = build_protocol(CONFIG, REPOSITORY)
    second = build_protocol(CONFIG, REPOSITORY)
    assert first == second
    assert len(first["protocol_sha256"]) == 64
    assert {
        "rsqaoa/circuits.py",
        "rsqaoa/graphs.py",
        "rsqaoa/operator.py",
        "rsqaoa/randqb.py",
    } <= set(first["source_files"])
    validate_protocol(first, CONFIG, REPOSITORY)


def test_protocol_tampering_is_detected():
    payload = build_protocol(CONFIG, REPOSITORY)
    payload["config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="config_sha256"):
        validate_protocol(payload, CONFIG, REPOSITORY)
