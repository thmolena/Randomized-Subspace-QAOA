from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import rsqaoa
import rsqaoa_repro
from rsqaoa_repro.cli import _verify_files, full_rerun, replay


CODE = Path(__file__).resolve().parents[1]


def test_versions_match() -> None:
    assert rsqaoa_repro.__version__ == rsqaoa.__version__ == "0.3.0"


def test_source_tree_is_byte_identical() -> None:
    subprocess.run(
        [sys.executable, str(CODE / "tools/check_source_sync.py")],
        check=True,
    )


def test_replay_validates_all_committed_evidence() -> None:
    report = replay()
    assert report["mode"] == "byte-hash-replay"
    assert report["legacy_rows"] == 216
    assert report["amortized"]["amortized_development"]["rows"] == 896
    assert report["amortized"]["amortized_shot_development"]["rows"] == 576
    assert report["files"] == 82
    assert report["asset_files"] == 23
    assert report["confirmatory_design_status"] == (
        "design_only_incomplete_unregistered"
    )


def test_replay_ignores_interpreter_caches(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("released\n", encoding="utf-8")
    cache = tmp_path / "__pycache__/script.cpython-313.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"interpreter cache")
    payload = evidence.read_bytes()
    report = _verify_files(
        tmp_path,
        {
            "files": {
                "evidence.txt": {
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size": len(payload),
                }
            },
            "manifest_sha256": "test",
        },
    )
    assert report["files"] == 1


def test_full_dry_run_names_every_study(tmp_path: Path) -> None:
    report = full_rerun(tmp_path / "rerun", quick=False, dry_run=True)
    serialized = json.dumps(report)
    assert "maxcut_small.yaml" in serialized
    assert "amortized_development.yaml" in serialized
    assert "amortized_shot_development.yaml" in serialized
    assert "validate_nmi_design.py" in serialized
    assert report["bitwise_identity_claimed"] is False
