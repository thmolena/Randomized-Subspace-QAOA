"""Deterministic configuration and implementation hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def source_hash(paths: Iterable[Path], root: Path) -> tuple[str, dict]:
    root = Path(root).resolve()
    files = {}
    digest = hashlib.sha256()
    for path in sorted((Path(item).resolve() for item in paths), key=str):
        relative = path.relative_to(root).as_posix()
        value = sha256_file(path)
        files[relative] = value
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest(), files


def git_commit(repository: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def build_protocol(config_path: Path, repository: Path) -> dict:
    import yaml

    repository = Path(repository).resolve()
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text())
    # Hash the complete package source tree because the amortized workflow
    # calls inherited circuit, graph, sensitivity and randomized-QB modules.
    # Restricting the fingerprint to the new subpackage would miss uncommitted
    # drift in those transitive dependencies.
    source_paths = list((repository / "rsqaoa").rglob("*.py"))
    source_paths.extend([
        repository / "experiments/run_amortized.py",
        repository / "experiments/analyze_amortized.py",
    ])
    implementation, source_files = source_hash(source_paths, repository)
    payload = {
        "protocol_schema": 1,
        "config": config,
        "config_path": config_path.relative_to(repository).as_posix(),
        "config_sha256": sha256_file(config_path),
        "implementation_sha256": implementation,
        "source_files": source_files,
        "git_commit_at_freeze": git_commit(repository),
    }
    payload["protocol_sha256"] = sha256_bytes(
        canonical_json(payload).encode("utf-8")
    )
    return payload


def validate_protocol(payload: dict, config_path: Path,
                      repository: Path) -> None:
    expected = build_protocol(config_path, repository)
    for key in ("protocol_schema", "config", "config_sha256",
                "implementation_sha256", "source_files",
                "protocol_sha256"):
        if payload.get(key) != expected.get(key):
            raise ValueError(f"frozen protocol mismatch for {key}")
