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


def recorded_source_hash(source_files: dict[str, str]) -> str:
    """Recompute an implementation hash from an immutable source-hash record."""
    digest = hashlib.sha256()
    for relative, value in sorted(source_files.items()):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe source path in frozen protocol: {relative}")
        if not re_full_sha256(value):
            raise ValueError(
                f"invalid source SHA-256 in frozen protocol: {relative}"
            )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def re_full_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def protocol_hash(payload: dict) -> str:
    """Hash a frozen protocol without its self-referential digest field."""
    record = dict(payload)
    record.pop("protocol_sha256", None)
    return sha256_bytes(canonical_json(record).encode("utf-8"))


def validate_protocol_record(
    payload: dict,
    config_path: Path | None = None,
    repository: Path | None = None,
    *,
    require_preserved_sources: bool = False,
) -> dict[str, int]:
    """Validate a historical protocol record without requiring old source bytes.

    The embedded source-file hashes establish the implementation identity used
    by the recorded run.  They are deliberately not replaced by hashes of a
    later checkout.  Passing ``config_path`` also verifies that the retained
    configuration still matches the frozen snapshot.
    """
    required = {
        "protocol_schema", "config", "config_path", "config_sha256",
        "implementation_sha256", "source_files", "git_commit_at_freeze",
        "protocol_sha256",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(
            "frozen protocol is missing fields: " + ", ".join(missing)
        )
    if payload["protocol_schema"] != 1:
        raise ValueError("unsupported frozen protocol schema")
    if not isinstance(payload["source_files"], dict) or not payload["source_files"]:
        raise ValueError("frozen protocol has no source-file hashes")
    implementation = recorded_source_hash(payload["source_files"])
    if payload["implementation_sha256"] != implementation:
        raise ValueError("frozen protocol implementation hash is inconsistent")
    live_sources = 0
    snapshot_sources = 0
    if repository is not None:
        repository = Path(repository).resolve()
        snapshot_root = (
            repository / "experiments" / "protocol" / "frozen_source"
            / payload["implementation_sha256"]
        )
        for relative, expected_sha256 in payload["source_files"].items():
            live_path = repository / relative
            live_matches = (
                live_path.is_file()
                and sha256_file(live_path) == expected_sha256
            )
            if live_matches:
                live_sources += 1
                if not require_preserved_sources:
                    continue
            snapshot_path = snapshot_root / relative
            if (
                not snapshot_path.is_file()
                or sha256_file(snapshot_path) != expected_sha256
            ):
                raise ValueError(
                    "recorded source bytes are unavailable or inconsistent: "
                    f"{relative}"
                )
            snapshot_sources += 1
    if config_path is not None:
        import yaml

        config_path = Path(config_path).resolve()
        if sha256_file(config_path) != payload["config_sha256"]:
            raise ValueError(
                "current config_sha256 differs from frozen protocol"
            )
        if yaml.safe_load(config_path.read_text()) != payload["config"]:
            raise ValueError("current configuration values differ from frozen protocol")
        if repository is not None:
            relative = config_path.relative_to(repository).as_posix()
            if relative != payload["config_path"]:
                raise ValueError("configuration path differs from frozen protocol")
    if payload["protocol_sha256"] != protocol_hash(payload):
        raise ValueError("frozen protocol self-hash is inconsistent")
    return {
        "recorded_sources": len(payload["source_files"]),
        "live_sources": live_sources,
        "snapshot_sources": snapshot_sources,
    }


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
    payload["protocol_sha256"] = protocol_hash(payload)
    return payload


def validate_protocol(payload: dict, config_path: Path,
                      repository: Path) -> None:
    validate_protocol_record(payload, config_path, repository)
    expected = build_protocol(config_path, repository)
    for key in ("implementation_sha256", "source_files", "protocol_sha256"):
        if payload.get(key) != expected.get(key):
            raise ValueError(
                f"current environment differs from frozen protocol for {key}; "
                "do not refreeze an existing evidence record"
            )
