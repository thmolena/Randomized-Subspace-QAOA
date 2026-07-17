"""Replay immutable RSQAOA evidence or launch all seeded studies again."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.resources as resources
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import yaml


PACKAGE = "rsqaoa_repro"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest() -> dict:
    manifest = json.loads(
        resources.files(PACKAGE).joinpath(
            "evidence_manifest.json"
        ).read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported RSQAOA evidence manifest")
    return manifest


def _verify_files(bundle: Path, manifest: dict) -> dict:
    expected = manifest.get("files", {})
    if not expected:
        raise RuntimeError("RSQAOA evidence manifest has no files")
    actual_paths = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }
    expected_paths = set(expected)
    errors = []
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        added = sorted(actual_paths - expected_paths)
        if missing:
            errors.append("missing files: " + ", ".join(missing))
        if added:
            errors.append("unmanifested files: " + ", ".join(added))
    for relative, record in sorted(expected.items()):
        path = bundle / relative
        if not path.is_file():
            continue
        if path.stat().st_size != record["size"]:
            errors.append(f"size mismatch for {relative}")
            continue
        if _sha256(path) != record["sha256"]:
            errors.append(f"SHA-256 mismatch for {relative}")
    if errors:
        raise RuntimeError(
            "RSQAOA evidence replay failed:\n- " + "\n- ".join(errors)
        )
    return {
        "files": len(expected),
        "bytes": sum(item["size"] for item in expected.values()),
        "manifest_sha256": manifest["manifest_sha256"],
    }


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _recorded_implementation_hash(source_files: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, value in sorted(source_files.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_semantics(bundle: Path) -> dict:
    results = bundle / "experiments/results"
    legacy_csv = results / "maxcut_small.csv"
    legacy = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    if len(_rows(legacy_csv)) != legacy["n_rows"]:
        raise RuntimeError("legacy CSV row count does not match summary.json")
    if _sha256(legacy_csv) != legacy["source_csv_sha256"]:
        raise RuntimeError("legacy source CSV hash does not match summary.json")

    amortized = {}
    for stem in ("amortized_development", "amortized_shot_development"):
        csv_path = results / f"{stem}.csv"
        protocol = json.loads(
            (
                bundle / "experiments/protocol" / f"{stem}.json"
            ).read_text(encoding="utf-8")
        )
        claimed_protocol_hash = protocol["protocol_sha256"]
        hash_payload = dict(protocol)
        hash_payload.pop("protocol_sha256")
        actual_protocol_hash = hashlib.sha256(
            json.dumps(
                hash_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        if claimed_protocol_hash != actual_protocol_hash:
            raise RuntimeError(f"{stem} frozen protocol self-hash is invalid")
        if (
            _recorded_implementation_hash(protocol["source_files"])
            != protocol["implementation_sha256"]
        ):
            raise RuntimeError(
                f"{stem} frozen implementation record is inconsistent"
            )
        config_path = bundle / protocol["config_path"]
        if _sha256(config_path) != protocol["config_sha256"]:
            raise RuntimeError(f"{stem} frozen configuration hash is invalid")
        if yaml.safe_load(config_path.read_text(encoding="utf-8")) != protocol["config"]:
            raise RuntimeError(f"{stem} frozen configuration values differ")
        snapshot_root = (
            bundle / "experiments/protocol/frozen_source"
            / protocol["implementation_sha256"]
        )
        for relative, expected in protocol["source_files"].items():
            snapshot = snapshot_root / relative
            if not snapshot.is_file() or _sha256(snapshot) != expected:
                raise RuntimeError(
                    f"{stem} preserved source mismatch: {relative}"
                )
        summary = json.loads(
            (results / f"{stem}_summary.json").read_text(encoding="utf-8")
        )
        rows = _rows(csv_path)
        if len(rows) != summary["n_rows"]:
            raise RuntimeError(f"{stem} row count does not match its summary")
        protocol_hashes = sorted({row["protocol_sha256"] for row in rows})
        if protocol_hashes != summary["protocol_sha256"]:
            raise RuntimeError(f"{stem} protocol hashes do not match")
        if protocol_hashes != [claimed_protocol_hash]:
            raise RuntimeError(f"{stem} rows do not match protocol JSON")
        if {row["config_sha256"] for row in rows} != {
            protocol["config_sha256"]
        }:
            raise RuntimeError(f"{stem} rows do not match frozen configuration")
        if {row["implementation_sha256"] for row in rows} != {
            protocol["implementation_sha256"]
        }:
            raise RuntimeError(f"{stem} rows do not match frozen implementation")
        if summary["implementation_sha256"] != [
            protocol["implementation_sha256"]
        ]:
            raise RuntimeError(f"{stem} summary implementation hash differs")
        amortized[stem] = {
            "rows": len(rows),
            "sampling_units": summary["n_sampling_units"],
            "protocol_sha256": protocol_hashes,
            "preserved_source_files": len(protocol["source_files"]),
        }

    asset_manifest = json.loads(
        (bundle / "paper/tables/amortized_asset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    confirmatory_design = yaml.safe_load(
        (bundle / "experiments/configs/nmi_confirmatory_design.yaml").read_text(
            encoding="utf-8"
        )
    )
    for relative, digest in {
        **asset_manifest["source_hashes"],
        **asset_manifest["asset_hashes"],
    }.items():
        path = bundle / relative
        if not path.is_file() or _sha256(path) != digest:
            raise RuntimeError(f"asset-manifest mismatch: {relative}")
    return {
        "legacy_rows": legacy["n_rows"],
        "amortized": amortized,
        "asset_files": len(asset_manifest["asset_hashes"]),
        "confirmatory_design_status": confirmatory_design["status"],
    }


def replay(output: Path | None = None) -> dict:
    manifest = _load_manifest()
    bundle_ref = resources.files(PACKAGE).joinpath("bundle")
    with resources.as_file(bundle_ref) as bundle:
        file_report = _verify_files(bundle, manifest)
        semantic_report = _validate_semantics(bundle)
        if output is not None:
            destination = output.expanduser().resolve()
            if destination.exists() and any(destination.iterdir()):
                raise RuntimeError(
                    f"replay output must be absent or empty: {destination}"
                )
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                bundle,
                destination,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )
            _verify_files(destination, manifest)
    return {"mode": "byte-hash-replay", **file_report, **semantic_report}


def _run_commands(
    commands: Iterable[list[str]], *, cwd: Path, dry_run: bool
) -> None:
    for command in commands:
        print("+", " ".join(command))
        if not dry_run:
            subprocess.run(command, cwd=cwd, check=True)


def _prepare_full_tree(output: Path) -> None:
    bundle_ref = resources.files(PACKAGE).joinpath("bundle")
    with resources.as_file(bundle_ref) as bundle:
        experiments = bundle / "experiments"
        for path in experiments.rglob("*"):
            if (
                not path.is_file()
                or "__pycache__" in path.parts
                or path.suffix in {".pyc", ".pyo"}
            ):
                continue
            relative = path.relative_to(experiments)
            if "results" in relative.parts or "protocol" in relative.parts:
                continue
            destination = output / "experiments" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    # The frozen-protocol code deliberately fingerprints the complete package
    # source relative to the experiment root. Materialize the wheel's canonical
    # source bytes so a clean-wheel rerun preserves that contract.
    package_ref = resources.files("rsqaoa")
    with resources.as_file(package_ref) as package:
        shutil.copytree(
            package,
            output / "rsqaoa",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
    (output / "experiments/results").mkdir(parents=True, exist_ok=True)
    (output / "experiments/protocol").mkdir(parents=True, exist_ok=True)
    (output / "paper/figures").mkdir(parents=True, exist_ok=True)
    (output / "paper/tables").mkdir(parents=True, exist_ok=True)


def _script(destination: Path, name: str, *arguments: object) -> list[str]:
    return [
        sys.executable,
        str(destination / "experiments" / name),
        *(str(value) for value in arguments),
    ]


def full_rerun(
    output: Path,
    *,
    quick: bool,
    dry_run: bool,
) -> dict:
    destination = output.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"full-rerun output must be absent or empty: {destination}")
    if not dry_run:
        destination.mkdir(parents=True, exist_ok=True)
        _prepare_full_tree(destination)

    legacy_extra: list[object] = []
    amortized_extra: list[object] = []
    if quick:
        legacy_extra = [
            "--families", "regular", "er", "ring", "--n", 8, 10,
            "--p", 1, 2,
            "--tols", 0.1, 0.01,
            "--seeds", 1, "--steps", 2,
        ]
        amortized_extra = ["--limit-jobs", 1]

    legacy_csv = destination / "experiments/results/maxcut_small.csv"
    exact_config = destination / "experiments/configs/amortized_development.yaml"
    exact_protocol = destination / "experiments/protocol/amortized_development.json"
    exact_csv = destination / "experiments/results/amortized_development.csv"
    exact_json = destination / "experiments/results/amortized_development_summary.json"
    shot_config = destination / "experiments/configs/amortized_shot_development.yaml"
    shot_protocol = (
        destination / "experiments/protocol/amortized_shot_development.json"
    )
    shot_csv = destination / "experiments/results/amortized_shot_development.csv"
    shot_json = (
        destination / "experiments/results/amortized_shot_development_summary.json"
    )
    commands = [
        _script(
            destination, "run_experiment.py",
            "--config", destination / "experiments/configs/maxcut_small.yaml",
            "--out", legacy_csv, *legacy_extra,
        ),
        _script(
            destination, "summarize_results.py",
            "--csv", legacy_csv, "--paper", destination / "paper",
        ),
        _script(
            destination, "freeze_amortized_protocol.py",
            "--config", exact_config, "--output", exact_protocol,
        ),
        _script(
            destination, "run_amortized.py",
            "--config", exact_config, "--protocol", exact_protocol,
            "--output", exact_csv, *amortized_extra,
        ),
        _script(
            destination, "analyze_amortized.py",
            "--csv", exact_csv, "--output", exact_json,
        ),
        _script(
            destination, "freeze_amortized_protocol.py",
            "--config", shot_config, "--output", shot_protocol,
        ),
        _script(
            destination, "run_amortized.py",
            "--config", shot_config, "--protocol", shot_protocol,
            "--output", shot_csv, *amortized_extra,
        ),
        _script(
            destination, "analyze_amortized.py",
            "--csv", shot_csv, "--output", shot_json,
        ),
        _script(destination, "make_amortized_paper_assets.py"),
        _script(
            destination, "validate_nmi_design.py",
            "--protocol",
            destination / "experiments/configs/nmi_confirmatory_design.yaml",
        ),
    ]
    _run_commands(commands, cwd=destination, dry_run=dry_run)
    report = {
        "mode": "full-seeded-rerun",
        "quick": quick,
        "dry_run": dry_run,
        "commands": commands,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "bitwise_identity_claimed": False,
        "numerical_drift_note": (
            "PyTorch, BLAS, compiler, and platform differences may change "
            "floating-point results despite frozen seeds."
        ),
    }
    if not dry_run:
        (destination / "rerun_metadata.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rsqaoa-reproduce-all",
        description=(
            "Verify/copy committed RSQAOA bytes or launch every seeded study."
        ),
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    replay_parser = subparsers.add_parser(
        "replay", help="hash-verify committed artifacts without recomputation"
    )
    replay_parser.add_argument(
        "--output", type=Path, default=None,
        help="optional empty directory receiving an exact byte copy",
    )
    full_parser = subparsers.add_parser(
        "full", help="run a new seeded execution (not expected to be bitwise portable)"
    )
    full_parser.add_argument("--output", type=Path, required=True)
    full_parser.add_argument(
        "--quick", action="store_true", help="limit each study to a smoke grid"
    )
    full_parser.add_argument(
        "--dry-run", action="store_true", help="print the execution plan"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mode == "replay":
        report = replay(args.output)
    else:
        report = full_rerun(args.output, quick=args.quick, dry_run=args.dry_run)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
