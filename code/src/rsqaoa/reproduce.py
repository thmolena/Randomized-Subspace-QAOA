"""Repository-level reproduction command for the RSQAOA evidence package."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .amortized.protocol import validate_protocol


def _run(root: Path, relative: str, *arguments: str) -> None:
    command = [sys.executable, str(root / relative), *arguments]
    subprocess.run(command, cwd=root, check=True)


def _preflight_frozen_protocols(root: Path) -> None:
    """Refuse a rerun before writes unless both frozen environments match."""
    for stem in ("amortized_development", "amortized_shot_development"):
        config = root / "experiments" / "configs" / f"{stem}.yaml"
        protocol_path = root / "experiments" / "protocol" / f"{stem}.json"
        payload = json.loads(protocol_path.read_text())
        validate_protocol(payload, config, root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rsqaoa-reproduce",
        description=(
            "Regenerate and validate all committed RSQAOA manuscript displays; "
            "optionally rerun the frozen experiment protocols first."
        ),
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path.cwd(),
        help="cloned Randomized-Subspace-QAOA repository (default: cwd)",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="rerun the frozen legacy and amortized experiments before analysis",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="audit committed results and displays without regenerating them",
    )
    args = parser.parse_args(argv)
    if args.rerun and args.validate_only:
        parser.error("--rerun and --validate-only are mutually exclusive")
    root = args.repository.expanduser().resolve()
    required = [
        root / "pyproject.toml",
        root / "experiments" / "summarize_results.py",
        root / "experiments" / "make_amortized_paper_assets.py",
        root / "experiments" / "validate_release.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("repository evidence files are missing: " + ", ".join(missing))

    if args.rerun:
        try:
            _preflight_frozen_protocols(root)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(
                "rerun refused before any result write because the current "
                "environment does not match the preserved frozen protocols: "
                f"{error}. Do not refreeze an existing evidence record; use "
                "the recorded source checkout or create a separately named "
                "new protocol and output set."
            )
        _run(root, "experiments/run_experiment.py")
        for stem in ("amortized_development", "amortized_shot_development"):
            config = root / "experiments" / "configs" / f"{stem}.yaml"
            protocol = root / "experiments" / "protocol" / f"{stem}.json"
            csv = root / "experiments" / "results" / f"{stem}.csv"
            summary = (
                root / "experiments" / "results" / f"{stem}_summary.json"
            )
            _run(
                root, "experiments/run_amortized.py",
                "--config", str(config),
                "--protocol", str(protocol),
                "--output", str(csv),
            )
            _run(
                root, "experiments/analyze_amortized.py",
                "--csv", str(csv),
                "--output", str(summary),
            )
    if not args.validate_only:
        _run(root, "experiments/summarize_results.py")
        _run(root, "experiments/make_amortized_paper_assets.py")
    _run(root, "experiments/validate_release.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
