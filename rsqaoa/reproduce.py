"""Repository-level reproduction command for the RSQAOA evidence package."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(root: Path, relative: str, *arguments: str) -> None:
    command = [sys.executable, str(root / relative), *arguments]
    subprocess.run(command, cwd=root, check=True)


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
        _run(root, "experiments/run_experiment.py")
        _run(root, "experiments/freeze_amortized_protocol.py")
        _run(root, "experiments/run_amortized.py")
        _run(root, "experiments/analyze_amortized.py")
    if not args.validate_only:
        _run(root, "experiments/summarize_results.py")
        _run(root, "experiments/make_amortized_paper_assets.py")
    _run(root, "experiments/validate_release.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
