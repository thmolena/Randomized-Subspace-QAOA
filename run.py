"""Single entry point for RSQ experiments and release verification.

Examples
--------
Run one configurable experiment directly through the public package CLI::

    python run.py demo --family ring --n 8 --p 1 --steps 20 --json

Regenerate or verify the committed evidence bundle::

    python run.py summarize
    python run.py validate
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COMMANDS = {
    "experiment": ROOT / "experiments" / "run_experiment.py",
    "summarize": ROOT / "experiments" / "summarize_results.py",
    "validate": ROOT / "experiments" / "validate_release.py",
    "merge": ROOT / "experiments" / "merge_shards.py",
}


def _run_script(path: Path, arguments: list[str]) -> None:
    """Execute a repository script with its directory importable."""
    sys.argv = [str(path), *arguments]
    sys.path.insert(0, str(path.parent))
    runpy.run_path(str(path), run_name="__main__")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run RSQ demos, experiments, summarization, and validation."
    )
    parser.add_argument(
        "command",
        choices=["demo", *COMMANDS],
        help="demo uses the installed CLI; other commands operate on repository artifacts",
    )
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if args.command == "demo":
        from rsqaoa._cli import main as cli_main

        cli_main(args.arguments)
        return
    _run_script(COMMANDS[args.command], args.arguments)


if __name__ == "__main__":
    main()
