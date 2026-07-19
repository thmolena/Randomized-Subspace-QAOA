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
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COMMANDS = {
    "experiment": ROOT / "experiments" / "run_experiment.py",
    "summarize": ROOT / "experiments" / "summarize_results.py",
    "validate": ROOT / "experiments" / "validate_release.py",
    "merge": ROOT / "experiments" / "merge_shards.py",
}


def _checked(command: list[str], *, cwd: Path = ROOT) -> None:
    """Run one release command and stop immediately on drift or failure."""
    print("+", " ".join(command), flush=True)
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUTF8"] = "1"
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def _validate_bundle_sync() -> None:
    """Require every bundled manuscript/evidence byte to match the repository."""
    bundle = ROOT / "code/src/rsqaoa_repro/bundle"
    mismatches: list[str] = []
    checked = 0
    for subtree in ("experiments", "paper"):
        for packaged in sorted((bundle / subtree).rglob("*")):
            if (
                not packaged.is_file()
                or "__pycache__" in packaged.parts
                or packaged.suffix in {".pyc", ".pyo"}
            ):
                continue
            relative = packaged.relative_to(bundle)
            canonical = ROOT / relative
            checked += 1
            if not canonical.is_file() or canonical.read_bytes() != packaged.read_bytes():
                mismatches.append(relative.as_posix())
    if mismatches:
        raise RuntimeError(
            "bundled evidence/source drift: " + ", ".join(mismatches)
        )
    print(f"[bundle-sync] {checked} evidence/source files are byte-identical")


def _sync_release_outputs() -> None:
    """Mirror only validated, reproducible outputs into the installed bundle."""
    bundle = ROOT / "code/src/rsqaoa_repro/bundle"
    relative_paths = (
        Path("experiments/results/summary.json"),
        Path("paper/figures"),
        Path("paper/tables"),
        Path("paper/legacy_evidence_claims.tex"),
        Path("paper/main.bbl"),
        Path("paper/main.pdf"),
        Path("paper/arxiv-source-rsq.zip"),
    )
    copied = 0
    for relative in relative_paths:
        source = ROOT / relative
        if source.is_dir():
            for item in sorted(source.rglob("*")):
                if not item.is_file() or item.name == ".DS_Store":
                    continue
                destination = bundle / item.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, destination)
                copied += 1
        elif source.is_file():
            destination = bundle / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied += 1
        else:
            raise RuntimeError(f"missing validated release output: {relative}")
    print(f"[bundle-sync] mirrored {copied} validated release outputs")


def _remove_build_intermediates() -> None:
    """Remove successful-build files that do not belong in a public release."""
    for suffix in (".aux", ".blg", ".log", ".out"):
        (ROOT / "paper/main").with_suffix(suffix).unlink(missing_ok=True)


def _compile_and_test_archive() -> None:
    tectonic = shutil.which("tectonic")
    if tectonic is None:
        raise RuntimeError("tectonic is required for the release build")
    paper = ROOT / "paper"
    _checked(
        [tectonic, "--keep-logs", "--keep-intermediates", "main.tex"],
        cwd=paper,
    )
    _checked([sys.executable, "experiments/validate_manuscript.py", "--skip-archive"])
    _checked([sys.executable, "experiments/build_arxiv_source.py"])
    archive = paper / "arxiv-source-rsq.zip"
    with tempfile.TemporaryDirectory(prefix="rsq-source-build-") as temporary:
        extracted = Path(temporary)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(extracted)
        _checked([tectonic, "--keep-logs", "main.tex"], cwd=extracted)
        pdf = extracted / "main.pdf"
        if not pdf.is_file() or pdf.stat().st_size < 100_000:
            raise RuntimeError("extracted RSQ source archive did not build a valid PDF")
    _checked([sys.executable, "experiments/validate_manuscript.py"])


def _release() -> None:
    """Regenerate every feasible manuscript asset and enforce the release gate."""
    _checked(
        [
            sys.executable,
            "experiments/summarize_results.py",
            "--csv",
            "experiments/results/maxcut_small.csv",
            "--paper",
            "paper",
        ]
    )
    for stem in ("amortized_development", "amortized_shot_development"):
        _checked(
            [
                sys.executable,
                "experiments/analyze_amortized.py",
                "--csv",
                f"experiments/results/{stem}.csv",
                "--output",
                f"experiments/results/{stem}_summary.json",
            ]
        )
    _checked([sys.executable, "experiments/make_amortized_paper_assets.py"])
    _checked([sys.executable, "experiments/validate_nmi_design.py"])
    _checked([sys.executable, "experiments/validate_release.py"])
    _checked([sys.executable, "code/tools/check_source_sync.py"])
    _compile_and_test_archive()
    _sync_release_outputs()
    _checked([sys.executable, "code/tools/build_evidence_manifest.py"])
    _validate_bundle_sync()
    package_environment = dict(os.environ)
    package_environment["PYTHONPATH"] = str(ROOT / "code/src")
    package_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    package_environment["PYTHONUTF8"] = "1"
    print("+ rsqaoa-reproduce-all replay", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from rsqaoa_repro.cli import replay; import json; "
            "print(json.dumps(replay(), sort_keys=True))",
        ],
        cwd=ROOT,
        env=package_environment,
        check=True,
    )
    _remove_build_intermediates()
    print("[release] RSQ evidence, manuscript, package mirror, and archive passed")


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
        choices=["demo", "release", *COMMANDS],
        help=(
            "release runs the complete publication gate; demo uses the installed "
            "CLI; other commands operate on repository artifacts"
        ),
    )
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if args.command == "demo":
        from rsqaoa._cli import main as cli_main

        cli_main(args.arguments)
        return
    if args.command == "release":
        if args.arguments:
            parser.error("release does not accept additional arguments")
        _release()
        return
    _run_script(COMMANDS[args.command], args.arguments)


if __name__ == "__main__":
    main()
