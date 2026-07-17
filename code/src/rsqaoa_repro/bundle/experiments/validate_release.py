"""Validate the committed RSQ evidence chain without rerunning optimization.

Checks the row schema and complete paired grid, source and artifact hashes,
implementation fingerprint, sampling-unit counts, refresh audit, and the
headline values rendered in the manuscript.  This script is intentionally
repository-only because the experiment data are not bundled in the wheel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent
sys.path.insert(0, str(REPOSITORY))

from rsqaoa import __version__
from rsqaoa.amortized.protocol import (
    validate_protocol,
    validate_protocol_record,
)
from run_experiment import runner_fingerprint
from summarize_results import sha256, validate_rows


def _released_source_tree_fingerprint() -> str:
    """Hash experiment-bearing modules while excluding the later CLI shim."""
    root = REPOSITORY / "rsqaoa"
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        if path.name == "reproduce.py":
            continue
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _require(text: str, phrase: str, context: str) -> None:
    if phrase not in text:
        raise ValueError(f"{context} is missing generated claim {phrase!r}")


def _validate_frozen_amortized_protocols(repository: Path) -> dict:
    """Audit historical protocol identity separately from rerun compatibility."""
    reports = {}
    for stem in ("amortized_development", "amortized_shot_development"):
        config_path = repository / "experiments/configs" / f"{stem}.yaml"
        protocol_path = repository / "experiments/protocol" / f"{stem}.json"
        csv_path = repository / "experiments/results" / f"{stem}.csv"
        summary_path = (
            repository / "experiments/results" / f"{stem}_summary.json"
        )
        protocol = json.loads(protocol_path.read_text())
        source_report = validate_protocol_record(
            protocol,
            config_path,
            repository,
            require_preserved_sources=True,
        )
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        summary = json.loads(summary_path.read_text())
        if not rows or len(rows) != summary["n_rows"]:
            raise ValueError(f"{stem} row count is inconsistent")
        expected_protocol = {protocol["protocol_sha256"]}
        if {row["protocol_sha256"] for row in rows} != expected_protocol:
            raise ValueError(f"{stem} rows do not match the frozen protocol")
        if summary["protocol_sha256"] != sorted(expected_protocol):
            raise ValueError(f"{stem} summary does not match the frozen protocol")
        expected_config = {protocol["config_sha256"]}
        if {row["config_sha256"] for row in rows} != expected_config:
            raise ValueError(f"{stem} rows do not match the frozen configuration")
        expected_implementation = {protocol["implementation_sha256"]}
        if {row["implementation_sha256"] for row in rows} != expected_implementation:
            raise ValueError(f"{stem} rows do not match the frozen implementation")
        if summary["implementation_sha256"] != sorted(expected_implementation):
            raise ValueError(
                f"{stem} summary does not match the frozen implementation"
            )
        execution_compatible = True
        incompatibility = None
        try:
            validate_protocol(protocol, config_path, repository)
        except ValueError as error:
            execution_compatible = False
            incompatibility = str(error)
        reports[stem] = {
            "protocol_sha256": protocol["protocol_sha256"],
            "implementation_sha256": protocol["implementation_sha256"],
            "rows": len(rows),
            "record_integrity": True,
            "current_execution_compatible": execution_compatible,
            "current_execution_incompatibility": incompatibility,
            **source_report,
        }
    return reports


def _validate_amortized_assets(repository: Path, paper_path: Path) -> int:
    manifest_path = paper_path / "tables/amortized_asset_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing amortized asset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != 1:
        raise ValueError("amortized asset manifest schema is not 1")

    for section in ("source_hashes", "asset_hashes"):
        for relative, expected in manifest[section].items():
            path = repository / relative
            if not path.is_file():
                raise ValueError(f"missing amortized {section}: {path}")
            if sha256(path) != expected:
                raise ValueError(
                    f"amortized {section} hash mismatch: {path}"
                )
            if section == "asset_hashes" and path.suffix == ".pdf":
                payload = path.read_bytes()
                if b"/CreationDate" in payload or b"/ModDate" in payload:
                    raise ValueError(
                        f"amortized PDF contains volatile date metadata: {path}"
                    )

    required_assets = {
        "paper/tables/table_amortized_exact_audit.tex",
        "paper/tables/table_amortized_shot_audit.tex",
        "paper/tables/table_development_gate.tex",
        "paper/tables/table_protocol.tex",
        "paper/tables/table_reproduction.tex",
    }
    missing_assets = required_assets - set(manifest["asset_hashes"])
    if missing_assets:
        raise ValueError(
            "amortized asset manifest is missing required tables: "
            + ", ".join(sorted(missing_assets))
        )

    exact_summary = json.loads(
        (repository / "experiments/results/amortized_development_summary.json")
        .read_text()
    )
    gate = exact_summary["thesis_gate"]
    gate_table = (
        paper_path / "tables/table_development_gate.tex"
    ).read_text()
    for claim in (
        f"{gate['quality_delta_vs_full']:+.5f}",
        f"{gate['quality_delta_vs_random_basis']:+.5f}",
        f"{gate['forward_cost_ratio_vs_full']:.5f}",
        r"\textbf{fail}",
    ):
        _require(
            gate_table,
            claim,
            "paper/tables/table_development_gate.tex",
        )
    return len(manifest["asset_hashes"])


def validate_release(csv_path: Path, summary_path: Path, paper_path: Path) -> dict:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    validate_rows(rows)
    summary = json.loads(summary_path.read_text())

    if summary["source_csv_sha256"] != sha256(csv_path):
        raise ValueError("summary source_csv_sha256 does not match the CSV")
    if summary.get("schema_version") != 3:
        raise ValueError("summary schema_version is not 3")
    if summary["n_rows"] != len(rows):
        raise ValueError("summary row count does not match the CSV")
    if {row["experiment_schema"] for row in rows} != {"3"}:
        raise ValueError("CSV experiment_schema is not uniformly 3")
    if summary["protocol"]["experiment_schema"] != "3":
        raise ValueError("summary protocol experiment_schema is not 3")

    first = rows[0]
    effective_config = {
        "families": json.loads(first["design_families"]),
        "n": json.loads(first["design_n"]),
        "p": json.loads(first["design_p"]),
        "tols": json.loads(first["design_tols"]),
        "seeds": int(first["design_restarts"]),
        "steps": int(first["steps"]),
        "learning_rate": float(first["learning_rate"]),
        "refresh_every": int(first["refresh_every"]),
        "eps_refresh": float(first["eps_refresh"]),
        "block": int(first["block"]),
        "residual_probes": int(first["residual_probes"]),
        "fd_eps": float(first["fd_eps"]),
        "spsa_eps": float(first["spsa_eps"]),
        "af_rank": int(first["af_rank"]),
        "indicator": first["indicator"],
        "graph_seed_offset": int(first["graph_seed_offset"]),
        "init_seed_offset": int(first["init_seed_offset"]),
        "sketch_seed_offset": int(first["sketch_seed_offset"]),
    }
    expected_config_sha256 = hashlib.sha256(json.dumps(
        effective_config, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    if {row["experiment_config_sha256"] for row in rows} != {
            expected_config_sha256}:
        raise ValueError("CSV experiment_config_sha256 is inconsistent")
    if (summary["protocol"]["experiment_config_sha256"] !=
            expected_config_sha256):
        raise ValueError("summary experiment_config_sha256 is inconsistent")
    current_source = _released_source_tree_fingerprint()
    if {row["implementation_sha256"] for row in rows} != {current_source}:
        raise ValueError("CSV rows were generated from a different implementation")
    if {row["runner_sha256"] for row in rows} != {runner_fingerprint()}:
        raise ValueError("CSV rows were generated by a different experiment driver")
    if {row["rsqaoa_version"] for row in rows} != {__version__}:
        raise ValueError("CSV rows do not match the installed package version")
    if summary["protocol"]["implementation_sha256"] != current_source:
        raise ValueError("results were generated from a different implementation")
    if summary["protocol"]["runner_sha256"] != runner_fingerprint():
        raise ValueError("results were generated by a different experiment driver")
    if summary["protocol"]["rsqaoa_version"] != __version__:
        raise ValueError("summary does not match the installed package version")
    if summary["summarizer_sha256"] != sha256(
            Path(__file__).with_name("summarize_results.py")):
        raise ValueError("summary was generated by a different summarizer")
    if summary["merge_shards_sha256"] != sha256(
            Path(__file__).with_name("merge_shards.py")):
        raise ValueError("summary does not match the released shard merger")
    deterministic = {row["torch_deterministic"] for row in rows}
    if deterministic != {"True"} or summary["protocol"]["torch_deterministic"] != "True":
        raise ValueError("released run did not enable deterministic PyTorch algorithms")
    if {row["torch_num_threads"] for row in rows} != {"1"}:
        raise ValueError("released run did not use one PyTorch intra-op thread")
    if {row["torch_num_interop_threads"] for row in rows} != {"1"}:
        raise ValueError("released run did not use one PyTorch inter-op thread")

    repository = paper_path.parent
    frozen_protocols = _validate_frozen_amortized_protocols(repository)
    amortized_artifacts = _validate_amortized_assets(
        repository, paper_path
    )
    for relative, expected in summary["generated_artifact_sha256"].items():
        artifact = repository / relative
        if not artifact.is_file():
            raise ValueError(f"missing generated artifact: {artifact}")
        if sha256(artifact) != expected:
            raise ValueError(f"generated artifact hash mismatch: {artifact}")
        if artifact.suffix == ".pdf":
            payload = artifact.read_bytes()
            if b"/CreationDate" in payload or b"/ModDate" in payload:
                raise ValueError(
                    f"generated PDF contains volatile date metadata: {artifact}")

    manuscript = (paper_path / "main.tex").read_text()
    legacy_claims_path = paper_path / "legacy_evidence_claims.tex"
    if not legacy_claims_path.is_file():
        raise ValueError(
            "missing generated legacy evidence claims: "
            f"{legacy_claims_path}"
        )
    legacy_claims = legacy_claims_path.read_text()
    readme = (repository / "README.md").read_text()
    homepage = (repository / "index.html").read_text()
    generated_table = (paper_path / "tables/table1_summary.tex").read_text()
    for context, text in [
        ("paper/legacy_evidence_claims.tex", legacy_claims),
        ("paper/tables/table1_summary.tex", generated_table),
    ]:
        if "$+-" in text:
            raise ValueError(f"{context} contains malformed signed TeX '$+-'")
    p1 = summary["p1:rsq:0.01"]
    p2 = summary["p2:rsq:0.01"]
    generated_claims = [
        f"${100 * p1['optimized_parameter_reduction_mean']:.1f}\\%$",
        f"${100 * p2['optimized_parameter_reduction_mean']:.1f}\\%$",
        f"${100 * p1['paired_delta_mean']:+.2f}\\pm"
        f"{100 * p1['paired_delta_2se']:.2f}$",
        f"${100 * p2['paired_delta_mean']:+.2f}\\pm"
        f"{100 * p2['paired_delta_2se']:.2f}$",
        f"{summary['n_paired_runs']} paired",
        f"{summary['n_unique_topologies']} unique topologies",
    ]
    for claim in generated_claims:
        _require(
            legacy_claims, claim, "paper/legacy_evidence_claims.tex"
        )

    audits = summary["refresh_audit"]
    total_checks = sum(item["n_residual_checks"] for item in audits.values())
    total_refreshes = sum(item["n_triggered_refreshes"] for item in audits.values())
    _require(
        legacy_claims, f"{total_refreshes} of {total_checks}",
        "paper/legacy_evidence_claims.tex",
    )

    p1_af = summary["p1:rsq_adjoint_free:none"]
    p2_af = summary["p2:rsq_adjoint_free:none"]
    readme_claims = [
        f"{100 * p1['optimized_parameter_reduction_mean']:.1f}%",
        f"{100 * p2['optimized_parameter_reduction_mean']:.1f}%",
        f"{100 * p1['paired_delta_mean']:+.2f} +/- "
        f"{100 * p1['paired_delta_2se']:.2f}",
        f"{100 * p2['paired_delta_mean']:+.2f} +/- "
        f"{100 * p2['paired_delta_2se']:.2f}",
        f"{100 * p1_af['optimized_parameter_reduction_mean']:.1f}%",
        f"{100 * p2_af['optimized_parameter_reduction_mean']:.1f}%",
        f"{total_refreshes} of {total_checks}",
    ]
    homepage_claims = [
        f"{100 * p1['optimized_parameter_reduction_mean']:.1f}%",
        f"{100 * p2['optimized_parameter_reduction_mean']:.1f}%",
        f"\\({100 * p1['paired_delta_mean']:+.2f}\\pm"
        f"{100 * p1['paired_delta_2se']:.2f}\\)",
        f"\\({100 * p2['paired_delta_mean']:+.2f}\\pm"
        f"{100 * p2['paired_delta_2se']:.2f}\\)",
        f"{100 * p1_af['optimized_parameter_reduction_mean']:.1f}%",
        f"{100 * p2_af['optimized_parameter_reduction_mean']:.1f}%",
        f"{total_refreshes} of {total_checks}",
    ]
    for claim in readme_claims:
        _require(readme, claim, "README.md")
    for claim in homepage_claims:
        _require(homepage, claim, "index.html")

    stale_claims = [
        "41.5%", "70.8%", "-0.21 +/- 0.89", "-2.86 +/- 1.43",
        "22.4 parameters", "44.9", "13.4 directions",
    ]
    for context, text in [
        ("paper/main.tex", manuscript), ("README.md", readme),
        ("index.html", homepage),
    ]:
        for stale in stale_claims:
            if stale in text:
                raise ValueError(f"{context} contains stale public value {stale!r}")
    for label, audit in audits.items():
        expected = audit["n_residual_checks"] == audit["n_triggered_refreshes"]
        if audit["all_checks_triggered"] != expected:
            raise ValueError(f"inconsistent all_checks_triggered flag for {label}")

    return {
        "rows": len(rows),
        "paired_runs": summary["n_paired_runs"],
        "graph_depth_clusters": summary["n_graph_depth_clusters"],
        "unique_topologies": summary["n_unique_topologies"],
        "source_csv_sha256": summary["source_csv_sha256"],
        "implementation_sha256": current_source,
        "generated_artifacts": len(summary["generated_artifact_sha256"]),
        "amortized_generated_artifacts": amortized_artifacts,
        "frozen_amortized_protocols": frozen_protocols,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", default=str(HERE / "results/maxcut_small.csv")
    )
    parser.add_argument(
        "--summary", default=str(HERE / "results/summary.json")
    )
    parser.add_argument("--paper", default=str(REPOSITORY / "paper"))
    args = parser.parse_args()
    report = validate_release(
        Path(args.csv), Path(args.summary), Path(args.paper)
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
