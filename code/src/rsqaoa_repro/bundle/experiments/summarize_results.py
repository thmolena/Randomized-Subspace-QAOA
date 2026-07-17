"""Build the manuscript's figures, table, and machine-readable summary.

The script consumes only the tidy CSV emitted by ``run_experiment.py``.  It
does not rerun or alter experiments, and every plotted point is a direct group
summary or a paired difference against full ma-QAOA on the same instance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform as py_platform
import statistics as stats
from collections import defaultdict
from pathlib import Path

import numpy as np

matplotlib = None
plt = None


METHODS = [
    ("full_maqaoa", "", "Full ma-QAOA"),
    ("symmetry", "", "Symmetry-tied"),
    ("fixed_rank_oracle", "", "Fixed-rank oracle"),
    ("rsq", "0.01", "RSQ ($\\varepsilon=0.01$)"),
    ("rsq", "0.1", "RSQ ($\\varepsilon=0.1$)"),
    ("rsq_adjoint_free", "", "Forward-only RSQ"),
]


def mean_sd(values):
    values = list(values)
    return stats.mean(values), stats.stdev(values) if len(values) > 1 else 0.0


def mean_se(values):
    values = list(values)
    mean, sd = mean_sd(values)
    return mean, sd / math.sqrt(len(values))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def edge_list_fingerprint(n, encoded):
    edges = json.loads(encoded)
    canonical = sorted([min(int(u), int(v)), max(int(u), int(v))]
                       for u, v in edges)
    payload = json.dumps(
        {"edges": canonical, "n": int(n)},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], canonical


def validate_rows(rows):
    required = {
        "method", "family", "n", "p", "seed", "graph_seed", "init_seed",
        "sketch_seed", "graph_id", "edge_list", "tol", "d", "m", "cut",
        "approx_ratio", "params_opt", "rank", "refreshes", "forward_F",
        "jvp", "vjp", "dense_jacobian", "wall_s", "steps",
        "best_cut", "best_approx_ratio", "best_step", "subspace_builds",
        "residual_steps", "residual_history", "refresh_steps",
        "build_history", "objective_history", "theta_final",
        "learning_rate", "refresh_every", "eps_refresh", "block",
        "residual_probes", "fd_eps", "spsa_eps", "af_rank", "indicator",
        "graph_seed_offset", "init_seed_offset", "sketch_seed_offset",
        "experiment_schema", "rsqaoa_version", "implementation_sha256",
        "runner_sha256", "experiment_config_sha256", "torch_deterministic",
        "torch_num_threads", "torch_num_interop_threads",
        "design_families", "design_n", "design_p", "design_tols",
        "design_restarts",
        "python_version", "numpy_version", "torch_version", "networkx_version",
        "platform",
    }
    if not rows:
        raise ValueError("the result CSV is empty")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"result CSV is missing columns: {sorted(missing)}")
    identities = set()
    for line, row in enumerate(rows, start=2):
        identity = (
            row["method"], row["family"], row["graph_id"], row["p"],
            row["seed"], row["tol"],
        )
        if identity in identities:
            raise ValueError(f"duplicate result identity at CSV line {line}: {identity}")
        identities.add(identity)
        fingerprint, edges = edge_list_fingerprint(row["n"], row["edge_list"])
        if row["graph_id"] != fingerprint:
            raise ValueError(
                f"graph hash does not match edge_list at CSV line {line}")
        if len(edges) != int(row["m"]):
            raise ValueError(f"edge count does not match m at CSV line {line}")
        if any(u < 0 or v >= int(row["n"]) or u >= v for u, v in edges):
            raise ValueError(f"invalid canonical edge at CSV line {line}")
        for column in ("n", "p", "seed", "graph_seed", "init_seed",
                       "sketch_seed", "d", "m", "cut", "approx_ratio",
                       "params_opt", "refreshes", "forward_F", "jvp", "vjp",
                       "dense_jacobian", "wall_s", "best_cut",
                       "best_approx_ratio", "best_step", "subspace_builds",
                       "steps", "learning_rate",
                       "refresh_every", "eps_refresh", "block",
                       "residual_probes", "fd_eps", "spsa_eps", "af_rank",
                       "graph_seed_offset", "init_seed_offset",
                       "sketch_seed_offset"):
            if row[column] == "" and column not in {"approx_ratio"}:
                raise ValueError(f"missing {column} at CSV line {line}")
            if row[column] and not math.isfinite(float(row[column])):
                raise ValueError(f"non-finite {column} at CSV line {line}")
        ratio = float(row["approx_ratio"])
        if not 0.0 <= ratio <= 1.0 + 1e-6:
            raise ValueError(f"approximation ratio outside [0, 1] at CSV line {line}")
        best_ratio = float(row["best_approx_ratio"])
        if not ratio - 1e-6 <= best_ratio <= 1.0 + 1e-6:
            raise ValueError(f"invalid best approximation ratio at CSV line {line}")
        if not 0 <= int(row["best_step"]) <= int(row["steps"]):
            raise ValueError(f"invalid best_step at CSV line {line}")

        try:
            residual_steps = json.loads(row["residual_steps"])
            residual_history = json.loads(row["residual_history"])
            refresh_steps = json.loads(row["refresh_steps"])
            build_history = json.loads(row["build_history"])
            objective_history = json.loads(row["objective_history"])
            theta_final = json.loads(row["theta_final"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSON audit field at CSV line {line}") from exc
        if len(objective_history) != int(row["steps"]):
            raise ValueError(f"objective history length mismatch at CSV line {line}")
        if len(theta_final) != int(row["d"]):
            raise ValueError(f"final parameter length mismatch at CSV line {line}")
        if not all(math.isfinite(float(value))
                   for value in objective_history + theta_final + residual_history):
            raise ValueError(f"non-finite audit value at CSV line {line}")
        if len(residual_steps) != len(residual_history):
            raise ValueError(f"residual audit length mismatch at CSV line {line}")
        if len(refresh_steps) != int(row["refreshes"]):
            raise ValueError(f"refresh audit length mismatch at CSV line {line}")
        if row["method"] in {"rsq", "rsq_adjoint_free"}:
            every = int(row["refresh_every"])
            expected_steps = (list(range(every, int(row["steps"]), every))
                              if every else [])
            if residual_steps != expected_steps:
                raise ValueError(f"residual schedule mismatch at CSV line {line}")
            if not set(refresh_steps).issubset(residual_steps):
                raise ValueError(f"refresh outside a residual check at CSV line {line}")
            if int(row["subspace_builds"]) != 1 + int(row["refreshes"]):
                raise ValueError(f"subspace build count mismatch at CSV line {line}")
            if len(build_history) != int(row["subspace_builds"]):
                raise ValueError(f"build history length mismatch at CSV line {line}")
        elif any((residual_steps, residual_history, refresh_steps, build_history)):
            raise ValueError(f"baseline row contains RSQ audit data at CSV line {line}")
        elif int(row["subspace_builds"]) != 0:
            raise ValueError(f"baseline row has subspace builds at CSV line {line}")

    design_families = json.loads(rows[0]["design_families"])
    design_n = [int(value) for value in json.loads(rows[0]["design_n"])]
    design_p = [int(value) for value in json.loads(rows[0]["design_p"])]
    design_tols = [str(float(value)) for value in json.loads(rows[0]["design_tols"])]
    design_restarts = int(rows[0]["design_restarts"])
    for field in ("design_families", "design_n", "design_p", "design_tols",
                  "design_restarts"):
        if len({row[field] for row in rows}) != 1:
            raise ValueError(f"mixed {field} values in result CSV")

    actual_cells = {
        (row["family"], int(row["n"]), int(row["p"]), int(row["seed"]))
        for row in rows
    }
    expected_cells = set(itertools.product(
        design_families, design_n, design_p, range(design_restarts)
    ))
    if actual_cells != expected_cells:
        missing = sorted(expected_cells - actual_cells)
        extra = sorted(actual_cells - expected_cells)
        raise ValueError(
            f"incomplete design cells: missing={missing[:3]}, extra={extra[:3]}"
        )

    rsq_tols = {row["tol"] for row in rows if row["method"] == "rsq"}
    if rsq_tols != set(design_tols):
        raise ValueError(
            f"RSQ tolerances {sorted(rsq_tols)} do not match design "
            f"{sorted(design_tols)}"
        )
    if not rsq_tols or "" in rsq_tols:
        raise ValueError("at least one explicit RSQ tolerance is required")
    expected = {
        ("full_maqaoa", ""), ("symmetry", ""),
        ("fixed_rank_oracle", ""), ("rsq_adjoint_free", ""),
    } | {("rsq", tol) for tol in rsq_tols}
    grouped = defaultdict(set)
    for row in rows:
        base = (
            row["family"], row["n"], row["graph_id"], row["p"], row["seed"],
            row["init_seed"], row["sketch_seed"],
        )
        grouped[base].add((row["method"], row["tol"]))
        if int(row["graph_seed"]) != (
                int(row["graph_seed_offset"]) + int(row["seed"])):
            raise ValueError(f"inconsistent graph seed for {base}")
        if int(row["init_seed"]) != (
                int(row["init_seed_offset"]) + int(row["seed"])):
            raise ValueError(f"inconsistent initialization seed for {base}")
        if int(row["sketch_seed"]) != (
                int(row["sketch_seed_offset"]) + int(row["seed"])):
            raise ValueError(f"inconsistent sketch seed for {base}")
    incomplete = {base: signatures for base, signatures in grouped.items()
                  if signatures != expected}
    if incomplete:
        base, signatures = next(iter(incomplete.items()))
        raise ValueError(
            f"incomplete method grid for {base}: got {sorted(signatures)}, "
            f"expected {sorted(expected)}"
        )


def select(rows, method, tol="", depth=None):
    return [
        row for row in rows
        if row["method"] == method and row["tol"] == tol
        and (depth is None or int(row["p"]) == depth)
    ]


def key(row):
    return (row["graph_id"], int(row["p"]), int(row["init_seed"]),
            int(row["sketch_seed"]))


def cluster_key(row):
    """Scientific sampling unit: one graph topology at one QAOA depth."""
    return row["graph_id"], int(row["p"])


def clustered_values(rows, value):
    grouped = defaultdict(list)
    for row in rows:
        grouped[cluster_key(row)].append(float(value(row)))
    return [stats.mean(values) for values in grouped.values()]


def paired_delta(rows, method, tol, depth=None, family=None):
    full = {
        key(row): float(row["approx_ratio"])
        for row in select(rows, "full_maqaoa", depth=depth)
        if family is None or row["family"] == family
    }
    paired = [
        row for row in select(rows, method, tol, depth)
        if key(row) in full and (family is None or row["family"] == family)
    ]
    return clustered_values(
        paired, lambda row: float(row["approx_ratio"]) - full[key(row)]
    )


def configure_plotting():
    global matplotlib, plt
    if plt is None:
        import matplotlib as mpl
        mpl.use("Agg")
        import matplotlib.pyplot as pyplot
        matplotlib = mpl
        plt = pyplot
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.dpi": 160,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 240,
        "savefig.bbox": "tight",
    })


def save(fig, outdir, stem):
    # Matplotlib otherwise inserts the wall-clock creation time into every PDF.
    # Omitting volatile dates makes figure bytes and their recorded release
    # hashes reproducible when the same rows are summarized again.
    fig.savefig(
        outdir / f"{stem}.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(outdir / f"{stem}.png")
    plt.close(fig)


def make_tradeoff(rows, outdir):
    colors = {1: "#2563eb", 2: "#db2777"}
    fig, ax = plt.subplots(figsize=(6.7, 3.8))
    for depth, marker in [(1, "o"), (2, "s")]:
        full = select(rows, "full_maqaoa", depth=depth)
        full_d = {key(row): float(row["d"]) for row in full}
        plotted = [entry for entry in METHODS[1:]
                   if not (entry[0] == "rsq" and entry[1] == "0.1")]
        for method, tol, label in plotted:
            current = select(rows, method, tol, depth)
            compression = clustered_values(
                current,
                lambda row: 100.0 * (
                    1.0 - float(row["params_opt"]) / full_d[key(row)]),
            )
            delta = paired_delta(rows, method, tol, depth)
            xm, xse = mean_se(compression)
            ym, yse = mean_se(delta)
            ax.errorbar(xm, 100 * ym, xerr=2 * xse, yerr=200 * yse,
                        marker=marker, ms=7, capsize=3, color=colors[depth],
                        markeredgecolor="white", markeredgewidth=0.7,
                        alpha=0.9)
            short = {"Symmetry-tied": "Symmetry-tied",
                     "Fixed-rank oracle": "Fixed-rank oracle",
                     "RSQ ($\\varepsilon=0.01$)": "RSQ",
                     "Forward-only RSQ": "Forward-only RSQ"}[label]
            offset = {
                (1, "rsq"): (4, 6), (2, "rsq"): (4, -16),
                (1, "rsq_adjoint_free"): (4, 8),
                (2, "rsq_adjoint_free"): (4, 7),
                (1, "symmetry"): (4, 7), (2, "symmetry"): (4, 7),
                (1, "fixed_rank_oracle"): (4, 7),
                (2, "fixed_rank_oracle"): (4, 7),
            }[(depth, method)]
            ax.annotate(short,
                        (xm, 100 * ym), xytext=offset,
                        textcoords="offset points", fontsize=7)
    ax.axhline(0, color="#111827", lw=1, ls="--")
    ax.set_xlabel("optimized-parameter reduction relative to full ma-QAOA (%)")
    ax.set_ylabel("paired approximation-ratio change (percentage points)")
    ax.set_title("Compression-quality frontier")
    ax.text(0.99, 0.03, "circle: p=1   square: p=2", transform=ax.transAxes,
            ha="right", va="bottom", color="#4b5563")
    save(fig, outdir, "figure1_tradeoff")


def make_family(rows, outdir):
    families = ["regular", "er", "ring"]
    labels = ["3-regular", "Erdos-Renyi", "ring"]
    x = np.arange(len(families))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.7, 3.8))
    for offset, depth, color in [(-width / 2, 1, "#2563eb"),
                                  (width / 2, 2, "#db2777")]:
        means, errors = [], []
        for family in families:
            delta = paired_delta(rows, "rsq", "0.01", depth, family)
            mean, se = mean_se(delta)
            means.append(100 * mean)
            errors.append(200 * se)
        ax.bar(x + offset, means, width, yerr=errors, capsize=3,
               label=f"p={depth}", color=color, alpha=0.86)
    ax.axhline(0, color="#111827", lw=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("RSQ minus full ma-QAOA (percentage points)")
    ax.set_title(r"Paired quality change for RSQ ($\varepsilon=0.01$)")
    ax.legend(frameon=False)
    save(fig, outdir, "figure2_family")


def make_budget(rows, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), sharey=True)
    entries = [("full_maqaoa", "", "Full"),
               ("rsq", "0.01", "RSQ"),
               ("rsq_adjoint_free", "", "Forward-only")]
    colors = ["#9ca3af", "#2563eb", "#0f766e"]
    for ax, depth in zip(axes, [1, 2]):
        x = np.arange(len(entries))
        fwd, jvp, vjp = [], [], []
        for method, tol, _ in entries:
            current = select(rows, method, tol, depth)
            fwd.append(stats.mean(clustered_values(
                current, lambda row: row["forward_F"])))
            jvp.append(stats.mean(clustered_values(
                current, lambda row: row["jvp"])))
            vjp.append(stats.mean(clustered_values(
                current, lambda row: row["vjp"])))
        width = 0.24
        ax.bar(x - width, fwd, width, color="#2563eb", label="forward F")
        ax.bar(x, jvp, width, color="#0f766e", label="JVP")
        ax.bar(x + width, vjp, width, color="#9f1239", label="VJP")
        ax.set_xticks(x, [entry[2] for entry in entries], rotation=18)
        ax.set_title(f"p={depth}")
    axes[0].set_ylabel("mean counted operator applications")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 0.91))
    fig.suptitle("Instrumented operator counts", y=0.99,
                 fontweight="bold")
    fig.text(0.5, 0.01, "Categories are reported separately and are not additive.",
             ha="center", color="#4b5563", fontsize=8)
    fig.subplots_adjust(top=0.74, bottom=0.24, wspace=0.22)
    save(fig, outdir, "figure3_operator_budget")


def make_rank(rows, outdir):
    fig, ax = plt.subplots(figsize=(6.7, 3.8))
    xlabels, dims, ranks = [], [], []
    for depth in [1, 2]:
        for n in [8, 10]:
            current = [row for row in select(rows, "rsq", "0.01", depth)
                       if int(row["n"]) == n]
            xlabels.append(f"n={n}, p={depth}")
            dims.append(stats.mean(clustered_values(current, lambda row: row["d"])))
            ranks.append(stats.mean(clustered_values(current, lambda row: row["rank"])))
    x = np.arange(len(xlabels))
    ax.bar(x, dims, color="#d1d5db", label="full parameter dimension")
    ax.bar(x, ranks, color="#2563eb", label="residual-controlled RSQ rank")
    ax.set_xticks(x, xlabels, rotation=16)
    ax.set_ylabel("dimension")
    ax.set_title(r"Retained dimension for RSQ ($\varepsilon=0.01$)")
    ax.legend(frameon=False)
    save(fig, outdir, "figure4_rank")


def write_outputs(rows, table_path, summary_path, source_path, paper_path):
    protocol_fields = [
        "steps", "learning_rate", "refresh_every", "eps_refresh", "block",
        "residual_probes", "fd_eps", "spsa_eps", "af_rank", "indicator",
        "python_version", "numpy_version", "torch_version", "networkx_version",
        "platform", "graph_seed_offset", "init_seed_offset",
        "sketch_seed_offset", "experiment_schema", "rsqaoa_version",
        "implementation_sha256", "runner_sha256", "experiment_config_sha256",
        "torch_deterministic", "torch_num_threads",
        "torch_num_interop_threads",
        "design_families", "design_n", "design_p", "design_tols",
        "design_restarts",
    ]
    protocol = {}
    for field in protocol_fields:
        values = {row[field] for row in rows}
        if len(values) != 1:
            raise ValueError(f"released CSV mixes protocol values for {field}: {values}")
        protocol[field] = next(iter(values))
    summary = {
        "schema_version": 3,
        "source_csv": source_path.name,
        "source_csv_sha256": sha256(source_path),
        "summarizer_sha256": sha256(Path(__file__)),
        "merge_shards_sha256": sha256(Path(__file__).with_name("merge_shards.py")),
        "n_rows": len(rows),
        "n_paired_runs": len({key(row) for row in select(rows, "full_maqaoa")}),
        "n_graph_depth_clusters": len({
            cluster_key(row) for row in select(rows, "full_maqaoa")
        }),
        "n_unique_topologies": len({
            row["graph_id"] for row in select(rows, "full_maqaoa")
        }),
        "design": {
            "families": sorted({row["family"] for row in rows}),
            "n": sorted({int(row["n"]) for row in rows}),
            "depths": sorted({int(row["p"]) for row in rows}),
            "rsq_tolerances": sorted({
                float(row["tol"]) for row in rows if row["method"] == "rsq"
            }),
            "restart_ids": sorted({int(row["seed"]) for row in rows}),
        },
        "summarizer_environment": {
            "python_version": py_platform.python_version(),
            "platform": py_platform.platform(),
            "numpy_version": np.__version__,
            "matplotlib_version": matplotlib.__version__,
        },
        "operator_count_convention": (
            "forward_F, JVP, VJP, and dense-Jacobian materializations are "
            "separate overlapping categories and must not be summed"
        ),
        "scope": (
            "descriptive CPU exact-statevector validation on n=8,10 and p=1,2; "
            "no hardware, noise, or asymptotic scaling claim"
        ),
        "uncertainty_convention": {
            "sampling_unit": (
                "graph topology by QAOA depth; repeated initializations and "
                "sketches are averaged within each unit"
            ),
            "raw_approximation_ratio": (
                "sample standard deviation across graph-depth cluster means"
            ),
            "paired_change": (
                "two standard errors across graph-depth cluster means; descriptive"
            ),
            "compression": "mean of graph-depth cluster-mean fractions",
        },
        "protocol": protocol,
    }
    summary["refresh_audit"] = {}
    for method, tol, label in [
        ("rsq", "0.01", "two_sided_tol_0.01"),
        ("rsq", "0.1", "two_sided_tol_0.1"),
        ("rsq_adjoint_free", "", "forward_only"),
    ]:
        current = select(rows, method, tol)
        checks = sum(len(json.loads(row["residual_steps"])) for row in current)
        refreshes = sum(int(row["refreshes"]) for row in current)
        builds = [entry for row in current
                  for entry in json.loads(row["build_history"])]
        stop_reasons = defaultdict(int)
        for entry in builds:
            stop_reasons[str(entry["stop_reason"])] += 1
        summary["refresh_audit"][label] = {
            "n_runs": len(current),
            "n_residual_checks": checks,
            "n_triggered_refreshes": refreshes,
            "all_checks_triggered": checks == refreshes,
            "n_subspace_builds": len(builds),
            "build_stop_reasons": dict(sorted(stop_reasons.items())),
        }
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Depth & Parameters & Approx. ratio & Paired change \\",
        r"\midrule",
    ]
    for depth in [1, 2]:
        for method, tol, label in METHODS:
            current = select(rows, method, tol, depth)
            ratios = clustered_values(current, lambda row: row["approx_ratio"])
            params = clustered_values(current, lambda row: row["params_opt"])
            compression = clustered_values(
                current,
                lambda row: 1.0 - float(row["params_opt"]) / float(row["d"]),
            )
            rm, rsd = mean_sd(ratios)
            pm, _ = mean_sd(params)
            cm, csd = mean_sd(compression)
            if method == "full_maqaoa":
                delta_text = "--"
            else:
                delta, se = mean_se(paired_delta(rows, method, tol, depth))
                delta_text = f"{100*delta:+.2f} $\\pm$ {200*se:.2f} pp"
            lines.append(
                f"{label} & {depth} & {pm:.1f} & {rm:.3f} $\\pm$ {rsd:.3f} & {delta_text} \\\\"
            )
            summary[f"p{depth}:{method}:{tol or 'none'}"] = {
                "n_runs": len(current),
                "n_graph_depth_clusters": len({cluster_key(row) for row in current}),
                "parameters_mean": pm,
                "optimized_parameter_reduction_mean": cm,
                "optimized_parameter_reduction_sd": csd,
                "approx_ratio_mean": rm, "approx_ratio_sd": rsd,
                "forward_F_mean": stats.mean(clustered_values(
                    current, lambda row: row["forward_F"])),
                "jvp_mean": stats.mean(clustered_values(
                    current, lambda row: row["jvp"])),
                "vjp_mean": stats.mean(clustered_values(
                    current, lambda row: row["vjp"])),
                "dense_jacobian_mean": stats.mean(clustered_values(
                    current, lambda row: row["dense_jacobian"])),
                "refreshes_mean": stats.mean(clustered_values(
                    current, lambda row: row["refreshes"])),
                "paired_delta_mean": 0.0 if method == "full_maqaoa" else delta,
                "paired_delta_2se": 0.0 if method == "full_maqaoa" else 2 * se,
            }
        if depth == 1:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    table_path.write_text("\n".join(lines) + "\n")

    # Preserve the legacy benchmark's generated manuscript claims in a
    # dedicated immutable evidence artifact.  The current main manuscript can
    # then discuss the separate amortized experiment without silently dropping
    # or mixing the original release claims.
    p1 = summary["p1:rsq:0.01"]
    p2 = summary["p2:rsq:0.01"]
    total_checks = sum(
        item["n_residual_checks"] for item in summary["refresh_audit"].values()
    )
    total_refreshes = sum(
        item["n_triggered_refreshes"]
        for item in summary["refresh_audit"].values()
    )
    legacy_claims_path = paper_path / "legacy_evidence_claims.tex"
    legacy_claims_path.write_text("\n".join([
        "% Auto-generated by experiments/summarize_results.py.",
        "% This file preserves the locked pre-amortization evidence track.",
        r"\section*{Legacy single-objective evidence claims}",
        (
            f"The legacy exact-statevector benchmark contains "
            f"{summary['n_paired_runs']} paired optimizer runs on "
            f"{summary['n_unique_topologies']} unique topologies."
        ),
        (
            f"At depth one, RSQ reduces the optimized parameter count by "
            f"${100 * p1['optimized_parameter_reduction_mean']:.1f}\\%$ and "
            f"has paired approximation-ratio change "
            f"${100 * p1['paired_delta_mean']:+.2f}\\pm"
            f"{100 * p1['paired_delta_2se']:.2f}$ percentage points."
        ),
        (
            f"At depth two, RSQ reduces the optimized parameter count by "
            f"${100 * p2['optimized_parameter_reduction_mean']:.1f}\\%$ and "
            f"has paired approximation-ratio change "
            f"${100 * p2['paired_delta_mean']:+.2f}\\pm"
            f"{100 * p2['paired_delta_2se']:.2f}$ percentage points."
        ),
        (
            f"The legacy refresh audit records {total_refreshes} of "
            f"{total_checks} scheduled checks triggering refresh."
        ),
        (
            "These are descriptive CPU exact-statevector results from the "
            "original single-objective release; they are not evidence for "
            "the separate amortized task-stream hypothesis."
        ),
    ]) + "\n")
    artifacts = [
        paper_path / "figures" / f"figure{index}_{stem}.{extension}"
        for index, stem in [
            (1, "tradeoff"), (2, "family"), (3, "operator_budget"), (4, "rank")
        ]
        for extension in ("pdf", "png")
    ] + [table_path, legacy_claims_path]
    summary["generated_artifact_sha256"] = {
        str(path.relative_to(paper_path.parent)): sha256(path)
        for path in artifacts
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="experiments/results/maxcut_small.csv")
    parser.add_argument("--paper", default="paper")
    args = parser.parse_args()
    source_path = Path(args.csv)
    with open(source_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    validate_rows(rows)
    paper = Path(args.paper)
    figures = paper / "figures"
    tables = paper / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    make_tradeoff(rows, figures)
    make_family(rows, figures)
    make_budget(rows, figures)
    make_rank(rows, figures)
    write_outputs(
        rows,
        tables / "table1_summary.tex",
        source_path.with_name("summary.json"),
        source_path,
        paper,
    )
    print(f"summarized {len(rows)} rows into {figures} and {tables}")


if __name__ == "__main__":
    main()
