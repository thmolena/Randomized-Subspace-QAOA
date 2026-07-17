"""Regenerate every display in the amortized-RSQ paper from frozen results.

The script aggregates configured graph-topology/depth analysis cells and
averages finite-shot measurement repetitions within those cells.  The exact
grid reuses each of eight topologies across two depths, so its 16-cell error
bars are descriptive and do not model cross-depth dependence.  The script does
not rerun or alter either experiment.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
RESULTS = REPOSITORY / "experiments/results"
PAPER = REPOSITORY / "paper"
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"

EXACT_CSV = RESULTS / "amortized_development.csv"
EXACT_JSON = RESULTS / "amortized_development_summary.json"
SHOT_CSV = RESULTS / "amortized_shot_development.csv"
SHOT_JSON = RESULTS / "amortized_shot_development_summary.json"

DISPLAY = {
    "full_spsa": "Full SPSA",
    "amortized_none": "No refresh",
    "amortized_fixed": "Fixed refresh",
    "amortized_gated": "Residual gated",
    "amortized_per_task": "Per-task refresh",
    "amortized_random": "Random refresh",
    "amortized_random_basis": "Random basis",
}
COLORS = {
    "full_spsa": "#303030",
    "amortized_none": "#9A9A9A",
    "amortized_fixed": "#4C78A8",
    "amortized_gated": "#D1495B",
    "amortized_per_task": "#2A9D8F",
    "amortized_random": "#F4A261",
    "amortized_random_basis": "#7B61A8",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _two_se(values: list[float]) -> float:
    return 0.0 if len(values) < 2 else 2.0 * stdev(values) / math.sqrt(len(values))


def _topology_key(row: dict[str, str]) -> tuple:
    return (
        row["family"], int(row["n"]), int(row["p"]),
        int(row["graph_seed"]), row["graph_id"],
    )


def _task_curves(rows: list[dict[str, str]], methods: list[str]) -> dict:
    """Return task-index means/2SE after nesting measurement repetitions."""
    repeated: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        key = (
            _topology_key(row), row["method"], int(row["task_index"]),
            int(row.get("measurement_repeat", 0)),
        )
        repeated[key].append(float(row["approximation_ratio"]))
    nested: dict[tuple, list[float]] = defaultdict(list)
    for (unit, method, task, _repeat), values in repeated.items():
        nested[(unit, method, task)].append(mean(values))
    output = {}
    for method in methods:
        means, errors = [], []
        tasks = sorted({key[2] for key in nested if key[1] == method})
        for task in tasks:
            values = [
                mean(repetitions)
                for (unit, candidate, index), repetitions in nested.items()
                if candidate == method and index == task
            ]
            means.append(mean(values))
            errors.append(_two_se(values))
        output[method] = {"tasks": tasks, "mean": means, "two_se": errors}
    return output


def _refresh_curves(rows: list[dict[str, str]], methods: list[str]) -> dict:
    repeated: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] not in methods:
            continue
        key = (_topology_key(row), row["method"], int(row["task_index"]))
        repeated[key].append(float(row["refreshed"] in {"True", "true", "1"}))
    output = {}
    for method in methods:
        tasks = sorted({key[2] for key in repeated if key[1] == method})
        output[method] = {
            "tasks": tasks,
            "rate": [
                mean(mean(values) for (unit, candidate, index), values in repeated.items()
                     if candidate == method and index == task)
                for task in tasks
            ],
        }
    return output


def _style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.color": "#D7D7D7",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig: plt.Figure, stem: str) -> None:
    metadata = {"Creator": "make_amortized_paper_assets.py",
                "CreationDate": None, "ModDate": None}
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", metadata=metadata)
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def _box(ax, xy, width, height, text, *, face, edge="#333333", fontsize=8.0):
    patch = mpl.patches.FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.018,rounding_size=0.018",
        facecolor=face, edgecolor=edge, linewidth=1.0,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=fontsize, linespacing=1.25)


def figure_protocol() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.35),
                             gridspec_kw={"width_ratios": [1.28, 1.0]})
    ax, theorem = axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("a  Leakage-resistant task-stream protocol", loc="left", pad=8)

    _box(ax, (0.03, 0.79), 0.25, 0.13,
         "Fixed weight model\nbase plus rank-3\nloadings", face="#E8EEF7",
         fontsize=6.2)
    _box(ax, (0.03, 0.49), 0.25, 0.15,
         "Independent\ntraining bank\n8 latent-drift tasks", face="#DCEAF7",
         fontsize=6.2)
    _box(ax, (0.03, 0.17), 0.25, 0.15,
         "Independent\nevaluation stream\n8 latent-drift tasks", face="#E2F2ED",
         fontsize=6.2)
    _box(ax, (0.39, 0.50), 0.24, 0.14,
         "$\\widehat M=W^{\\mathsf{T}}W/8$\n$A=J(\\theta)^{\\mathsf{T}}L$", face="#FFF1D6",
         fontsize=7.1)
    _box(ax, (0.71, 0.50), 0.26, 0.14,
         "Randomized QB\n$Q_r$: reusable basis", face="#F9E2E5",
         fontsize=6.7)
    _box(ax, (0.49, 0.18), 0.43, 0.14,
         "Matched 40-step SPSA\nfull versus reduced coordinates", face="#E2F2ED",
         fontsize=6.7)

    arrow = dict(arrowstyle="-|>", lw=1.2, color="#4B4B4B",
                 mutation_scale=10)
    ax.annotate("", xy=(0.155, 0.65), xytext=(0.155, 0.79), arrowprops=arrow)
    ax.annotate("", xy=(0.155, 0.33), xytext=(0.155, 0.79), arrowprops=arrow)
    ax.annotate("", xy=(0.39, 0.57), xytext=(0.28, 0.57), arrowprops=arrow)
    ax.annotate("", xy=(0.71, 0.57), xytext=(0.63, 0.57), arrowprops=arrow)
    ax.annotate("", xy=(0.64, 0.32), xytext=(0.84, 0.50), arrowprops=arrow)
    ax.annotate("", xy=(0.49, 0.25), xytext=(0.28, 0.25), arrowprops=arrow)
    ax.text(0.34, 0.72, "shared statistical model only", color="#555555",
            fontsize=6.3)
    ax.text(0.36, 0.085, "No evaluation weight enters basis training",
            color="#166B52", fontsize=6.4, fontweight="bold")

    theorem.axis("off")
    theorem.set_title("b  Task-weighted optimality and audit", loc="left", pad=8)
    theorem.text(
        0.03, 0.89,
        "$M=\\mathbb{E}[ww^{\\mathsf{T}}]=LL^{\\mathsf{T}}$,  "
        "$P_r=Q_rQ_r^{\\mathsf{T}}$",
        ha="left", va="top", fontsize=9.0,
    )
    theorem.text(
        0.03, 0.73,
        "$\\mathbb{E}\\,\\Vert(I-P_r)\\nabla C_w\\Vert_2^2$\n"
        "$\\qquad=\\Vert(I-P_r)J^{\\mathsf{T}}L\\Vert_F^2$",
        ha="left", va="top", fontsize=11.0,
        bbox=dict(boxstyle="round,pad=0.45", fc="#FFF6E5", ec="#D4A64A"),
    )
    theorem.text(0.03, 0.43,
                 "Top left singular vectors of $J^{\\mathsf{T}}L$ minimize\n"
                 "expected discarded local gradient energy.",
                 ha="left", va="top", fontsize=8.3)
    theorem.text(0.03, 0.25,
                 "Development gate:  FAILED", color="#A8323E",
                 fontsize=10.5, fontweight="bold")
    theorem.text(0.03, 0.16,
                 "Exact gated deficit: -1.57 +/- 1.04 points (2 s.e.)\n"
                 "Finite-shot per-task signal: +0.61 +/- 1.57 points",
                 fontsize=7.8, va="top", linespacing=1.4)
    theorem.text(0.03, 0.015,
                 "The second result is inconclusive and simulator-assisted.",
                 fontsize=7.5, color="#555555")
    _save(fig, "figure_amortized_protocol")


def figure_exact(summary: dict) -> None:
    order = [
        "amortized_none", "amortized_fixed", "amortized_random",
        "amortized_random_basis", "amortized_gated", "amortized_per_task",
    ]
    all_order = ["full_spsa"] + order
    fig = plt.figure(figsize=(7.25, 5.25), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.08, 1.0])
    delta_ax = fig.add_subplot(grid[0, :])
    cost_ax = fig.add_subplot(grid[1, 0])
    dim_ax = fig.add_subplot(grid[1, 1])

    y = np.arange(len(order))
    values = np.array([
        100 * summary["paired_comparisons"][f"{m}_minus_full_spsa"]["mean"]
        for m in order
    ])
    errors = np.array([
        100 * summary["paired_comparisons"][f"{m}_minus_full_spsa"]["two_se"]
        for m in order
    ])
    delta_ax.axvspan(-1.0, 0.0, color="#E8F2EC", zorder=0,
                     label="Prespecified quality band")
    delta_ax.axvline(-1.0, color="#56896B", lw=1.0, ls="--")
    delta_ax.axvline(0.0, color="#333333", lw=0.9)
    for i, method in enumerate(order):
        delta_ax.errorbar(values[i], y[i], xerr=errors[i], fmt="o",
                          ms=6.2, capsize=3, color=COLORS[method],
                          ecolor=COLORS[method], lw=1.4)
    delta_ax.set_yticks(y, [DISPLAY[m] for m in order])
    delta_ax.invert_yaxis()
    delta_ax.set_xlabel("Paired mean approximation-ratio difference vs full SPSA (percentage points; +/-2 s.e.)")
    delta_ax.set_title("a  Exact-statevector falsification audit", loc="left")
    delta_ax.grid(axis="x")
    delta_ax.legend(loc="upper right", frameon=False)

    x = np.arange(len(all_order))
    short_labels = ["Full", "None", "Fixed", "Random", "Rnd basis",
                    "Gated", "Per-task"]
    objective = np.array([
        summary["aggregate"][m]["objective_evaluations"]["mean"]
        for m in all_order
    ])
    observable = np.array([
        summary["aggregate"][m]["observable_evaluations"]["mean"]
        for m in all_order
    ])
    vjps = np.array([
        summary["aggregate"][m]["simulator_vjps"]["mean"]
        for m in all_order
    ])
    cost_ax.bar(x, objective, color="#516D8D", label="Scalar objective")
    cost_ax.bar(x, observable, bottom=objective, color="#B6CBE0",
                label="Observable-map")
    cost_ax.set_xticks(x, short_labels, rotation=30, ha="right")
    cost_ax.set_ylabel("Forward circuit evaluations")
    cost_ax.set_title("b  Disjoint forward ledger", loc="left")
    cost_ax.grid(axis="y")
    twin = cost_ax.twinx()
    twin.scatter(x, vjps, marker="D", s=22, color="#8F3D57",
                 label="Simulator-only VJP", zorder=5)
    twin.set_ylabel("Simulator-only VJPs", color="#8F3D57")
    twin.spines["right"].set_visible(True)
    handles, labels = cost_ax.get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    cost_ax.legend(handles + handles2, labels + labels2, frameon=False,
                   loc="upper left", ncol=1)

    dims = np.array([
        summary["aggregate"][m]["optimized_dimension"]["mean"]
        for m in all_order
    ])
    refreshes = np.array([
        summary["aggregate"][m]["refreshes"]["mean"] for m in all_order
    ])
    dim_ax.bar(x, dims, color=[COLORS[m] for m in all_order], alpha=0.9)
    dim_ax.set_xticks(x, short_labels, rotation=30, ha="right")
    dim_ax.set_ylabel("Mean optimized dimension")
    dim_ax.set_title("c  Compression and refresh burden", loc="left")
    dim_ax.grid(axis="y")
    twin_dim = dim_ax.twinx()
    twin_dim.plot(x, refreshes, "o--", color="#B75D20", lw=1.2, ms=4)
    twin_dim.set_ylabel("Mean refreshes per 8-task stream", color="#B75D20")
    twin_dim.spines["right"].set_visible(True)
    _save(fig, "figure_amortized_exact_audit")


def figure_shot(summary: dict) -> None:
    order = ["amortized_gated", "amortized_per_task"]
    all_order = ["full_spsa"] + order
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.85),
                             constrained_layout=True)

    ax = axes[0]
    values = [
        100 * summary["paired_comparisons"][f"{m}_minus_full_spsa"]["mean"]
        for m in order
    ]
    errors = [
        100 * summary["paired_comparisons"][f"{m}_minus_full_spsa"]["two_se"]
        for m in order
    ]
    ax.axvline(0, color="#333333", lw=0.9)
    for index, method in enumerate(order):
        ax.errorbar(values[index], index, xerr=errors[index], fmt="o",
                    color=COLORS[method], capsize=3, lw=1.5, ms=6)
    ax.set_yticks(range(len(order)), [DISPLAY[m] for m in order])
    ax.invert_yaxis()
    ax.set_xlabel("Paired difference\n(points; +/-2 s.e.)")
    ax.set_title("a  256-shot pilot", loc="left")
    ax.grid(axis="x")

    ax = axes[1]
    ratios = [summary["aggregate"][m]["mean_ratio"]["mean"] for m in all_order]
    ratio_err = [summary["aggregate"][m]["mean_ratio"]["two_se"] for m in all_order]
    x = np.arange(len(all_order))
    ax.bar(x, ratios, yerr=ratio_err, capsize=3,
           color=[COLORS[m] for m in all_order])
    ax.set_ylim(0.74, 0.92)
    ax.set_xticks(x, [DISPLAY[m].replace(" ", "\n") for m in all_order],
                  rotation=24, ha="right")
    ax.set_ylabel("Mean approximation ratio")
    ax.set_title("b  Nested repeats", loc="left")
    ax.grid(axis="y")

    ax = axes[2]
    shots = [summary["aggregate"][m]["shots"]["mean"] / 1000 for m in all_order]
    vjps = [summary["aggregate"][m]["simulator_vjps"]["mean"] for m in all_order]
    ax.bar(x, shots, color=[COLORS[m] for m in all_order], alpha=0.85)
    ax.set_xticks(x, [DISPLAY[m].replace(" ", "\n") for m in all_order],
                  rotation=24, ha="right")
    ax.set_ylabel("Objective shots (thousands)")
    ax.set_title("c  Hybrid cost ledger", loc="left")
    ax.grid(axis="y")
    twin = ax.twinx()
    twin.plot(x, vjps, "D--", color="#8F3D57", lw=1.2, ms=4)
    twin.set_ylabel("Simulator-only VJPs", color="#8F3D57")
    twin.spines["right"].set_visible(True)
    _save(fig, "figure_amortized_shot_audit")


def figure_streams(rows: list[dict[str, str]]) -> None:
    methods = [
        "full_spsa", "amortized_gated", "amortized_per_task",
        "amortized_random_basis",
    ]
    curves = _task_curves(rows, methods)
    refresh_methods = [
        "amortized_fixed", "amortized_gated", "amortized_per_task",
        "amortized_random",
    ]
    refresh = _refresh_curves(rows, refresh_methods)
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.95),
                             constrained_layout=True)
    ax = axes[0]
    for method in methods:
        task = np.asarray(curves[method]["tasks"])
        center = np.asarray(curves[method]["mean"])
        error = np.asarray(curves[method]["two_se"])
        ax.plot(task, center, marker="o", ms=3.5, lw=1.5,
                color=COLORS[method], label=DISPLAY[method])
        ax.fill_between(task, center - error, center + error,
                        color=COLORS[method], alpha=0.11, linewidth=0)
    ax.set_xlabel("Evaluation-task index")
    ax.set_ylabel("Mean approximation ratio (+/-2 s.e.)")
    ax.set_title("a  Performance across the exact task stream", loc="left")
    ax.grid()
    ax.legend(frameon=False, ncol=2)

    ax = axes[1]
    for method in refresh_methods:
        ax.plot(refresh[method]["tasks"], refresh[method]["rate"], marker="o",
                ms=3.5, lw=1.5, color=COLORS[method], label=DISPLAY[method])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Evaluation-task index")
    ax.set_ylabel("Fraction of streams refreshed")
    ax.set_title("b  The residual gate rarely avoids refreshing", loc="left")
    ax.grid()
    ax.legend(frameon=False, ncol=2)
    _save(fig, "figure_amortized_stream")


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def table_exact(summary: dict) -> None:
    order = [
        "full_spsa", "amortized_none", "amortized_fixed",
        "amortized_gated", "amortized_per_task", "amortized_random",
        "amortized_random_basis",
    ]
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Method & Ratio $\pm2\,$s.e. & $\Delta$ full (points) & Dim. & Obj. & Obs. & VJP & Refresh \\",
        r"\midrule",
    ]
    for method in order:
        aggregate = summary["aggregate"][method]
        ratio = aggregate["mean_ratio"]
        if method == "full_spsa":
            delta = r"0.00 (ref.)"
        else:
            comparison = summary["paired_comparisons"][f"{method}_minus_full_spsa"]
            delta = (
                f"{100 * comparison['mean']:+.2f} $\\pm$ "
                f"{100 * comparison['two_se']:.2f}"
            )
        lines.append(
            f"{DISPLAY[method]} & {ratio['mean']:.4f} $\\pm$ {ratio['two_se']:.4f} "
            f"& {delta} & {aggregate['optimized_dimension']['mean']:.1f} "
            f"& {aggregate['objective_evaluations']['mean']:.0f} "
            f"& {aggregate['observable_evaluations']['mean']:.1f} "
            f"& {aggregate['simulator_vjps']['mean']:.1f} "
            f"& {aggregate['refreshes']['mean']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / "table_amortized_exact_audit.tex").write_text(
        "\n".join(lines) + "\n"
    )


def table_shot(summary: dict) -> None:
    order = ["full_spsa", "amortized_gated", "amortized_per_task"]
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & Ratio $\pm2\,$s.e. & $\Delta$ full (points) & Forward & Shots & VJP & Refresh \\",
        r"\midrule",
    ]
    for method in order:
        aggregate = summary["aggregate"][method]
        ratio = aggregate["mean_ratio"]
        if method == "full_spsa":
            delta = r"0.00 (ref.)"
        else:
            comparison = summary["paired_comparisons"][f"{method}_minus_full_spsa"]
            delta = (
                f"{100 * comparison['mean']:+.2f} $\\pm$ "
                f"{100 * comparison['two_se']:.2f}"
            )
        lines.append(
            f"{DISPLAY[method]} & {ratio['mean']:.4f} $\\pm$ {ratio['two_se']:.4f} "
            f"& {delta} & {aggregate['forward_circuit_evaluations']['mean']:.1f} "
            f"& {aggregate['shots']['mean']:.0f} "
            f"& {aggregate['simulator_vjps']['mean']:.1f} "
            f"& {aggregate['refreshes']['mean']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / "table_amortized_shot_audit.tex").write_text(
        "\n".join(lines) + "\n"
    )


def table_development_gate(summary: dict) -> None:
    gate = summary["thesis_gate"]
    rows = [
        (
            "Quality versus full SPSA",
            r"$\geq -0.010$",
            f"{gate['quality_delta_vs_full']:+.5f}",
            gate["quality_delta_vs_full"] >= -0.01,
        ),
        (
            "Quality versus random basis",
            r"$>0$",
            f"{gate['quality_delta_vs_random_basis']:+.5f}",
            gate["quality_delta_vs_random_basis"] > 0.0,
        ),
        (
            "Forward-cost ratio versus full",
            r"$\leq 1.25$",
            f"{gate['forward_cost_ratio_vs_full']:.5f}",
            gate["forward_cost_ratio_vs_full"] <= 1.25,
        ),
        (
            "Conjunctive development gate",
            "all pass",
            "observed",
            gate["survives_first_pilot"],
        ),
    ]
    lines = [
        r"\begin{tabular}{@{}lrrc@{}}",
        r"\toprule",
        r"Criterion & Required & Observed & Result \\",
        r"\midrule",
    ]
    for label, required, observed, passed in rows:
        result = "pass" if passed else r"\textbf{fail}"
        lines.append(
            f"{label} & {required} & {observed} & {result} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / "table_development_gate.tex").write_text(
        "\n".join(lines) + "\n"
    )


def table_protocol(exact: dict, shot: dict) -> None:
    lines = [
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"Track & Topologies & Depths & Tasks & Repeats & Evaluator \\",
        r"\midrule",
        (
            f"Exact development & {exact['n_sampling_units'] // 2} & 2 & "
            f"{exact['n_tasks_per_unit']} & 1 & exact statevector \\\\"
        ),
        (
            f"Finite-shot pilot & {shot['n_sampling_units']} & 1 & "
            f"{shot['n_tasks_per_unit']} & "
            f"{shot['measurement_repeats_per_method_unit'][0]} & 256 shots \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (TABLES / "table_protocol.tex").write_text("\n".join(lines) + "\n")


def table_reproduction() -> None:
    sources = [
        ("Legacy results", RESULTS / "maxcut_small.csv"),
        ("Exact task streams", EXACT_CSV),
        ("Shot task streams", SHOT_CSV),
        ("Legacy summary", LEGACY_JSON),
        ("Exact summary", EXACT_JSON),
        ("Shot summary", SHOT_JSON),
    ]
    lines = [
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Artifact & Repository path & SHA-256 prefix \\",
        r"\midrule",
    ]
    for label, path in sources:
        relative = path.relative_to(REPOSITORY).as_posix().replace("_", r"\_")
        lines.append(f"{label} & \\texttt{{{relative}}} & "
                     f"\\texttt{{{_sha256(path)[:12]}}} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / "table_reproduction.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    exact_rows = _load_rows(EXACT_CSV)
    shot_rows = _load_rows(SHOT_CSV)
    exact = _load_json(EXACT_JSON)
    shot = _load_json(SHOT_JSON)
    if len(exact_rows) != exact["n_rows"] or len(shot_rows) != shot["n_rows"]:
        raise RuntimeError("CSV and JSON row counts disagree")
    for rows, summary, name in (
        (exact_rows, exact, "exact"), (shot_rows, shot, "shot")
    ):
        hashes = sorted({row["protocol_sha256"] for row in rows})
        if hashes != summary["protocol_sha256"]:
            raise RuntimeError(f"{name} protocol hashes disagree")

    _style()
    figure_protocol()
    figure_exact(exact)
    figure_shot(shot)
    figure_streams(exact_rows)
    table_exact(exact)
    table_shot(shot)
    table_development_gate(exact)
    table_protocol(exact, shot)
    table_reproduction()

    assets = [
        FIGURES / f"{stem}.{suffix}"
        for stem in (
            "figure_amortized_protocol", "figure_amortized_exact_audit",
            "figure_amortized_shot_audit", "figure_amortized_stream",
        )
        for suffix in ("pdf", "png")
    ] + [
        TABLES / "table_amortized_exact_audit.tex",
        TABLES / "table_amortized_shot_audit.tex",
        TABLES / "table_development_gate.tex",
        TABLES / "table_protocol.tex",
        TABLES / "table_reproduction.tex",
    ]
    manifest = {
        "schema": 1,
        "source_hashes": {
            path.relative_to(REPOSITORY).as_posix(): _sha256(path)
            for path in (EXACT_CSV, EXACT_JSON, SHOT_CSV, SHOT_JSON)
        },
        "protocol_hashes": {
            "exact": exact["protocol_sha256"], "shot": shot["protocol_sha256"],
        },
        "sampling": {
            "exact_units": exact["n_sampling_units"],
            "shot_units": shot["n_sampling_units"],
            "shot_measurement_repeats_nested": shot[
                "measurement_repeats_per_method_unit"
            ],
            "tasks_per_stream": exact["n_tasks_per_unit"],
        },
        "asset_hashes": {
            path.relative_to(REPOSITORY).as_posix(): _sha256(path)
            for path in assets
        },
    }
    destination = TABLES / "amortized_asset_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
