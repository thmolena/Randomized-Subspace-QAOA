"""Build the manuscript's figures, table, and machine-readable summary.

The script consumes only the tidy CSV emitted by ``run_experiment.py``.  It
does not rerun or alter experiments, and every plotted point is a direct group
summary or a paired difference against full ma-QAOA on the same instance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as stats
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    ("full_maqaoa", "", "Full ma-QAOA"),
    ("symmetry", "", "Symmetry-tied"),
    ("fixed_rank_oracle", "", "Fixed-rank oracle"),
    ("rsq", "0.1", "RSQ ($\\varepsilon=0.1$)"),
    ("rsq", "0.01", "RSQ ($\\varepsilon=0.01$)"),
    ("rsq_adjoint_free", "", "Forward-only RSQ"),
]


def mean_sd(values):
    values = list(values)
    return stats.mean(values), stats.stdev(values) if len(values) > 1 else 0.0


def mean_se(values):
    values = list(values)
    mean, sd = mean_sd(values)
    return mean, sd / math.sqrt(len(values))


def select(rows, method, tol="", depth=None):
    return [
        row for row in rows
        if row["method"] == method and row["tol"] == tol
        and (depth is None or int(row["p"]) == depth)
    ]


def key(row):
    return row["family"], int(row["n"]), int(row["p"]), int(row["seed"])


def paired_delta(rows, method, tol, depth=None, family=None):
    full = {
        key(row): float(row["approx_ratio"])
        for row in select(rows, "full_maqaoa", depth=depth)
        if family is None or row["family"] == family
    }
    return [
        float(row["approx_ratio"]) - full[key(row)]
        for row in select(rows, method, tol, depth)
        if key(row) in full and (family is None or row["family"] == family)
    ]


def configure_plotting():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.dpi": 160,
        "savefig.dpi": 240,
        "savefig.bbox": "tight",
    })


def save(fig, outdir, stem):
    fig.savefig(outdir / f"{stem}.pdf")
    fig.savefig(outdir / f"{stem}.png")
    plt.close(fig)


def make_tradeoff(rows, outdir):
    colors = {1: "#2563eb", 2: "#db2777"}
    fig, ax = plt.subplots(figsize=(6.7, 3.8))
    for depth, marker in [(1, "o"), (2, "s")]:
        full = select(rows, "full_maqaoa", depth=depth)
        full_d = {key(row): float(row["d"]) for row in full}
        plotted = [entry for entry in METHODS[1:]
                   if not (entry[0] == "rsq" and entry[1] == "0.01")]
        for method, tol, label in plotted:
            current = select(rows, method, tol, depth)
            compression = [
                100.0 * (1.0 - float(row["params_opt"]) / full_d[key(row)])
                for row in current
            ]
            delta = paired_delta(rows, method, tol, depth)
            xm, xse = mean_se(compression)
            ym, yse = mean_se(delta)
            ax.errorbar(xm, 100 * ym, xerr=2 * xse, yerr=200 * yse,
                        marker=marker, ms=7, capsize=3, color=colors[depth],
                        markeredgecolor="white", markeredgewidth=0.7,
                        alpha=0.9)
            short = {"Symmetry-tied": "Symmetry-tied",
                     "Fixed-rank oracle": "Fixed-rank oracle",
                     "RSQ ($\\varepsilon=0.1$)": "RSQ",
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
            delta = paired_delta(rows, "rsq", "0.1", depth, family)
            mean, se = mean_se(delta)
            means.append(100 * mean)
            errors.append(200 * se)
        ax.bar(x + offset, means, width, yerr=errors, capsize=3,
               label=f"p={depth}", color=color, alpha=0.86)
    ax.axhline(0, color="#111827", lw=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("RSQ minus full ma-QAOA (percentage points)")
    ax.set_title("Paired quality change across graph families")
    ax.legend(frameon=False)
    save(fig, outdir, "figure2_family")


def make_budget(rows, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), sharey=True)
    entries = [("full_maqaoa", "", "Full"),
               ("rsq", "0.1", "RSQ"),
               ("rsq_adjoint_free", "", "Forward-only")]
    colors = ["#9ca3af", "#2563eb", "#0f766e"]
    for ax, depth in zip(axes, [1, 2]):
        x = np.arange(len(entries))
        fwd, jvp, vjp = [], [], []
        for method, tol, _ in entries:
            current = select(rows, method, tol, depth)
            fwd.append(stats.mean(float(row["forward_F"]) for row in current))
            jvp.append(stats.mean(float(row["jvp"]) for row in current))
            vjp.append(stats.mean(float(row["vjp"]) for row in current))
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
            current = [row for row in select(rows, "rsq", "0.1", depth)
                       if int(row["n"]) == n]
            xlabels.append(f"n={n}, p={depth}")
            dims.append(stats.mean(float(row["d"]) for row in current))
            ranks.append(stats.mean(float(row["rank"]) for row in current))
    x = np.arange(len(xlabels))
    ax.bar(x, dims, color="#d1d5db", label="full parameter dimension")
    ax.bar(x, ranks, color="#2563eb", label="certified RSQ rank")
    ax.set_xticks(x, xlabels, rotation=16)
    ax.set_ylabel("dimension")
    ax.set_title("Certified active dimension grows with observables, not layers")
    ax.legend(frameon=False)
    save(fig, outdir, "figure4_rank")


def write_outputs(rows, outdir, table_path, summary_path):
    summary = {"n_rows": len(rows), "n_instances": len({key(row) for row in rows})}
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Depth & Parameters & Approx. ratio & Paired change \\",
        r"\midrule",
    ]
    for depth in [1, 2]:
        for method, tol, label in METHODS:
            current = select(rows, method, tol, depth)
            ratios = [float(row["approx_ratio"]) for row in current]
            params = [float(row["params_opt"]) for row in current]
            rm, rsd = mean_sd(ratios)
            pm, _ = mean_sd(params)
            if method == "full_maqaoa":
                delta_text = "--"
            else:
                delta, se = mean_se(paired_delta(rows, method, tol, depth))
                delta_text = f"{100*delta:+.2f} $\\pm$ {200*se:.2f} pp"
            lines.append(
                f"{label} & {depth} & {pm:.1f} & {rm:.3f} $\\pm$ {rsd:.3f} & {delta_text} \\\\"
            )
            summary[f"p{depth}:{method}:{tol or 'none'}"] = {
                "n": len(current), "parameters_mean": pm,
                "approx_ratio_mean": rm, "approx_ratio_sd": rsd,
                "paired_delta_mean": 0.0 if method == "full_maqaoa" else delta,
                "paired_delta_2se": 0.0 if method == "full_maqaoa" else 2 * se,
            }
        if depth == 1:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    table_path.write_text("\n".join(lines) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/maxcut_small.csv")
    parser.add_argument("--paper", default="../paper")
    args = parser.parse_args()
    rows = list(csv.DictReader(open(args.csv, newline="")))
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
    write_outputs(rows, figures, tables / "table1_summary.tex",
                  Path(args.csv).with_name("summary.json"))
    print(f"summarized {len(rows)} rows into {figures} and {tables}")


if __name__ == "__main__":
    main()
