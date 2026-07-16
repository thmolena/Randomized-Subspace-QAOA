"""Merge deterministic RSQ experiment shards into one validated tidy CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from run_experiment import FIELDS
from summarize_results import validate_rows


FAMILY_ORDER = {"regular": 0, "er": 1, "ring": 2}
METHOD_ORDER = {
    "full_maqaoa": 0,
    "symmetry": 1,
    "fixed_rank_oracle": 2,
    "rsq": 3,
    "rsq_adjoint_free": 4,
}


def row_key(row):
    tolerance_order = -float(row["tol"]) if row["tol"] else 0.0
    return (
        FAMILY_ORDER.get(row["family"], 99), int(row["n"]), int(row["p"]),
        int(row["seed"]), METHOD_ORDER.get(row["method"], 99), tolerance_order,
    )


def merge(inputs, output):
    rows = []
    for path in inputs:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != FIELDS:
                raise ValueError(f"unexpected columns in shard {path}")
            rows.extend(reader)
    rows.sort(key=row_key)
    validate_rows(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"merged {len(inputs)} shards and {len(rows)} validated rows into {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    merge(args.inputs, args.out)


if __name__ == "__main__":
    main()
