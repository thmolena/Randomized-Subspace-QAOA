"""Freeze an amortized-RSQ experiment configuration and source hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from rsqaoa.amortized.protocol import build_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPOSITORY / "experiments/configs/amortized_development.yaml"),
    )
    parser.add_argument(
        "--output",
        default=str(REPOSITORY / "experiments/protocol/amortized_development.json"),
    )
    args = parser.parse_args()
    payload = build_protocol(Path(args.config), REPOSITORY)
    destination = Path(args.output)
    if destination.is_file():
        existing = json.loads(destination.read_text())
        if existing == payload:
            print(destination)
            print(payload["protocol_sha256"])
            return
        raise SystemExit(
            "refusing to overwrite a different frozen protocol; choose a "
            "new --output path so the existing evidence identity is retained"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(destination)
    print(payload["protocol_sha256"])


if __name__ == "__main__":
    main()
