"""Freeze an amortized-RSQ experiment configuration and source hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsqaoa.amortized.protocol import build_protocol


REPOSITORY = Path(__file__).resolve().parents[1]


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
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(destination)
    print(payload["protocol_sha256"])


if __name__ == "__main__":
    main()
