"""Disjoint resource accounting for amortized RSQ experiments.

The legacy release reports overlapping forward/JVP/VJP counters.  That is
useful for implementation audits but unsuitable for a physical-query claim.
This module keeps forward circuit evaluations and simulator-only reverse-mode
actions separate.  No conversion from a VJP to hardware circuits is assumed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass
class ResourceLedger:
    """Additive, non-overlapping experiment-resource ledger.

    ``objective_evaluations`` counts forward parameter points at which only a
    scalar weighted objective is requested. ``observable_evaluations`` counts
    forward parameter points at which the complete observable vector is
    requested. Their sum is the number of forward circuit settings in the
    exact/shot backends. Reverse-mode VJPs are recorded separately and are not
    silently assigned a hardware-query cost.
    """

    objective_evaluations: int = 0
    observable_evaluations: int = 0
    shots: int = 0
    sensitivity_jvps: int = 0
    simulator_vjps: int = 0
    subspace_builds: int = 0
    residual_checks: int = 0
    refreshes: int = 0

    @property
    def forward_circuit_evaluations(self) -> int:
        return self.objective_evaluations + self.observable_evaluations

    def copy(self) -> "ResourceLedger":
        return ResourceLedger(**asdict(self))

    def add(self, other: "ResourceLedger") -> "ResourceLedger":
        for name in asdict(self):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        return self

    def difference(self, earlier: "ResourceLedger") -> "ResourceLedger":
        values = {
            name: getattr(self, name) - getattr(earlier, name)
            for name in asdict(self)
        }
        if any(value < 0 for value in values.values()):
            raise ValueError("resource ledgers are not monotonically ordered")
        return ResourceLedger(**values)

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["forward_circuit_evaluations"] = self.forward_circuit_evaluations
        return payload

    @classmethod
    def from_mapping(cls, values: Mapping[str, int]) -> "ResourceLedger":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values) - allowed - {"forward_circuit_evaluations"}
        if unknown:
            raise ValueError(f"unknown resource fields: {sorted(unknown)}")
        payload = {name: int(values.get(name, 0)) for name in allowed}
        if any(value < 0 for value in payload.values()):
            raise ValueError("resource counts must be nonnegative")
        return cls(**payload)
