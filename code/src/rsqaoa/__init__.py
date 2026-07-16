"""rsqaoa -- Randomized Subspace QAOA.

Adaptive, matrix-free, adjoint-free low-rank subspace optimization for
multi-angle QAOA. Public API is intentionally small and flat.
"""

from .circuits import (
    MaxCutProblem,
    cut_table,
    cut_value,
    edge_expectations,
    n_params,
    statevector,
    unpack,
)
from .operator import QAOASensitivity, OpCounts, adjointness_gap
from .randqb import (
    QBResult,
    randqb,
    spectral_residual,
    active_subspace,
    active_subspace_adjoint_free,
    certified_residual,
)
from .subspace_opt import RSQResult, optimize_rsq
from . import baselines
from . import graphs

__version__ = "0.1.0"

__all__ = [
    "MaxCutProblem", "cut_table", "cut_value", "edge_expectations", "n_params",
    "statevector", "unpack",
    "QAOASensitivity", "OpCounts", "adjointness_gap",
    "QBResult", "randqb", "spectral_residual", "active_subspace",
    "active_subspace_adjoint_free", "certified_residual",
    "RSQResult", "optimize_rsq",
    "baselines", "graphs",
    "__version__",
]
