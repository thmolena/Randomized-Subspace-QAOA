"""rsqaoa -- Randomized Subspace QAOA.

Adaptive, matrix-free two-sided subspace optimization plus a distinct
forward-only fixed-rank mode for multi-angle QAOA. Public API is intentionally
small and flat.
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
from .operator import (MatrixFreeSensitivity, QAOASensitivity, OpCounts,
                       adjointness_gap)
from .randqb import (
    QBResult,
    randqb,
    spectral_residual,
    active_subspace,
    active_subspace_adjoint_free,
    randomized_residual,
    randomized_residual_forward_only,
    certified_residual,
    certified_residual_forward_only,
    residual_ratio_confidence,
    ResidualRatioConfidence,
)
from .subspace_opt import RSQResult, optimize_rsq
from . import baselines
from . import graphs

__version__ = "0.3.0"

__all__ = [
    "MaxCutProblem", "cut_table", "cut_value", "edge_expectations", "n_params",
    "statevector", "unpack",
    "MatrixFreeSensitivity", "QAOASensitivity", "OpCounts", "adjointness_gap",
    "QBResult", "randqb", "spectral_residual", "active_subspace",
    "active_subspace_adjoint_free", "randomized_residual",
    "randomized_residual_forward_only",
    "certified_residual", "certified_residual_forward_only",
    "residual_ratio_confidence", "ResidualRatioConfidence",
    "RSQResult", "optimize_rsq",
    "baselines", "graphs",
    "__version__",
]
