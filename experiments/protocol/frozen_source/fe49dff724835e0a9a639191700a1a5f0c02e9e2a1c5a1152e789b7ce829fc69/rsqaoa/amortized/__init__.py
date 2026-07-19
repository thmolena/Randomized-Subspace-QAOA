"""Amortized objective-bank optimization extensions for RSQ."""

from .budget import ResourceLedger
from .evaluators import (ExactEvaluator, ObservableEstimate, ShotEvaluator,
                         make_evaluator)
from .operators import (ResidualEstimate, TaskSubspace,
                        build_task_subspace, empirical_weight_factor,
                        objective_conditioned_residual)
from .optimizers import (SPSAConfig, SPSAResult, full_space_spsa,
                         optimize_spsa, reduced_space_spsa)
from .protocol import (build_protocol, canonical_json, sha256_file,
                       source_hash, validate_protocol)
from .stream_opt import (StreamResult, TaskRecord, optimize_amortized_stream,
                         optimize_full_stream)
from .task_streams import (LowRankWeightModel, WeightStream,
                           iid_weight_stream, low_rank_drift_stream,
                           make_low_rank_weight_model,
                           sample_low_rank_drift_stream)

__all__ = [
    "ResourceLedger", "ExactEvaluator", "ObservableEstimate",
    "ShotEvaluator", "make_evaluator", "ResidualEstimate", "TaskSubspace",
    "build_task_subspace", "empirical_weight_factor",
    "objective_conditioned_residual", "SPSAConfig", "SPSAResult",
    "full_space_spsa", "optimize_spsa", "reduced_space_spsa",
    "build_protocol", "canonical_json", "sha256_file", "source_hash",
    "validate_protocol",
    "StreamResult", "TaskRecord", "optimize_amortized_stream",
    "optimize_full_stream", "LowRankWeightModel", "WeightStream",
    "iid_weight_stream", "low_rank_drift_stream",
    "make_low_rank_weight_model", "sample_low_rank_drift_stream",
]
