"""Task-bank construction and second-moment identities."""

import numpy as np  # load before torch on macOS
import pytest
import torch

from rsqaoa.amortized.task_streams import (WeightStream, iid_weight_stream,
                                           low_rank_drift_stream,
                                           make_low_rank_weight_model,
                                           sample_low_rank_drift_stream)


def test_weight_factor_reconstructs_empirical_second_moment():
    stream = low_rank_drift_stream(
        7, n_tasks=9, latent_rank=3, drift_scale=0.2, seed=11
    )
    factor = stream.factor()
    assert factor.shape == (7, 9)
    assert torch.allclose(factor @ factor.t(), stream.second_moment())
    assert torch.allclose(
        stream.weights.mean(dim=1), torch.ones(9, dtype=torch.float64)
    )
    assert torch.all(stream.weights > 0)


def test_streams_are_deterministic_and_iid_is_adverse_control():
    left = low_rank_drift_stream(6, n_tasks=5, seed=3)
    right = low_rank_drift_stream(6, n_tasks=5, seed=3)
    other = low_rank_drift_stream(6, n_tasks=5, seed=4)
    assert torch.equal(left.weights, right.weights)
    assert not torch.equal(left.weights, other.weights)
    iid = iid_weight_stream(6, n_tasks=5, seed=3)
    assert iid.mode == "iid" and iid.latent_rank == 5


def test_training_and_evaluation_streams_share_only_the_weight_model():
    model = make_low_rank_weight_model(7, latent_rank=3, seed=20)
    train = sample_low_rank_drift_stream(model, 7, n_tasks=8, seed=21)
    evaluation = sample_low_rank_drift_stream(model, 7, n_tasks=8, seed=22)
    assert not torch.equal(train.weights, evaluation.weights)
    assert train.latent_rank == evaluation.latent_rank == model.latent_rank
    # Removing each bank's mean leaves variation predominantly in the same
    # fixed edge-loading span, while trajectory innovations remain disjoint.
    centered_train = train.weights - train.weights.mean(dim=0)
    centered_eval = evaluation.weights - evaluation.weights.mean(dim=0)
    projector = model.loadings @ model.loadings.t()
    identity = torch.eye(7, dtype=torch.float64)
    assert torch.norm(centered_train @ (identity - projector)) < 0.35
    assert torch.norm(centered_eval @ (identity - projector)) < 0.35


def test_invalid_streams_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        WeightStream(
            weights=torch.tensor([[1.0, 0.0]]), mode="bad", seed=0,
            latent_rank=1, drift_scale=0.0,
        )
    with pytest.raises(ValueError, match="changepoint"):
        low_rank_drift_stream(4, n_tasks=3, changepoint=3)
