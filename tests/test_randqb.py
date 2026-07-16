"""Adaptive randomized-QB behavior: tolerance tracking, rank pruning, and
active-subspace quality on a real ma-QAOA operator."""

import numpy as np
import pytest
import torch

from rsqaoa.circuits import MaxCutProblem
from rsqaoa.operator import MatrixFreeSensitivity, QAOASensitivity
from rsqaoa.randqb import (active_subspace, active_subspace_adjoint_free,
                           certified_residual, randomized_residual, randqb,
                           residual_ratio_confidence)


def _synthetic(m=40, d=30, true_r=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(m, m, generator=g, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    s = torch.cat([torch.logspace(0, -4, true_r, dtype=torch.float64),
                   1e-10 * torch.ones(min(m, d) - true_r, dtype=torch.float64)])
    A = (U[:, :min(m, d)] * s) @ V[:, :min(m, d)].t()
    return A


def test_tolerance_and_rank_monotone():
    A = _synthetic()
    mv = lambda u: A @ u
    rmv = lambda y: A.t() @ y
    ranks = []
    for tol in [1e-1, 1e-2, 1e-3]:
        gen = torch.Generator().manual_seed(3)
        res = randqb(mv, rmv, dout=A.shape[0], din=A.shape[1],
                     tol=tol, block=2, generator=gen)
        true_rel = float(torch.norm(A - res.Q @ res.B) / torch.norm(A))
        assert true_rel <= tol * 2.0, (tol, true_rel)     # meets prescribed tolerance
        assert res.converged and res.stop_reason == "tolerance_met"
        ranks.append(res.rank)
    # tighter tolerance never needs fewer directions
    assert ranks[0] <= ranks[1] <= ranks[2]


def test_matrix_free_matches_dense():
    A = _synthetic(m=20, d=16, true_r=4, seed=2)
    mv = lambda u: A @ u
    rmv = lambda y: A.t() @ y
    gen = torch.Generator().manual_seed(1)
    res = randqb(mv, rmv, dout=20, din=16, tol=1e-3, block=4, generator=gen)
    # Q orthonormal and B == Q^T A
    assert torch.norm(res.Q.t() @ res.Q - torch.eye(res.rank, dtype=torch.float64)) < 1e-8
    assert torch.norm(res.B - res.Q.t() @ A) < 1e-8


def test_active_subspace_on_qaoa():
    n, p = 5, 2
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 3)]
    prob = MaxCutProblem(n=n, edges=edges, p=p)
    g = torch.Generator().manual_seed(0)
    op = QAOASensitivity(prob, prob.random_theta(generator=g))
    res = active_subspace(op, tol=1e-2, block=4, jvp_mode="autograd",
                          generator=torch.Generator().manual_seed(4))
    assert res.rank >= 1
    # orthonormal basis in parameter space
    Q = res.Q
    assert torch.norm(Q.t() @ Q - torch.eye(res.rank, dtype=torch.float64)) < 1e-6
    # residual diagnostic on the same operator should report a small value
    rel = randomized_residual(op, Q, generator=torch.Generator().manual_seed(6))
    assert rel <= 5e-2 + 1e-6


def test_finite_probe_confidence_envelope():
    loose = residual_ratio_confidence(12, failure_probability=0.05)
    assert not loose.informative
    assert np.isinf(loose.upper_multiplier)

    informative = residual_ratio_confidence(2000, failure_probability=0.05)
    assert informative.informative
    assert 0 < informative.lower_multiplier < 1
    assert informative.upper_multiplier > 1


def test_gaussian_probe_moment_identity_empirically():
    generator = torch.Generator().manual_seed(17)
    residual = torch.randn(4, 7, generator=generator, dtype=torch.float64)
    probes = torch.randn(7, 20000, generator=generator, dtype=torch.float64)
    estimate = (residual @ probes).pow(2).sum(dim=0).mean()
    truth = residual.pow(2).sum()
    assert abs(float(estimate / truth) - 1.0) < 0.03


def test_maxrank_status_exposes_unmet_tolerance():
    A = torch.diag(torch.tensor([1.0, 0.8, 0.6, 0.4], dtype=torch.float64))
    result = randqb(
        lambda x: A @ x, lambda y: A.t() @ y,
        dout=4, din=4, tol=1e-6, block=1, maxrank=1,
        generator=torch.Generator().manual_seed(21),
    )
    assert not result.converged
    assert result.stop_reason == "maxrank"
    assert result.rel_residual > 1e-6


def test_forward_only_recycled_refresh_is_orthonormal():
    A0 = _synthetic(m=12, d=8, true_r=4, seed=22)
    A1 = A0 + 0.02 * _synthetic(m=12, d=8, true_r=4, seed=23)
    first = MatrixFreeSensitivity(d=8, m=12, jvp=lambda x: A0 @ x)
    Q0, _ = active_subspace_adjoint_free(
        first, rank=4, oversamp=2,
        generator=torch.Generator().manual_seed(24),
    )
    current = MatrixFreeSensitivity(d=8, m=12, jvp=lambda x: A1 @ x)
    Q1, _ = active_subspace_adjoint_free(
        current, rank=4, oversamp=2, Q_init=Q0,
        generator=torch.Generator().manual_seed(25),
    )
    assert Q1.shape == (8, 4)
    assert torch.norm(Q1.t() @ Q1 - torch.eye(4, dtype=torch.float64)) < 1e-8
    assert current.counts.jvp == 6


def test_legacy_certificate_name_is_a_deprecated_alias():
    prob = MaxCutProblem(n=4, edges=[(0, 1), (1, 2), (2, 3)], p=1)
    theta = prob.random_theta(generator=torch.Generator().manual_seed(18))
    op = QAOASensitivity(prob, theta)
    result = active_subspace(
        op, tol=1e-2, generator=torch.Generator().manual_seed(19)
    )
    with pytest.deprecated_call(match="randomized_residual"):
        legacy = certified_residual(
            op, result.Q, generator=torch.Generator().manual_seed(20)
        )
    current = randomized_residual(
        op, result.Q, generator=torch.Generator().manual_seed(20)
    )
    assert legacy == pytest.approx(current)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tol": 0.0},
        {"tol": 1.0},
        {"indicator": "invalid"},
        {"block": 0},
        {"maxrank": 0},
    ],
)
def test_randqb_rejects_invalid_controls(kwargs):
    A = torch.eye(3, dtype=torch.float64)
    with pytest.raises(ValueError):
        randqb(lambda x: A @ x, lambda y: A.t() @ y,
               dout=3, din=3, **kwargs)
