"""Matrix-free operator correctness: JVP/VJP adjointness and FD-vs-autograd."""

import numpy as np
import pytest
import torch

from rsqaoa.circuits import MaxCutProblem
from rsqaoa.operator import (MatrixFreeSensitivity, QAOASensitivity,
                             adjointness_gap)
from rsqaoa.randqb import active_subspace, active_subspace_adjoint_free


def _problem(seed=0):
    n, p = 5, 2
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 3)]
    return MaxCutProblem(n=n, edges=edges, p=p)


def test_adjointness_autograd():
    prob = _problem()
    g = torch.Generator().manual_seed(0)
    op = QAOASensitivity(prob, prob.random_theta(generator=g))
    # <J v, w> == <v, J^T w> to ~machine precision with exact forward-mode
    assert adjointness_gap(op, seed=3, jvp_mode="autograd") < 1e-8


def test_adjointness_finite_difference():
    prob = _problem()
    g = torch.Generator().manual_seed(1)
    op = QAOASensitivity(prob, prob.random_theta(generator=g), fd_eps=1e-4)
    # FD JVP is adjoint-free; agreement limited by O(eps^2)
    assert adjointness_gap(op, seed=5, jvp_mode="fd") < 1e-4


def test_fd_matches_autograd_jvp():
    prob = _problem()
    g = torch.Generator().manual_seed(2)
    op = QAOASensitivity(prob, prob.random_theta(generator=g), fd_eps=1e-5)
    v = torch.randn(op.d, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    jv_fd = op.jvp(v, mode="fd")
    jv_ad = op.jvp(v, mode="autograd")
    assert torch.norm(jv_fd - jv_ad) / torch.norm(jv_ad).clamp_min(1e-12) < 1e-4


def test_dense_jacobian_consistency():
    prob = _problem()
    g = torch.Generator().manual_seed(3)
    op = QAOASensitivity(prob, prob.random_theta(generator=g))
    J = op.dense_jacobian()                 # (m, d)
    v = torch.randn(op.d, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    w = torch.randn(op.m, generator=torch.Generator().manual_seed(11), dtype=torch.float64)
    assert torch.norm(J @ v - op.jvp(v, mode="autograd")) < 1e-8
    assert torch.norm(J.t() @ w - op.vjp(w)) < 1e-8


def test_weighted_objective_gradient_is_jacobian_transpose_times_weights():
    base = _problem()
    weights = torch.linspace(0.5, 1.5, base.m, dtype=torch.float64)
    prob = MaxCutProblem(
        n=base.n, edges=base.edges, p=base.p, weights=weights
    )
    theta = prob.random_theta(generator=torch.Generator().manual_seed(21))
    variable = theta.clone().requires_grad_(True)
    gradient = torch.autograd.grad(prob.cut(variable), variable)[0]
    jacobian = QAOASensitivity(prob, theta).dense_jacobian()
    assert torch.allclose(gradient, jacobian.t() @ weights, atol=1e-10, rtol=1e-10)


def test_discarded_direction_obeys_first_order_residual_bound():
    prob = _problem()
    theta = prob.random_theta(generator=torch.Generator().manual_seed(22))
    op = QAOASensitivity(prob, theta)
    jacobian = op.dense_jacobian()
    _, _, vh = torch.linalg.svd(jacobian, full_matrices=False)
    q = vh[:3].t()
    projector = q @ q.t()
    direction = torch.randn(
        prob.dim, generator=torch.Generator().manual_seed(23), dtype=torch.float64
    )
    weights = torch.ones(prob.m, dtype=torch.float64)
    directional_difference = torch.dot(
        weights, jacobian @ ((torch.eye(prob.dim) - projector) @ direction)
    ).abs()
    residual = torch.linalg.matrix_norm(
        jacobian @ (torch.eye(prob.dim) - projector), ord=2
    )
    bound = weights.norm() * residual * direction.norm()
    assert directional_difference <= bound + 1e-12


def test_counts_increment():
    prob = _problem()
    op = QAOASensitivity(prob, torch.zeros(prob.dim, dtype=torch.float64))
    op.vjp(torch.ones(op.m, dtype=torch.float64))
    op.jvp(torch.ones(op.d, dtype=torch.float64), mode="fd")
    assert op.counts.vjp == 1 and op.counts.jvp == 1 and op.counts.forward_F >= 2


def test_backend_neutral_sensitivity_actions():
    generator = torch.Generator().manual_seed(13)
    J = torch.randn(5, 9, generator=generator, dtype=torch.float64)
    op = MatrixFreeSensitivity(
        d=9, m=5, jvp=lambda vector: J @ vector,
        vjp=lambda vector: J.t() @ vector,
    )
    result = active_subspace(
        op, tol=1e-6, block=2, jvp_mode="callback",
        generator=torch.Generator().manual_seed(14),
    )
    relative = torch.norm(J.t() - result.Q @ result.B) / torch.norm(J)
    assert relative < 1e-5
    assert op.counts.jvp > 0 and op.counts.vjp > 0

    forward = MatrixFreeSensitivity(d=9, m=5, jvp=lambda vector: J @ vector)
    basis, _ = active_subspace_adjoint_free(
        forward, rank=3, oversamp=2,
        generator=torch.Generator().manual_seed(15),
    )
    assert basis.shape == (9, 3)
    assert forward.counts.jvp > 0 and forward.counts.vjp == 0


def test_backend_neutral_contract_checks_shape_dtype_device_and_costs():
    identity = torch.eye(3, dtype=torch.float64)
    op = MatrixFreeSensitivity(
        d=3,
        m=3,
        jvp=lambda vector: identity @ vector,
        vjp=lambda vector: identity @ vector,
        forward_evals_per_jvp=2,
        forward_evals_per_vjp=1,
    )
    assert op.jvp(torch.ones(3, dtype=torch.float32)).dtype == torch.float64
    assert op.vjp(torch.ones(3)).device.type == "cpu"
    assert op.counts.as_dict() == {"forward_F": 3, "jvp": 1, "vjp": 1}

    with pytest.raises(ValueError, match="shape"):
        op.jvp(torch.ones(2))
    with pytest.raises(ValueError, match="CPU float64"):
        MatrixFreeSensitivity(
            d=3, m=3, jvp=lambda vector: vector, dtype=torch.float32
        )
