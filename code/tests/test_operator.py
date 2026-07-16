"""Matrix-free operator correctness: JVP/VJP adjointness and FD-vs-autograd."""

import numpy as np
import torch

from rsqaoa.circuits import MaxCutProblem
from rsqaoa.operator import QAOASensitivity, adjointness_gap


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


def test_counts_increment():
    prob = _problem()
    op = QAOASensitivity(prob, torch.zeros(prob.dim, dtype=torch.float64))
    op.vjp(torch.ones(op.m, dtype=torch.float64))
    op.jvp(torch.ones(op.d, dtype=torch.float64), mode="fd")
    assert op.counts.vjp == 1 and op.counts.jvp == 1 and op.counts.forward_F >= 2
