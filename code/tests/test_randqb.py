"""Adaptive randomized-QB behavior: tolerance tracking, rank pruning, and
active-subspace quality on a real ma-QAOA operator."""

import numpy as np
import torch

from rsqaoa.circuits import MaxCutProblem
from rsqaoa.operator import QAOASensitivity
from rsqaoa.randqb import randqb, active_subspace, certified_residual


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
    # certificate on the same operator should report small residual for this Q
    rel = certified_residual(op, Q, generator=torch.Generator().manual_seed(6))
    assert rel <= 5e-2 + 1e-6
