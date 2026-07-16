"""Tests for the spectral-norm residual indicator and subspace recycling."""

import torch

from rsqaoa.randqb import randqb, spectral_residual


def _synthetic(m=40, d=30, true_r=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(m, m, generator=g, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    s = torch.cat([torch.logspace(0, -4, true_r, dtype=torch.float64),
                   1e-10 * torch.ones(min(m, d) - true_r, dtype=torch.float64)])
    A = (U[:, :min(m, d)] * s) @ V[:, :min(m, d)].t()
    return A


def test_spectral_indicator_matches_dense():
    A = _synthetic()
    mv = lambda u: A @ u
    rmv = lambda y: A.t() @ y
    U, s, Vh = torch.linalg.svd(A, full_matrices=False)
    for r in [0, 2, 3, 5]:
        Q = U[:, :r]
        B = (torch.diag(s[:r]) @ Vh[:r]) if r > 0 else torch.zeros(0, A.shape[1], dtype=torch.float64)
        true = float(torch.linalg.svdvals(A - Q @ B)[0])
        est = spectral_residual(mv, rmv, Q, B, din=A.shape[1], iters=40,
                                gen=torch.Generator().manual_seed(1))
        assert abs(est - true) <= 0.05 * true + 1e-9, (r, est, true)


def test_spectral_indicator_stops_randqb():
    A = _synthetic()
    mv = lambda u: A @ u
    rmv = lambda y: A.t() @ y
    res = randqb(mv, rmv, dout=A.shape[0], din=A.shape[1], tol=1e-2,
                 indicator="spec", block=2, generator=torch.Generator().manual_seed(2))
    true_rel = float(torch.linalg.svdvals(A - res.Q @ res.B)[0] / torch.linalg.svdvals(A)[0])
    assert true_rel <= 2e-2, true_rel


def test_recycling_reduces_added_columns():
    A = _synthetic()
    mv = lambda u: A @ u
    rmv = lambda y: A.t() @ y
    coarse = randqb(mv, rmv, dout=A.shape[0], din=A.shape[1], tol=1e-1, block=2,
                    generator=torch.Generator().manual_seed(3))
    scratch = randqb(mv, rmv, dout=A.shape[0], din=A.shape[1], tol=1e-3, block=2,
                     generator=torch.Generator().manual_seed(4))
    recycled = randqb(mv, rmv, dout=A.shape[0], din=A.shape[1], tol=1e-3, block=2,
                      Q_init=coarse.Q, generator=torch.Generator().manual_seed(4))
    # both reach the tolerance
    for r in (scratch, recycled):
        rel = float(torch.norm(A - r.Q @ r.B) / torch.norm(A))
        assert rel <= 2e-3, rel
    # recycling adds fewer new columns than starting from scratch
    assert recycled.rank <= scratch.rank
