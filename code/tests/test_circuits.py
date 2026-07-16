"""Validate the pure-torch ma-QAOA simulator against an independent dense
(Kronecker-product) implementation, and check trivial-angle sanity."""

import numpy as np
import torch

from rsqaoa.circuits import MaxCutProblem, edge_expectations


def _dense_edge_expectations(n, edges, gammas, betas):
    """Independent dense reference: explicit Kronecker products."""
    dim = 2 ** n
    x = np.arange(dim)
    bits = ((x[None, :] >> np.arange(n)[:, None]) & 1).astype(np.int8)
    cuts = np.stack([bits[i] ^ bits[j] for (i, j) in edges], 0).astype(float)
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    I2 = np.eye(2, dtype=np.complex128)

    def kron_list(mats):
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    p = gammas.shape[0]
    for layer in range(p):
        D = np.exp(-1j * (gammas[layer][:, None] * cuts).sum(0))
        psi = D * psi
        for k in range(n):
            b = betas[layer, k]
            RX = np.array([[np.cos(b), -1j * np.sin(b)],
                           [-1j * np.sin(b), np.cos(b)]], dtype=np.complex128)
            mats = [I2] * n
            mats[n - 1 - k] = RX          # qubit k is LSB -> kron position n-1-k
            psi = kron_list(mats) @ psi
    probs = np.abs(psi) ** 2
    return cuts @ probs


def test_matches_dense_reference():
    rng = np.random.default_rng(1)
    n, p = 5, 2
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]
    gammas = rng.uniform(0, np.pi, (p, len(edges)))
    betas = rng.uniform(0, np.pi, (p, n))
    theta = torch.tensor(np.concatenate([gammas.ravel(), betas.ravel()]),
                         dtype=torch.float64)
    got = edge_expectations(theta, n, edges, p).numpy()
    ref = _dense_edge_expectations(n, edges, gammas, betas)
    assert np.max(np.abs(got - ref)) < 1e-10


def test_zero_angles_give_half():
    n, p = 4, 1
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    prob = MaxCutProblem(n=n, edges=edges, p=p)
    theta = torch.zeros(prob.dim, dtype=torch.float64)
    fe = prob.F(theta).numpy()
    assert np.allclose(fe, 0.5, atol=1e-12)


def test_cut_value_in_range():
    n, p = 6, 2
    edges = [(i, (i + 1) % n) for i in range(n)]
    prob = MaxCutProblem(n=n, edges=edges, p=p)
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        theta = prob.random_theta(generator=g)
        c = float(prob.cut(theta))
        assert -1e-9 <= c <= len(edges) + 1e-9
