"""Pure-PyTorch multi-angle QAOA state-vector simulator for MaxCut / Ising.

Everything here is a differentiable function of a flat real parameter vector
``theta``. Keeping the whole forward pass in native torch ops (no custom
autograd, no external QNode) is what makes the sensitivity operator in
``operator.py`` accessible through ordinary autodiff and finite differences,
which is exactly what the matrix-free / adjoint-free randomized subspace method
needs.

Conventions
-----------
* ``n`` qubits, computational basis indexed ``x = 0 .. 2**n - 1`` with qubit ``k``
  the ``k``-th least-significant bit.
* Initial state ``|+>^{\\otimes n}`` (uniform amplitudes).
* Multi-angle phase separator: each edge ``(i, j)`` gets its own angle
  ``gamma`` and contributes ``exp(-i * gamma * C_ij)`` where
  ``C_ij(x) = (1 - z_i z_j)/2`` is the cut indicator (1 if the edge is cut).
* Multi-angle mixer: each qubit ``k`` gets its own angle ``beta`` and applies
  ``exp(-i * beta * X_k)``.
* Layers ``p``: ``(phase, mixer)`` repeated ``p`` times, with independent angles
  per layer.

The per-edge cut expectations ``<C_ij>`` form the vector-valued observable
``F(theta) in R^{|E|}`` whose Jacobian is the object we compress. The scalar
MaxCut objective is ``sum_e w_e <C_e>``.

Validated in ``tests/test_circuits.py`` against an independent dense
(Kronecker-product) simulator to machine precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

Edge = Tuple[int, int]

# real / complex working precision -- float64 keeps finite-difference JVPs clean
RDTYPE = torch.float64
CDTYPE = torch.complex128


def cut_table(n: int, edges: Sequence[Edge], device="cpu") -> torch.Tensor:
    """Return a ``(|E|, 2**n)`` float table ``C[e, x] = 1`` iff edge ``e`` is cut
    in basis state ``x``. Static (no gradient); precompute once per graph."""
    x = np.arange(2 ** n, dtype=np.int64)
    bits = ((x[None, :] >> np.arange(n)[:, None]) & 1).astype(np.int8)  # (n, 2**n)
    rows = [bits[i] ^ bits[j] for (i, j) in edges]
    C = np.stack(rows, axis=0).astype(np.float64) if rows else np.zeros((0, 2 ** n))
    return torch.tensor(C, dtype=RDTYPE, device=device)


def n_params(n: int, n_edges: int, p: int) -> int:
    """Dimension of the multi-angle parameter vector: ``p * (|E| + n)``."""
    return p * (n_edges + n)


def unpack(theta: torch.Tensor, n: int, n_edges: int, p: int):
    """Split flat ``theta`` into ``(gammas[p, |E|], betas[p, n])``."""
    g = theta[: p * n_edges].reshape(p, n_edges)
    b = theta[p * n_edges:].reshape(p, n)
    return g, b


def _apply_mixer_qubit(psi: torch.Tensor, k: int, n: int,
                       cos_b: torch.Tensor, sin_b: torch.Tensor) -> torch.Tensor:
    """Apply ``exp(-i * beta * X_k)`` to the state vector, no in-place ops
    (so the map stays autodiff- and functorch-friendly)."""
    psi_r = psi.reshape([2] * n)
    ax = n - 1 - k                       # qubit k is the k-th least-significant bit
    moved = torch.movedim(psi_r, ax, 0)  # shape (2, ...)
    a0, a1 = moved[0], moved[1]
    # [[cos, -i sin], [-i sin, cos]]
    n0 = cos_b * a0 - (sin_b * a1) * 1j
    n1 = cos_b * a1 - (sin_b * a0) * 1j
    out = torch.stack([n0, n1], dim=0)
    return torch.movedim(out, 0, ax).reshape(-1)


def statevector(theta: torch.Tensor, n: int, edges: Sequence[Edge], p: int,
                C: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Return the ma-QAOA state vector ``|psi(theta)>`` as a complex128 tensor."""
    if C is None:
        C = cut_table(n, edges, device=theta.device)
    n_edges = len(edges)
    dim = 2 ** n
    psi = torch.full((dim,), (1.0 / np.sqrt(dim)), dtype=CDTYPE, device=theta.device)
    g, b = unpack(theta, n, n_edges, p)
    for layer in range(p):
        # phase separator: exp(-i * sum_e gamma_e * C_e(x))
        ang = (g[layer].unsqueeze(1) * C).sum(dim=0)          # (2**n,) real
        phase = torch.exp(torch.complex(torch.zeros_like(ang), -ang))
        psi = phase * psi
        # multi-angle mixer
        for k in range(n):
            bk = b[layer, k]
            psi = _apply_mixer_qubit(psi, k, n, torch.cos(bk), torch.sin(bk))
    return psi


def edge_expectations(theta: torch.Tensor, n: int, edges: Sequence[Edge], p: int,
                      C: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Vector observable ``F(theta) = (<C_e>)_{e in E} in R^{|E|}`` (differentiable)."""
    if C is None:
        C = cut_table(n, edges, device=theta.device)
    psi = statevector(theta, n, edges, p, C)
    probs = (psi.conj() * psi).real                          # (2**n,)
    return C @ probs                                         # (|E|,)


def cut_value(theta: torch.Tensor, n: int, edges: Sequence[Edge], p: int,
              weights: Optional[torch.Tensor] = None,
              C: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Scalar (weighted) expected cut ``sum_e w_e <C_e>``."""
    fe = edge_expectations(theta, n, edges, p, C)
    if weights is None:
        return fe.sum()
    return (weights * fe).sum()


@dataclass
class MaxCutProblem:
    """Container bundling a graph instance with its precomputed cut table."""
    n: int
    edges: List[Edge]
    p: int
    weights: Optional[torch.Tensor] = None
    device: str = "cpu"

    def __post_init__(self):
        self.edges = [tuple(int(v) for v in e) for e in self.edges]
        self.C = cut_table(self.n, self.edges, device=self.device)
        if self.weights is not None and not torch.is_tensor(self.weights):
            self.weights = torch.tensor(self.weights, dtype=RDTYPE, device=self.device)

    @property
    def dim(self) -> int:
        return n_params(self.n, len(self.edges), self.p)

    @property
    def m(self) -> int:
        return len(self.edges)

    def F(self, theta: torch.Tensor) -> torch.Tensor:
        return edge_expectations(theta, self.n, self.edges, self.p, self.C)

    def cut(self, theta: torch.Tensor) -> torch.Tensor:
        return cut_value(theta, self.n, self.edges, self.p, self.weights, self.C)

    def random_theta(self, generator: Optional[torch.Generator] = None,
                     scale: float = np.pi) -> torch.Tensor:
        return scale * torch.rand(self.dim, dtype=RDTYPE, generator=generator)
