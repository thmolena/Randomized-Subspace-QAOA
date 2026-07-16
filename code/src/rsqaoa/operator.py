"""The ma-QAOA sensitivity operator as a *matrix-free* linear map.

Let ``F(theta) in R^m`` be the vector of per-edge cut expectations and
``J(theta) = dF/dtheta in R^{m x d}`` its Jacobian. We never build ``J``.
We expose it only through two actions:

* ``jvp(v)  = J v``     -- forward directional derivative (observable space),
* ``vjp(w)  = J^T w``   -- reverse (parameter space).

Two ways to get ``J v`` are provided:

* ``mode="fd"`` (default): a central finite difference
  ``(F(theta + eps v) - F(theta - eps v)) / (2 eps)``. This uses **only forward
  evaluations of F** -- no adjoint, no backprop -- so it is the operation that
  survives on shot-based hardware where you cannot differentiate through the
  device. This is the "adjoint-free" access mode.
* ``mode="autograd"``: exact forward-mode via double-backward, for reference.

``vjp`` uses reverse-mode autodiff (one backward pass), which is cheap on a
simulator and is the fast path used by the two-sided method.

Every call is counted so experiments can report operator-application budgets
(the currency the source randomized-QB work minimizes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from .circuits import MaxCutProblem, RDTYPE


@dataclass
class OpCounts:
    """Bookkeeping for operator-application budgets."""
    forward_F: int = 0   # scalar-vector forward evaluations of F
    jvp: int = 0         # J v applications
    vjp: int = 0         # J^T w applications

    def as_dict(self):
        return {"forward_F": self.forward_F, "jvp": self.jvp, "vjp": self.vjp}


class QAOASensitivity:
    """Matrix-free ``J(theta)`` at a fixed expansion point ``theta0``.

    Parameters
    ----------
    problem : MaxCutProblem
    theta0 : torch.Tensor
        Expansion point (shape ``(d,)``). Copied and detached.
    fd_eps : float
        Step for the finite-difference (adjoint-free) JVP.
    """

    def __init__(self, problem: MaxCutProblem, theta0: torch.Tensor,
                 fd_eps: float = 1e-4):
        self.problem = problem
        self.theta0 = theta0.detach().clone().to(RDTYPE)
        self.fd_eps = float(fd_eps)
        self.m = problem.m
        self.d = problem.dim
        self.counts = OpCounts()

    # -- forward observable -------------------------------------------------
    def F(self, theta: torch.Tensor) -> torch.Tensor:
        self.counts.forward_F += 1
        return self.problem.F(theta)

    # -- J v : forward directional derivative -------------------------------
    def jvp(self, v: torch.Tensor, mode: str = "fd") -> torch.Tensor:
        v = v.to(RDTYPE)
        if mode == "fd":
            eps = self.fd_eps
            fp = self.problem.F(self.theta0 + eps * v)
            fm = self.problem.F(self.theta0 - eps * v)
            self.counts.forward_F += 2
            self.counts.jvp += 1
            return (fp - fm) / (2.0 * eps)
        elif mode == "autograd":
            def f(t):
                return self.problem.F(t)
            _, jv = torch.autograd.functional.jvp(
                f, self.theta0, v, create_graph=False, strict=False)
            self.counts.jvp += 1
            return jv.detach()
        raise ValueError(f"unknown jvp mode {mode!r}")

    # -- J^T w : reverse-mode ----------------------------------------------
    def vjp(self, w: torch.Tensor) -> torch.Tensor:
        w = w.to(RDTYPE)
        t = self.theta0.clone().requires_grad_(True)
        out = self.problem.F(t)
        (out * w).sum().backward()
        self.counts.vjp += 1
        return t.grad.detach()

    # -- convenience: dense Jacobian (small instances / baselines only) -----
    def dense_jacobian(self) -> torch.Tensor:
        """Explicitly form ``J`` (``m x d``). Only for small instances / oracle
        baselines -- defeats the matrix-free point, used just for validation."""
        def f(t):
            return self.problem.F(t)
        return torch.autograd.functional.jacobian(f, self.theta0).detach()


def adjointness_gap(op: QAOASensitivity, seed: int = 0, jvp_mode: str = "fd") -> float:
    """Return ``|<J v, w> - <v, J^T w>| / (|Jv||w|)`` for random ``v, w``.
    Should be ~machine-eps for ``mode='autograd'`` and ~O(eps^2) for FD."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(op.d, generator=g, dtype=RDTYPE)
    w = torch.randn(op.m, generator=g, dtype=RDTYPE)
    Jv = op.jvp(v, mode=jvp_mode)
    JTw = op.vjp(w)
    lhs = torch.dot(Jv, w)
    rhs = torch.dot(v, JTw)
    denom = (Jv.norm() * w.norm()).clamp_min(1e-30)
    return float((lhs - rhs).abs() / denom)
