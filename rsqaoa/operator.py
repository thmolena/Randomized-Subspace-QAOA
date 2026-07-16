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
from typing import Callable, Optional, Union

import torch

from .circuits import MaxCutProblem, RDTYPE


@dataclass
class OpCounts:
    """Bookkeeping for overlapping operator-application budgets.

    ``forward_F`` counts evaluations of the observable map needed by an
    algorithm.  ``jvp`` and ``vjp`` count derivative actions and are reported
    separately; they are not added to ``forward_F`` to estimate physical shots.
    A finite-difference JVP contributes two forward evaluations, while a
    reverse-mode VJP contributes one forward evaluation.
    """
    forward_F: int = 0   # forward evaluations of F (including those in JVP/VJP)
    jvp: int = 0         # J v applications
    vjp: int = 0         # J^T w applications

    def as_dict(self):
        return {"forward_F": self.forward_F, "jvp": self.jvp, "vjp": self.vjp}


class MatrixFreeSensitivity:
    """Backend-neutral sensitivity actions for a user-supplied observable map.

    This adapter lets callers use the randomized subspace builders with
    parameter-shift rules, hardware services, differentiable simulators, or
    domain-specific linear operators. ``jvp`` maps a parameter-space vector of
    length ``d`` to ``m`` observable sensitivities. ``vjp`` is optional; omit it
    for the forward-only builders.
    """

    def __init__(
        self,
        d: int,
        m: int,
        jvp: Callable[[torch.Tensor], torch.Tensor],
        vjp: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        *,
        dtype: torch.dtype = RDTYPE,
        device: Union[str, torch.device] = "cpu",
        forward_evals_per_jvp: int = 0,
        forward_evals_per_vjp: int = 0,
    ):
        if int(d) != d or d < 1 or int(m) != m or m < 1:
            raise ValueError("d and m must be positive integers")
        if not callable(jvp) or (vjp is not None and not callable(vjp)):
            raise TypeError("jvp must be callable and vjp must be callable or None")
        if not dtype.is_floating_point:
            raise TypeError("dtype must be a real floating-point torch dtype")
        for name, value in (
            ("forward_evals_per_jvp", forward_evals_per_jvp),
            ("forward_evals_per_vjp", forward_evals_per_vjp),
        ):
            if int(value) != value or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        self.d = int(d)
        self.m = int(m)
        requested_device = torch.device(device)
        if requested_device.type != "cpu" or dtype != RDTYPE:
            raise ValueError(
                "the current randomized core uses CPU float64 operator actions"
            )
        self.dtype = dtype
        self.device = requested_device
        self._jvp = jvp
        self._vjp = vjp
        self._forward_evals_per_jvp = int(forward_evals_per_jvp)
        self._forward_evals_per_vjp = int(forward_evals_per_vjp)
        self.counts = OpCounts()

    def _checked(self, value, expected: int, name: str) -> torch.Tensor:
        out = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        if out.ndim != 1 or out.numel() != expected:
            raise ValueError(f"{name} callback must return shape ({expected},)")
        if not torch.isfinite(out).all():
            raise ValueError(f"{name} callback returned a non-finite value")
        return out

    def jvp(self, vector: torch.Tensor, mode: str = "callback") -> torch.Tensor:
        del mode  # accepted for compatibility with QAOASensitivity
        vector = self._checked(vector, self.d, "jvp input")
        out = self._checked(self._jvp(vector), self.m, "jvp")
        self.counts.jvp += 1
        self.counts.forward_F += self._forward_evals_per_jvp
        return out

    def vjp(self, vector: torch.Tensor) -> torch.Tensor:
        if self._vjp is None:
            raise RuntimeError(
                "this sensitivity is forward-only; use an adjoint-free builder")
        vector = self._checked(vector, self.m, "vjp input")
        out = self._checked(self._vjp(vector), self.d, "vjp")
        self.counts.vjp += 1
        self.counts.forward_F += self._forward_evals_per_vjp
        return out


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
        if theta0.ndim != 1 or theta0.numel() != problem.dim:
            raise ValueError(f"theta0 must have shape ({problem.dim},)")
        if fd_eps <= 0:
            raise ValueError("fd_eps must be positive")
        self.theta0 = theta0.detach().clone().to(
            device=problem.C.device, dtype=RDTYPE
        )
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
        if v.ndim != 1 or v.numel() != self.d:
            raise ValueError(f"v must have shape ({self.d},)")
        v = v.to(device=self.theta0.device, dtype=RDTYPE)
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
            self.counts.forward_F += 1
            self.counts.jvp += 1
            return jv.detach()
        raise ValueError(f"unknown jvp mode {mode!r}")

    # -- J^T w : reverse-mode ----------------------------------------------
    def vjp(self, w: torch.Tensor) -> torch.Tensor:
        if w.ndim != 1 or w.numel() != self.m:
            raise ValueError(f"w must have shape ({self.m},)")
        w = w.to(device=self.theta0.device, dtype=RDTYPE)
        t = self.theta0.clone().requires_grad_(True)
        out = self.problem.F(t)
        (out * w).sum().backward()
        self.counts.forward_F += 1
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
    v = torch.randn(op.d, generator=g, dtype=RDTYPE).to(op.theta0.device)
    w = torch.randn(op.m, generator=g, dtype=RDTYPE).to(op.theta0.device)
    Jv = op.jvp(v, mode=jvp_mode)
    JTw = op.vjp(w)
    lhs = torch.dot(Jv, w)
    rhs = torch.dot(v, JTw)
    denom = (Jv.norm() * w.norm()).clamp_min(1e-30)
    return float((lhs - rhs).abs() / denom)
