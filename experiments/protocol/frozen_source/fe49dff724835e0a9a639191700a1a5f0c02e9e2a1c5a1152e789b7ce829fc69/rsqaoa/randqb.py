"""Adaptive, matrix-free randomized QB with fixed-tolerance stopping, a choice of
Frobenius- or spectral-norm residual indicator, rank pruning, and subspace
recycling.

Given a linear operator ``A`` accessible only through ``matvec`` (``A u``) and
``rmatvec`` (``A^T y``), build orthonormal ``Q`` and ``B = Q^T A`` until a
randomized estimate of ``||A - Q B|| / ||A||`` meets ``tol`` or the requested
rank cap is reached.  The returned residual is an indicator, not a deterministic
upper bound.

Design choices (validated in ``tests/``):

* **Direct residual indicator (Frobenius).** ``E||(A-QB) w||^2 = ||A-QB||_F^2``;
  self-correcting as the approximation improves.
* **Spectral indicator.** Randomized power iteration on the residual operator
  ``R = A - QB`` estimates ``||R||_2`` without forming ``R`` -- a worst-direction
  diagnostic rather than an average one.
* **Rank pruning.** Directions carrying negligible energy are dropped, decoupling
  the retained rank from the (BLAS-friendly) block size.
* **Recycling.** A previous basis ``Q_init`` seeds the factorization; ``B`` is
  recomputed at the current operator and only missing directions are added, so
  subspace *refresh* during optimization is cheap (see ``subspace_opt``).

For the ma-QAOA active subspace we factor ``A = J^T`` (``d x m``), so ``Q``
(``d x r``) spans directions in *parameter* space.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional, Tuple
import warnings

import torch

from .circuits import RDTYPE
from .operator import QAOASensitivity


@dataclass
class QBResult:
    Q: torch.Tensor          # (dout, r) orthonormal
    B: torch.Tensor          # (r, din),  B = Q^T A
    rank: int
    rel_residual: float      # estimated ||A - QB|| / ||A|| in the chosen norm
    indicator: str           # "fro" or "spec"
    matvecs: int
    rmatvecs: int
    converged: bool          # whether the estimated residual met tol
    stop_reason: str         # "tolerance_met", "maxrank", or "no_progress"


@dataclass(frozen=True)
class ResidualRatioConfidence:
    """Chebyshev/union-bound envelope for a Gaussian probe ratio.

    If ``informative`` is true, multiplying an observed Frobenius residual
    ratio by ``lower_multiplier`` and ``upper_multiplier`` gives a simultaneous
    confidence envelope for the true ratio under the exact linear-operator
    model.  The bound is deliberately distribution-light and can be loose.
    """

    n_probe: int
    failure_probability: float
    relative_radius: float
    lower_multiplier: float
    upper_multiplier: float
    informative: bool


def residual_ratio_confidence(n_probe: int,
                              failure_probability: float = 0.05
                              ) -> ResidualRatioConfidence:
    """Return a rigorous finite-probe envelope for a Frobenius residual ratio.

    For Gaussian probes, each trace estimator has relative Chebyshev radius
    ``sqrt(2 / (s * delta))``.  Applying a union bound to numerator and
    denominator gives ``t = 2 / sqrt(s * delta)``.  When ``t < 1``, the true
    norm ratio lies between

    ``sqrt((1-t)/(1+t)) * observed`` and
    ``sqrt((1+t)/(1-t)) * observed``

    with probability at least ``1-delta``.  When ``t >= 1`` the elementary
    bound is non-informative; the returned upper multiplier is infinite.  This
    makes the distinction between a useful randomized diagnostic and a
    run-wise confidence statement explicit to callers.
    """
    if int(n_probe) != n_probe or n_probe < 1:
        raise ValueError("n_probe must be a positive integer")
    if not 0.0 < failure_probability < 1.0:
        raise ValueError("failure_probability must lie in (0, 1)")
    t = 2.0 / math.sqrt(float(n_probe) * failure_probability)
    if t >= 1.0:
        return ResidualRatioConfidence(
            n_probe=int(n_probe),
            failure_probability=float(failure_probability),
            relative_radius=t,
            lower_multiplier=0.0,
            upper_multiplier=math.inf,
            informative=False,
        )
    lower = math.sqrt((1.0 - t) / (1.0 + t))
    upper = 1.0 / lower
    return ResidualRatioConfidence(
        n_probe=int(n_probe),
        failure_probability=float(failure_probability),
        relative_radius=t,
        lower_multiplier=lower,
        upper_multiplier=upper,
        informative=True,
    )


# --------------------------------------------------------------------------
# residual estimators
# --------------------------------------------------------------------------
def _frob_norm2(matvec, din, n_probe, gen):
    P = torch.randn(din, n_probe, generator=gen, dtype=RDTYPE)
    return sum(float(matvec(P[:, i]).pow(2).sum()) for i in range(n_probe)) / n_probe


def _frob_residual(matvec, Q, B, din, n_probe, gen):
    W = torch.randn(din, n_probe, generator=gen, dtype=RDTYPE)
    AW = torch.stack([matvec(W[:, i]) for i in range(n_probe)], dim=1)
    R = AW - (Q @ (B @ W) if Q.shape[1] else torch.zeros_like(AW))
    return float((R.pow(2).sum() / n_probe).sqrt())


def spectral_residual(matvec, rmatvec, Q, B, din, iters=20, gen=None) -> float:
    """Estimate ``||A - QB||_2`` by power iteration on ``R^T R`` using only
    operator products (never forming ``R``). Validated to ratio ~1.0 against a
    dense reference in ``tests/test_extensions.py``."""
    if gen is None:
        gen = torch.Generator().manual_seed(0)
    v = torch.randn(din, generator=gen, dtype=RDTYPE)
    v = v / v.norm().clamp_min(1e-30)
    has_Q = Q.shape[1] > 0
    for _ in range(iters):
        Rv = matvec(v) - (Q @ (B @ v) if has_Q else 0.0)
        Mv = rmatvec(Rv) - (B.t() @ (Q.t() @ Rv) if has_Q else 0.0)   # R^T R v
        nv = Mv.norm()
        if nv < 1e-30:
            return 0.0
        v = Mv / nv
    Rv = matvec(v) - (Q @ (B @ v) if has_Q else 0.0)
    return float(Rv.norm() / v.norm().clamp_min(1e-30))


# --------------------------------------------------------------------------
# adaptive matrix-free QB
# --------------------------------------------------------------------------
def randqb(matvec: Callable[[torch.Tensor], torch.Tensor],
           rmatvec: Callable[[torch.Tensor], torch.Tensor],
           dout: int, din: int,
           tol: float = 1e-2, block: int = 4, maxrank: Optional[int] = None,
           indicator: str = "fro", n_norm: int = 20, n_res: int = 10,
           spec_iters: int = 20, prune_ratio: float = 1e-10,
           Q_init: Optional[torch.Tensor] = None,
           generator: Optional[torch.Generator] = None) -> QBResult:
    """Adaptive matrix-free QB factorization of ``A`` (``dout x din``).

    ``indicator`` selects the stopping norm: ``"fro"`` (direct Frobenius sketch)
    or ``"spec"`` (spectral power iteration). ``Q_init`` recycles a previous
    basis: ``B`` is recomputed at the current operator and only missing
    directions are added.
    """
    if not callable(matvec) or not callable(rmatvec):
        raise TypeError("matvec and rmatvec must be callable")
    if int(dout) != dout or int(din) != din or dout < 1 or din < 1:
        raise ValueError("dout and din must be positive integers")
    dout, din = int(dout), int(din)
    if not 0.0 < tol < 1.0:
        raise ValueError("tol must lie in (0, 1)")
    if indicator not in {"fro", "spec"}:
        raise ValueError("indicator must be 'fro' or 'spec'")
    for name, value in (("block", block), ("n_norm", n_norm),
                        ("n_res", n_res), ("spec_iters", spec_iters)):
        if int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if maxrank is not None and (int(maxrank) != maxrank or maxrank < 1):
        raise ValueError("maxrank must be a positive integer when supplied")
    if prune_ratio < 0:
        raise ValueError("prune_ratio must be nonnegative")
    if Q_init is not None:
        if Q_init.ndim != 2 or Q_init.shape[0] != dout:
            raise ValueError(f"Q_init must have shape ({dout}, r)")
        if not torch.isfinite(Q_init).all():
            raise ValueError("Q_init must contain only finite values")
    if maxrank is None:
        maxrank = min(dout, din)
    target_maxrank = min(int(maxrank), dout, din)
    if generator is None:
        generator = torch.Generator().manual_seed(0)
    mv = 0
    rmv = 0

    normA2 = max(_frob_norm2(matvec, din, n_norm, generator), 1e-300)
    mv += n_norm
    if indicator == "spec":
        normA = spectral_residual(matvec, rmatvec,
                                  torch.zeros(dout, 0, dtype=RDTYPE),
                                  torch.zeros(0, din, dtype=RDTYPE),
                                  din, iters=spec_iters, gen=generator)
        mv += spec_iters + 1
        rmv += spec_iters
        normA = max(normA, 1e-300)
    else:
        normA = normA2 ** 0.5

    # recycling: seed with a previous basis, recompute B at this operator
    if Q_init is not None and Q_init.shape[1] > 0:
        Q, _ = torch.linalg.qr(Q_init.to(RDTYPE))
        Q = Q[:, :target_maxrank]
        B = torch.stack([rmatvec(Q[:, i]) for i in range(Q.shape[1])], dim=0)
        rmv += Q.shape[1]
        # A recycled basis can already occupy the requested output rank while
        # missing directions of the *current* operator.  Permit a temporary
        # augmented basis, then recompress it below.  Without this workspace a
        # saturated recycled basis can never rotate and silently fails to meet
        # the new tolerance.
        work_maxrank = dout
    else:
        Q = torch.zeros(dout, 0, dtype=RDTYPE)
        B = torch.zeros(0, din, dtype=RDTYPE)
        work_maxrank = target_maxrank

    def rel_res():
        nonlocal mv, rmv
        if indicator == "spec":
            mv += spec_iters + 1
            rmv += spec_iters
            return spectral_residual(matvec, rmatvec, Q, B, din,
                                     iters=spec_iters, gen=generator) / normA
        mv += n_res
        return _frob_residual(matvec, Q, B, din, n_res, generator) / normA

    est = rel_res()
    no_progress = False
    while Q.shape[1] < work_maxrank and est > tol:
        Om = torch.randn(din, block, generator=generator, dtype=RDTYPE)
        Y = torch.stack([matvec(Om[:, i]) for i in range(block)], dim=1)
        mv += block
        if Q.shape[1]:
            Y = Y - Q @ (B @ Om)
            Y = Y - Q @ (Q.t() @ Y)
        Qi, _ = torch.linalg.qr(Y)
        Bi = torch.stack([rmatvec(Qi[:, i]) for i in range(Qi.shape[1])], dim=0)
        rmv += Qi.shape[1]
        keep = Bi.norm(dim=1) > prune_ratio * normA2 ** 0.5
        if keep.any():
            Qi, Bi = Qi[:, keep], Bi[keep]
            Q = torch.cat([Q, Qi], dim=1)
            B = torch.cat([B, Bi], dim=0)
            if Q.shape[1] > work_maxrank:
                Q, B = Q[:, :work_maxrank], B[:work_maxrank]
        else:
            # No numerically independent direction was found.  Continuing with
            # an unchanged basis would make the fixed-tolerance loop infinite.
            no_progress = True
            break
        est = rel_res()

    if Q.shape[1] > target_maxrank:
        # ``A ~= Q B`` and Q is orthonormal, so the SVD of the small B rotates
        # the augmented recycled space to the best target-rank basis available
        # in that space.  This is the step that makes recycling an update rather
        # than a frozen warm start.
        Ub, sb, Vhb = torch.linalg.svd(B, full_matrices=False)
        r = target_maxrank
        Q = Q @ Ub[:, :r]
        B = sb[:r, None] * Vhb[:r, :]
        est = rel_res()

    converged = bool(est <= tol)
    if converged:
        stop_reason = "tolerance_met"
    elif no_progress:
        stop_reason = "no_progress"
    else:
        stop_reason = "maxrank"
    return QBResult(Q=Q, B=B, rank=Q.shape[1], rel_residual=est,
                    indicator=indicator, matvecs=mv, rmatvecs=rmv,
                    converged=converged, stop_reason=stop_reason)


# --------------------------------------------------------------------------
# active-subspace builders for the ma-QAOA operator
# --------------------------------------------------------------------------
def active_subspace(op: QAOASensitivity, tol: float = 1e-2, block: int = 4,
                    maxrank: Optional[int] = None, indicator: str = "fro",
                    jvp_mode: str = "fd", Q_init: Optional[torch.Tensor] = None,
                    generator: Optional[torch.Generator] = None) -> QBResult:
    """Two-sided active subspace of ``J`` in parameter space (factor ``A=J^T``).

    ``matvec = J^T`` (a ``vjp``), ``rmatvec = J`` (a ``jvp``). ``Q_init`` recycles
    a prior basis for cheap refresh.
    """
    matvec = lambda u: op.vjp(u)
    rmatvec = lambda y: op.jvp(y, mode=jvp_mode)
    return randqb(matvec, rmatvec, dout=op.d, din=op.m, tol=tol, block=block,
                  maxrank=maxrank, indicator=indicator, Q_init=Q_init,
                  generator=generator)


def active_subspace_adjoint_free(op: QAOASensitivity, rank: int, oversamp: int = 10,
                                 jvp_mode: str = "fd",
                                 Q_init: Optional[torch.Tensor] = None,
                                 generator: Optional[torch.Generator] = None
                                 ) -> Tuple[torch.Tensor, float]:
    """Forward-only (adjoint-free) active subspace via a Rayleigh--Ritz sketch of
    ``J^T J`` using only ``jvp``. No adjoint, so no subspace iteration: a single
    pass that trades accuracy for forward-only access (e.g. QPU shots). The
    accuracy cost is measured, not assumed. When ``Q_init`` is supplied, its
    span is augmented with fresh orthogonal trial directions and the combined
    space is recompressed at the current operator. This gives a JVP-only
    recycled refresh rather than restarting from an unrelated sketch.
    """
    if int(rank) != rank or rank < 1:
        raise ValueError("rank must be a positive integer")
    if int(oversamp) != oversamp or oversamp < 0:
        raise ValueError("oversamp must be a nonnegative integer")
    if generator is None:
        generator = torch.Generator().manual_seed(0)
    # A thin QR factorization of a ``d x k`` sketch has only ``min(d, k)``
    # columns.  Cap the requested sketch width before indexing those columns;
    # this matters for the small instances used by the smoke test and for any
    # hardware problem whose parameter dimension is below ``rank+oversamp``.
    k = min(op.d, rank + oversamp)
    rank = min(rank, k)
    if Q_init is not None:
        Q_init = torch.as_tensor(Q_init, dtype=RDTYPE)
        if (Q_init.ndim != 2 or Q_init.shape[0] != op.d
                or not torch.isfinite(Q_init).all()):
            raise ValueError("Q_init must be a finite matrix with op.d rows")
        Q_seed, _ = torch.linalg.qr(Q_init, mode="reduced")
        Q_seed = Q_seed[:, :min(Q_seed.shape[1], k)]
    else:
        Q_seed = torch.zeros(op.d, 0, dtype=RDTYPE)

    n_fresh = k - Q_seed.shape[1]
    if n_fresh:
        omega = torch.randn(op.d, n_fresh, generator=generator, dtype=RDTYPE)
        if Q_seed.shape[1]:
            omega = omega - Q_seed @ (Q_seed.t() @ omega)
        Q_fresh, _ = torch.linalg.qr(omega, mode="reduced")
        Q0, _ = torch.linalg.qr(
            torch.cat([Q_seed, Q_fresh[:, :n_fresh]], dim=1), mode="reduced")
    else:
        Q0 = Q_seed
    Y = torch.stack([op.jvp(Q0[:, i], mode=jvp_mode) for i in range(k)], dim=1)
    S = Y.t() @ Y
    evals, evecs = torch.linalg.eigh(S)
    idx = torch.argsort(evals, descending=True)[:rank]
    Qz, _ = torch.linalg.qr(Q0 @ evecs[:, idx], mode="reduced")
    captured = float(evals[idx].clamp_min(0).sum()
                     / evals.clamp_min(0).sum().clamp_min(1e-300))
    return Qz, captured


def randomized_residual(op: QAOASensitivity, Q: torch.Tensor, n_probe: int = 12,
                        indicator: str = "fro", spec_iters: int = 20,
                        generator: Optional[torch.Generator] = None) -> float:
    """Randomized indicator of whether ``Q`` still spans the active subspace of the
    *current* ``J``: estimate ``||J^T - Q Q^T J^T|| / ||J^T||`` in the chosen
    norm. Drives the subspace-refresh trigger.

    With ``A = J^T``: ``matvec(u)=J^T u`` (a vjp), ``rmatvec(y)=J y`` (a jvp),
    and ``B = Q^T A`` has rows ``B[i]=J Q[:,i]`` (a jvp of each basis column).
    """
    if int(n_probe) != n_probe or n_probe < 1:
        raise ValueError("n_probe must be a positive integer")
    if indicator not in {"fro", "spec"}:
        raise ValueError("indicator must be 'fro' or 'spec'")
    if int(spec_iters) != spec_iters or spec_iters < 1:
        raise ValueError("spec_iters must be a positive integer")
    if generator is None:
        generator = torch.Generator().manual_seed(0)
    matvec = lambda u: op.vjp(u)
    rmatvec = lambda y: op.jvp(y)
    if Q.shape[1]:
        B = torch.stack([op.jvp(Q[:, i]) for i in range(Q.shape[1])], dim=0)  # (r, m)
    else:
        B = torch.zeros(0, op.m, dtype=RDTYPE)
    if indicator == "spec":
        empty_Q = torch.zeros(op.d, 0, dtype=RDTYPE)
        empty_B = torch.zeros(0, op.m, dtype=RDTYPE)
        num = spectral_residual(matvec, rmatvec, Q, B, din=op.m,
                                iters=spec_iters, gen=generator)
        den = spectral_residual(matvec, rmatvec, empty_Q, empty_B, din=op.m,
                                iters=spec_iters, gen=generator)
        return num / max(den, 1e-30)
    num = _frob_residual(matvec, Q, B, din=op.m, n_probe=n_probe, gen=generator)
    den = _frob_norm2(matvec, din=op.m, n_probe=n_probe, gen=generator) ** 0.5
    return num / max(den, 1e-30)


def randomized_residual_forward_only(
        op: QAOASensitivity, Q: torch.Tensor, n_probe: int = 12,
        generator: Optional[torch.Generator] = None) -> float:
    """Estimate discarded sensitivity using only forward ``Jv`` products.

    For Gaussian parameter-space probes ``v``, compare ``Jv`` with
    ``JQQ^T v``.  The root ratio of their accumulated squared norms estimates
    ``||J(I-QQ^T)||_F / ||J||_F`` without a VJP or a differentiable device.
    This is the drift diagnostic used by the adjoint-free optimizer.
    """
    if int(n_probe) != n_probe or n_probe < 1:
        raise ValueError("n_probe must be a positive integer")
    if generator is None:
        generator = torch.Generator().manual_seed(0)
    num = 0.0
    den = 0.0
    for _ in range(n_probe):
        v = torch.randn(op.d, generator=generator, dtype=RDTYPE)
        jv = op.jvp(v, mode="fd")
        projected = Q @ (Q.t() @ v) if Q.shape[1] else torch.zeros_like(v)
        jv_projected = op.jvp(projected, mode="fd")
        num += float((jv - jv_projected).pow(2).sum())
        den += float(jv.pow(2).sum())
    return (num / max(den, 1e-30)) ** 0.5


def certified_residual(*args, **kwargs) -> float:
    """Deprecated alias for :func:`randomized_residual`.

    The older name could be read as a deterministic certificate even though a
    finite randomized probe estimate need not upper-bound the true residual.
    """
    warnings.warn(
        "certified_residual is deprecated; use randomized_residual",
        DeprecationWarning,
        stacklevel=2,
    )
    return randomized_residual(*args, **kwargs)


def certified_residual_forward_only(*args, **kwargs) -> float:
    """Deprecated alias for :func:`randomized_residual_forward_only`."""
    warnings.warn(
        "certified_residual_forward_only is deprecated; use "
        "randomized_residual_forward_only",
        DeprecationWarning,
        stacklevel=2,
    )
    return randomized_residual_forward_only(*args, **kwargs)
