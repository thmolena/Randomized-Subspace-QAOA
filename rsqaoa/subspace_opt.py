"""Randomized Subspace QAOA (RSQ): optimize ma-QAOA inside an adaptively
discovered low-dimensional active subspace, refreshing it only when a randomized
residual diagnostic says it has drifted, and reusing the previous basis when it does.

Loop
----
1. At an expansion point ``theta0`` build the matrix-free ``J(theta0)`` and get an
   active-subspace basis ``Q`` (``d x r``) at tolerance ``tol`` via ``randqb``.
2. Optimize the reduced coordinates ``z in R^r`` (``theta = theta0 + Q z``) with
   Adam, optionally under a **trust-region step cap** so the first-order
   truncation bound stays valid step to step.
3. Every ``refresh_every`` steps, evaluate the randomized residual diagnostic of
   the *current* ``Q`` against the *current* ``J``. If it exceeds ``eps_refresh``,
   re-anchor ``theta0``, **recycle** the current ``Q`` to rebuild cheaply, and
   reset ``z``.

The residual-gated, recycled refresh is what makes the number and cost of
subspace rebuilds adapt to optimization progress rather than a fixed schedule.
When ``adjoint_free=True``, both subspace discovery and reduced optimization
avoid reverse-mode differentiation: the latter uses a two-evaluation SPSA
gradient estimate.  Everything is reported: objective trajectory, retained
rank over time, refresh count, and the overlapping operator-application budget
(forward F / JVP / VJP).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .circuits import MaxCutProblem, RDTYPE
from .operator import QAOASensitivity
from .randqb import (
    active_subspace,
    active_subspace_adjoint_free,
    randomized_residual,
    randomized_residual_forward_only,
)


@dataclass
class RSQResult:
    theta: torch.Tensor
    cut: float
    history: List[float] = field(default_factory=list)      # -objective per step
    cut_history: List[float] = field(default_factory=list)  # cut per step
    rank_history: List[int] = field(default_factory=list)
    refreshes: int = 0
    counts: dict = field(default_factory=dict)
    final_rank: int = 0
    indicator: str = "fro"
    residual_history: List[float] = field(default_factory=list)
    residual_steps: List[int] = field(default_factory=list)
    refresh_steps: List[int] = field(default_factory=list)
    subspace_builds: int = 0
    build_history: List[dict] = field(default_factory=list)
    best_theta: Optional[torch.Tensor] = None
    best_cut: float = float("-inf")
    best_step: int = -1


def optimize_rsq(problem: MaxCutProblem,
                 theta0: Optional[torch.Tensor] = None,
                 tol: float = 1e-2,
                 maxrank: Optional[int] = None,
                 steps: int = 200,
                 inner_lr: float = 0.05,
                 refresh_every: int = 25,
                 eps_refresh: float = 5e-2,
                 block: int = 4,
                 indicator: str = "fro",
                 recycle: bool = True,
                 step_cap: Optional[float] = None,
                 adjoint_free: bool = False,
                 af_rank: int = 8,
                 residual_probes: int = 12,
                 fd_eps: float = 1e-4,
                 spsa_eps: float = 1e-4,
                 seed: int = 0,
                 verbose: bool = False) -> RSQResult:
    """Run RSQ on a MaxCut problem (maximizes the weighted cut).

    Parameters of note
    -------------------
    indicator : "fro" or "spec"   -- residual norm used for rank + refresh.
    recycle   : reuse the current basis when refreshing (cheaper rebuilds).
    step_cap  : if set, cap ||Delta z|| per step (trust region tied to the bound).
    residual_probes : Gaussian probes used by each Frobenius refresh diagnostic.
    adjoint_free : use only forward observable evaluations and finite-difference
        JVPs, including a two-evaluation SPSA update in reduced coordinates.
    fd_eps : central-difference step for observable JVP actions.
    spsa_eps : perturbation radius for the forward-only reduced objective update.

    ``cut`` and ``theta`` describe the final iterate. ``best_step`` is the
    zero-based number of completed optimizer updates at the observation (and
    equals ``steps`` for the final iterate). In forward-only mode the best
    observed point may be one of the two SPSA probe points, not a center iterate.
    Residual, refresh, and build-status histories make every adaptive decision
    auditable; a two-sided build can stop at a rank cap without meeting ``tol``.
    """
    if not 0.0 < tol < 1.0:
        raise ValueError("tol must lie in (0, 1)")
    if maxrank is not None and (int(maxrank) != maxrank or maxrank < 1):
        raise ValueError("maxrank must be a positive integer when supplied")
    if int(steps) != steps or steps < 1:
        raise ValueError("steps must be a positive integer")
    if inner_lr <= 0:
        raise ValueError("inner_lr must be positive")
    if int(refresh_every) != refresh_every or refresh_every < 0:
        raise ValueError("refresh_every must be a nonnegative integer")
    if eps_refresh < 0:
        raise ValueError("eps_refresh must be nonnegative")
    if int(block) != block or block < 1:
        raise ValueError("block must be a positive integer")
    if indicator not in {"fro", "spec"}:
        raise ValueError("indicator must be 'fro' or 'spec'")
    if adjoint_free and indicator != "fro":
        raise ValueError("adjoint_free uses the forward-only Frobenius diagnostic")
    if adjoint_free and maxrank is not None:
        raise ValueError("use af_rank, not maxrank, with adjoint_free=True")
    if step_cap is not None and step_cap <= 0:
        raise ValueError("step_cap must be positive when supplied")
    if int(af_rank) != af_rank or af_rank < 1:
        raise ValueError("af_rank must be a positive integer")
    if int(residual_probes) != residual_probes or residual_probes < 1:
        raise ValueError("residual_probes must be a positive integer")
    if fd_eps <= 0:
        raise ValueError("fd_eps must be positive")
    if spsa_eps <= 0:
        raise ValueError("spsa_eps must be positive")

    gen = torch.Generator().manual_seed(seed)
    if theta0 is None:
        theta0 = problem.random_theta(generator=gen)
    theta0 = theta0.detach().clone().to(
        device=problem.C.device, dtype=RDTYPE
    )
    if theta0.ndim != 1 or theta0.numel() != problem.dim:
        raise ValueError(f"theta0 must have shape ({problem.dim},)")
    if not torch.isfinite(theta0).all():
        raise ValueError("theta0 must contain only finite values")

    total_counts = {"forward_F": 0, "jvp": 0, "vjp": 0}

    def build_subspace(anchor, Q_prev=None):
        op = QAOASensitivity(problem, anchor, fd_eps=fd_eps)
        if adjoint_free:
            Q, captured = active_subspace_adjoint_free(
                op, rank=af_rank,
                Q_init=(Q_prev if recycle else None), generator=gen)
            status = {
                "mode": "forward_fixed_rank",
                "rank": int(Q.shape[1]),
                "captured_fraction_in_trial_space": float(captured),
                "converged": None,
                "stop_reason": "fixed_rank",
            }
        else:
            qb = active_subspace(op, tol=tol, block=block, maxrank=maxrank,
                                 indicator=indicator,
                                 Q_init=(Q_prev if recycle else None), generator=gen)
            Q = (qb.Q if qb.rank > 0 else torch.eye(
                problem.dim, dtype=RDTYPE, device=problem.C.device)[:, :1])
            status = {
                "mode": "two_sided_adaptive",
                "rank": int(Q.shape[1]),
                "estimated_relative_residual": float(qb.rel_residual),
                "converged": bool(qb.converged),
                "stop_reason": qb.stop_reason,
            }
        for k in total_counts:
            total_counts[k] += op.counts.as_dict()[k]
        return Q, status

    Q, initial_status = build_subspace(theta0)
    z = torch.zeros(Q.shape[1], dtype=RDTYPE, requires_grad=True)
    opt = torch.optim.Adam([z], lr=inner_lr)

    res = RSQResult(
        theta=theta0.clone(), cut=float("nan"),
        indicator=("fro-forward" if adjoint_free else indicator),
        best_theta=None, best_cut=float("-inf"), best_step=-1,
        subspace_builds=1, build_history=[initial_status],
    )
    refreshes = 0

    for step in range(steps):
        z_prev = z.detach().clone()
        opt.zero_grad()
        if adjoint_free:
            # Simultaneous perturbation keeps each reduced-coordinate update at
            # two objective evaluations, independently of the retained rank.
            # For Rademacher perturbations, elementwise reciprocal equals the
            # perturbation itself.  No autograd graph or transpose action is
            # created on this branch.
            delta = (2 * torch.randint(
                0, 2, z.shape, generator=gen, dtype=torch.int64
            ) - 1).to(RDTYPE)
            with torch.no_grad():
                plus = theta0 + Q @ (z + spsa_eps * delta)
                minus = theta0 + Q @ (z - spsa_eps * delta)
                loss_plus = -problem.cut(plus)
                loss_minus = -problem.cut(minus)
                grad = ((loss_plus - loss_minus) / (2.0 * spsa_eps)) * delta
                neg = 0.5 * (loss_plus + loss_minus)
                if float(loss_plus) <= float(loss_minus):
                    observed_theta = plus.detach().clone()
                    observed_cut = float(-loss_plus)
                else:
                    observed_theta = minus.detach().clone()
                    observed_cut = float(-loss_minus)
            z.grad = grad.detach().clone()
            total_counts["forward_F"] += 2
        else:
            theta = theta0 + Q @ z
            neg = -problem.cut(theta)
            neg.backward()
            observed_theta = theta.detach().clone()
            observed_cut = float(-neg.detach())
            total_counts["forward_F"] += 1
            total_counts["vjp"] += 1
        opt.step()

        # trust-region: cap the parameter-space step (||Q dz|| = ||dz|| since Q orthonormal)
        if step_cap is not None:
            with torch.no_grad():
                dz = z - z_prev
                nd = dz.norm()
                if float(nd) > step_cap:
                    z.copy_(z_prev + dz * (step_cap / nd.clamp_min(1e-30)))

        res.history.append(float(neg.detach()))
        res.cut_history.append(observed_cut)
        res.rank_history.append(Q.shape[1])
        if observed_cut > res.best_cut:
            res.best_cut = observed_cut
            res.best_theta = observed_theta
            res.best_step = step

        # residual-gated, recycled subspace refresh
        completed_step = step + 1
        if (refresh_every and completed_step < steps
                and completed_step % refresh_every == 0):
            anchor = (theta0 + Q @ z).detach().clone()
            op_c = QAOASensitivity(problem, anchor, fd_eps=fd_eps)
            if adjoint_free:
                rel = randomized_residual_forward_only(
                    op_c, Q, n_probe=residual_probes, generator=gen)
            else:
                rel = randomized_residual(
                    op_c, Q, n_probe=residual_probes,
                    indicator=indicator, generator=gen)
            for k in total_counts:
                total_counts[k] += op_c.counts.as_dict()[k]
            res.residual_steps.append(completed_step)
            res.residual_history.append(float(rel))
            if rel > eps_refresh:
                theta0 = anchor
                Q, build_status = build_subspace(theta0, Q_prev=Q)
                z = torch.zeros(Q.shape[1], dtype=RDTYPE, requires_grad=True)
                opt = torch.optim.Adam([z], lr=inner_lr)
                refreshes += 1
                res.refresh_steps.append(completed_step)
                res.subspace_builds += 1
                res.build_history.append(build_status)

    with torch.no_grad():
        theta_final = (theta0 + Q @ z).detach()
        res.theta = theta_final
        res.cut = float(problem.cut(theta_final))
    total_counts["forward_F"] += 1
    if res.cut > res.best_cut:
        res.best_cut = res.cut
        res.best_theta = theta_final.clone()
        res.best_step = steps
    res.refreshes = refreshes
    res.counts = total_counts
    res.final_rank = Q.shape[1]
    if verbose:
        print(f"RSQ[{indicator}]: cut={res.cut:.4f} rank={res.final_rank} "
              f"refreshes={refreshes} counts={total_counts}")
    return res
