"""Randomized Subspace QAOA (RSQ): optimize ma-QAOA inside an adaptively
discovered low-dimensional active subspace, refreshing it only when a randomized
certificate says it has drifted, and reusing the previous basis when it does.

Loop
----
1. At an expansion point ``theta0`` build the matrix-free ``J(theta0)`` and get an
   active-subspace basis ``Q`` (``d x r``) at tolerance ``tol`` via ``randqb``.
2. Optimize the reduced coordinates ``z in R^r`` (``theta = theta0 + Q z``) with
   Adam, optionally under a **trust-region step cap** so the first-order
   truncation bound stays valid step to step.
3. Every ``refresh_every`` steps, evaluate the randomized residual certificate of
   the *current* ``Q`` against the *current* ``J``. If it exceeds ``eps_refresh``,
   re-anchor ``theta0``, **recycle** the current ``Q`` to rebuild cheaply, and
   reset ``z``.

The certificate-gated, recycled refresh is what makes the number and cost of
subspace rebuilds adapt to optimization progress rather than a fixed schedule.
Everything is reported: objective trajectory, retained rank over time, refresh
count, and the operator-application budget (forward F / jvp / vjp).
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
    certified_residual,
    certified_residual_forward_only,
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
                 fd_eps: float = 1e-4,
                 seed: int = 0,
                 verbose: bool = False) -> RSQResult:
    """Run RSQ on a MaxCut problem (maximizes the weighted cut).

    Parameters of note
    -------------------
    indicator : "fro" or "spec"   -- residual norm used for rank + refresh.
    recycle   : reuse the current basis when refreshing (cheaper rebuilds).
    step_cap  : if set, cap ||Delta z|| per step (trust region tied to the bound).
    """
    gen = torch.Generator().manual_seed(seed)
    if theta0 is None:
        theta0 = problem.random_theta(generator=gen)
    theta0 = theta0.detach().clone().to(RDTYPE)

    total_counts = {"forward_F": 0, "jvp": 0, "vjp": 0}

    def build_subspace(anchor, Q_prev=None):
        op = QAOASensitivity(problem, anchor, fd_eps=fd_eps)
        if adjoint_free:
            Q, _ = active_subspace_adjoint_free(op, rank=af_rank, generator=gen)
        else:
            res = active_subspace(op, tol=tol, block=block, maxrank=maxrank,
                                  indicator=indicator,
                                  Q_init=(Q_prev if recycle else None), generator=gen)
            Q = res.Q if res.rank > 0 else torch.eye(problem.dim, dtype=RDTYPE)[:, :1]
        for k in total_counts:
            total_counts[k] += op.counts.as_dict()[k]
        return Q

    Q = build_subspace(theta0)
    z = torch.zeros(Q.shape[1], dtype=RDTYPE, requires_grad=True)
    opt = torch.optim.Adam([z], lr=inner_lr)

    res = RSQResult(theta=theta0.clone(), cut=float(problem.cut(theta0)),
                    indicator=indicator)
    refreshes = 0

    for step in range(steps):
        z_prev = z.detach().clone()
        opt.zero_grad()
        theta = theta0 + Q @ z
        neg = -problem.cut(theta)
        neg.backward()
        opt.step()

        # trust-region: cap the parameter-space step (||Q dz|| = ||dz|| since Q orthonormal)
        if step_cap is not None:
            with torch.no_grad():
                dz = z - z_prev
                nd = dz.norm()
                if float(nd) > step_cap:
                    z.copy_(z_prev + dz * (step_cap / nd.clamp_min(1e-30)))

        with torch.no_grad():
            theta_now = theta0 + Q @ z
            cut_now = float(problem.cut(theta_now))
        res.history.append(float(neg.detach()))
        res.cut_history.append(cut_now)
        res.rank_history.append(Q.shape[1])

        # certificate-gated, recycled subspace refresh
        if refresh_every and step > 0 and step % refresh_every == 0:
            anchor = (theta0 + Q @ z).detach().clone()
            op_c = QAOASensitivity(problem, anchor, fd_eps=fd_eps)
            if adjoint_free:
                rel = certified_residual_forward_only(op_c, Q, generator=gen)
            else:
                rel = certified_residual(op_c, Q, indicator=indicator, generator=gen)
            for k in total_counts:
                total_counts[k] += op_c.counts.as_dict()[k]
            if rel > eps_refresh:
                theta0 = anchor
                Q = build_subspace(theta0, Q_prev=Q)
                z = torch.zeros(Q.shape[1], dtype=RDTYPE, requires_grad=True)
                opt = torch.optim.Adam([z], lr=inner_lr)
                refreshes += 1

    with torch.no_grad():
        theta_final = (theta0 + Q @ z).detach()
        res.theta = theta_final
        res.cut = float(problem.cut(theta_final))
    res.refreshes = refreshes
    res.counts = total_counts
    res.final_rank = Q.shape[1]
    if verbose:
        print(f"RSQ[{indicator}]: cut={res.cut:.4f} rank={res.final_rank} "
              f"refreshes={refreshes} counts={total_counts}")
    return res
