"""Optional PyTorch Lightning wrapper for orchestration only.

Lightning is used here purely for experiment management -- deterministic seeding,
metric logging, and checkpointing of the reduced coordinates. It is **not** part
of the numerical method and none of the RSQ algorithm lives here. The scientific
code is in `rsqaoa/`; this file just shows how to drive it under Lightning if you
prefer that harness over `run_experiment.py`.

Requires the `experiments` extra:  pip install '.[experiments]'
"""

from __future__ import annotations

from typing import Optional

import torch

try:
    import lightning as L
    _Base = L.LightningModule
except Exception:  # pragma: no cover - lightning optional
    L = None
    _Base = object

from rsqaoa.circuits import MaxCutProblem
from rsqaoa.operator import QAOASensitivity
from rsqaoa.randqb import active_subspace, certified_residual


class RSQLightning(_Base):
    """Optimize ma-QAOA in a randomized active subspace under Lightning's manual
    optimization loop. One 'training step' == one reduced-coordinate update."""

    def __init__(self, problem: MaxCutProblem, tol: float = 1e-2,
                 inner_lr: float = 0.05, refresh_every: int = 25,
                 eps_refresh: float = 5e-2, seed: int = 0):
        if L is None:
            raise ImportError("pip install '.[experiments]' to use RSQLightning")
        super().__init__()
        self.automatic_optimization = False
        self.problem = problem
        self.tol = tol
        self.inner_lr = inner_lr
        self.refresh_every = refresh_every
        self.eps_refresh = eps_refresh
        self.gen = torch.Generator().manual_seed(seed)
        self.theta0 = problem.random_theta(generator=self.gen)
        self._build_subspace(self.theta0)
        self.refreshes = 0

    def _build_subspace(self, anchor):
        op = QAOASensitivity(self.problem, anchor)
        res = active_subspace(op, tol=self.tol, generator=self.gen)
        self.register_parameter(
            "z", torch.nn.Parameter(torch.zeros(max(res.rank, 1), dtype=torch.float64)))
        self.Q = res.Q if res.rank > 0 else torch.eye(self.problem.dim, dtype=torch.float64)[:, :1]
        self.theta0 = anchor.detach().clone()

    def configure_optimizers(self):
        return torch.optim.Adam([self.z], lr=self.inner_lr)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        theta = self.theta0 + self.Q @ self.z
        loss = -self.problem.cut(theta)
        loss.backward()
        opt.step()
        self.log("neg_cut", float(loss.detach()), prog_bar=True)
        self.log("rank", float(self.Q.shape[1]))
        if self.refresh_every and batch_idx > 0 and batch_idx % self.refresh_every == 0:
            anchor = (self.theta0 + self.Q @ self.z).detach().clone()
            op = QAOASensitivity(self.problem, anchor)
            if certified_residual(op, self.Q, generator=self.gen) > self.eps_refresh:
                self._build_subspace(anchor)
                self.trainer.optimizers = [self.configure_optimizers()]
                self.refreshes += 1
        return loss
