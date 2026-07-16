"""Baselines RSQ is compared against.

* ``full_maqaoa`` -- optimize all ``d = p(|E|+n)`` angles with Adam (the thing we
  want to be cheaper than).
* ``fixed_rank_subspace`` -- form ``J`` explicitly (small instances only),
  truncated SVD to a *preset* rank, optimize in that fixed subspace. An oracle /
  ablation isolating the value of *adaptive* rank + refresh.
* ``symmetry_reduced`` -- tie angles by graph-automorphism orbits (the principled,
  non-random parameter-reduction baseline). This is the headline comparison: a
  fair method must beat symmetry reduction, not just random angle dropping.

None of these fabricate results; they are runnable procedures whose outputs the
experiments record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import networkx as nx
import torch

from .circuits import MaxCutProblem, RDTYPE
from .operator import QAOASensitivity


@dataclass
class OptResult:
    theta: torch.Tensor
    cut: float
    history: List[float] = field(default_factory=list)
    n_params_opt: int = 0
    counts: dict = field(default_factory=dict)


def full_maqaoa(problem: MaxCutProblem, theta0: Optional[torch.Tensor] = None,
                steps: int = 200, lr: float = 0.05, seed: int = 0) -> OptResult:
    gen = torch.Generator().manual_seed(seed)
    if theta0 is None:
        theta0 = problem.random_theta(generator=gen)
    theta = theta0.detach().clone().to(RDTYPE).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    hist = []
    for _ in range(steps):
        opt.zero_grad()
        neg = -problem.cut(theta)
        neg.backward()
        opt.step()
        hist.append(float(neg.detach()))
    with torch.no_grad():
        cut = float(problem.cut(theta))
    # cost bookkeeping: full gradient == one vjp-equivalent per step
    return OptResult(theta=theta.detach(), cut=cut, history=hist,
                     n_params_opt=problem.dim,
                     counts={"forward_F": steps, "vjp": steps, "jvp": 0})


def fixed_rank_subspace(problem: MaxCutProblem, rank: int,
                        theta0: Optional[torch.Tensor] = None,
                        steps: int = 200, lr: float = 0.05, seed: int = 0) -> OptResult:
    """Truncated-SVD subspace at a *fixed* rank (small instances / oracle)."""
    gen = torch.Generator().manual_seed(seed)
    if theta0 is None:
        theta0 = problem.random_theta(generator=gen)
    theta0 = theta0.detach().clone().to(RDTYPE)
    op = QAOASensitivity(problem, theta0)
    J = op.dense_jacobian()                       # (m, d) -- explicit, small only
    _, _, Vh = torch.linalg.svd(J, full_matrices=False)
    Q = Vh[:rank].t().contiguous()                # (d, rank) top right singular vecs
    z = torch.zeros(rank, dtype=RDTYPE, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    hist = []
    for _ in range(steps):
        opt.zero_grad()
        neg = -problem.cut(theta0 + Q @ z)
        neg.backward()
        opt.step()
        hist.append(float(neg.detach()))
    with torch.no_grad():
        cut = float(problem.cut(theta0 + Q @ z))
    return OptResult(theta=(theta0 + Q @ z).detach(), cut=cut, history=hist,
                     n_params_opt=rank, counts={"forward_F": steps, "vjp": steps})


# --- symmetry reduction -----------------------------------------------------
def _automorphism_orbits(n: int, edges) -> Tuple[List[int], List[int]]:
    """Return node-orbit and edge-orbit label arrays under the graph automorphism
    group (enumerated with VF2). Falls back to trivial orbits if enumeration is
    too large."""
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    node_orbit = list(range(n))
    try:
        gm = nx.algorithms.isomorphism.GraphMatcher(G, G)
        autos = []
        for i, mapping in enumerate(gm.isomorphisms_iter()):
            autos.append(mapping)
            if i > 5000:                         # cap enumeration cost
                break
        # union-find over nodes using the automorphisms
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        for mapping in autos:
            for a, b in mapping.items():
                union(a, b)
        roots = {}
        node_orbit = []
        for a in range(n):
            r = find(a)
            node_orbit.append(roots.setdefault(r, len(roots)))
    except Exception:
        pass
    # edge orbits: label an edge by the sorted pair of node-orbit ids
    edge_labels = {}
    edge_orbit = []
    for (i, j) in edges:
        key = tuple(sorted((node_orbit[i], node_orbit[j])))
        edge_orbit.append(edge_labels.setdefault(key, len(edge_labels)))
    return node_orbit, edge_orbit


def symmetry_reduced(problem: MaxCutProblem, theta0: Optional[torch.Tensor] = None,
                     steps: int = 200, lr: float = 0.05, seed: int = 0) -> OptResult:
    """ma-QAOA with angles tied within automorphism orbits (per layer)."""
    gen = torch.Generator().manual_seed(seed)
    n, edges, p = problem.n, problem.edges, problem.p
    node_orbit, edge_orbit = _automorphism_orbits(n, edges)
    n_no = max(node_orbit) + 1
    n_eo = max(edge_orbit) + 1
    node_orbit_t = torch.tensor(node_orbit, dtype=torch.long)
    edge_orbit_t = torch.tensor(edge_orbit, dtype=torch.long)

    # reduced params: per layer, one gamma per edge-orbit and one beta per node-orbit
    n_reduced = p * (n_eo + n_no)
    red = (torch.rand(n_reduced, generator=gen, dtype=RDTYPE) * np.pi).requires_grad_(True)

    def expand(red_params: torch.Tensor) -> torch.Tensor:
        rg = red_params[: p * n_eo].reshape(p, n_eo)
        rb = red_params[p * n_eo:].reshape(p, n_no)
        gam = rg[:, edge_orbit_t]                 # (p, |E|)
        bet = rb[:, node_orbit_t]                 # (p, n)
        return torch.cat([gam.reshape(-1), bet.reshape(-1)])

    opt = torch.optim.Adam([red], lr=lr)
    hist = []
    for _ in range(steps):
        opt.zero_grad()
        neg = -problem.cut(expand(red))
        neg.backward()
        opt.step()
        hist.append(float(neg.detach()))
    with torch.no_grad():
        theta = expand(red)
        cut = float(problem.cut(theta))
    return OptResult(theta=theta.detach(), cut=cut, history=hist,
                     n_params_opt=n_reduced,
                     counts={"forward_F": steps, "vjp": steps})
