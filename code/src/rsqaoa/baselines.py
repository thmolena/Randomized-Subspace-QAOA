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
    # Every optimization step evaluates F once and applies one reverse-mode
    # objective gradient; the final reported cut adds one forward evaluation.
    return OptResult(theta=theta.detach(), cut=cut, history=hist,
                     n_params_opt=problem.dim,
                     counts={"forward_F": steps + 1, "vjp": steps, "jvp": 0})


def fixed_rank_subspace(problem: MaxCutProblem, rank: int,
                        theta0: Optional[torch.Tensor] = None,
                        steps: int = 200, lr: float = 0.05, seed: int = 0) -> OptResult:
    """Truncated-SVD subspace at a *fixed* rank (small instances / oracle)."""
    gen = torch.Generator().manual_seed(seed)
    if theta0 is None:
        theta0 = problem.random_theta(generator=gen)
    theta0 = theta0.detach().clone().to(RDTYPE)
    if int(rank) != rank or rank < 1:
        raise ValueError("rank must be a positive integer")
    op = QAOASensitivity(problem, theta0)
    J = op.dense_jacobian()                       # (m, d) -- explicit, small only
    _, _, Vh = torch.linalg.svd(J, full_matrices=False)
    actual_rank = min(int(rank), Vh.shape[0])
    Q = Vh[:actual_rank].t().contiguous()         # (d, rank) top right singular vecs
    z = torch.zeros(actual_rank, dtype=RDTYPE, requires_grad=True)
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
                     n_params_opt=actual_rank,
                     counts={"forward_F": steps + 1, "vjp": steps, "jvp": 0,
                             "dense_jacobian": 1})


# --- symmetry reduction -----------------------------------------------------
def _automorphism_orbits(n: int, edges, weights=None) -> Tuple[List[int], List[int]]:
    """Return node-orbit and edge-orbit label arrays under the graph automorphism
    group (enumerated with VF2). Falls back to trivial orbits if enumeration is
    too large."""
    G = nx.Graph()
    G.add_nodes_from(range(n))
    normalized_edges = [tuple(sorted((int(i), int(j)))) for i, j in edges]
    if weights is None:
        numeric_weights = [1.0] * len(normalized_edges)
    else:
        numeric_weights = [float(value) for value in weights]
        if len(numeric_weights) != len(normalized_edges):
            raise ValueError("weights and edges must have the same length")
    G.add_weighted_edges_from([
        (i, j, weight)
        for (i, j), weight in zip(normalized_edges, numeric_weights)
    ])
    edge_index = {edge: idx for idx, edge in enumerate(normalized_edges)}
    node_parent = list(range(n))
    edge_parent = list(range(len(normalized_edges)))

    def find(parent, item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(parent, a, b):
        ra, rb = find(parent, a), find(parent, b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    try:
        edge_match = nx.algorithms.isomorphism.numerical_edge_match(
            "weight", 1.0
        )
        gm = nx.algorithms.isomorphism.GraphMatcher(
            G, G, edge_match=edge_match
        )
        for i, mapping in enumerate(gm.isomorphisms_iter()):
            for a, b in mapping.items():
                union(node_parent, a, b)
            for edge_id, (a, b) in enumerate(normalized_edges):
                mapped = tuple(sorted((mapping[a], mapping[b])))
                union(edge_parent, edge_id, edge_index[mapped])
            if i >= 5000:                        # safe under-refinement if capped
                break
    except Exception:
        # The initialized singleton partitions are a valid conservative
        # fallback: they never tie parameters that are not symmetry-equivalent.
        node_parent = list(range(n))
        edge_parent = list(range(len(normalized_edges)))

    node_roots = {}
    node_orbit = [
        node_roots.setdefault(find(node_parent, a), len(node_roots))
        for a in range(n)
    ]
    edge_roots = {}
    edge_orbit = [
        edge_roots.setdefault(find(edge_parent, e), len(edge_roots))
        for e in range(len(normalized_edges))
    ]
    return node_orbit, edge_orbit


def symmetry_reduced(problem: MaxCutProblem, theta0: Optional[torch.Tensor] = None,
                     steps: int = 200, lr: float = 0.05, seed: int = 0) -> OptResult:
    """ma-QAOA with angles tied within automorphism orbits (per layer)."""
    gen = torch.Generator().manual_seed(seed)
    n, edges, p = problem.n, problem.edges, problem.p
    weights = None if problem.weights is None else problem.weights.detach().cpu().tolist()
    node_orbit, edge_orbit = _automorphism_orbits(n, edges, weights=weights)
    n_no = max(node_orbit) + 1
    n_eo = max(edge_orbit) + 1
    node_orbit_t = torch.tensor(
        node_orbit, dtype=torch.long, device=problem.C.device
    )
    edge_orbit_t = torch.tensor(
        edge_orbit, dtype=torch.long, device=problem.C.device
    )

    # reduced params: per layer, one gamma per edge-orbit and one beta per node-orbit
    n_reduced = p * (n_eo + n_no)
    if theta0 is None:
        red = torch.rand(n_reduced, generator=gen, dtype=RDTYPE) * np.pi
        red = red.to(problem.C.device)
    else:
        if theta0.ndim != 1 or theta0.numel() != problem.dim:
            raise ValueError(f"theta0 must have shape ({problem.dim},)")
        theta0 = theta0.detach().to(device=problem.C.device, dtype=RDTYPE)
        gammas = theta0[: p * len(edges)].reshape(p, len(edges))
        betas = theta0[p * len(edges):].reshape(p, n)
        reduced_gammas = torch.stack([
            torch.stack([
                gammas[layer, edge_orbit_t == orbit].mean()
                for orbit in range(n_eo)
            ])
            for layer in range(p)
        ])
        reduced_betas = torch.stack([
            torch.stack([
                betas[layer, node_orbit_t == orbit].mean()
                for orbit in range(n_no)
            ])
            for layer in range(p)
        ])
        red = torch.cat([reduced_gammas.reshape(-1), reduced_betas.reshape(-1)])
    red = red.detach().clone().requires_grad_(True)

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
                     counts={"forward_F": steps + 1, "vjp": steps, "jvp": 0})
