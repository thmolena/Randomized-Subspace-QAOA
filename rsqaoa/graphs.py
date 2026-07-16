"""Graph instance generators and exact small-instance references.

Graph families used in the study: random 3-regular, Erdos--Renyi, ring, and
weighted (Ising / Sherrington--Kirkpatrick-style) variants. Brute-force MaxCut
is provided for small ``n`` so approximation ratios are well defined.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx

Edge = Tuple[int, int]


def _edges(G: nx.Graph) -> List[Edge]:
    return [(int(u), int(v)) for u, v in G.edges()]


def random_regular(n: int, degree: int = 3, seed: int = 0) -> List[Edge]:
    return _edges(nx.random_regular_graph(degree, n, seed=seed))


def erdos_renyi(n: int, prob: float = 0.5, seed: int = 0) -> List[Edge]:
    G = nx.gnp_random_graph(n, prob, seed=seed)
    # ensure connectivity is not required, but avoid isolated-only graphs
    return _edges(G)


def ring(n: int) -> List[Edge]:
    return [(i, (i + 1) % n) for i in range(n)]


def complete(n: int) -> List[Edge]:
    return _edges(nx.complete_graph(n))


def random_weights(edges: Sequence[Edge], seed: int = 0,
                   low: float = -1.0, high: float = 1.0) -> np.ndarray:
    """Weighted (Ising/spin-glass) edge weights in ``[low, high]``."""
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=len(edges))


def brute_force_maxcut(n: int, edges: Sequence[Edge],
                       weights: Optional[np.ndarray] = None) -> float:
    """Exact maximum (weighted) cut by enumeration. Use only for small ``n``."""
    if weights is None:
        weights = np.ones(len(edges))
    x = np.arange(2 ** n, dtype=np.int64)
    bits = ((x[None, :] >> np.arange(n)[:, None]) & 1).astype(np.int8)  # (n, 2**n)
    cut = np.zeros(2 ** n)
    for w, (i, j) in zip(weights, edges):
        cut += w * (bits[i] ^ bits[j])
    return float(cut.max())


def approximation_ratio(cut_value: float, n: int, edges: Sequence[Edge],
                        weights: Optional[np.ndarray] = None) -> float:
    best = brute_force_maxcut(n, edges, weights)
    return cut_value / best if best != 0 else float("nan")
