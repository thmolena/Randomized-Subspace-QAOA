"""Optional PennyLane (``lightning.qubit``) cross-check backend.

This is *not* used by the core method; it exists so users can independently
verify the pure-torch state-vector simulator against a standard CPU backend and
so finite-shot experiments can reuse a well-tested sampler.

The circuit reproduces the exact convention in ``circuits.py``:
    phase separator   e^{-i gamma_e (1 - Z_i Z_j)/2}   ==  IsingZZ(-gamma_e)  (up to global phase)
    mixer             e^{-i beta_k X_k}                ==  RX(2 beta_k)
    observable        <C_e> = (1 - <Z_i Z_j>)/2

Import lazily; raises a clear error if PennyLane is not installed.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _require_pennylane():
    try:
        import pennylane as qml  # noqa: F401
        return qml
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "PennyLane backend requested but not installed. "
            "Install with:  pip install '.[pennylane]'"
        ) from exc


def edge_expectations_pennylane(theta, n: int, edges: Sequence, p: int,
                                shots: Optional[int] = None):
    """Return per-edge cut expectations ``<C_e>`` via ``lightning.qubit``.

    ``theta`` is the same flat layout as the torch core:
    ``[gammas (p*|E|), betas (p*n)]``.
    """
    qml = _require_pennylane()
    theta = np.asarray(theta, dtype=float)
    n_edges = len(edges)
    gammas = theta[: p * n_edges].reshape(p, n_edges)
    betas = theta[p * n_edges:].reshape(p, n)

    dev = qml.device("lightning.qubit", wires=n, shots=shots)

    @qml.qnode(dev)
    def circuit():
        for k in range(n):
            qml.Hadamard(wires=k)                       # |+>^n
        for layer in range(p):
            for e, (i, j) in enumerate(edges):
                qml.IsingZZ(-gammas[layer, e], wires=[i, j])
            for k in range(n):
                qml.RX(2.0 * betas[layer, k], wires=k)
        return [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for (i, j) in edges]

    zz = np.array(circuit(), dtype=float)
    return (1.0 - zz) / 2.0


def cut_value_pennylane(theta, n: int, edges, p: int, weights=None,
                        shots: Optional[int] = None) -> float:
    fe = edge_expectations_pennylane(theta, n, edges, p, shots=shots)
    if weights is None:
        return float(fe.sum())
    return float(np.asarray(weights) @ fe)
