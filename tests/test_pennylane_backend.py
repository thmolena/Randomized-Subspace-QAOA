"""Optional cross-check: pure-torch core vs PennyLane lightning.qubit.
Skipped automatically if PennyLane is not installed."""

import numpy as np
import pytest
import torch

pennylane = pytest.importorskip("pennylane")

from rsqaoa.circuits import edge_expectations
from rsqaoa.backends_pennylane import edge_expectations_pennylane


def test_core_matches_pennylane():
    rng = np.random.default_rng(0)
    n, p = 4, 2
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    theta_np = rng.uniform(0, np.pi, p * (len(edges) + n))
    theta = torch.tensor(theta_np, dtype=torch.float64)
    core = edge_expectations(theta, n, edges, p).numpy()
    pl = edge_expectations_pennylane(theta_np, n, edges, p)
    assert np.max(np.abs(core - pl)) < 1e-8
