"""Smoke + sanity tests for the RSQ optimizer and baselines."""

import numpy as np
import pytest
import torch

from rsqaoa.circuits import MaxCutProblem
from rsqaoa import graphs
from rsqaoa.subspace_opt import optimize_rsq
from rsqaoa.baselines import full_maqaoa, fixed_rank_subspace, symmetry_reduced


def _ring(n=6, p=1):
    return MaxCutProblem(n=n, edges=graphs.ring(n), p=p)


def test_rsq_improves_objective():
    prob = _ring(6, 1)
    theta0 = prob.random_theta(generator=torch.Generator().manual_seed(0))
    init_cut = float(prob.cut(theta0))
    res = optimize_rsq(prob, theta0=theta0.clone(), tol=1e-2, steps=150,
                       inner_lr=0.08, refresh_every=30, seed=0)
    assert max(res.cut_history) >= init_cut + 1e-6      # subspace search made progress
    assert 0 <= res.cut <= prob.m + 1e-9
    assert res.final_rank <= prob.dim
    assert set(res.counts) == {"forward_F", "jvp", "vjp"}


def test_rsq_adjoint_free_runs():
    prob = _ring(6, 1)
    res = optimize_rsq(prob, tol=1e-2, steps=60, adjoint_free=True, af_rank=4, seed=1)
    assert res.counts["vjp"] == 0   # build, optimization, and refresh are forward-only
    assert res.counts["jvp"] > 0
    assert res.counts["forward_F"] >= 2 * 60 + 1
    assert 0 <= res.cut <= prob.m + 1e-9


def test_baselines_run():
    prob = _ring(6, 1)
    theta0 = prob.random_theta(generator=torch.Generator().manual_seed(2))
    full = full_maqaoa(prob, theta0=theta0.clone(), steps=80)
    fr = fixed_rank_subspace(prob, rank=3, theta0=theta0.clone(), steps=80)
    sym = symmetry_reduced(prob, steps=80)
    for r in (full, fr, sym):
        assert 0 <= r.cut <= prob.m + 1e-9
    # ring has a large automorphism group -> symmetry should compress parameters
    assert sym.n_params_opt <= prob.dim


def test_edge_orbits_follow_automorphism_action():
    from rsqaoa.baselines import _automorphism_orbits

    # The triangular prism is node-transitive but not edge-transitive: triangle
    # and matching edges form different orbits.  Endpoint node-orbit labels
    # alone would incorrectly merge all nine edges.
    edges = [
        (0, 1), (1, 2), (2, 0),
        (3, 4), (4, 5), (5, 3),
        (0, 3), (1, 4), (2, 5),
    ]
    node_orbit, edge_orbit = _automorphism_orbits(6, edges)
    assert len(set(node_orbit)) == 1
    assert len(set(edge_orbit)) == 2


def test_weighted_edge_orbits_do_not_tie_unequal_cost_terms():
    from rsqaoa.baselines import _automorphism_orbits

    edges = graphs.ring(4)
    weights = [1.0, 1.0, 2.0, 2.0]
    _, edge_orbit = _automorphism_orbits(4, edges, weights=weights)
    assert len(set(edge_orbit)) == 2
    for left in range(len(edges)):
        for right in range(len(edges)):
            if edge_orbit[left] == edge_orbit[right]:
                assert weights[left] == weights[right]


def test_configurable_cli_emits_strict_json(capsys):
    import json
    from rsqaoa._cli import main

    main([
        "--family", "ring", "--n", "4", "--p", "1", "--steps", "1",
        "--residual-probes", "2", "--json",
    ])
    output = capsys.readouterr().out
    assert "Infinity" not in output and "NaN" not in output
    payload = json.loads(output)
    assert payload["problem"]["ambient_dimension"] == 8
    assert len(payload["results"]) == 3
    assert "residual_checks" in payload["rsq"]


def test_cli_reports_version(capsys):
    from rsqaoa._cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == "rsqaoa 0.3.0"


def test_rsq_reports_best_iterate_and_residual_audit_trail():
    prob = _ring(6, 1)
    res = optimize_rsq(
        prob, steps=12, refresh_every=5, residual_probes=3,
        eps_refresh=0.0, seed=4,
    )
    assert res.best_theta is not None
    assert res.best_cut >= res.cut - 1e-12
    assert res.best_cut >= max(res.cut_history) - 1e-12
    assert res.residual_steps == [5, 10]
    assert len(res.residual_history) == 2
    assert len(res.refresh_steps) == res.refreshes
    assert res.subspace_builds == 1 + res.refreshes
    assert len(res.build_history) == res.subspace_builds
    assert all("stop_reason" in item for item in res.build_history)


def test_forward_only_path_never_calls_backward(monkeypatch):
    def fail_backward(*args, **kwargs):
        raise AssertionError("reverse-mode backward was called")

    monkeypatch.setattr(torch.Tensor, "backward", fail_backward)
    result = optimize_rsq(
        _ring(4, 1), steps=2, refresh_every=0,
        adjoint_free=True, af_rank=2, seed=8,
    )
    assert result.counts["vjp"] == 0


def test_terminal_step_does_not_trigger_noop_rebuild():
    prob = _ring(6, 1)
    res = optimize_rsq(
        prob, steps=10, refresh_every=5, residual_probes=2,
        eps_refresh=0.0, seed=8,
    )
    assert res.residual_steps == [5]
    assert res.refresh_steps == [5]
    assert res.refreshes == 1
    assert res.subspace_builds == 2
    assert len(res.build_history) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tol": 0.0},
        {"indicator": "invalid"},
        {"residual_probes": 0},
        {"refresh_every": -1},
        {"step_cap": 0.0},
    ],
)
def test_rsq_rejects_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        optimize_rsq(_ring(4, 1), steps=1, **kwargs)
