"""Smoke test for the SAA convergence experiment on a tiny injected problem.

The full run (N up to 160, 10 seeds, 1,000 evaluation scenarios) is heavy and runs
only on request, never in CI; this exercises the same code end to end in seconds.
"""
import numpy as np
import pandas as pd

from experiments.saa_convergence import plot, saa_table, summarize
from milo.simulate import ordered_demand
from tests.test_optimize import _random_case

RATIO_MULTIPLIERS = np.array([[1.0, 0.8, 0.6, 0.5],
                              [1.0, 0.9, 0.7, 0.4],
                              [1.0, 0.7, 0.5, 0.3]])
LAM = np.array([4.0, 2.0, 6.0])


def _make_demand_fn(n_scenarios, seed):
    rng = np.random.default_rng(seed)
    return ordered_demand(LAM, RATIO_MULTIPLIERS, n_scenarios, rng)


def test_saa_smoke(tmp_path):
    products, prices, _ = _random_case(3, 4, seed=6)
    table = saa_table(products, prices, _make_demand_fn, ns=(4, 8), seeds=(1, 2),
                      n_eval=40, eval_seed=99, budget=200.0, time_limit=30)
    assert len(table) == 4
    expected_columns = ["n_scenarios", "seed", "in_sample", "out_of_sample",
                        "solver_objective", "optimal", "mip_gap",
                        "build_seconds", "solve_seconds"]
    assert list(table.columns) == expected_columns
    assert np.all(np.isfinite(table["in_sample"].to_numpy()))
    assert np.all(np.isfinite(table["out_of_sample"].to_numpy()))
    # the solver objective IS the in-sample value of the chosen plan
    in_sample = table["in_sample"].to_numpy()
    solver = table["solver_objective"].to_numpy()
    assert np.allclose(in_sample, solver, rtol=1e-6, atol=1e-6)

    summary = summarize(table)
    assert list(summary["n_scenarios"]) == [4, 8]
    assert list(summary["n_seeds"]) == [2, 2]

    png_path = tmp_path / "saa.png"
    plot(table, png_path, n_eval=40)
    assert png_path.exists()
    assert png_path.stat().st_size > 0

    csv_path = tmp_path / "saa.csv"
    table.to_csv(csv_path, index=False)
    back = pd.read_csv(csv_path)
    assert len(back) == 4
