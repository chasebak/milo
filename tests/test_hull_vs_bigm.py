"""hull-vs-bigm experiment table on a small random case."""
import numpy as np

from experiments.hull_vs_bigm import hull_vs_bigm
from tests.test_optimize import _random_case


def test_table_objectives_agree_and_hull_bound_is_tighter():
    products, prices, demand = _random_case(4, 10, seed=1)
    problem = {"products": products, "prices": prices, "demand": demand}
    table = hull_vs_bigm(problem, budgets=(100.0, 300.0), time_limit=30)
    expected_columns = ["budget", "transform", "objective", "lp_relaxation_objective",
                        "build_seconds", "solve_seconds", "optimal", "both_optimal",
                        "objectives_agree"]
    assert list(table.columns) == expected_columns
    assert len(table) == 4
    assert table["both_optimal"].all()
    assert table["objectives_agree"].all()
    budgets = sorted(table["budget"].unique())
    for budget in budgets:
        part = table[table["budget"] == budget].set_index("transform")
        hull_lp = float(part.at["hull", "lp_relaxation_objective"])
        bigm_lp = float(part.at["bigm", "lp_relaxation_objective"])
        hull_obj = float(part.at["hull", "objective"])
        scale = max(1.0, abs(hull_obj))
        # maximization: every LP relaxation is an upper bound, and hull is tighter
        assert hull_lp >= hull_obj - 1e-6 * scale
        assert bigm_lp >= float(part.at["bigm", "objective"]) - 1e-6 * scale
        assert hull_lp <= bigm_lp + 1e-6 * scale
    assert np.all(table["solve_seconds"].to_numpy() >= 0.0)
