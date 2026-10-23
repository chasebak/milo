"""Greedy baseline tests on small hand-checkable and random cases."""
import numpy as np

from experiments import common
from experiments.greedy_baseline import gap_table, greedy_plan
from milo.optimize import solve_plan
from tests.test_optimize import _hand_case, _random_case


def test_hand_case_shows_the_joint_optimization_gap():
    # Product b's best STANDALONE level is k=0 (buy 2 for profit 80), but under the
    # shared budget the MILP prices b at k=1 and reaches 109; the greedy, having
    # fixed b at k=0 independently, can only afford 1 unit there and reaches 91.
    products, prices, demand = _hand_case()
    buy, levels, spend = greedy_plan(products, prices, demand, budget=100.0)
    assert levels == {"a": 1, "b": 0}
    assert buy == {"a": 3, "b": 1}
    assert spend == 80.0
    value, _ = common.expected_value(products, prices, demand, buy, levels)
    assert abs(value - 91.0) < 1e-9
    milp = solve_plan(products, prices, demand, budget=100.0, mip_gap=0)
    assert milp["objective"] >= value


def test_greedy_respects_budget_and_min_order():
    products, prices, demand = _random_case(6, 10, seed=7)
    budgets = [0.0, 60.0, 200.0, 800.0]
    for budget in budgets:
        buy, levels, spend = greedy_plan(products, prices, demand, budget=budget)
        assert spend <= budget + 1e-9
        bought_ids = list(buy.keys())
        for i in bought_ids:
            min_order = int(products.at[i, "min_order"])
            max_units = int(products.at[i, "max_units"])
            assert buy[i] <= max_units
            assert buy[i] == 0 or buy[i] >= min_order


def test_greedy_never_beats_the_exact_milp():
    products, prices, demand = _random_case(5, 12, seed=9)
    budgets = [100.0, 400.0]
    for budget in budgets:
        buy, levels, spend = greedy_plan(products, prices, demand, budget=budget)
        value, _ = common.expected_value(products, prices, demand, buy, levels)
        milp = solve_plan(products, prices, demand, budget=budget, mip_gap=0,
                          lp_relaxation=False)
        assert value <= milp["objective"] + 1e-6 * max(1.0, abs(milp["objective"]))


def test_gap_table_columns_and_signs():
    products, prices, demand = _random_case(4, 8, seed=2)
    problem = {"products": products, "prices": prices, "demand": demand}
    table = gap_table(problem, budgets=(50.0, 150.0), time_limit=30)
    expected_columns = ["budget", "milp_objective", "greedy_objective", "gap",
                        "gap_pct", "milp_spend", "greedy_spend"]
    assert list(table.columns) == expected_columns
    assert len(table) == 2
    slack = 2e-4 * np.maximum(1.0, np.abs(table["milp_objective"].to_numpy()))
    assert np.all(table["gap"].to_numpy() >= -slack)
