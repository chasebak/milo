import numpy as np
import pandas as pd

from milo.simulate import evaluate_plan, ordered_demand, strategy_table


def _products():
    df = pd.DataFrame({
        "cost": [10.0, 50.0], "on_hand": [1, 0], "min_order": [1, 1],
        "max_units": [10, 5], "set_code": ["A", "B"], "salvage_frac": [0.5, 0.5],
    }, index=["a", "b"])
    return df


def _prices():
    return pd.DataFrame({
        "i": ["a", "a", "b", "b"], "k": [0, 1, 0, 1],
        "ratio": [1.0, 1.5, 1.0, 1.2], "price": [20.0, 30.0, 100.0, 120.0],
        "net_cash": [18.0, 27.0, 90.0, 108.0],
    })


def test_evaluate_plan_hand_calc():
    # demand (2 products, 2 levels, 2 scenarios)
    demand = np.array([[[5, 2], [3, 1]], [[2, 0], [1, 0]]])
    profit, summary = evaluate_plan(_products(), _prices(), demand,
                                    {"a": 3, "b": 1}, {"a": 1, "b": 0})
    # scen 0: a stock 4 sold 3 left 1 -> 81 + 5 ; b sold 1 -> 90 ; cost 30 + 50
    s0 = 27 * 3 + 5 * 1 + 90 * 1 - 30 - 50
    # scen 1: a sold 1 left 3 -> 27 + 15 ; b sold 0 left 1 -> 25 ; cost 80
    s1 = 27 * 1 + 5 * 3 + 0 + 25 - 80
    assert np.allclose(profit, [s0, s1])
    assert abs(summary["expected"] - (s0 + s1) / 2) < 1e-9
    assert abs(summary["sell_through"] - ((4 + 1) / 2) / 5) < 1e-9
    assert abs(summary["leftover_value"] - (5 + 15 + 25) / 2) < 1e-9
    assert summary["p10"] == min(s0, s1)


def test_strategy_table_rows():
    demand = np.array([[[5, 2], [3, 1]], [[2, 0], [1, 0]]])
    table = strategy_table(_products(), _prices(), demand, {
        "none": ({}, {"a": 0}),
        "plan": ({"a": 3, "b": 1}, {"a": 1, "b": 0}),
    })
    assert list(table["strategy"]) == ["none", "plan"]
    assert "p90" in table.columns


def test_ordered_demand_monotonic():
    rng = np.random.default_rng(7)
    lam = rng.uniform(0.1, 20, size=30)
    ratios = np.linspace(0.8, 1.15, 8)
    elasticity = rng.uniform(-4, -0.2, size=30)
    mult = ratios[None, :] ** elasticity[:, None]
    demand = ordered_demand(lam, mult, 200, rng)
    assert demand.shape == (30, 8, 200)
    assert demand.dtype.kind == "i"
    assert np.all(np.diff(demand, axis=1) <= 0)
