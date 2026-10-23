import numpy as np
import pandas as pd
import pytest

from milo.config import EXAMPLE_CONFIG
from milo.fees import net_cash_per_unit
from milo.optimize import budget_curve, solve_plan
from milo.simulate import ordered_demand

RATIOS = [0.80, 0.90, 1.00, 1.10]


def _hand_case():
    products = pd.DataFrame({
        "cost": [10.0, 50.0], "on_hand": [0, 0], "min_order": [1, 1],
        "max_units": [10, 5], "set_code": ["A", "B"], "salvage_frac": [0.5, 0.5],
    }, index=["a", "b"])
    prices = pd.DataFrame({
        "i": ["a", "a", "b", "b"], "k": [0, 1, 0, 1],
        "ratio": [1.0, 1.5, 1.0, 1.2], "price": [20.0, 30.0, 100.0, 120.0],
        "net_cash": [18.0, 27.0, 90.0, 108.0],
    })
    demand = np.array([[[5.0], [3.0]], [[2.0], [1.0]]])
    return products, prices, demand


def _random_case(n_products, n_scen, seed, flat=False):
    rng = np.random.default_rng(seed)
    ids = []
    product_nums = list(range(n_products))
    for j in product_nums:
        ids.append(f"p{j}")
    ref = rng.uniform(5, 120, size=n_products)
    products = pd.DataFrame({
        "cost": np.round(ref * rng.uniform(0.5, 0.75, size=n_products), 2),
        "on_hand": rng.integers(0, 3, size=n_products),
        "min_order": rng.choice([1, 1, 3], size=n_products),
        "max_units": rng.integers(4, 15, size=n_products),
        "set_code": rng.choice(["s1", "s2"], size=n_products),
        "salvage_frac": 0.7,
    }, index=ids)
    rows = []
    level_nums = list(range(len(RATIOS)))
    for j in product_nums:
        for k in level_nums:
            p = round(RATIOS[k] * ref[j], 2)
            rows.append({"i": ids[j], "k": k, "ratio": RATIOS[k], "price": p,
                         "net_cash": net_cash_per_unit(p, EXAMPLE_CONFIG)})
    prices = pd.DataFrame(rows)
    if flat:
        elasticity = np.full(n_products, -0.05)
    else:
        elasticity = rng.uniform(-4, -1.5, size=n_products)
    mult = np.array(RATIOS)[None, :] ** elasticity[:, None]
    lam = rng.uniform(1, 8, size=n_products)
    demand = ordered_demand(lam, mult, n_scen, rng)
    return products, prices, demand


def test_hand_checkable_optimum():
    products, prices, demand = _hand_case()
    out = solve_plan(products, prices, demand, budget=100.0)
    assert out["optimal"]
    units = dict(zip(out["buy"]["i"], out["buy"]["units"]))
    levels = dict(zip(out["prices"]["i"], out["prices"]["level"]))
    # a at k=1: 3*27 - 30 = 51 ; b at k=1: 108 - 50 = 58 ; spend 80 <= 100
    assert units == {"a": 3, "b": 1}
    assert levels == {"a": 1, "b": 1}
    assert abs(out["objective"] - 109.0) < 1e-6


def test_hull_and_bigm_agree():
    products, prices, demand = _random_case(4, 10, seed=1)
    hull = solve_plan(products, prices, demand, budget=300.0, transform="hull", mip_gap=0)
    bigm = solve_plan(products, prices, demand, budget=300.0, transform="bigm", mip_gap=0)
    assert hull["optimal"] and bigm["optimal"]
    rel = abs(hull["objective"] - bigm["objective"]) / max(1.0, abs(hull["objective"]))
    assert rel < 1e-6


@pytest.mark.parametrize("budget", [0.0, 40.0, 150.0, 400.0, 2000.0])
def test_budget_never_exceeded(budget):
    products, prices, demand = _random_case(5, 8, seed=3)
    out = solve_plan(products, prices, demand, budget=budget, lp_relaxation=False)
    assert out["buy"]["spend"].sum() <= budget + 1e-6


def test_zero_budget_buys_nothing_and_flags_m4():
    products, prices, demand = _random_case(4, 5, seed=2)
    out = solve_plan(products, prices, demand, budget=0.0)
    assert out["buy"]["units"].sum() == 0
    assert "M4" in set(out["flags"]["code"])


def test_m1_drops_unprofitable_level():
    products, prices, demand = _hand_case()
    # a level 0 net cash 4 <= salvage 0.5*10 = 5
    prices.loc[(prices["i"] == "a") & (prices["k"] == 0), "net_cash"] = 4.0
    out = solve_plan(products, prices, demand, budget=100.0)
    flags = out["flags"]
    m1 = flags[(flags["code"] == "M1") & (flags["i"] == "a")]
    assert len(m1) == 1
    assert "level 0 dropped" in m1.iloc[0]["message"]
    level_a = out["prices"].set_index("i").at["a", "level"]
    assert level_a == 1


def test_flat_demand_picks_top_price_and_flags_edge():
    products, prices, demand = _random_case(3, 20, seed=5, flat=True)
    products["on_hand"] = 2
    out = solve_plan(products, prices, demand, budget=500.0)
    top = max(range(len(RATIOS)))
    assert set(out["prices"]["level"]) == {top}
    m2 = out["flags"][out["flags"]["code"] == "M2"]
    assert set(m2["i"]) == set(products.index)


def test_lumpy_item_lp_gap():
    products = pd.DataFrame({
        "cost": [600.0], "on_hand": [0], "min_order": [1], "max_units": [3],
        "set_code": ["s"], "salvage_frac": [0.5],
    }, index=["case"])
    prices = pd.DataFrame({"i": ["case"], "k": [0], "ratio": [1.0],
                           "price": [1000.0], "net_cash": [900.0]})
    demand = np.array([[[3.0]]])
    out = solve_plan(products, prices, demand, budget=1000.0)
    assert out["buy"]["units"].iloc[0] == 1
    assert abs(out["objective"] - 300.0) < 1e-6
    assert out["lp_relaxation_objective"] >= out["objective"] + 100.0
    assert "M5" in set(out["flags"]["code"])


def test_min_order_respected():
    for d in [1, 2, 3, 4, 5, 8]:
        products = pd.DataFrame({
            "cost": [10.0], "on_hand": [0], "min_order": [6], "max_units": [12],
            "set_code": ["s"], "salvage_frac": [0.3],
        }, index=["display"])
        prices = pd.DataFrame({"i": ["display"], "k": [0], "ratio": [1.0],
                               "price": [25.0], "net_cash": [22.0]})
        demand = np.array([[[float(d)]]])
        out = solve_plan(products, prices, demand, budget=500.0)
        units = out["buy"]["units"].iloc[0]
        assert units == 0 or units >= 6
    # d=1: buying 6 loses money -> 0 ; d=8: buys 8
    assert units == 8


def test_demand_must_be_non_increasing():
    products, prices, demand = _hand_case()
    demand[0, 1, 0] = 9.0
    with pytest.raises(ValueError):
        solve_plan(products, prices, demand, budget=100.0)


def test_budget_curve_monotone():
    products, prices, demand = _random_case(4, 6, seed=4)
    curve = budget_curve(products, prices, demand, [0, 100, 300])
    assert list(curve.columns) == ["budget", "objective", "spend"]
    assert np.all(np.diff(curve["objective"]) >= -1e-6)
