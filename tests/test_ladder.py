"""Constraint ladder (U2-U4) logic checks.

Exact checks run on small random cases at mip_gap = 0; on the committed problem the
tolerance is mip_gap x |bound| (about 0.16 at the pinned 1e-4 gap), and the committed
reports/constraint_ladder.json is re-valued and re-checked without solving.
Each rung kind gets the test that fits it: restrictions can't raise profit, the
accounting rung is worth >= 0, and the controls show each check can fail.
"""
import copy
import json

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest

from experiments import common, constraint_ladder
from milo import ladder
from milo.optimize import _build_model, _prepare, solve_plan
from tests.test_optimize import _random_case

EXACT = 1e-6


def _spaced(products, seed=0):
    rng = np.random.default_rng(seed)
    out = products.copy()
    out["space"] = rng.choice([0.5, 1.0, 4.0, 8.0], size=len(out.index))
    return out


def _maps(result):
    return common.plan_maps(result)


def _close(a, b):
    return abs(a - b) <= EXACT * max(1.0, abs(b))


@pytest.fixture(scope="module")
def small():
    products, prices, demand = _random_case(6, 10, seed=11)
    return _spaced(products), prices, demand


@pytest.fixture(scope="module")
def committed():
    with open(constraint_ladder.LADDER_PATH, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def problem():
    return constraint_ladder.ladder_problem()


# ---- rules off = the existing model -------------------------------------------------

def test_rules_off_is_the_same_model_and_optimum(small):
    products, prices, demand = small
    plain = solve_plan(products, prices, demand, budget=300.0, mip_gap=0, lp_relaxation=False)
    ruled = ladder.solve_with_rules(products, prices, demand, ladder.Rules(), budget=300.0, mip_gap=0)
    assert ladder.model_size(plain["model"]) == (ruled["n_variables"], ruled["n_integer_variables"],
                                                 ruled["n_constraints"])
    assert _close(ruled["objective"], plain["objective"])
    assert list(ruled["flags"]["code"]) == list(plain["flags"]["code"])


def test_rules_off_matches_the_committed_model_counts_and_rp(problem, committed):
    products = problem["products"]
    setup = _prepare(products, problem["prices"], problem["demand"], None, common.MAIN_BUDGET, "hull")
    plain = _build_model(products, setup["demand"], setup["probs"], setup["levels"], setup["valid"],
                         setup["net_cash"], common.MAIN_BUDGET, 1.0, setup["flags"])
    pyo.TransformationFactory("gdp.hull").apply_to(plain)
    ruled, _, _, _ = ladder.build_with_rules(products, problem["prices"], problem["demand"],
                                             ladder.Rules(), budget=common.MAIN_BUDGET)
    record = committed["records"]["R0b"]
    counts = (record["n_variables"], record["n_integer_variables"], record["n_constraints"])
    assert ladder.model_size(plain) == ladder.model_size(ruled) == counts
    with open(common.REPORTS_DIR / "numbers.json", encoding="utf-8") as handle:
        numbers = json.load(handle)
    assert record["solver_objective"] == pytest.approx(numbers["evpi_vss"]["rp"], rel=1e-9)


# ---- kind-typed checks and their controls -------------------------------------------

def test_restriction_and_accounting_checks_can_fail():
    assert ladder.restriction_holds(100.0, 100.05, 100.1, 1e-4)
    assert not ladder.restriction_holds(100.5, 100.0, 100.1, 1e-4)
    assert ladder.accounting_holds(100.0, 99.0, 100.1, 1e-4)
    assert not ladder.accounting_holds(98.0, 99.0, 100.1, 1e-4)
    assert ladder.rung_tolerance(1e-4, 1568.53) == pytest.approx(0.156853)
    assert set(ladder.RUNG_KINDS) == {"restriction", "accounting"}


def test_committed_rung_kinds_get_their_own_test(committed):
    rows = committed["derived"]["ladder"]
    kinds = {}
    for row in rows:
        kinds[row["rung"]] = row["kind"]
        assert row["kind"] in ladder.RUNG_KINDS
    assert kinds == {"R0a": "accounting", "R0b": "accounting", "R1": "restriction", "R2": "restriction"}
    for row in rows:
        if row["kind"] == "restriction":
            assert row["price_of_rule"] is not None and row["value_of_modeling"] is None
        if row["rung"] == "R0b":
            assert row["value_of_modeling"] is not None and row["price_of_rule"] is None


def test_committed_ladder_is_current_and_its_logic_checks_hold(committed, problem):
    assert constraint_ladder.check(stored=committed, problem=problem) == []


def test_control_a_rule_that_raises_profit_fails_the_check(committed, problem):
    tampered = copy.deepcopy(committed)
    record = tampered["records"]["shelf_0.70"]
    record["buy"] = copy.deepcopy(tampered["records"]["R0b"]["buy"])
    record["levels"] = copy.deepcopy(tampered["records"]["R0b"]["levels"])
    problems = constraint_ladder.check(stored=tampered, problem=problem)
    assert any("shelf use over capacity" in item for item in problems)
    tampered = copy.deepcopy(committed)
    tampered["records"]["edits_0"]["levels"] = copy.deepcopy(tampered["records"]["R0b"]["levels"])
    tampered["records"]["edits_0"]["buy"] = copy.deepcopy(tampered["records"]["R0b"]["buy"])
    problems = constraint_ladder.check(stored=tampered, problem=problem)
    assert any("price edits over the limit" in item for item in problems)
    assert any("restriction raised profit" in item for item in problems)


def test_committed_tolerance_is_about_016(committed):
    bound = committed["records"]["R0b"]["objective_bound"]
    tolerance = ladder.rung_tolerance(committed["settings"]["mip_gap"], bound)
    assert 0.15 < tolerance < 0.17
    assert committed["derived"]["tolerance_at_rp"] == pytest.approx(tolerance, rel=1e-9)


# ---- grid edge (M2) and the minimum-order rule's reach ------------------------------

def _edge_case():
    products = pd.DataFrame({"cost": [10.0, 10.0, 10.0], "on_hand": [2, 2, 0], "min_order": [1, 1, 1],
                             "max_units": [5, 5, 5], "set_code": ["A", "A", "A"],
                             "salvage_frac": [0.5, 0.5, 0.5]}, index=["a", "b", "c"])
    prices = pd.DataFrame({"i": ["a", "a", "a", "b", "b", "b", "c", "c", "c"], "k": [0, 1, 2] * 3,
                           "ratio": [0.9, 1.0, 1.1] * 3,
                           "price": [18.0, 20.0, 22.0] * 3, "net_cash": [16.0, 18.0, 20.0] * 3})
    return products, prices


def test_grid_edge_moves_hand_fixture():
    products, prices = _edge_case()
    # a at the top level (moved, edge), b at the bottom (moved, bottom edge), c unstocked at the top
    counts = ladder.grid_edge_moves(products, prices, {"a": 0, "b": 0, "c": 0}, {"a": 2, "b": 0, "c": 2})
    assert counts == {"moved": 2, "moved_top": 1, "moved_bottom": 1, "m2": 1, "m3": 1}
    # once c is bought it counts too
    counts = ladder.grid_edge_moves(products, prices, {"a": 0, "b": 0, "c": 3}, {"a": 2, "b": 0, "c": 2})
    assert counts["moved_top"] == 2 and counts["m2"] == 2


def test_control_grid_edge_follows_the_m1_filter():
    products, prices = _edge_case()
    # drop a's top level below salvage: level 1 (today) becomes a's highest kept level
    prices.loc[(prices["i"] == "a") & (prices["k"] == 2), "net_cash"] = 4.0
    counts = ladder.grid_edge_moves(products, prices, {}, {"a": 1, "b": 1})
    assert counts["m2"] == 1 and counts["moved_top"] == 0   # at the edge, but not a move


def test_committed_edge_recount_matches_the_solver_flag(committed, problem):
    for name in sorted(committed["records"].keys()):
        if name == "R0a":
            continue
        record = committed["records"][name]
        counts = ladder.grid_edge_moves(problem["products"], problem["prices"], record["buy"], record["levels"])
        assert ("M2" in record["flag_codes"]) == (counts["m2"] > 0), name
    derived = committed["derived"]
    assert derived["m2_records"] <= derived["n_records"]
    assert derived["base_price_moves_at_top_edge"] <= derived["base_price_edits"]


def test_control_a_dropped_m2_flag_fails_the_check(committed, problem):
    tampered = copy.deepcopy(committed)
    tampered["records"]["R0b"]["flag_codes"] = ["M5"]
    problems = constraint_ladder.check(stored=tampered, problem=problem)
    assert "logic check: R0b: the M2 recount disagrees with the solver's flag" in problems


def test_min_order_facts_say_whether_the_rule_can_bind(committed, problem):
    row = None
    for entry in committed["derived"]["existing_rules"]:
        if entry["without"] == "no_min_order":
            row = entry
    above = 0
    for i in problem["products"].index:
        if int(problem["products"].at[i, "min_order"]) > 1:
            above += 1
    assert row["products_with_min_order_above_1"] == above
    assert row["n_products"] == len(problem["products"].index)
    records = committed["records"]
    assert row["identical_buys"] == (records["R0b"]["buy"] == records["no_min_order"]["buy"])
    # control: a plan that buys a product with a real minimum is counted
    fake = {"R0b": {"buy": {}}, "no_min_order": {"buy": {}}}
    for i in problem["products"].index:
        if int(problem["products"].at[i, "min_order"]) > 1:
            fake["no_min_order"]["buy"][i] = int(problem["products"].at[i, "min_order"])
    facts = constraint_ladder.min_order_facts(problem, fake)
    assert facts["bought_without_rule"] == above and facts["bought_with_rule"] == 0
    assert facts["identical_buys"] is (above == 0)


def test_rule_labels_follow_the_ladder_shelf_fraction(committed):
    share = constraint_ladder.shelf_label(constraint_ladder.LADDER_SHELF_FRACTION)
    assert share == "70%"
    rungs = {}
    for row in committed["derived"]["ladder"]:
        rungs[row["rung"]] = row
    assert share in rungs["R1"]["rule"]
    assert committed["derived"]["alone"][0]["rule"] == "shelf space alone (" + share + ")"


def test_ladder_output_has_its_own_file_and_numbers_json_is_frozen():
    with open(common.REPORTS_DIR / "numbers.json", encoding="utf-8") as handle:
        numbers = json.load(handle)
    assert "constraint_ladder" not in numbers and "scorecard" not in numbers
    assert numbers["evpi_vss"]["rp"] == pytest.approx(1568.3775260403797, rel=1e-12)


# ---- R1 shelf space -----------------------------------------------------------------

def test_shelf_use_recount_and_monotone_in_capacity(small):
    products, prices, demand = small
    base = ladder.solve_with_rules(products, prices, demand, ladder.Rules(), budget=400.0, mip_gap=0)
    buy, _ = _maps(base)
    full_use = ladder.shelf_use(products, buy)
    previous = None
    for fraction in (0.4, 0.6, 0.8, 1.0):
        capacity = fraction * full_use
        result = ladder.solve_with_rules(products, prices, demand,
                                         ladder.Rules(shelf_capacity=capacity), budget=400.0, mip_gap=0)
        used, _ = _maps(result)
        assert ladder.shelf_use(products, used) <= capacity + 1e-9
        assert result["rule_report"]["shelf"]["used"] <= capacity + 1e-9
        assert result["objective"] <= base["objective"] + EXACT * max(1.0, abs(base["objective"]))
        if previous is not None:
            assert result["objective"] >= previous - EXACT * max(1.0, abs(previous))
        previous = result["objective"]
    assert _close(previous, base["objective"])


def test_infinite_shelf_gives_the_base_exactly(small):
    products, prices, demand = small
    base = ladder.solve_with_rules(products, prices, demand, ladder.Rules(), budget=400.0, mip_gap=0)
    wide = ladder.solve_with_rules(products, prices, demand, ladder.Rules(shelf_capacity=1e9),
                                   budget=400.0, mip_gap=0)
    assert _close(wide["objective"], base["objective"])
    assert not wide["rule_report"]["shelf"]["binds"]


def test_on_hand_over_capacity_raises_a_shelf_flag_not_an_infeasible_solve(small):
    products, prices, demand = small
    stocked = products.copy()
    stocked["on_hand"] = 5
    result = ladder.solve_with_rules(stocked, prices, demand, ladder.Rules(shelf_capacity=1.0),
                                     budget=400.0, mip_gap=0)
    assert "SHELF" in list(result["flags"]["code"])
    assert result["optimal"]
    assert int(result["buy"]["units"].sum()) == 0


def test_shelf_rule_needs_a_space_column():
    products, prices, demand = _random_case(3, 4, seed=1)
    with pytest.raises(ValueError):
        ladder.solve_with_rules(products, prices, demand, ladder.Rules(shelf_capacity=10.0), budget=50.0)


def test_space_defaults_cover_every_committed_product_type(problem):
    assert problem["products"]["space"].notna().all()
    bad = pd.DataFrame({"unit": ["u"], "product_type": ["mystery"]})
    with pytest.raises(ValueError):
        ladder.space_per_unit(bad)


# ---- R2 weekly price-edit limit -----------------------------------------------------

def test_edit_recount_and_monotone_in_limit(small):
    products, prices, demand = small
    base = ladder.solve_with_rules(products, prices, demand, ladder.Rules(), budget=400.0, mip_gap=0)
    previous = None
    limits = list(range(0, len(products.index) + 1))
    for limit in limits:
        result = ladder.solve_with_rules(products, prices, demand,
                                         ladder.Rules(price_edit_limit=limit), budget=400.0, mip_gap=0)
        buy, levels = _maps(result)
        edited = ladder.count_price_edits(products, prices, buy, levels)
        assert len(edited) <= limit
        assert result["rule_report"]["price_edits"]["used"] == len(edited)
        if previous is not None:
            assert result["objective"] >= previous - EXACT * max(1.0, abs(previous))
        previous = result["objective"]
    assert _close(previous, base["objective"])


def test_zero_edits_keep_every_stocked_product_at_today(small):
    products, prices, demand = small
    result = ladder.solve_with_rules(products, prices, demand, ladder.Rules(price_edit_limit=0),
                                     budget=400.0, mip_gap=0)
    buy, levels = _maps(result)
    now = ladder.today_levels(prices)
    product_ids = list(products.index)
    for i in product_ids:
        stock = float(products.at[i, "on_hand"]) + buy.get(i, 0)
        if stock > 0:
            assert levels[i] == now[i]


def test_unstocked_product_never_counts_as_an_edit():
    products = pd.DataFrame({"cost": [10.0, 10.0], "on_hand": [0, 2], "min_order": [1, 1],
                             "max_units": [5, 5], "set_code": ["A", "A"], "salvage_frac": [0.5, 0.5],
                             "space": [1.0, 1.0]}, index=["a", "b"])
    prices = pd.DataFrame({"i": ["a", "a", "b", "b"], "k": [0, 1, 0, 1], "ratio": [1.0, 1.1, 1.0, 1.1],
                           "price": [20.0, 22.0, 20.0, 22.0], "net_cash": [18.0, 19.8, 18.0, 19.8]})
    # a is unstocked and not bought, at ratio 1.1: no edit. b is stocked at 1.1: one edit.
    assert ladder.count_price_edits(products, prices, {"a": 0, "b": 0}, {"a": 1, "b": 1}) == ["b"]
    # once a is bought at 1.1 it counts
    assert ladder.count_price_edits(products, prices, {"a": 2, "b": 0}, {"a": 1, "b": 0}) == ["a"]


def test_dropped_today_level_counts_as_an_edit():
    # b's ratio-1.00 level nets less than salvage, so M1 drops it; b is stocked, so it
    # must move: P = 0 cannot hold, the limit is raised to 1 with an EDITS flag.
    products = pd.DataFrame({"cost": [10.0, 10.0], "on_hand": [0, 2], "min_order": [1, 1],
                             "max_units": [5, 5], "set_code": ["A", "A"], "salvage_frac": [0.5, 0.9],
                             "space": [1.0, 1.0]}, index=["a", "b"])
    prices = pd.DataFrame({"i": ["a", "a", "b", "b"], "k": [0, 1, 0, 1], "ratio": [1.0, 1.2, 1.0, 1.2],
                           "price": [20.0, 24.0, 9.0, 12.0], "net_cash": [18.0, 21.6, 8.5, 10.8]})
    demand = np.array([[[3.0, 2.0], [2.0, 1.0]], [[2.0, 2.0], [2.0, 1.0]]])
    result = ladder.solve_with_rules(products, prices, demand, ladder.Rules(price_edit_limit=0),
                                     budget=50.0, mip_gap=0)
    assert "EDITS" in list(result["flags"]["code"])
    assert result["rule_report"]["price_edits"]["forced"] == ["b"]
    buy, levels = _maps(result)
    edited = ladder.count_price_edits(products, prices, buy, levels)
    assert "b" in edited


def test_fee_blind_objective_is_at_least_its_true_value_on_a_small_case(small):
    products, prices, demand = small
    gross = ladder.fee_blind_prices(prices)
    blind = ladder.solve_with_rules(products, gross, demand, ladder.Rules(), budget=300.0, mip_gap=0)
    buy, levels = _maps(blind)
    true_value, _ = common.expected_value(products, prices, demand, buy, levels)
    aware = ladder.solve_with_rules(products, prices, demand, ladder.Rules(), budget=300.0, mip_gap=0)
    assert blind["objective"] >= true_value - 1e-9
    assert ladder.accounting_holds(aware["objective"], true_value, aware["objective_bound"], 0.0)


def test_hull_and_bigm_agree_with_rules_on(small):
    products, prices, demand = small
    rules = ladder.Rules(shelf_capacity=20.0, price_edit_limit=2)
    hull = ladder.solve_with_rules(products, prices, demand, rules, budget=300.0, mip_gap=0)
    bigm = ladder.solve_with_rules(products, prices, demand, rules, budget=300.0, mip_gap=0,
                                   transform="bigm")
    assert hull["optimal"] and bigm["optimal"]
    assert _close(hull["objective"], bigm["objective"])
