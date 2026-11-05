"""Out-of-sample scorecard (U1) logic checks, on the committed file and on small cases.

The committed reports/scorecard.json is never re-solved here (that is the ~75 s heavy
run); instead every stored plan is re-valued on regenerated inputs (--check) and the
invariants are asserted on it. Out-of-sample values have no sign guarantee, so they
are reported, never asserted.
"""
import copy
import json

import numpy as np
import pytest

from experiments import common, scorecard
from milo import ladder
from milo.optimize import solve_plan
from tests.test_optimize import _random_case


@pytest.fixture(scope="module")
def problem():
    return common.load_problem()


@pytest.fixture(scope="module")
def card():
    with open(scorecard.SCORECARD_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def _row(card, policy, budget):
    for entry in card["table"]:
        if entry["policy"] == policy and entry["budget"] == budget:
            return entry
    raise KeyError((policy, budget))


def _plan(card, policy, budget):
    position = card["settings"]["budgets"].index(budget)
    return card["plans"][policy][position]


def test_eval_seed_is_never_a_training_seed():
    seeds = scorecard.training_seeds()
    assert scorecard.EVAL_SEED == 999
    assert scorecard.EVAL_SEED not in seeds
    # control: the set really holds the seeds the experiments train on
    assert common.SCENARIO_SEED in seeds and 101 in seeds and 15 in seeds


def test_rord_hand_fixture_and_na_at_zero_spend():
    assert scorecard.rord(110.0, 100.0, 20.0) == pytest.approx(0.5)
    assert scorecard.rord(90.0, 100.0, 20.0) == pytest.approx(-0.5)
    assert scorecard.rord(100.0, 100.0, 0.0) is None
    assert scorecard.rord(100.0, 100.0, None) is None


def test_committed_scorecard_is_current(card, problem):
    assert scorecard.check(stored=card, problem=problem) == []


def test_control_a_tampered_scorecard_fails_the_check(card, problem):
    tampered = copy.deepcopy(card)
    tampered["headline"]["optimizer_rord_out"] = tampered["headline"]["optimizer_rord_out"] + 0.01
    tampered["plans"]["greedy"][3]["buy"][next(iter(tampered["plans"]["greedy"][3]["buy"]))] += 1
    problems = scorecard.check(stored=tampered, problem=problem)
    assert any("headline.optimizer_rord_out" in item for item in problems)
    assert any("table" in item for item in problems)


def test_rord_recomputes_from_the_table(card):
    for entry in card["table"]:
        if entry["budget"] == 0.0:
            assert entry["rord_in"] is None and entry["rord_out"] is None
            continue
        expected_in = (entry["in_sample"] - entry["base_in_sample"]) / entry["spend"]
        expected_out = (entry["out_of_sample"] - entry["base_out_of_sample"]) / entry["spend"]
        assert entry["rord_in"] == pytest.approx(expected_in, rel=1e-8)
        assert entry["rord_out"] == pytest.approx(expected_out, rel=1e-8)
        assert entry["curse_gap"] == pytest.approx(entry["in_sample"] - entry["out_of_sample"], abs=1e-6)


def test_no_policy_beats_the_milp_in_sample_within_the_gap(card):
    gap = card["settings"]["mip_gap"]
    for budget in card["settings"]["budgets"]:
        milp = _plan(card, "optimizer", budget)
        milp_value = _row(card, "optimizer", budget)["in_sample"]
        for policy in ("greedy", "today_prices"):
            value = _row(card, policy, budget)["in_sample"]
            assert ladder.restriction_holds(value, milp_value, milp["objective_bound"], gap), (policy, budget)


def test_control_never_beats_check_can_fail():
    assert not ladder.restriction_holds(1600.0, 1568.38, 1568.53, 1e-4)
    assert ladder.restriction_holds(1568.40, 1568.38, 1568.53, 1e-4)


def test_more_budget_never_lowers_the_optimizer_in_sample(card):
    gap = card["settings"]["mip_gap"]
    budgets = card["settings"]["budgets"]
    positions = list(range(1, len(budgets)))
    for position in positions:
        lower = _plan(card, "optimizer", budgets[position - 1])
        higher = _plan(card, "optimizer", budgets[position])
        # the smaller budget's plan is feasible at the larger budget, so it is the restriction
        assert ladder.restriction_holds(lower["solver_objective"], higher["solver_objective"],
                                        higher["objective_bound"], gap)


def test_spend_never_exceeds_budget_and_solves_reached_the_gap(card):
    for policy in scorecard.POLICIES:
        budgets = card["settings"]["budgets"]
        numbered = list(enumerate(budgets))
        for position, budget in numbered:
            record = card["plans"][policy][position]
            assert record["spend"] <= budget + 1e-6
            if "optimal" in record:
                assert record["optimal"]


def test_solver_objective_is_the_in_sample_value(card):
    for policy in ("optimizer", "today_prices"):
        budgets = card["settings"]["budgets"]
        numbered = list(enumerate(budgets))
        for position, budget in numbered:
            record = card["plans"][policy][position]
            assert _row(card, policy, budget)["in_sample"] == pytest.approx(record["solver_objective"], rel=1e-8)


def test_fee_blind_objective_is_at_least_its_true_fee_value(card):
    fee = card["fee_blind"]
    assert fee["own_objective"] >= fee["true_in_sample"]
    assert fee["own_objective"] == pytest.approx(card["fee_blind_plan"]["solver_objective"], rel=1e-8)
    assert fee["overstatement"] == pytest.approx(fee["own_objective"] - fee["true_in_sample"], abs=1e-6)


def test_marginal_rows_recompute_and_carry_their_tolerance(card):
    for row in card["marginal"]:
        assert row["tolerance_per_dollar"] > 0
        assert row["step"] == scorecard.MARGINAL_STEP
        assert row["marginal_in"] >= -row["tolerance_per_dollar"]


def test_today_prices_baseline_equals_the_zero_edit_rung():
    # The fair baseline (ratio-1.00 rows only, buys re-optimized) must be the same
    # problem as the price-edit limit P = 0; exact at mip_gap = 0 on a small case.
    products, prices, demand = _random_case(5, 10, seed=4)
    prices = prices.copy()
    now_prices, now_demand = scorecard.today_problem(prices, demand)
    assert now_demand.shape == (5, 1, 10)
    today = solve_plan(products, now_prices, now_demand, budget=250.0, mip_gap=0, lp_relaxation=False)
    zero = ladder.solve_with_rules(products, prices, demand, ladder.Rules(price_edit_limit=0),
                                   budget=250.0, mip_gap=0)
    assert today["objective"] == pytest.approx(zero["objective"], rel=1e-7, abs=1e-6)


def test_committed_today_prices_matches_the_committed_zero_edit_rung(card):
    with open(common.REPORTS_DIR / "constraint_ladder.json", encoding="utf-8") as handle:
        ladder_file = json.load(handle)
    zero = None
    for row in ladder_file["derived"]["edit_sweep"]:
        if row["limit"] == 0:
            zero = row
    today = _row(card, "today_prices", card["settings"]["headline_budget"])
    bound = _plan(card, "today_prices", card["settings"]["headline_budget"])["objective_bound"]
    tolerance = ladder.rung_tolerance(card["settings"]["mip_gap"], bound)
    assert abs(today["in_sample"] - zero["expected_profit"]) <= tolerance


def test_policy_plans_smoke_on_a_small_case():
    products, prices, demand = _random_case(4, 8, seed=3)
    small = {"products": products, "prices": prices, "demand": demand}
    plans = scorecard.policy_plans(small, budgets=(0.0, 120.0))
    for policy in scorecard.POLICIES:
        assert len(plans[policy]) == 2
        assert plans[policy][0]["spend"] == 0.0
    eval_demand = demand[:, :, ::-1].copy()
    marginal = scorecard.marginal_plans(small, budgets=(0.0, 120.0))
    fee_blind = scorecard.fee_blind_plan(small, budget=120.0)
    values = scorecard.valuations(small, eval_demand, plans, marginal, fee_blind, budgets=(0.0, 120.0))
    table, marginal_rows, fee_row, paired_rows = scorecard.derive((0.0, 120.0), plans, values, marginal, fee_blind)
    assert len(table) == 6 and len(marginal_rows) == 1 and len(paired_rows) == 2
    assert fee_row["own_objective"] >= fee_row["true_in_sample"] - 1e-9
    assert np.isfinite(table[1]["out_of_sample"])
    # the paired lead is the difference of the two out-of-sample means (same scenarios)
    rows = {}
    for entry in table:
        rows[(entry["policy"], entry["budget"])] = entry
    expected = rows[("optimizer", 120.0)]["out_of_sample"] - rows[("greedy", 120.0)]["out_of_sample"]
    assert paired_rows[1]["lead_vs_greedy_out"] == pytest.approx(expected, abs=1e-9)
    assert fee_row["out_of_sample_minus_fee_aware"] == pytest.approx(
        fee_row["true_out_of_sample"] - rows[("optimizer", 120.0)]["out_of_sample"], abs=1e-9)


# ---- review follow-ups: stored differences, paired errors, guards, greedy freshness ----

def test_derived_differences_are_stored_and_recompute(card):
    head = card["headline"]
    assert head["curse_rord_gap"] == pytest.approx(head["optimizer_rord_in"] - head["optimizer_rord_out"], rel=1e-8)
    context = card["context"]
    today = _row(card, "today_prices", context["budget"])
    assert context["straw_man_loss_out"] == pytest.approx(
        today["out_of_sample"] - context["same_buys_today_out_of_sample"], abs=1e-6)
    for row in card["marginal"]:
        assert row["average_minus_marginal"] == pytest.approx(row["average_in"] - row["marginal_in"], abs=1e-9)


def test_pinned_seed_position_recomputes_from_numbers_json(card):
    with open(common.REPORTS_DIR / "numbers.json", encoding="utf-8") as handle:
        spread = json.load(handle)["rp_seed_spread"]
    seed = card["context"]["pinned_seed"]
    position = spread["seeds"].index(common.SCENARIO_SEED)
    pinned = spread["rp_values"][position]
    higher = 0
    for value in spread["rp_values"]:
        if value > pinned:
            higher += 1
    assert seed["rank_from_top"] == higher + 1
    assert seed["minus_seed_mean"] == pytest.approx(pinned - spread["mean"], abs=1e-6)


def test_paired_se_hand_fixture_and_control():
    first = np.array([3.0, 5.0, 7.0])
    second = np.array([1.0, 2.0, 3.0])
    # differences 2, 3, 4: sd 1, n 3 -> 1 / sqrt(3)
    assert scorecard.paired_se(first, second) == pytest.approx(1.0 / np.sqrt(3.0))
    # control: identical shifts have zero paired error although each series varies
    assert scorecard.paired_se(first, first - 10.0) == pytest.approx(0.0)
    assert scorecard.paired_se(np.array([1.0]), np.array([0.0])) is None


def test_headline_lead_has_its_paired_error(card):
    head = card["headline"]
    optimizer = _row(card, "optimizer", head["budget"])
    greedy = _row(card, "greedy", head["budget"])
    assert head["lead_vs_greedy_out"] == pytest.approx(optimizer["out_of_sample"] - greedy["out_of_sample"], abs=1e-6)
    assert head["lead_vs_greedy_out_se"] > 0 and head["rord_lift_vs_greedy_out_se"] > 0


def test_today_problem_guard_fires_when_the_today_level_fails_m1():
    products, prices, demand = _random_case(4, 8, seed=3)
    assert scorecard.today_level_dropped(products, prices) == []
    scorecard.today_problem(prices, demand, products)   # the guard passes on clean data
    broken = prices.copy()
    first = products.index[0]
    salvage = float(products.at[first, "salvage_frac"]) * float(products.at[first, "cost"])
    today = (broken["i"] == first) & ((broken["ratio"].astype(float) - 1.0).abs() < 1e-9)
    broken.loc[today, "net_cash"] = salvage - 0.01
    assert scorecard.today_level_dropped(products, broken) == [first]
    with pytest.raises(ValueError, match="ratio-1.00 level fails the M1 filter"):
        scorecard.today_problem(broken, demand, products)


def test_committed_data_drops_no_today_level(problem):
    assert scorecard.today_level_dropped(problem["products"], problem["prices"]) == []


def test_control_a_stale_greedy_plan_fails_the_check(card, problem):
    tampered = copy.deepcopy(card)
    record = tampered["plans"]["greedy"][2]
    unit = sorted(record["levels"].keys())[0]
    record["levels"][unit] = int(record["levels"][unit]) + 1
    problems = scorecard.check(stored=tampered, problem=problem)
    assert "scorecard greedy plan stale at $500" in problems


def test_control_compare_reports_extra_keys():
    problems = []
    scorecard._compare({"a": 1.0, "b": 2.0}, {"a": 1.0, "c": 3.0}, "x", problems)
    assert "x.c missing" in problems and "x.b is extra" in problems
