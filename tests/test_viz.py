"""Restock Explorer page data (docs/data/*.json) checks: the page computes nothing, so its truth is here.

What is checked, and why each can fail:
- the committed files equal a fresh re-valuation of the committed plans (--check, no solves),
  with controls that tamper a value and a plan;
- invariants on every cell (spend <= budget, p10 <= p50 <= p90, RoRD recomputes and is n/a at
  $0, the optimizer never loses to a baseline in-sample beyond mip_gap x |bound|, more budget
  never lowers the optimizer in-sample);
- parity with the README's sources: the grid at the scorecard's budgets equals
  reports/scorecard.json, and the bracket, SAA and hull rows equal reports/numbers.json;
- one fresh solve at $500 (the plan's slow smoke test: 801.95 in numbers.json), compared
  within mip_gap x |bound| so a HiGHS version change cannot fail it;
- the page kit contract: tokens.css hash, .nojekyll, the exact Chart.js tag, 600 KB budget,
  key allowlists, no forbidden keys, no pipeline unit keys, no data-row-length list.
"""
import base64
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import common, scorecard, viz
from milo import jsonout, ladder
from milo.why import WHY
from tests.test_optimize import _random_case

REPO = Path(__file__).resolve().parents[1]
KIT_TOKENS = REPO.parent / "_build" / "pagekit" / "tokens.css"
CONVENTIONS_TAG = (
    '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js" '
    'integrity="sha512-CQBWl4fJHWbryGE+Pc7UAxWMUMNMWzWxF4SQo9CgkJIN1kx6djDQZjh3Y8SZ1d+6I+1zze6Z7kHXO7q3UyZAWw==" '
    'crossorigin="anonymous" referrerpolicy="no-referrer"></script>'
)


@pytest.fixture(scope="module")
def problem():
    return common.load_problem()


@pytest.fixture(scope="module")
def files():
    return viz.load_files()


@pytest.fixture(scope="module")
def grid(files):
    return files[viz.GRID_PATH]


@pytest.fixture(scope="module")
def method(files):
    return files[viz.METHOD_PATH]


@pytest.fixture(scope="module")
def saa(files):
    return files[viz.SAA_PATH]


def _numbers():
    with open(common.REPORTS_DIR / "numbers.json", encoding="utf-8") as handle:
        return json.load(handle)


def _scorecard():
    with open(scorecard.SCORECARD_PATH, encoding="utf-8") as handle:
        return json.load(handle)


# ---- freshness -----------------------------------------------------------------------

def test_committed_page_data_is_current(files, problem):
    assert viz.check(stored=files, problem=problem) == []


def test_control_a_tampered_value_fails_the_check(files, problem):
    tampered = copy.deepcopy(files)
    position = tampered[viz.GRID_PATH]["budgets"].index(viz.DEFAULT_BUDGET)
    tampered[viz.GRID_PATH]["cells"][position]["policies"]["optimizer"]["rord_out"] += 0.01
    problems = viz.check(stored=tampered, problem=problem)
    expected = "budget_grid.json.cells[" + str(position) + "].policies.optimizer.rord_out is stale"
    assert expected in problems


def test_control_a_tampered_plan_fails_the_check(files, problem):
    tampered = copy.deepcopy(files)
    position = tampered[viz.GRID_PATH]["budgets"].index(viz.DEFAULT_BUDGET)
    plan = tampered[viz.GRID_PATH]["plans"]["optimizer"][position]
    first = sorted(plan["buy"].keys())[0]
    plan["buy"][first] += 1
    problems = viz.check(stored=tampered, problem=problem)
    # the plan itself is the input, so the values derived from it go stale
    expected = "budget_grid.json.cells[" + str(position) + "].policies.optimizer.in_sample is stale"
    assert expected in problems


def test_control_compare_reports_missing_and_extra_keys():
    problems = []
    viz.compare({"a": 1.0, "b": 2.0}, {"a": 1.0, "c": 3.0}, "x", problems)
    assert "x.c missing" in problems and "x.b is extra" in problems
    problems = []
    viz.compare({"a": None}, {"a": 0.5}, "x", problems)
    assert problems == ["x.a is stale"]


# ---- invariants on every cell --------------------------------------------------------

def test_grid_is_the_36_solved_budgets_and_ticks_sit_on_it(grid, method):
    assert tuple(grid["budgets"]) == viz.GRID_BUDGETS
    assert len(grid["budgets"]) == 36
    assert grid["budgets"][0] == 0.0 and grid["budgets"][-1] == 3000.0
    assert method["provenance"]["slider"] == "snaps to 36 solved budgets"
    assert method["default_budget"] in grid["budgets"]
    for value in method["tick_budgets"]:
        assert value in grid["budgets"]


def test_spend_within_budget_and_quantiles_ordered(grid):
    for cell in grid["cells"]:
        for policy in viz.POLICIES:
            entry = cell["policies"][policy]
            assert entry["spend"] <= cell["budget"] + 1e-9
            assert entry["out_of_sample_p10"] <= entry["out_of_sample_p50"] <= entry["out_of_sample_p90"]


def test_rord_recomputes_and_is_na_at_zero(grid):
    for cell in grid["cells"]:
        for policy in viz.POLICIES:
            entry = cell["policies"][policy]
            if cell["budget"] == 0.0:
                assert entry["rord_in"] is None and entry["rord_out"] is None
                continue
            expected = (entry["out_of_sample"] - entry["base_out_of_sample"]) / entry["spend"]
            assert entry["rord_out"] == pytest.approx(expected, rel=1e-8)
            expected_in = (entry["in_sample"] - entry["base_in_sample"]) / entry["spend"]
            assert entry["rord_in"] == pytest.approx(expected_in, rel=1e-8)


def baseline_never_wins(cell, optimizer_plan, policy, mip_gap=common.EXPERIMENT_GAP):
    """The per-cell invariant: a baseline's in-sample value <= the optimizer's + mip_gap x |bound|.

    Both the real test and its control call this one helper (via ladder.restriction_holds,
    the same rule the scorecard and the ladder use), so a wrong field or tolerance here
    fails the control too.
    """
    return ladder.restriction_holds(cell["policies"][policy]["in_sample"], cell["policies"]["optimizer"]["in_sample"],
                                    optimizer_plan["objective_bound"], mip_gap)


def test_optimizer_never_loses_in_sample_beyond_the_gap(grid):
    positions = list(range(len(grid["budgets"])))
    for position in positions:
        cell = grid["cells"][position]
        plan = grid["plans"]["optimizer"][position]
        for policy in ("greedy", "today_prices"):
            assert baseline_never_wins(cell, plan, policy), (policy, cell["budget"])


def test_control_never_loses_check_can_fail(grid):
    position = grid["budgets"].index(viz.DEFAULT_BUDGET)
    planted = copy.deepcopy(grid["cells"][position])
    plan = grid["plans"]["optimizer"][position]
    assert baseline_never_wins(planted, plan, "greedy")
    tolerance = ladder.rung_tolerance(common.EXPERIMENT_GAP, plan["objective_bound"])
    planted["policies"]["greedy"]["in_sample"] = planted["policies"]["optimizer"]["in_sample"] + 2.0 * tolerance
    assert not baseline_never_wins(planted, plan, "greedy")
    # inside the tolerance it must still pass, so the helper uses the gap, not a strict <=
    planted["policies"]["greedy"]["in_sample"] = planted["policies"]["optimizer"]["in_sample"] + 0.5 * tolerance
    assert baseline_never_wins(planted, plan, "greedy")


def test_more_budget_never_lowers_the_optimizer_in_sample(grid):
    previous = None
    for cell in grid["cells"]:
        value = cell["policies"]["optimizer"]["in_sample"]
        if previous is not None:
            assert value >= previous - cell["tolerance"], cell["budget"]
        previous = value


def test_every_grid_solve_reached_the_gap(grid, method):
    for policy in ("optimizer", "today_prices"):
        for record in grid["plans"][policy]:
            assert record["optimal"] is True
    assert method["grid_facts"]["all_solves_optimal"] is True


def test_solver_objective_is_the_in_sample_value(grid):
    positions = list(range(len(grid["budgets"])))
    for policy in ("optimizer", "today_prices"):
        for position in positions:
            record = grid["plans"][policy][position]
            cell = grid["cells"][position]["policies"][policy]
            tolerance = ladder.rung_tolerance(common.EXPERIMENT_GAP, record["objective_bound"])
            assert abs(record["solver_objective"] - cell["in_sample"]) <= tolerance


# ---- buy lists -----------------------------------------------------------------------

def test_buy_lists_add_up_and_use_kept_levels(grid, problem):
    products = {}
    for row in grid["products"]:
        products[row["id"]] = row
    _, _, unit_by_id = viz.product_table(problem)
    price, _ = viz.price_lookups(problem)
    positions = list(range(len(grid["budgets"])))
    for policy in viz.POLICIES:
        for position in positions:
            record = grid["plans"][policy][position]
            listing = grid["cells"][position]["buy_lists"][policy]
            total = 0.0
            for row in listing["rows"]:
                assert row["spend"] == pytest.approx(row["buy"] * products[row["product"]]["cost"])
                assert row["buy"] > 0 or row["price_moved"]
                level = record["levels"][row["product"]]
                assert row["price"] == pytest.approx(price[(unit_by_id[row["product"]], level)])
                total += row["spend"]
            assert total == pytest.approx(record["spend"], abs=1e-6)
            assert listing["summary"]["spend"] == record["spend"]


def test_today_prices_never_moves_a_price(grid):
    for cell in grid["cells"]:
        assert cell["policies"]["today_prices"]["price_edits"] == 0
        for row in cell["buy_lists"]["today_prices"]["rows"]:
            assert row["price_moved"] is False


def test_edge_counts_recount_and_stay_within_the_moves(grid, problem, method):
    _, _, unit_by_id = viz.product_table(problem)
    positions = list(range(len(grid["budgets"])))
    total = 0
    total_top = 0
    for position in positions:
        cell = grid["cells"][position]
        for policy in viz.POLICIES:
            summary = cell["buy_lists"][policy]["summary"]
            assert summary["price_edits_at_top_edge"] + summary["price_edits_at_bottom_edge"] <= summary["price_edits"]
        record = grid["plans"]["optimizer"][position]
        buy = viz._to_unit(record["buy"], unit_by_id)
        levels = viz._to_unit(record["levels"], unit_by_id)
        edges = ladder.grid_edge_moves(problem["products"], problem["prices"], buy, levels)
        assert cell["policies"]["optimizer"]["price_edits_at_top_edge"] == edges["moved_top"]
        total += edges["moved"]
        total_top += edges["moved_top"]
        assert cell["policies"]["today_prices"]["price_edits_at_top_edge"] == 0
    facts = method["grid_facts"]
    assert facts["optimizer_price_edits"] == total and facts["optimizer_price_edits_at_top_edge"] == total_top
    assert facts["n_plans"] == len(grid["budgets"]) * len(viz.POLICIES)
    assert method["settings"]["price_grid_high"] == 1.15 and method["settings"]["price_grid_low"] == 0.80


def test_curse_gap_shape_hand_fixture_and_committed(saa):
    rows = [{"n_scenarios": 40, "curse_gap": 7.0}, {"n_scenarios": 20, "curse_gap": 15.0},
            {"n_scenarios": 80, "curse_gap": 9.0}, {"n_scenarios": 160, "curse_gap": 3.0}]
    shape = viz.curse_gap_shape(rows)
    assert shape["monotone"] is False
    assert shape["rises"] == [{"from_n": 40, "to_n": 80, "from_gap": 7.0, "to_gap": 9.0}]
    # control: a smoothly falling series has no rise
    smooth = viz.curse_gap_shape([{"n_scenarios": 20, "curse_gap": 5.0}, {"n_scenarios": 40, "curse_gap": 4.0}])
    assert smooth == {"monotone": True, "rises": []}
    assert saa["curse_gap_shape"] == jsonout.rounded(viz.curse_gap_shape(_numbers()["saa_convergence"]))


def test_control_a_stale_greedy_list_fails_the_page_check(files, problem):
    tampered = copy.deepcopy(files)
    plan = tampered[viz.GRID_PATH]["plans"]["greedy"][3]
    first = sorted(plan["levels"].keys())[0]
    plan["levels"][first] = int(plan["levels"][first]) + 1
    problems = viz.check(stored=tampered, problem=problem)
    budget = format(tampered[viz.GRID_PATH]["budgets"][3], ",.0f")
    assert "budget_grid.json greedy plan stale at $" + budget in problems


def test_builder_lives_outside_the_installed_package():
    # pyproject packages only `milo`; nothing in it may import experiments/
    package = REPO / "milo"
    for path in sorted(package.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            assert not stripped.startswith("from experiments"), path.name
            assert not stripped.startswith("import experiments"), path.name


def test_buy_list_counts_edits_like_the_ladder(grid, problem):
    _, _, unit_by_id = viz.product_table(problem)
    position = grid["budgets"].index(viz.DEFAULT_BUDGET)
    record = grid["plans"]["optimizer"][position]
    buy = viz._to_unit(record["buy"], unit_by_id)
    levels = viz._to_unit(record["levels"], unit_by_id)
    edited = ladder.count_price_edits(problem["products"], problem["prices"], buy, levels)
    assert grid["cells"][position]["policies"]["optimizer"]["price_edits"] == len(edited)


# ---- parity with the README's sources ------------------------------------------------

def test_grid_equals_the_scorecard_at_its_budgets(grid):
    card = _scorecard()
    for entry in card["table"]:
        position = grid["budgets"].index(entry["budget"])
        cell = grid["cells"][position]["policies"][entry["policy"]]
        bound = grid["plans"]["optimizer"][position]["objective_bound"]
        tolerance = ladder.rung_tolerance(common.EXPERIMENT_GAP, bound)
        assert abs(cell["in_sample"] - entry["in_sample"]) <= tolerance


def test_page_tiles_at_the_default_budget_print_the_readme_headline(grid, method):
    head = _scorecard()["headline"]
    position = grid["budgets"].index(head["budget"])
    cell = grid["cells"][position]["policies"]
    assert format(cell["optimizer"]["rord_out"], ".3f") == format(head["optimizer_rord_out"], ".3f")
    assert format(cell["greedy"]["rord_out"], ".3f") == format(head["greedy_rord_out"], ".3f")
    assert format(cell["today_prices"]["rord_out"], ".3f") == format(head["today_prices_rord_out"], ".3f")
    assert method["headline"] == head


def test_bracket_saa_and_hull_equal_numbers_json(method, saa):
    numbers = _numbers()
    ev = numbers["evpi_vss"]
    b = method["bracket"]
    assert b["eev"] <= b["rp"] <= b["ws"]
    for key in ("eev", "rp", "ws", "vss", "evpi"):
        assert b[key] == pytest.approx(ev[key], rel=1e-9)
    span = b["axis_max"] - b["axis_min"]
    for key in ("eev", "rp", "ws"):
        position = (b[key] - b["axis_min"]) / span
        assert b["positions"][key] == pytest.approx(position, rel=1e-8)
        assert 0.0 < position < 1.0
    assert len(saa["rows"]) == len(numbers["saa_convergence"])
    pairs = list(zip(saa["rows"], numbers["saa_convergence"]))
    for page_row, report_row in pairs:
        assert page_row["curse_gap"] == pytest.approx(report_row["curse_gap"], rel=1e-9)
    assert len(method["hull_bigm"]) == len(numbers["hull_vs_bigm"])
    assert method["settings"]["set_cap_share"] == 1.0 and method["settings"]["n_scenarios"] == 24


def test_bracket_hand_fixture():
    out = viz.bracket({"eev": 120.0, "rp": 160.0, "ws": 230.0, "vss": 40.0, "evpi": 70.0,
                       "vss_pct_of_rp": 25.0, "evpi_pct_of_rp": 43.75, "budget": 1.0}, step=50.0)
    assert out["axis_min"] == 50.0 and out["axis_max"] == 300.0
    assert out["positions"]["rp"] == pytest.approx(110.0 / 250.0)
    assert out["ticks"][0] == {"value": 50.0, "position": 0.0}
    assert out["ticks"][-1] == {"value": 300.0, "position": 1.0}


def test_slow_smoke_one_fresh_budget_matches_the_committed_cell(grid, problem):
    """The plan's build smoke test: a fresh $500 solve equals the committed grid and 801.95."""
    plans = viz.solve_grid(problem, budgets=(0.0, 500.0))
    eval_demand, fingerprint = viz._inputs(problem)
    fresh = viz.bundle(problem, plans, (0.0, 500.0), fingerprint, eval_demand)[viz.GRID_PATH]
    fresh_cell = fresh["cells"][1]["policies"]["optimizer"]
    position = grid["budgets"].index(500.0)
    committed = grid["cells"][position]["policies"]["optimizer"]
    tolerance = grid["cells"][position]["tolerance"]
    assert abs(fresh_cell["in_sample"] - committed["in_sample"]) <= tolerance
    milp_500 = None
    greedy_rows = _numbers()["greedy_vs_milp"]["budgets"]
    for row in greedy_rows:
        if row["budget"] == 500.0:
            milp_500 = row["milp_objective"]
    assert format(milp_500, ".2f") == "801.95"
    assert abs(fresh_cell["in_sample"] - milp_500) <= tolerance


# ---- small cases: the derivation itself ----------------------------------------------

def test_bundle_on_a_small_case_without_unit_metadata():
    products, prices, demand = _random_case(4, 8, seed=3)
    small = {"products": products, "prices": prices, "demand": demand}
    budgets = (0.0, 60.0, 120.0)
    plans = viz.solve_grid(small, budgets=budgets, mip_gap=0)
    eval_demand = demand[:, :, ::-1].copy()
    values = viz.value_grid(small, eval_demand, plans)
    cells = viz.derive_cells(small, budgets, plans, values, mip_gap=0)
    rows, _, _ = viz.product_table(small)
    assert rows[0]["label"] == "Product 01"
    for cell in cells:
        optimizer = cell["policies"]["optimizer"]["in_sample"]
        assert cell["policies"]["greedy"]["in_sample"] <= optimizer + 1e-6
        assert cell["policies"]["today_prices"]["in_sample"] <= optimizer + 1e-6
    assert cells[2]["policies"]["optimizer"]["in_sample"] >= cells[1]["policies"]["optimizer"]["in_sample"] - 1e-6


def test_grid_facts_hand_fixture():
    budgets = (0.0, 100.0, 200.0, 300.0)
    plans = {"optimizer": [{"buy": {}, "solve_seconds": 1.0, "optimal": True},
                           {"buy": {"p01": 3}, "solve_seconds": 2.0, "optimal": True},
                           {"buy": {"p01": 1, "p02": 5}, "solve_seconds": 1.5, "optimal": True},
                           {"buy": {"p01": 2, "p02": 5}, "solve_seconds": 1.0, "optimal": True}],
             "today_prices": [{"solve_seconds": 0.5, "optimal": True}] * 4}
    in_values = (10.0, 50.0, 60.0, 100.0)
    values = {"optimizer": []}
    for value in in_values:
        values["optimizer"].append({"in_sample": value, "out_of_sample_p10": 0.0, "percentile_p10_linear": 0.0})
    cells = []
    for budget in budgets:
        cells.append({"budget": budget, "lead_vs_greedy_in_pct": budget / 100.0, "lead_vs_greedy_in": 1.0,
                      "lead_vs_greedy_out_pct": 1.0, "lead_vs_greedy_out": 1.0,
                      "policies": {"optimizer": {"rord_out": 0.5}, "greedy": {"rord_out": 0.4}}})
    old_default = viz.DEFAULT_BUDGET
    viz.DEFAULT_BUDGET = 100.0
    try:
        facts = viz.grid_facts(budgets, plans, values, cells)
    finally:
        viz.DEFAULT_BUDGET = old_default
    # neighbours of 100 are 10 and 60 -> line 35, error 15; of 200: 50 and 100 -> 75, error 15
    assert facts["max_interpolation_error"] == {"budget": 100.0, "error": 15.0}
    # p01 falls from 3 to 1 between 100 and 200: one non-nested step
    assert facts["non_nested_steps"] == 1
    assert facts["widest_lead_in"]["budget"] == 300.0
    assert facts["greedy_rord_out_wins"] == []
    assert facts["n_solves"] == 8 and facts["solve_seconds_total"] == pytest.approx(7.5)


# ---- the page kit contract -----------------------------------------------------------

def test_tokens_css_is_a_byte_copy_of_the_kit():
    digest = base64.b64encode(hashlib.sha256(viz.TOKENS_PATH.read_bytes()).digest()).decode("ascii")
    assert digest == viz.TOKENS_SHA256_B64
    if not KIT_TOKENS.is_file():
        pytest.skip("page kit not present (public clone); the pinned hash above still holds")
    assert viz.TOKENS_PATH.read_bytes() == KIT_TOKENS.read_bytes()


def test_control_static_check_catches_a_changed_tokens_file(tmp_path, monkeypatch):
    changed = tmp_path / "tokens.css"
    changed.write_bytes(viz.TOKENS_PATH.read_bytes() + b"\n")
    monkeypatch.setattr(viz, "TOKENS_PATH", changed)
    problems = viz.static_problems()
    assert "docs/tokens.css is not the page kit's tokens.css" in problems


def test_nojekyll_and_chartjs_tag_exactly_as_the_conventions(method):
    assert viz.NOJEKYLL_PATH.is_file()
    assert viz.CHART_JS_TAG == CONVENTIONS_TAG
    assert method["chartjs_tag"] == CONVENTIONS_TAG
    html = viz.INDEX_PATH.read_text(encoding="utf-8")
    assert html.count(CONVENTIONS_TAG) == 1
    assert html.count("<script src=") == 2   # Chart.js and app.js, nothing else
    assert '<script src="app.js"></script>' in html
    assert html.count("chart unavailable, table shown") == 2


def test_static_checks_pass_and_data_fit_the_budget():
    assert viz.static_problems() == []
    total = 0
    for path in viz.DATA_PATHS:
        total += path.stat().st_size
    assert total <= viz.DATA_BUDGET_BYTES == 600_000


def test_page_text_comes_from_the_one_why_dict_and_every_hook_exists(method):
    html = viz.INDEX_PATH.read_text(encoding="utf-8")
    for key in viz.PAGE_WHY_KEYS:
        assert method["why"][key] == WHY[key]
        assert 'data-why="' + key + '"' in html
    hooks = html.split('data-prov="')
    for chunk in hooks[1:]:
        assert chunk.split('"')[0] in method["provenance"]
    assert method["footer"] == viz.FOOTER
    assert method["footer"] == ("Every number was computed by this repo's tested Python and copied; "
                                "the page computes nothing.")


def _strings(obj, out):
    if isinstance(obj, dict):
        values = list(obj.values())
        for value in values:
            _strings(value, out)
    elif isinstance(obj, list):
        for item in obj:
            _strings(item, out)
    elif isinstance(obj, str):
        out.append(obj)


def test_page_data_carry_no_forbidden_keys_unit_keys_or_row_dumps(files, problem):
    manifest = _numbers()["dataset_manifest"]
    unit_keys = list(problem["products"].index)
    assert manifest["seed"] == 19
    for path in viz.DATA_PATHS:
        data = files[path]
        keys = set()
        viz.collect_keys(data, keys)
        for forbidden in viz.FORBIDDEN_KEYS:
            assert forbidden not in keys, path.name
        strings = []
        _strings(data, strings)
        for text in strings:
            for unit in unit_keys:
                assert unit not in text, path.name
        assert viz.longest_list(data) < manifest["n_events"], path.name
    for row in files[viz.GRID_PATH]["products"]:
        assert sorted(row.keys()) == sorted(viz.PRODUCT_KEYS)


def test_control_key_walk_and_list_length_find_planted_items():
    keys = set()
    viz.collect_keys({"a": [{"b": {"cost_source": 1}}]}, keys)
    assert "cost_source" in keys
    assert viz.longest_list({"a": [{"b": list(np.arange(5))}]}) == 5
