"""The Restock Explorer page data: python -m experiments.viz build [--check].

Run from the repo root, like every experiment (it lives in experiments/, not in the
installed `milo` package, so `pip install .` never ships a module that imports
experiments/). Writes:

  docs/data/budget_grid.json  every policy's plan at each of the 36 solved budgets
                              ($0-$500 in $50 steps, then $100 steps to $3,000), its
                              in-sample and out-of-sample value, profit per restock
                              dollar, and the buy list the page shows
  docs/data/method.json       the EEV -> RP -> WS bracket, hull vs big-M, the README
                              headline, grid facts, why-box text, provenance labels,
                              the footer and the page settings
  docs/data/saa.json          the "is 24 scenarios enough?" rows and the pinned seed

Policies (the same three as the scorecard, experiments/scorecard.py):
  optimizer     the MILP: buys and prices chosen jointly;
  greedy        the careful greedy list (experiments/greedy_baseline.py);
  today_prices  the fair baseline: today's prices held, buys re-optimized by the MILP.

Headline per policy p and budget B, profit per restock dollar (RoRD):
    RoRD_p(B) = (V_p(B) - V_p(0)) / spend_p(B),  None ("n/a") when nothing is spent,
valued in-sample on the pinned 24 scenarios (seed 11) and out-of-sample on 1,000 fresh
scenarios (eval seed 999). Quantiles (p10/p50/p90) are computed here with
simulate._weighted_quantile, never in the browser, because a JavaScript percentile
interpolates and would print different numbers.

The page (docs/index.html + docs/app.js) computes nothing: it looks values up by
budget and policy and formats them. So the tests are in Python (tests/test_viz.py).

The build solves 72 MILPs (about 3 minutes) and never runs in CI; `build --reuse-plans`
rewrites the files from the committed plans after a text or report change. --check solves
nothing: it re-runs the deterministic greedy list at every budget (it must equal the
stored one), re-values the committed plans on regenerated inputs, re-derives every
number and compares it to the files at the JSON's 10-significant-digit rounding, then
checks docs/tokens.css against the kit hash, docs/.nojekyll, the 600 KB data budget,
each file's key allowlist and the Chart.js tag in docs/index.html.
Failure mode: --check proves the JSON is fresh for the committed plans, not that a new
HiGHS would pick the same plans (rerun the build for that), and not that the page reads
the JSON correctly (the screenshot click-through covers that, outside CI).
"""
import argparse
import base64
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments import common, greedy_baseline, saa_convergence, scorecard
from milo import jsonout, ladder
from milo.pricing import PRICE_LEVELS
from milo.simulate import _weighted_quantile
from milo.why import WHY

REPO = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO / "docs"
DATA_DIR = DOCS_DIR / "data"
GRID_PATH = DATA_DIR / "budget_grid.json"
METHOD_PATH = DATA_DIR / "method.json"
SAA_PATH = DATA_DIR / "saa.json"
DATA_PATHS = (GRID_PATH, METHOD_PATH, SAA_PATH)
TOKENS_PATH = DOCS_DIR / "tokens.css"
NOJEKYLL_PATH = DOCS_DIR / ".nojekyll"
INDEX_PATH = DOCS_DIR / "index.html"
APP_PATH = DOCS_DIR / "app.js"

# Plan, Shared standards (d): milo's page data stay under 600 KB.
DATA_BUDGET_BYTES = 600_000
# sha256 of the portfolio page kit's tokens.css, base64 like an SRI hash (a 64-hex
# digest would trip the public privacy test). docs/tokens.css must be a byte copy.
TOKENS_SHA256_B64 = "J3texCt1out7QFMk3jA+rC1vsZ5U8WlE3PnDZFY8OO8="
CHART_JS_TAG = (
    '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js" '
    'integrity="sha512-CQBWl4fJHWbryGE+Pc7UAxWMUMNMWzWxF4SQo9CgkJIN1kx6djDQZjh3Y8SZ1d+6I+1zze6Z7k'
    'HXO7q3UyZAWw==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>'
)
FOOTER = "Every number was computed by this repo's tested Python and copied; the page computes nothing."
TITLE = "Restock Explorer"


def _grid_budgets():
    """$0-$500 in $50 steps (where the plan changes fastest), then $100 steps to $3,000."""
    budgets = []
    fine = list(range(0, 501, 50))
    for value in fine:
        budgets.append(float(value))
    coarse = list(range(600, 3001, 100))
    for value in coarse:
        budgets.append(float(value))
    return tuple(budgets)


GRID_BUDGETS = _grid_budgets()
DEFAULT_BUDGET = scorecard.HEADLINE_BUDGET
TICK_LABEL_BUDGETS = (0.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0)
POLICIES = scorecard.POLICIES
POLICY_LABELS = scorecard.POLICY_LABELS
POLICY_SHORT_LABELS = {"optimizer": "Optimizer", "greedy": "Greedy list", "today_prices": "Today's prices"}
PAGE_WHY_KEYS = ("rord", "fair_baseline", "curse_gap", "budget_curve", "buy_list", "evpi_vss", "saa",
                 "hull_bigm")
BRACKET_STEP = 50.0      # tick spacing of the EEV -> RP -> WS axis, dollars

TYPE_LABELS = {
    "booster_box": "Booster box",
    "booster_bundle": "Booster bundle",
    "booster_pack": "Booster pack",
    "collection": "Collection box",
    "etb": "Elite trainer box",
    "tin": "Tin",
    "single": "Singles",
}
BAND_LABELS = {"<5": "under $5", "5-20": "$5-20", "20+": "$20+"}

# Key allowlists (privacy layer, Shared standards (c) 6): a new key must be added here
# on purpose, so a cost source or a store id cannot slip into the page data.
FORBIDDEN_KEYS = ("tcg_id", "sku", "cost_source")
GRID_KEYS = ("budgets", "cells", "generated_by", "input_fingerprint", "plans", "products")
METHOD_KEYS = ("bracket", "chartjs_tag", "dataset", "default_budget", "footer", "generated_by",
               "grid_facts", "headline", "hull_bigm", "label", "policies", "policy_labels",
               "policy_short_labels", "provenance", "settings", "tick_budgets", "title",
               "tokens_css_sha256_base64", "why")
SAA_KEYS = ("curse_gap_shape", "generated_by", "pinned", "provenance", "rows", "seed_spread")
PRODUCT_KEYS = ("cost", "id", "label", "on_hand", "price_now")


# ---- products ----------------------------------------------------------------------

def product_label(unit_id, product_type, set_code):
    """A readable synthetic label, e.g. "Booster box · Shadow Summit" or "Singles $5-20 · Lunar Bloom"."""
    type_label = TYPE_LABELS.get(product_type, str(product_type))
    if product_type == "single":
        band = str(unit_id).split("|")[-1]
        type_label = type_label + " " + BAND_LABELS.get(band, band)
    return type_label + " · " + str(set_code).title()


def product_table(problem):
    """(rows, id_by_unit, unit_by_id): one row per optimizer product, short ids p01, p02, ...

    Short ids keep the JSON small and keep the pipeline's unit keys out of the page.
    Raises if two products would share a label (the buy list would be ambiguous).
    """
    products = problem["products"]
    now = ladder.today_levels(problem["prices"])
    price = {}
    price_rows = problem["prices"][["i", "k", "price"]].itertuples(index=False)
    for row in price_rows:
        price[(row.i, int(row.k))] = float(row.price)
    units = problem.get("units")
    types = {}
    sets = {}
    if units is not None:
        unit_rows = units[["unit", "product_type", "set_code"]].itertuples(index=False)
        for row in unit_rows:
            types[row.unit] = row.product_type
            sets[row.unit] = row.set_code
    rows = []
    id_by_unit = {}
    unit_by_id = {}
    labels = set()
    unit_ids = list(products.index)
    numbered = list(enumerate(unit_ids, start=1))
    for number, unit in numbered:
        short = "p" + format(number, "02d")
        if unit in types:
            label = product_label(unit, types[unit], sets[unit])
        else:
            label = "Product " + format(number, "02d")
        if label in labels:
            raise ValueError("two products share the label " + label)
        labels.add(label)
        price_now = None
        if unit in now:
            price_now = price[(unit, now[unit])]
        rows.append({"id": short, "label": label, "on_hand": float(products.at[unit, "on_hand"]),
                     "cost": float(products.at[unit, "cost"]), "price_now": price_now})
        id_by_unit[unit] = short
        unit_by_id[short] = unit
    return rows, id_by_unit, unit_by_id


def _to_short(mapping, id_by_unit):
    out = {}
    pairs = list(mapping.items())
    for unit, value in pairs:
        out[id_by_unit[unit]] = int(value)
    return dict(sorted(out.items()))


def _to_unit(mapping, unit_by_id):
    out = {}
    pairs = list(mapping.items())
    for short, value in pairs:
        out[unit_by_id[short]] = int(value)
    return out


# ---- solving (build only) ----------------------------------------------------------

def _solved_record(result, id_by_unit):
    buy, levels = common.plan_maps(result)
    nonzero = {}
    pairs = list(buy.items())
    for unit, units in pairs:
        if units > 0:
            nonzero[unit] = units
    return {"buy": _to_short(nonzero, id_by_unit), "levels": _to_short(levels, id_by_unit),
            "spend": float(result["buy"]["spend"].sum()), "solver_objective": float(result["objective"]),
            "objective_bound": float(result["objective_bound"]), "optimal": bool(result["optimal"]),
            "solve_seconds": float(result["solve_seconds"])}


def _greedy_record(buy, levels, spend, id_by_unit):
    nonzero = {}
    pairs = list(buy.items())
    for unit, units in pairs:
        if units > 0:
            nonzero[unit] = units
    return {"buy": _to_short(nonzero, id_by_unit), "levels": _to_short(levels, id_by_unit),
            "spend": float(spend), "solver_objective": None, "objective_bound": None, "optimal": None,
            "solve_seconds": None}


def solve_grid(problem, budgets=GRID_BUDGETS, log_rows=None, mip_gap=common.EXPERIMENT_GAP,
               time_limit=common.EXPERIMENT_TIME_LIMIT):
    """Dict policy -> list of plan records, one per budget (the heavy part: 2 MILPs per budget)."""
    if log_rows is None:
        log_rows = []
    _, id_by_unit, _ = product_table(problem)
    products = problem["products"]
    prices = problem["prices"]
    demand_array = problem["demand"]
    now_prices, now_demand = scorecard.today_problem(prices, demand_array, products)
    plans = {}
    for policy in POLICIES:
        plans[policy] = []
    for budget in budgets:
        label = "B=" + format(budget, ".0f")
        result = ladder.solve_with_rules(products, prices, demand_array, budget=budget,
                                         mip_gap=mip_gap, time_limit=time_limit)
        common.log_solve(log_rows, "explorer", "optimizer " + label, result, demand_array.shape[2], budget)
        plans["optimizer"].append(_solved_record(result, id_by_unit))
        buy, levels, spend = greedy_baseline.greedy_plan(products, prices, demand_array, budget)
        plans["greedy"].append(_greedy_record(buy, levels, spend, id_by_unit))
        today = ladder.solve_with_rules(products, now_prices, now_demand, budget=budget,
                                        mip_gap=mip_gap, time_limit=time_limit)
        common.log_solve(log_rows, "explorer", "today_prices " + label, today, now_demand.shape[2], budget)
        plans["today_prices"].append(_solved_record(today, id_by_unit))
    return plans


# ---- valuing and deriving (build and --check) --------------------------------------

def value_plan(problem, demand_array, record, unit_by_id):
    """(expected profit, per-scenario profit) of a stored plan on the given scenarios."""
    buy = _to_unit(record["buy"], unit_by_id)
    levels = _to_unit(record["levels"], unit_by_id)
    value, profit = common.expected_value(problem["products"], problem["prices"], demand_array, buy, levels)
    return value, profit


def value_grid(problem, eval_demand, plans):
    """Every stored plan valued in-sample and out-of-sample (no solves)."""
    _, _, unit_by_id = product_table(problem)
    uniform = np.full(eval_demand.shape[2], 1.0 / eval_demand.shape[2])
    values = {}
    for policy in POLICIES:
        rows = []
        for record in plans[policy]:
            in_value, _ = value_plan(problem, problem["demand"], record, unit_by_id)
            out_value, out_profit = value_plan(problem, eval_demand, record, unit_by_id)
            rows.append({"in_sample": in_value, "out_of_sample": out_value,
                         "out_of_sample_p10": _weighted_quantile(out_profit, uniform, 0.10),
                         "out_of_sample_p50": _weighted_quantile(out_profit, uniform, 0.50),
                         "out_of_sample_p90": _weighted_quantile(out_profit, uniform, 0.90),
                         "percentile_p10_linear": float(np.percentile(out_profit, 10))})
        values[policy] = rows
    return values


def price_lookups(problem):
    """Dicts (unit, k) -> shelf price and (unit, k) -> ratio to today's price."""
    price = {}
    ratio = {}
    price_rows = problem["prices"][["i", "k", "ratio", "price"]].itertuples(index=False)
    for row in price_rows:
        price[(row.i, int(row.k))] = float(row.price)
        ratio[(row.i, int(row.k))] = float(row.ratio)
    return price, ratio


def buy_list(problem, record, products_rows, unit_by_id, price, ratio):
    """The rows the page lists for one plan: every product bought, then every price move.

    A row per product that is bought (units > 0) or that is stocked and moved off today's
    price. Spend = units x unit cost; the price is the chosen level's shelf price. The
    page looks the label, on-hand count and today's price up in the product table.
    Sorted by spend (largest first), then label. Stocked products that are neither bought
    nor repriced are counted in the summary, not listed.
    Failure mode: a product outside the model (no level passed the M1 filter) keeps
    today's price and is never listed, which matches how the solver counts it.
    """
    buy_units = _to_unit(record["buy"], unit_by_id)
    levels = _to_unit(record["levels"], unit_by_id)
    edited = set(ladder.count_price_edits(problem["products"], problem["prices"], buy_units, levels))
    edges = ladder.grid_edge_moves(problem["products"], problem["prices"], buy_units, levels)
    rows = []
    kept_today = 0
    units_bought = 0
    for product in products_rows:
        unit = unit_by_id[product["id"]]
        units = int(buy_units.get(unit, 0))
        stock = product["on_hand"] + units
        if stock <= 0 or unit not in levels:
            continue
        moved = unit in edited
        if units == 0 and not moved:
            kept_today += 1
            continue
        level = levels[unit]
        chosen_price = price[(unit, level)]
        change = None
        if product["price_now"]:
            change = (chosen_price / product["price_now"] - 1.0) * 100.0
        units_bought += units
        rows.append({"product": product["id"], "buy": units, "spend": units * product["cost"],
                     "ratio": ratio[(unit, level)], "price": chosen_price,
                     "price_change_pct": change, "price_moved": moved, "_label": product["label"]})
    ordered = sorted(rows, key=_row_order)
    for row in ordered:
        del row["_label"]
    products_bought = 0
    for row in ordered:
        if row["buy"] > 0:
            products_bought += 1
    summary = {"products_bought": products_bought, "units_bought": units_bought,
               "price_edits": len(edited), "price_edits_at_top_edge": edges["moved_top"],
               "price_edits_at_bottom_edge": edges["moved_bottom"],
               "stocked_at_today_price": kept_today, "spend": record["spend"]}
    return ordered, summary


def _row_order(row):
    return (-row["spend"], row["_label"])


def derive_cells(problem, budgets, plans, values, mip_gap=common.EXPERIMENT_GAP):
    """One cell per budget: every policy's metrics, the leads and the buy lists (arithmetic only).

    Per policy: rord_in / rord_out = scorecard.rord(V(B), V(0), spend), baseline the same
    policy's budget-0 plan on the same scenarios (n/a at zero spend); curse_gap = V_in - V_out
    of the same plan, no sign guarantee. Leads (optimizer minus a baseline):
      lead_vs_greedy_in / _out = V_opt - V_greedy in dollars, and _pct = that / |V_opt| x 100;
      lead_vs_today_out = V_opt,out - V_today,out, against the fair today's-prices baseline;
      tolerance = mip_gap x |bound| of the optimizer's solve, the slack every in-sample
      comparison allows.
    Failure modes: in-sample leads flatter the optimizer (it was fitted to those scenarios);
    out-of-sample leads are on the model's own generator and carry evaluation noise (the
    scorecard prints their paired SE); a lead built on moves at the grid edge
    (price_edits_at_top_edge) is bounded by the price grid, not by the data.
    """
    products_rows, _, unit_by_id = product_table(problem)
    price, ratio = price_lookups(problem)
    cells = []
    positions = list(range(len(budgets)))
    for position in positions:
        cell = {"budget": float(budgets[position]), "policies": {}, "buy_lists": {}}
        for policy in POLICIES:
            record = plans[policy][position]
            value = values[policy][position]
            base = values[policy][0]
            spend = record["spend"]
            rows, summary = buy_list(problem, record, products_rows, unit_by_id, price, ratio)
            cell["policies"][policy] = {
                "spend": spend, "in_sample": value["in_sample"], "out_of_sample": value["out_of_sample"],
                "out_of_sample_p10": value["out_of_sample_p10"], "out_of_sample_p50": value["out_of_sample_p50"],
                "out_of_sample_p90": value["out_of_sample_p90"],
                "base_in_sample": base["in_sample"], "base_out_of_sample": base["out_of_sample"],
                "rord_in": scorecard.rord(value["in_sample"], base["in_sample"], spend),
                "rord_out": scorecard.rord(value["out_of_sample"], base["out_of_sample"], spend),
                "curse_gap": value["in_sample"] - value["out_of_sample"],
                "solver_objective": record["solver_objective"], "optimal": record["optimal"],
                "products_bought": summary["products_bought"], "units_bought": summary["units_bought"],
                "price_edits": summary["price_edits"],
                "price_edits_at_top_edge": summary["price_edits_at_top_edge"],
            }
            cell["buy_lists"][policy] = {"rows": rows, "summary": summary}
        optimizer = cell["policies"]["optimizer"]
        greedy = cell["policies"]["greedy"]
        today = cell["policies"]["today_prices"]
        bound = plans["optimizer"][position]["objective_bound"]
        cell["tolerance"] = ladder.rung_tolerance(mip_gap, bound)
        cell["lead_vs_greedy_in"] = optimizer["in_sample"] - greedy["in_sample"]
        cell["lead_vs_greedy_in_pct"] = cell["lead_vs_greedy_in"] / abs(optimizer["in_sample"]) * 100.0
        cell["lead_vs_greedy_out"] = optimizer["out_of_sample"] - greedy["out_of_sample"]
        cell["lead_vs_greedy_out_pct"] = cell["lead_vs_greedy_out"] / abs(optimizer["out_of_sample"]) * 100.0
        cell["lead_vs_today_out"] = optimizer["out_of_sample"] - today["out_of_sample"]
        cells.append(cell)
    return cells


def grid_facts(budgets, plans, values, cells):
    """Facts about the whole grid that the page and DECISIONS.md print (arithmetic only).

    Failure mode: every fact is about this one 36-budget grid on the pinned problem; a
    "widest lead" is the maximum over solved budgets only, and timings are one machine's.

    widest_lead_in / widest_lead_out: the budget (> 0) where the optimizer's lead over the
        greedy list is largest, in % of the optimizer's value;
    greedy_rord_out_wins / greedy_profit_out_wins: budgets where greedy's out-of-sample
        per-dollar return (or profit) beats the optimizer's (printed, never hidden);
    max_interpolation_error: the largest gap between a solved budget's in-sample value and
        the straight line through its two neighbours (what interpolating would get wrong);
    non_nested_steps: neighbouring budget pairs where some product's optimizer units FALL
        as the budget rises (why each budget stores its own buy list);
    quantile_method_difference_p10: linear-interpolated numpy p10 minus the model's
        weighted-quantile p10 at the default budget (why quantiles stay in Python);
    n_plans: stored plans the --check re-values (budgets x policies);
    optimizer_price_edits / _at_top_edge: the optimizer's recounted price moves summed
        over the grid and how many sit on the grid's upper edge (M2).
    """
    widest_in = None
    widest_out = None
    rord_wins = []
    profit_wins = []
    for cell in cells:
        if cell["budget"] <= 0:
            continue
        if widest_in is None or cell["lead_vs_greedy_in_pct"] > widest_in["pct"]:
            widest_in = {"budget": cell["budget"], "pct": cell["lead_vs_greedy_in_pct"],
                         "dollars": cell["lead_vs_greedy_in"]}
        if widest_out is None or cell["lead_vs_greedy_out_pct"] > widest_out["pct"]:
            widest_out = {"budget": cell["budget"], "pct": cell["lead_vs_greedy_out_pct"],
                          "dollars": cell["lead_vs_greedy_out"]}
        optimizer_rord = cell["policies"]["optimizer"]["rord_out"]
        greedy_rord = cell["policies"]["greedy"]["rord_out"]
        if optimizer_rord is not None and greedy_rord is not None and greedy_rord > optimizer_rord:
            rord_wins.append(cell["budget"])
        if cell["lead_vs_greedy_out"] < 0:
            profit_wins.append(cell["budget"])
    worst_interpolation = {"budget": None, "error": 0.0}
    inner = list(range(1, len(budgets) - 1))
    for position in inner:
        low = values["optimizer"][position - 1]["in_sample"]
        high = values["optimizer"][position + 1]["in_sample"]
        middle = values["optimizer"][position]["in_sample"]
        width = budgets[position + 1] - budgets[position - 1]
        share = (budgets[position] - budgets[position - 1]) / width
        line = low + share * (high - low)
        error = abs(middle - line)
        if error > worst_interpolation["error"]:
            worst_interpolation = {"budget": float(budgets[position]), "error": error}
    non_nested = 0
    steps = list(range(1, len(budgets)))
    for position in steps:
        before = plans["optimizer"][position - 1]["buy"]
        after = plans["optimizer"][position]["buy"]
        fell = False
        bought_before = list(before.items())
        for product, units in bought_before:
            if after.get(product, 0) < units:
                fell = True
        if fell:
            non_nested += 1
    quantile_difference = None
    if DEFAULT_BUDGET in budgets:
        default_value = values["optimizer"][list(budgets).index(DEFAULT_BUDGET)]
        quantile_difference = default_value["percentile_p10_linear"] - default_value["out_of_sample_p10"]
    seconds = []
    for policy in ("optimizer", "today_prices"):
        for record in plans[policy]:
            seconds.append(record["solve_seconds"])
    all_optimal = True
    for policy in ("optimizer", "today_prices"):
        for record in plans[policy]:
            if not record["optimal"]:
                all_optimal = False
    moves = 0
    moves_top = 0
    for cell in cells:
        moves += cell["policies"]["optimizer"].get("price_edits", 0)
        moves_top += cell["policies"]["optimizer"].get("price_edits_at_top_edge", 0)
    n_plans = 0
    policies = list(plans.keys())
    for policy in policies:
        n_plans += len(plans[policy])
    return {
        "n_budgets": len(budgets), "n_solves": len(seconds), "n_plans": n_plans,
        "optimizer_price_edits": moves, "optimizer_price_edits_at_top_edge": moves_top,
        "solve_seconds_min": min(seconds), "solve_seconds_max": max(seconds),
        "solve_seconds_total": float(sum(seconds)), "all_solves_optimal": all_optimal,
        "widest_lead_in": widest_in, "widest_lead_out": widest_out,
        "greedy_rord_out_wins": rord_wins, "greedy_profit_out_wins": profit_wins,
        "max_interpolation_error": worst_interpolation, "non_nested_steps": non_nested,
        "quantile_method_difference_p10": quantile_difference,
    }


# ---- method and SAA files (copied from reports/, plus layout fractions) -------------

def _load_report(name):
    with open(common.REPORTS_DIR / name, encoding="utf-8") as handle:
        return json.load(handle)


def bracket(evpi_vss, step=BRACKET_STEP):
    """EEV -> RP -> WS with each value's position on an axis (0 = left edge, 1 = right edge).

    The axis runs from one step below EEV (rounded down to the step) to one step above WS
    (rounded up), so the SVG needs no arithmetic beyond placing the fractions.
    """
    low = (np.floor(evpi_vss["eev"] / step) - 1.0) * step
    high = (np.ceil(evpi_vss["ws"] / step) + 1.0) * step
    span = high - low
    ticks = []
    count = int(round(span / step))
    marks = list(range(count + 1))
    for mark in marks:
        value = low + mark * step
        ticks.append({"value": float(value), "position": float((value - low) / span)})
    positions = {}
    for key in ("eev", "rp", "ws"):
        positions[key] = float((evpi_vss[key] - low) / span)
    return {"eev": evpi_vss["eev"], "rp": evpi_vss["rp"], "ws": evpi_vss["ws"],
            "vss": evpi_vss["vss"], "evpi": evpi_vss["evpi"],
            "vss_pct_of_rp": evpi_vss["vss_pct_of_rp"], "evpi_pct_of_rp": evpi_vss["evpi_pct_of_rp"],
            "budget": evpi_vss["budget"], "axis_min": float(low), "axis_max": float(high),
            "positions": positions, "ticks": ticks}


def _money0(value):
    return format(value, ",.0f")


def provenance(budgets, settings):
    """The pill text per widget, written from the settings so it cannot drift."""
    n_budgets = str(len(budgets))
    in_sample = (format(settings["n_scenarios"], "d") + " scenarios, seed " + format(settings["scenario_seed"], "d"))
    out_sample = format(settings["n_eval"], ",d") + " scenarios, eval seed " + format(settings["eval_seed"], "d")
    return {
        "slider": "snaps to " + n_budgets + " solved budgets",
        "tiles": "synthetic · out-of-sample: " + out_sample + " · in-sample: " + in_sample,
        "curve": "precomputed at " + n_budgets + " budgets · synthetic",
        "fallback": "the same numbers as the chart",
        "buy_list": "synthetic · the stored plan at this budget",
        "bracket": "in-sample · recorded run (reports/numbers.json)",
        "saa": "recorded run (reports/numbers.json) · " + str(len(saa_convergence.SEEDS)) + " seeds per N",
        "hull_bigm": "recorded run (reports/numbers.json) · one machine",
        "page": "synthetic data; the page calls no solver",
    }


def build_method(budgets, plans, values, cells):
    numbers = _load_report("numbers.json")
    card = _load_report("scorecard.json")
    settings = {"budgets": list(budgets), "n_scenarios": common.N_SCENARIOS,
                "scenario_seed": common.SCENARIO_SEED, "n_eval": saa_convergence.N_EVAL,
                "eval_seed": saa_convergence.EVAL_SEED, "mip_gap": common.EXPERIMENT_GAP,
                "time_limit": common.EXPERIMENT_TIME_LIMIT, "set_cap_share": 1.0,
                "price_grid_low": float(PRICE_LEVELS[0]), "price_grid_high": float(PRICE_LEVELS[-1])}
    why = {}
    for key in PAGE_WHY_KEYS:
        why[key] = WHY[key]
    hull_rows = []
    for entry in numbers["hull_vs_bigm"]:
        hull_rows.append({"budget": entry["budget"], "transform": entry["transform"],
                          "objective": entry["objective"], "lp_relaxation_objective": entry["lp_relaxation_objective"],
                          "build_seconds": entry["build_seconds"], "solve_seconds": entry["solve_seconds"],
                          "optimal": entry["optimal"]})
    manifest = numbers["dataset_manifest"]
    return {
        "title": TITLE, "footer": FOOTER, "chartjs_tag": CHART_JS_TAG,
        "tokens_css_sha256_base64": TOKENS_SHA256_B64,
        "generated_by": "python -m experiments.viz build (committed synthetic dataset)",
        "label": ("The pinned problem: no set-spend cap, " + format(common.N_SCENARIOS, "d")
                  + " training scenarios (seed " + format(common.SCENARIO_SEED, "d") + "), rel gap "
                  + format(common.EXPERIMENT_GAP, "g") + ". Not the demo's settings."),
        "dataset": {"data_dir": "data/synthetic", "seed": manifest["seed"], "n_products": manifest["n_products"]},
        "settings": settings, "default_budget": DEFAULT_BUDGET, "tick_budgets": list(TICK_LABEL_BUDGETS),
        "policies": list(POLICIES), "policy_labels": dict(POLICY_LABELS),
        "policy_short_labels": dict(POLICY_SHORT_LABELS),
        "provenance": provenance(budgets, settings), "why": why,
        "headline": card["headline"], "bracket": bracket(numbers["evpi_vss"]), "hull_bigm": hull_rows,
        "grid_facts": grid_facts(budgets, plans, values, cells),
    }


def curse_gap_shape(rows):
    """Is the mean curse gap monotone in N? Every step where it RISES as N grows.

    Rule: rows sorted by n_scenarios; a rise is curse_gap(N_next) > curse_gap(N). The page
    note quotes the endpoints, so it must also say when the path between them is not
    smooth. Failure mode: with 10 seeds per N a rise is seed noise as often as signal.
    """
    ordered = sorted(rows, key=_by_n)
    rises = []
    steps = list(range(1, len(ordered)))
    for position in steps:
        before = ordered[position - 1]
        after = ordered[position]
        if after["curse_gap"] > before["curse_gap"]:
            rises.append({"from_n": before["n_scenarios"], "to_n": after["n_scenarios"],
                          "from_gap": before["curse_gap"], "to_gap": after["curse_gap"]})
    return {"monotone": not rises, "rises": rises}


def _by_n(row):
    return row["n_scenarios"]


def build_saa(budgets):
    numbers = _load_report("numbers.json")
    card = _load_report("scorecard.json")
    head = card["headline"]
    seed = card["context"]["pinned_seed"]
    settings = {"n_scenarios": common.N_SCENARIOS, "scenario_seed": common.SCENARIO_SEED,
                "n_eval": saa_convergence.N_EVAL, "eval_seed": saa_convergence.EVAL_SEED}
    return {
        "generated_by": "python -m experiments.viz build, copied from reports/numbers.json and reports/scorecard.json",
        "rows": numbers["saa_convergence"], "seed_spread": numbers["rp_seed_spread"],
        "curse_gap_shape": curse_gap_shape(numbers["saa_convergence"]),
        "pinned": {"n_scenarios": common.N_SCENARIOS, "seed": common.SCENARIO_SEED,
                   "in_sample": head["optimizer_in_sample"], "out_of_sample": head["optimizer_out_of_sample"],
                   "curse_gap": head["curse_gap"], "budget": head["budget"],
                   "minus_seed_mean": seed["minus_seed_mean"], "in_seed_sds": seed["in_seed_sds"],
                   "rank_from_top": seed["rank_from_top"], "n_seeds": seed["n_seeds"]},
        "provenance": provenance(budgets, settings)["saa"],
    }


def _inputs(problem):
    eval_demand = common.make_demand(problem, saa_convergence.N_EVAL, saa_convergence.EVAL_SEED)
    fingerprint = jsonout.fingerprint(problem["products"], problem["prices"], [problem["demand"], eval_demand])
    return eval_demand, fingerprint


def bundle(problem, plans, budgets, fingerprint, eval_demand):
    """The three page files from stored or fresh plans (shared by build and --check)."""
    products_rows, _, _ = product_table(problem)
    values = value_grid(problem, eval_demand, plans)
    cells = derive_cells(problem, budgets, plans, values)
    grid = {"generated_by": "python -m experiments.viz build (committed synthetic dataset)",
            "input_fingerprint": fingerprint, "budgets": list(budgets), "products": products_rows,
            "plans": plans, "cells": cells}
    method = build_method(budgets, plans, values, cells)
    saa = build_saa(budgets)
    return {GRID_PATH: grid, METHOD_PATH: method, SAA_PATH: saa}


def build(problem=None, log_rows=None):
    """Solve the grid and return {path: dict} for the three page files (heavy: minutes)."""
    if problem is None:
        problem = common.load_problem()
    if saa_convergence.EVAL_SEED in scorecard.training_seeds():
        raise ValueError("eval seed must never be a training seed")
    eval_demand, fingerprint = _inputs(problem)
    plans = solve_grid(problem, log_rows=log_rows)
    return bundle(problem, plans, GRID_BUDGETS, fingerprint, eval_demand)


def rebundle(problem=None):
    """The three page files re-derived from the committed plans (no solves)."""
    if problem is None:
        problem = common.load_problem()
    stored = json.loads(GRID_PATH.read_text(encoding="utf-8"))
    eval_demand, fingerprint = _inputs(problem)
    if stored["input_fingerprint"] != fingerprint:
        raise ValueError("inputs changed since the committed build; run the full build")
    return bundle(problem, stored["plans"], tuple(stored["budgets"]), fingerprint, eval_demand)


# ---- checks ------------------------------------------------------------------------

def tokens_hash(path):
    digest = hashlib.sha256(Path(path).read_bytes()).digest()
    return base64.b64encode(digest).decode("ascii")


def collect_keys(obj, out):
    """Every dict key anywhere in obj, into the set out."""
    if isinstance(obj, dict):
        keys = list(obj.keys())
        for key in keys:
            out.add(str(key))
            collect_keys(obj[key], out)
    elif isinstance(obj, list):
        for item in obj:
            collect_keys(item, out)


def longest_list(obj):
    """Length of the longest list anywhere in obj (a page file must never hold a data-row dump)."""
    longest = 0
    if isinstance(obj, dict):
        values = list(obj.values())
        for value in values:
            longest = max(longest, longest_list(value))
    elif isinstance(obj, list):
        longest = len(obj)
        for item in obj:
            longest = max(longest, longest_list(item))
    return longest


def static_problems():
    """Checks that need no data rebuild: kit hash, .nojekyll, size, allowlists, Chart.js tag."""
    problems = []
    if not TOKENS_PATH.is_file() or tokens_hash(TOKENS_PATH) != TOKENS_SHA256_B64:
        problems.append("docs/tokens.css is not the page kit's tokens.css")
    if not NOJEKYLL_PATH.is_file():
        problems.append("docs/.nojekyll is missing")
    total = 0
    for path in DATA_PATHS:
        if not path.is_file():
            problems.append(str(path.relative_to(REPO)) + " is missing")
            continue
        total += path.stat().st_size
    if total > DATA_BUDGET_BYTES:
        problems.append("page data is " + str(total) + " bytes, over the " + str(DATA_BUDGET_BYTES) + " budget")
    allowlists = ((GRID_PATH, GRID_KEYS), (METHOD_PATH, METHOD_KEYS), (SAA_PATH, SAA_KEYS))
    for path, allowed in allowlists:
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        top = sorted(data.keys())
        if top != sorted(allowed):
            problems.append(path.name + " top-level keys differ from its allowlist")
        keys = set()
        collect_keys(data, keys)
        for forbidden in FORBIDDEN_KEYS:
            if forbidden in keys:
                problems.append(path.name + " carries the forbidden key " + forbidden)
    if not INDEX_PATH.is_file() or CHART_JS_TAG not in INDEX_PATH.read_text(encoding="utf-8"):
        problems.append("docs/index.html lacks the pinned Chart.js tag")
    return problems


def _close(a, b):
    if a is None or b is None:
        return a is None and b is None
    return jsonout.round_sig(a) == jsonout.round_sig(b) or abs(a - b) <= 1e-9 * max(1.0, abs(a))


def compare(stored, fresh, path, problems):
    """Append "path is stale" for every value of fresh that differs from stored."""
    if isinstance(fresh, dict):
        if not isinstance(stored, dict):
            problems.append(path + " is stale")
            return
        keys = list(fresh.keys())
        for key in keys:
            if key not in stored:
                problems.append(path + "." + str(key) + " missing")
                continue
            compare(stored[key], fresh[key], path + "." + str(key), problems)
        stored_keys = list(stored.keys())
        for key in stored_keys:
            if key not in fresh:
                problems.append(path + "." + str(key) + " is extra")
    elif isinstance(fresh, list):
        if not isinstance(stored, list) or len(stored) != len(fresh):
            problems.append(path + " length differs")
            return
        positions = list(range(len(fresh)))
        for position in positions:
            compare(stored[position], fresh[position], path + "[" + str(position) + "]", problems)
    elif isinstance(fresh, bool) or isinstance(stored, bool):
        if stored is not fresh and stored != fresh:
            problems.append(path + " is stale")
    elif isinstance(fresh, float) or (fresh is None and not isinstance(stored, (str, list, dict))):
        if not _close(stored, fresh):
            problems.append(path + " is stale")
    elif stored != fresh:
        problems.append(path + " is stale")


def load_files():
    files = {}
    for path in DATA_PATHS:
        files[path] = json.loads(path.read_text(encoding="utf-8"))
    return files


def check(stored=None, problem=None):
    """Re-value the committed plans and re-derive every page number; returns a list of problems.

    No MILP is solved. Failure mode: see the module docstring.
    """
    problems = static_problems()
    if stored is None:
        missing = False
        for path in DATA_PATHS:
            if not path.is_file():
                missing = True
        if missing:
            return problems
        stored = load_files()
    if problem is None:
        problem = common.load_problem()
    eval_demand, fingerprint = _inputs(problem)
    grid = stored[GRID_PATH]
    if grid["input_fingerprint"] != fingerprint:
        problems.append("input_fingerprint differs: inputs changed since the committed build")
    budgets = tuple(grid["budgets"])
    if budgets != GRID_BUDGETS:
        problems.append("budget_grid.json was built on a different budget grid")
    problems.extend(greedy_problems(problem, budgets, grid["plans"]["greedy"]))
    fresh = bundle(problem, grid["plans"], budgets, fingerprint, eval_demand)
    paths = list(fresh.keys())
    for path in paths:
        compare(stored[path], jsonout.rounded(fresh[path]), path.name, problems)
    return problems


def greedy_problems(problem, budgets, stored_plans):
    """Re-run the deterministic greedy list at every stored budget (well under a second).

    The page stores short product ids and only nonzero buys, so they are mapped back to
    unit ids first; then scorecard.greedy_problems compares buys and levels.
    """
    _, _, unit_by_id = product_table(problem)
    mapped = []
    for record in stored_plans:
        mapped.append({"buy": _to_unit(record["buy"], unit_by_id),
                       "levels": _to_unit(record["levels"], unit_by_id)})
    return scorecard.greedy_problems(problem, budgets, mapped, "budget_grid.json")


def write(files):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = list(files.keys())
    for path in paths:
        jsonout.dump(files[path], path)
    if not NOJEKYLL_PATH.exists():
        NOJEKYLL_PATH.write_text("", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", nargs="?", default="build", choices=("build",),
                        help="build the page data (the default)")
    parser.add_argument("--check", action="store_true",
                        help="re-value the committed plans and fail if any page number is stale; solves nothing")
    parser.add_argument("--reuse-plans", action="store_true",
                        help="rewrite the files from the committed plans (text or report changes); solves nothing")
    args = parser.parse_args(argv)
    if args.check:
        problems = check()
        for problem in problems:
            print("STALE:", problem)
        if problems:
            return 1
        print("docs/data is current")
        return 0
    started = time.perf_counter()
    log_rows = []
    if args.reuse_plans:
        files = rebundle()
    else:
        files = build(log_rows=log_rows)
    write(files)
    if log_rows:
        common.save_solve_log(log_rows)
    total = 0
    for path in DATA_PATHS:
        size = path.stat().st_size
        total += size
        print("wrote", path.relative_to(REPO), format(size, ",d"), "bytes")
    print("page data total " + format(total, ",d") + " bytes (budget " + format(DATA_BUDGET_BYTES, ",d") + ")")
    print("total wall clock: " + format(time.perf_counter() - started, ".1f") + " s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
