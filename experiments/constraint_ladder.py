"""Constraint ladder (U2-U4): what each business rule costs, in dollars and solve seconds.

All rungs run on the pinned problem (experiments/common.py: 24 scenarios, seed 11,
budget $1,500, no set cap, rel gap 1e-4, 120 s) and are also valued on the 1,000-scenario
evaluation set (eval seed 999). Output goes to its own file, reports/constraint_ladder.json,
so the frozen reports/numbers.json and the goldens never move.

Rungs (cumulative):
  R0a textbook   accounting   plan solved on list prices (no fees), valued at true fees
  R0b today      (base)       the existing model, fees in the objective
  R1  shelf      restriction  sum_i v_i (q_i + x_i) <= V, V = 70% of R0b's shelf use
  R2  price edits restriction at most P = 4 stocked products off today's price (on top of R1)

Each kind gets the test that fits it (milo/ladder.py): a restriction must not raise
profit (obj <= previous + mip_gap x |bound|); an accounting rung must be worth >= 0
(VSS-style). The existing rules (35% set cap, min order, integrality) are priced the
same way as restrictions against a model without them, and two sweeps price R1 and R2
alone against R0b: shelf at 50/70/90/100% of R0b's use (plus the value of 5 more space
units) and P = 0, 1, 2, 4, 8, all.

    python -m experiments.constraint_ladder               # 19 MILPs + 1 LP, about 95 s, never in CI
    python -m experiments.constraint_ladder --reuse-plans # re-derive from committed plans; no solves
    python -m experiments.constraint_ladder --check       # re-value committed plans; no solves
"""
import argparse
import json
import sys
import time

import numpy as np

from experiments import common, scorecard
from milo import jsonout, ladder
from milo.simulate import _weighted_quantile
from milo.why import WHY

LADDER_BUDGET = common.MAIN_BUDGET
SHELF_FRACTIONS = (0.5, 0.7, 0.9, 1.0)
SHELF_STEP = 5.0
LADDER_SHELF_FRACTION = 0.7
EDIT_LIMITS = (0, 1, 2, 4, 8)          # plus "all" (every stocked or orderable product)
LADDER_EDIT_LIMIT = 4
SET_CAP_SHARE = 0.35                   # the operating default in pipeline.run
LADDER_PATH = common.REPORTS_DIR / "constraint_ladder.json"


def ladder_problem(problem=None):
    """The pinned problem with the illustrative shelf space column attached."""
    if problem is None:
        problem = common.load_problem()
    out = dict(problem)
    out["products"] = ladder.with_space(problem["products"], problem["units"])
    return out


def _types(problem):
    types = {}
    rows = problem["units"][["unit", "product_type"]].itertuples(index=False)
    for row in rows:
        types[row.unit] = row.product_type
    return types


def _solve(problem, prices, rules, products=None, set_cap_share=1.0, lp_relaxation=False,
           log_rows=None, label=""):
    """One solve_with_rules call, stored as a plan record (no valuations yet)."""
    if products is None:
        products = problem["products"]
    result = ladder.solve_with_rules(products, prices, problem["demand"], rules=rules,
                                     budget=LADDER_BUDGET, set_cap_share=set_cap_share,
                                     mip_gap=common.EXPERIMENT_GAP,
                                     time_limit=common.EXPERIMENT_TIME_LIMIT,
                                     lp_relaxation=lp_relaxation)
    if log_rows is not None:
        common.log_solve(log_rows, "constraint_ladder", label, result,
                         problem["demand"].shape[2], LADDER_BUDGET)
    buy, levels = common.plan_maps(result)
    flag_codes = sorted(set(result["flags"]["code"].astype(str)))
    record = {
        "buy": dict(sorted(buy.items())), "levels": dict(sorted(levels.items())),
        "spend": float(result["buy"]["spend"].sum()),
        "solver_objective": float(result["objective"]),
        "objective_bound": float(result["objective_bound"]),
        "optimal": bool(result["optimal"]), "termination": result["termination"],
        "build_seconds": float(result["build_seconds"]),
        "solve_seconds": float(result["solve_seconds"]),
        "n_variables": int(result["n_variables"]),
        "n_integer_variables": int(result["n_integer_variables"]),
        "n_constraints": int(result["n_constraints"]),
        "flag_codes": flag_codes,
        "rules": {"shelf_capacity": rules.shelf_capacity, "price_edit_limit": rules.price_edit_limit,
                  "set_cap_share": float(set_cap_share)},
    }
    if lp_relaxation:
        record["lp_relaxation_objective"] = float(result["lp_relaxation_objective"])
    return record


def solve_all(problem, log_rows=None):
    """Every solve the ladder needs; returns dict name -> plan record."""
    prices = problem["prices"]
    records = {}
    records["R0a"] = _solve(problem, ladder.fee_blind_prices(prices), ladder.Rules(),
                            log_rows=log_rows, label="R0a textbook")
    records["R0b"] = _solve(problem, prices, ladder.Rules(), lp_relaxation=True,
                            log_rows=log_rows, label="R0b today")
    base_use = ladder.shelf_use(problem["products"], records["R0b"]["buy"])
    for fraction in SHELF_FRACTIONS:
        capacity = fraction * base_use
        name = "shelf_" + format(fraction, ".2f")
        records[name] = _solve(problem, prices, ladder.Rules(shelf_capacity=capacity),
                               log_rows=log_rows, label=name)
        records[name + "_plus"] = _solve(problem, prices,
                                         ladder.Rules(shelf_capacity=capacity + SHELF_STEP),
                                         log_rows=log_rows, label=name + " +" + format(SHELF_STEP, ".0f"))
    limits = list(EDIT_LIMITS)
    limits.append(all_edits(problem))
    for limit in limits:
        name = "edits_" + str(limit)
        records[name] = _solve(problem, prices, ladder.Rules(price_edit_limit=limit),
                               log_rows=log_rows, label=name)
    records["R2"] = _solve(problem, prices,
                           ladder.Rules(shelf_capacity=LADDER_SHELF_FRACTION * base_use,
                                        price_edit_limit=LADDER_EDIT_LIMIT),
                           log_rows=log_rows, label="R2 shelf + edits")
    records["set_cap"] = _solve(problem, prices, ladder.Rules(), set_cap_share=SET_CAP_SHARE,
                                log_rows=log_rows, label="set cap 0.35")
    no_min = problem["products"].copy()
    no_min["min_order"] = 1
    records["no_min_order"] = _solve(problem, prices, ladder.Rules(), products=no_min,
                                     log_rows=log_rows, label="min order off")
    return records


def all_edits(problem):
    """P large enough to never bind: every product that is stocked or could be bought."""
    return int(len(problem["products"].index))


def _value(problem, demand_array, record):
    value, profit = common.expected_value(problem["products"], problem["prices"], demand_array,
                                          record["buy"], record["levels"])
    probs = np.full(len(profit), 1.0 / len(profit))
    return value, _weighted_quantile(profit, probs, 0.10)


def valuations(problem, eval_demand, records):
    """Per record: in-sample and out-of-sample value at the TRUE fees, worst-10%, recounts."""
    types = _types(problem)
    out = {}
    names = sorted(records.keys())
    for name in names:
        record = records[name]
        in_value, in_p10 = _value(problem, problem["demand"], record)
        out_value, _ = _value(problem, eval_demand, record)
        by_type = {}
        bought = list(record["buy"].items())
        for unit, units in bought:
            if units > 0:
                by_type[types[unit]] = by_type.get(types[unit], 0) + int(units)
        edited = ladder.count_price_edits(problem["products"], problem["prices"],
                                          record["buy"], record["levels"])
        edges = ladder.grid_edge_moves(problem["products"], problem["prices"], record["buy"],
                                       record["levels"])
        out[name] = {"in_sample": in_value, "in_sample_p10": in_p10, "out_of_sample": out_value,
                     "shelf_used": ladder.shelf_use(problem["products"], record["buy"]),
                     "price_edits": len(edited), "units_by_type": dict(sorted(by_type.items())),
                     "edges": edges}
    gross, _ = common.expected_value(problem["products"], ladder.fee_blind_prices(problem["prices"]),
                                     problem["demand"], records["R0a"]["buy"], records["R0a"]["levels"])
    out["R0a"]["own_objective"] = gross
    return out


def _row(rung, rule, kind, previous, name, records, values):
    """One ladder row: values, the price (restriction) or value (accounting) vs previous."""
    record = records[name]
    value = values[name]
    row = {"rung": rung, "rule": rule, "kind": kind, "source": name, "previous": previous,
           "expected_profit": value["in_sample"], "out_of_sample": value["out_of_sample"],
           "worst_10pct": value["in_sample_p10"], "shelf_used": value["shelf_used"],
           "price_edits": value["price_edits"], "spend": record["spend"],
           "solve_seconds": record["solve_seconds"], "build_seconds": record["build_seconds"],
           "optimal": record["optimal"], "n_variables": record["n_variables"],
           "n_integer_variables": record["n_integer_variables"],
           "n_constraints": record["n_constraints"], "units_by_type": value["units_by_type"],
           "price_of_rule": None, "price_of_rule_pct_of_rp": None, "value_of_modeling": None,
           "binds": None, "price_moves_at_top_edge": value["edges"]["moved_top"]}
    return row


def shelf_label(fraction):
    """"70%" from a shelf fraction, so every rule label follows LADDER_SHELF_FRACTION."""
    return format(fraction * 100.0, ".0f") + "%"


def min_order_facts(problem, records):
    """Does the all-or-nothing minimum order bind at all on this data?

    Counts the products whose min_order is above 1 (min_order = 1 is the same as no
    rule) and how many of them each plan buys. Failure mode guarded: a 0.00 price of the
    rule reads like a measured null, but when no plan buys a product with a real
    minimum the rung cannot show anything; these counts say which case it is.
    """
    products = problem["products"]
    binding = []
    product_ids = list(products.index)
    for i in product_ids:
        if int(products.at[i, "min_order"]) > 1:
            binding.append(i)
    bought_with = 0
    bought_without = 0
    for i in binding:
        if int(records["R0b"]["buy"].get(i, 0)) > 0:
            bought_with += 1
        if int(records["no_min_order"]["buy"].get(i, 0)) > 0:
            bought_without += 1
    same = records["R0b"]["buy"] == records["no_min_order"]["buy"]
    return {"products_with_min_order_above_1": len(binding), "n_products": len(product_ids),
            "bought_with_rule": bought_with, "bought_without_rule": bought_without,
            "identical_buys": bool(same)}


def derive(problem, records, values):
    """Ladder table, each-rule-alone table, existing-rule prices and both sweeps."""
    rp = values["R0b"]["in_sample"]
    base_use = values["R0b"]["shelf_used"]
    ladder_shelf = "shelf_" + format(LADDER_SHELF_FRACTION, ".2f")
    rows = []
    r0a = _row("R0a", "textbook: list price, no fees", "accounting", None, "R0a", records, values)
    r0a["own_objective"] = values["R0a"]["own_objective"]
    r0a["overstatement"] = values["R0a"]["own_objective"] - values["R0a"]["in_sample"]
    rows.append(r0a)
    r0b = _row("R0b", "+ store commission, card fee, per-order fee (today)", "accounting", "R0a",
               "R0b", records, values)
    r0b["value_of_modeling"] = values["R0b"]["in_sample"] - values["R0a"]["in_sample"]
    rows.append(r0b)
    r1 = _row("R1", "+ shelf space (" + shelf_label(LADDER_SHELF_FRACTION) + " of today's plan's use)",
              "restriction", "R0b", ladder_shelf, records, values)
    rows.append(r1)
    r2 = _row("R2", "+ weekly price-edit limit (P = " + str(LADDER_EDIT_LIMIT) + ")", "restriction",
              "R1", "R2", records, values)
    rows.append(r2)
    by_rung = {}
    for row in rows:
        by_rung[row["rung"]] = row
    for row in rows:
        if row["kind"] == "restriction":
            before = by_rung[row["previous"]]["expected_profit"]
            row["price_of_rule"] = before - row["expected_profit"]
            row["price_of_rule_pct_of_rp"] = row["price_of_rule"] / abs(rp) * 100.0
    r1["binds"] = bool(r1["price_of_rule"] > ladder.rung_tolerance(common.EXPERIMENT_GAP,
                                                                    records["R0b"]["objective_bound"]))
    r2["binds"] = bool(values["R2"]["price_edits"] >= LADDER_EDIT_LIMIT)

    alone = []
    alone_cases = [("R1", "shelf space alone (" + shelf_label(LADDER_SHELF_FRACTION) + ")", ladder_shelf),
                   ("R2", "price-edit limit alone (P = " + str(LADDER_EDIT_LIMIT) + ")",
                    "edits_" + str(LADDER_EDIT_LIMIT))]
    for rung, rule, name in alone_cases:
        price = rp - values[name]["in_sample"]
        alone.append({"rung": rung, "rule": rule, "source": name,
                      "expected_profit": values[name]["in_sample"],
                      "out_of_sample": values[name]["out_of_sample"],
                      "price_of_rule": price, "price_of_rule_pct_of_rp": price / abs(rp) * 100.0,
                      "solve_seconds": records[name]["solve_seconds"]})

    existing = []
    set_cap_price = rp - values["set_cap"]["in_sample"]
    existing.append({"rule": "35% per-set spend cap", "kind": "restriction", "with": "set_cap",
                     "without": "R0b", "price_of_rule": set_cap_price,
                     "price_of_rule_pct_of_rp": set_cap_price / abs(rp) * 100.0})
    min_price = values["no_min_order"]["in_sample"] - rp
    min_row = {"rule": "all-or-nothing minimum order", "kind": "restriction", "with": "R0b",
               "without": "no_min_order", "price_of_rule": min_price,
               "price_of_rule_pct_of_rp": min_price / abs(rp) * 100.0}
    facts = min_order_facts(problem, records)
    keys = list(facts.keys())
    for key in keys:
        min_row[key] = facts[key]
    existing.append(min_row)
    # Rounded first so the committed file (10 significant digits) re-derives exactly.
    lp_price = jsonout.round_sig(records["R0b"]["lp_relaxation_objective"]) - rp
    existing.append({"rule": "integrality (LP relaxation: integer units and every disjunct binary relaxed)",
                     "kind": "restriction", "with": "R0b", "without": "LP relaxation",
                     "price_of_rule": lp_price, "price_of_rule_pct_of_rp": lp_price / abs(rp) * 100.0})

    shelf_rows = []
    for fraction in SHELF_FRACTIONS:
        name = "shelf_" + format(fraction, ".2f")
        capacity = records[name]["rules"]["shelf_capacity"]
        plus = values[name + "_plus"]["in_sample"]
        tolerance = 2.0 * ladder.rung_tolerance(common.EXPERIMENT_GAP,
                                                records[name + "_plus"]["objective_bound"])
        slope = (plus - values[name]["in_sample"]) / SHELF_STEP
        shelf_rows.append({"value_exceeds_noise": bool(slope > tolerance / SHELF_STEP),
                           "fraction_of_today_use": fraction, "capacity": capacity,
                           "expected_profit": values[name]["in_sample"],
                           "out_of_sample": values[name]["out_of_sample"],
                           "price_of_rule": rp - values[name]["in_sample"],
                           "shelf_used": values[name]["shelf_used"],
                           "value_per_space_unit": slope,
                           "value_tolerance_per_unit": tolerance / SHELF_STEP,
                           "units_by_type": values[name]["units_by_type"],
                           "solve_seconds": records[name]["solve_seconds"],
                           "optimal": records[name]["optimal"]})
    edit_rows = []
    limits = list(EDIT_LIMITS)
    limits.append(all_edits(problem))
    for limit in limits:
        name = "edits_" + str(limit)
        edit_rows.append({"limit": int(limit), "is_all": bool(limit == all_edits(problem)),
                          "expected_profit": values[name]["in_sample"],
                          "out_of_sample": values[name]["out_of_sample"],
                          "price_of_rule": rp - values[name]["in_sample"],
                          "edits_used": values[name]["price_edits"],
                          "solve_seconds": records[name]["solve_seconds"],
                          "optimal": records[name]["optimal"]})
    m2_records = 0
    names = sorted(records.keys())
    for name in names:
        if "M2" in records[name]["flag_codes"]:
            m2_records += 1
    edges = values["R0b"]["edges"]
    return {"rp": rp, "base_shelf_use": base_use, "base_price_edits": values["R0b"]["price_edits"],
            "base_price_moves_at_top_edge": edges["moved_top"],
            "tolerance_at_rp": ladder.rung_tolerance(common.EXPERIMENT_GAP, records["R0b"]["objective_bound"]),
            "m2_records": m2_records, "n_records": len(names),
            "ladder": rows, "alone": alone, "existing_rules": existing,
            "shelf_sweep": shelf_rows, "edit_sweep": edit_rows}


def logic_checks(problem, records, values, derived):
    """Every kind-specific invariant; returns a list of failures (empty = all hold).

    Restriction pairs use ladder.restriction_holds with the looser model's bound; the
    accounting rung uses ladder.accounting_holds; recounts must respect each limit.
    """
    gap = common.EXPERIMENT_GAP
    failures = []
    pairs = [("R0b", "shelf_0.70", "R1 vs R0b"), ("shelf_0.70", "R2", "R2 vs R1"),
             ("R0b", "set_cap", "set cap vs R0b"), ("no_min_order", "R0b", "min order vs none")]
    for fraction_name in ("shelf_0.50", "shelf_0.70", "shelf_0.90", "shelf_1.00"):
        pairs.append(("R0b", fraction_name, fraction_name + " vs R0b"))
        pairs.append((fraction_name + "_plus", fraction_name, fraction_name + " vs +5"))
    pairs.append(("shelf_0.70", "shelf_0.50", "50% vs 70%"))
    pairs.append(("shelf_0.90", "shelf_0.70", "70% vs 90%"))
    pairs.append(("shelf_1.00", "shelf_0.90", "90% vs 100%"))
    limits = list(EDIT_LIMITS)
    limits.append(all_edits(problem))
    positions = list(range(1, len(limits)))
    for position in positions:
        looser = "edits_" + str(limits[position])
        tighter = "edits_" + str(limits[position - 1])
        pairs.append((looser, tighter, tighter + " vs " + looser))
    pairs.append(("R0b", "edits_" + str(all_edits(problem)), "R0b vs P=all"))
    for looser, tighter, label in pairs:
        ok = ladder.restriction_holds(values[tighter]["in_sample"], values[looser]["in_sample"],
                                      records[looser]["objective_bound"], gap)
        if not ok:
            failures.append("restriction raised profit: " + label)
    lp = records["R0b"]["lp_relaxation_objective"]
    if records["R0b"]["solver_objective"] > lp + 1e-6 * max(1.0, abs(lp)):
        failures.append("MILP above its LP relaxation")
    if not ladder.accounting_holds(values["R0b"]["in_sample"], values["R0a"]["in_sample"],
                                   records["R0b"]["objective_bound"], gap):
        failures.append("modeling fees lost money vs the fee-blind plan")
    if values["R0a"]["own_objective"] < values["R0a"]["in_sample"] - 1e-9:
        failures.append("fee-blind objective below its value at true fees")
    names = sorted(records.keys())
    for name in names:
        record = records[name]
        capacity = record["rules"]["shelf_capacity"]
        if capacity is not None and values[name]["shelf_used"] > capacity + 1e-6:
            failures.append(name + ": shelf use over capacity")
        limit = record["rules"]["price_edit_limit"]
        if limit is not None and values[name]["price_edits"] > limit:
            failures.append(name + ": price edits over the limit")
        if record["spend"] > LADDER_BUDGET + 1e-6:
            failures.append(name + ": spend over budget")
    if values["edits_0"]["price_edits"] != 0:
        failures.append("P = 0 moved a stocked product off today's price")
    for name in names:
        if name == "R0a":
            continue   # solved on fee-blind prices, whose M1 filter keeps other levels
        flagged = "M2" in records[name]["flag_codes"]
        recounted = values[name]["edges"]["m2"] > 0
        if flagged != recounted:
            failures.append(name + ": the M2 recount disagrees with the solver's flag")
    return failures


def _inputs(problem):
    eval_demand = common.make_demand(problem, scorecard.N_EVAL, scorecard.EVAL_SEED)
    fingerprint = jsonout.fingerprint(problem["products"], problem["prices"],
                                      [problem["demand"], eval_demand])
    return eval_demand, fingerprint


def build(problem=None, log_rows=None):
    """Solve every rung and sweep point, then assemble the file (heavy: about 95 s)."""
    problem = ladder_problem(problem)
    eval_demand, fingerprint = _inputs(problem)
    records = solve_all(problem, log_rows=log_rows)
    return assemble(problem, eval_demand, fingerprint, records)


def reuse_plans(stored=None, problem=None):
    """The ladder file re-derived from the committed records (no solves)."""
    if stored is None:
        with open(LADDER_PATH, encoding="utf-8") as handle:
            stored = json.load(handle)
    problem = ladder_problem(problem)
    eval_demand, fingerprint = _inputs(problem)
    if stored["input_fingerprint"] != fingerprint:
        raise ValueError("inputs changed since the committed run; run the full build")
    return assemble(problem, eval_demand, fingerprint, stored["records"])


def assemble(problem, eval_demand, fingerprint, records):
    """Values, derived tables and logic checks from the solved records."""
    values = valuations(problem, eval_demand, records)
    derived = derive(problem, records, values)
    failures = logic_checks(problem, records, values, derived)
    if failures:
        raise RuntimeError("ladder logic checks failed: " + "; ".join(failures))
    return {
        "generated_by": "experiments/constraint_ladder.py (committed synthetic dataset)",
        "label": ("in-sample on the pinned " + str(common.N_SCENARIOS) + "-scenario set (seed "
                  + str(common.SCENARIO_SEED) + ") unless a column says out-of-sample ("
                  + format(scorecard.N_EVAL, ",") + " fresh scenarios, eval seed " + str(scorecard.EVAL_SEED) + ")"),
        "settings": {"budget": LADDER_BUDGET, "shelf_fractions": list(SHELF_FRACTIONS),
                     "horizon_weeks": common.HORIZON_WEEKS,
                     "shelf_step": SHELF_STEP, "ladder_shelf_fraction": LADDER_SHELF_FRACTION,
                     "edit_limits": list(EDIT_LIMITS), "ladder_edit_limit": LADDER_EDIT_LIMIT,
                     "set_cap_share": SET_CAP_SHARE, "mip_gap": common.EXPERIMENT_GAP,
                     "time_limit": common.EXPERIMENT_TIME_LIMIT, "n_eval": scorecard.N_EVAL,
                     "eval_seed": scorecard.EVAL_SEED, "space_by_type": ladder.SPACE_BY_TYPE},
        "input_fingerprint": fingerprint,
        "records": records,
        "derived": derived,
        "why": WHY,
    }


def check(stored=None, problem=None):
    """Re-value the committed plans, recount, re-derive and re-run the logic checks."""
    if stored is None:
        with open(LADDER_PATH, encoding="utf-8") as handle:
            stored = json.load(handle)
    problem = ladder_problem(problem)
    problems = []
    eval_demand, fingerprint = _inputs(problem)
    if stored["input_fingerprint"] != fingerprint:
        problems.append("input_fingerprint differs: inputs changed since the committed run")
    records = stored["records"]
    values = valuations(problem, eval_demand, records)
    derived = derive(problem, records, values)
    failures = logic_checks(problem, records, values, derived)
    for failure in failures:
        problems.append("logic check: " + failure)
    fresh = {"derived": derived, "why": WHY}
    scorecard.compare_sections(stored, jsonout.rounded(fresh), "ladder", problems)
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="re-value the committed plans and fail if any number is stale")
    parser.add_argument("--reuse-plans", action="store_true",
                        help="rewrite the file from the committed plans (code or text change); solves nothing")
    args = parser.parse_args(argv)
    if args.check:
        problems = check()
        for problem in problems:
            print("STALE:", problem)
        if problems:
            return 1
        print("reports/constraint_ladder.json is current")
        return 0
    started = time.perf_counter()
    log_rows = []
    if args.reuse_plans:
        card = reuse_plans()
    else:
        card = build(log_rows=log_rows)
    common.REPORTS_DIR.mkdir(exist_ok=True)
    jsonout.dump(card, LADDER_PATH)
    if log_rows:
        common.save_solve_log(log_rows)
    for row in card["derived"]["ladder"]:
        print(row["rung"], format(row["expected_profit"], ",.2f"), row["kind"])
    print("wrote", LADDER_PATH)
    print("total wall clock: " + format(time.perf_counter() - started, ".1f") + " s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
