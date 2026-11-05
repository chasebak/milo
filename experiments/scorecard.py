"""Out-of-sample scorecard (U1): three policies x six budgets, in and out of sample.

Policies, all on the pinned problem (experiments/common.py) and the same accounting:
  optimizer     solve_plan: buys and prices chosen jointly (the MILP);
  greedy        the careful greedy list (experiments/greedy_baseline.py);
  today_prices  the FAIR baseline: today's prices (ratio 1.00) held fixed and the buys
                re-optimized by the same MILP, i.e. solve_plan on the ratio-1.00 rows only.
                (Same buys at today's prices would be a straw man.)

Every plan is valued twice with simulate.evaluate_plan: in-sample on the 24 training
scenarios (seed 11) and out-of-sample on 1,000 fresh scenarios (eval seed 999, the SAA
experiment's evaluation set, never a training seed).

Headline KPI, profit per restock dollar (RoRD), per policy p and budget B:
    RoRD_p(B) = (V_p(B) - V_p(0)) / spend_p(B)
where V_p(0) is the SAME policy's budget-0 plan valued on the same scenarios, so a
policy is credited only for what its buys add (repricing the shelf is in V_p(0)).
B = 0 (or a plan that spends nothing) gives None, shown as "n/a".
Failure mode: a policy whose budget-0 plan is poor gets a flattering RoRD; the
absolute values are printed beside it for that reason.

Also: the optimizer's-curse gap (in-sample - out-of-sample, same plan), the marginal
value of a budget dollar (obj(B+50) - obj(B-50)) / 100, the fee-blind row, the paired
standard error of every out-of-sample lead (the evaluation scenarios are shared, so
the per-scenario differences are paired), and how many of each plan's price moves sit
on the price grid's edge (the solver's M2 / M3 flags, recounted from the plan).

Heavy-ish (23 MILP solves, about 75 s): run on request, never in CI.
    python -m experiments.scorecard               # solve and write reports/scorecard.json
    python -m experiments.scorecard --reuse-plans # re-derive from the committed plans; no solves
    python -m experiments.scorecard --check       # re-value the committed plans; no solves
"""
import argparse
import json
import sys
import time

import numpy as np

from experiments import common, greedy_baseline, saa_convergence
from milo import jsonout, ladder
from milo.pricing import PRICE_LEVELS
from milo.simulate import _weighted_quantile
from milo.why import WHY

BUDGETS = (0.0, 250.0, 500.0, 1000.0, 1500.0, 2500.0)
HEADLINE_BUDGET = 1500.0
MARGINAL_STEP = 50.0
N_EVAL = saa_convergence.N_EVAL
EVAL_SEED = saa_convergence.EVAL_SEED
POLICIES = ("optimizer", "greedy", "today_prices")
POLICY_LABELS = {
    "optimizer": "optimizer (MILP: buys + prices)",
    "greedy": "careful greedy list",
    "today_prices": "today's prices, buys re-optimized",
}
SCORECARD_PATH = common.REPORTS_DIR / "scorecard.json"


def training_seeds():
    """Every seed any committed experiment trains on (eval seed must not be one)."""
    from experiments import build_numbers
    seeds = {common.SCENARIO_SEED}
    for seed in build_numbers.RP_SPREAD_SEEDS:
        seeds.add(int(seed))
    for seed in saa_convergence.SEEDS:
        seeds.add(int(seed))
    return seeds


def today_problem(prices, demand_array, products=None):
    """The ratio-1.00 rows only, with the matching demand column (shape I x 1 x W).

    Rule: this is the price-edit limit P = 0 problem only if every product the full
    model prices keeps its ratio-1.00 level through the M1 filter (net cash > salvage).
    Failure mode guarded here: if a product's 1.00 row netted <= salvage, M1 would drop
    that product from this model (and from its valuation) entirely, while P = 0 would
    force it onto another level, so the policies would no longer cover the same
    products. With `products` given, that case raises instead of passing silently.
    """
    if products is not None:
        dropped = today_level_dropped(products, prices)
        if dropped:
            raise ValueError("the ratio-1.00 level fails the M1 filter for " + str(len(dropped))
                             + " product(s); the today's-prices baseline would drop them")
    now_rows = prices[(prices["ratio"].astype(float) - ladder.TODAY_RATIO).abs() < 1e-9]
    levels = sorted(prices["k"].astype(int).unique())
    now_levels = sorted(now_rows["k"].astype(int).unique())
    if len(now_levels) != 1:
        raise ValueError("expected one shared level index for ratio 1.00, got " + str(now_levels))
    position = levels.index(now_levels[0])
    sliced = np.asarray(demand_array, dtype=float)[:, position:position + 1, :]
    return now_rows.reset_index(drop=True), sliced


def today_level_dropped(products, prices):
    """Products the full model prices (some level passes M1) whose ratio-1.00 level does not."""
    kept = ladder.kept_levels(products, prices)
    now = ladder.today_levels(prices)
    dropped = []
    units = sorted(kept.keys())
    for unit in units:
        if unit not in now or now[unit] not in kept[unit]:
            dropped.append(unit)
    return dropped


def rord(value, base_value, spend):
    """Profit per restock dollar: RoRD_p(B) = (V_p(B) - V_p(0)) / spend_p(B).

    value = V_p(B), the policy's plan at budget B valued on some scenario set;
    base_value = V_p(0), the SAME policy's budget-0 plan valued on the SAME scenarios
    (so repricing the shelf is not credited to the buys); spend = what the plan at B
    spent. Returns None ("n/a") when spend <= 1e-9 (B = 0 or a plan that buys nothing).
    Failure modes: it lies upward if V is in-sample (the curse gap), it explodes when
    the spend is tiny, and a policy with a poor budget-0 base gets a flattering ratio,
    so the absolute values are always printed beside it.
    """
    if spend is None or spend <= 1e-9:
        return None
    return (float(value) - float(base_value)) / float(spend)


def _plan_record(buy, levels, spend, solve=None):
    record = {"buy": dict(sorted(buy.items())), "levels": dict(sorted(levels.items())),
              "spend": float(spend)}
    if solve is not None:
        record["solver_objective"] = float(solve["objective"])
        record["optimal"] = bool(solve["optimal"])
        record["solve_seconds"] = float(solve["solve_seconds"])
    return record


def _solve(products, prices, demand_array, budget, log_rows, label):
    result = ladder.solve_with_rules(products, prices, demand_array, budget=budget,
                                     mip_gap=common.EXPERIMENT_GAP,
                                     time_limit=common.EXPERIMENT_TIME_LIMIT)
    common.log_solve(log_rows, "scorecard", label, result, demand_array.shape[2], budget)
    buy, levels = common.plan_maps(result)
    spend = float(result["buy"]["spend"].sum())
    record = _plan_record(buy, levels, spend, result)
    record["objective_bound"] = float(result["objective_bound"])
    return record


def policy_plans(problem, budgets=BUDGETS, log_rows=None):
    """Dict policy -> list of plan records, one per budget (budgets in order)."""
    if log_rows is None:
        log_rows = []
    products = problem["products"]
    prices = problem["prices"]
    demand_array = problem["demand"]
    now_prices, now_demand = today_problem(prices, demand_array, products)
    plans = {}
    for policy in POLICIES:
        plans[policy] = []
    for budget in budgets:
        label = "B=" + format(budget, ".0f")
        plans["optimizer"].append(_solve(products, prices, demand_array, budget, log_rows,
                                         "optimizer " + label))
        buy, levels, spend = greedy_baseline.greedy_plan(products, prices, demand_array, budget)
        plans["greedy"].append(_plan_record(buy, levels, spend))
        plans["today_prices"].append(_solve(products, now_prices, now_demand, budget, log_rows,
                                            "today_prices " + label))
    return plans


def marginal_plans(problem, budgets=BUDGETS, step=MARGINAL_STEP, log_rows=None):
    """Optimizer plans at B - step and B + step for every B > 0."""
    if log_rows is None:
        log_rows = []
    rows = []
    for budget in budgets:
        if budget <= 0:
            continue
        low = _solve(problem["products"], problem["prices"], problem["demand"], budget - step,
                     log_rows, "marginal B-" + format(step, ".0f"))
        high = _solve(problem["products"], problem["prices"], problem["demand"], budget + step,
                      log_rows, "marginal B+" + format(step, ".0f"))
        rows.append({"budget": float(budget), "step": float(step), "low": low, "high": high})
    return rows


def fee_blind_plan(problem, budget=HEADLINE_BUDGET, log_rows=None):
    """The textbook plan: solved on list prices, kept for valuation at the true fees."""
    if log_rows is None:
        log_rows = []
    gross = ladder.fee_blind_prices(problem["prices"])
    record = _solve(problem["products"], gross, problem["demand"], budget, log_rows,
                    "fee_blind B=" + format(budget, ".0f"))
    true_kept = set()
    salvage = problem["products"]["salvage_frac"] * problem["products"]["cost"]
    rows = problem["prices"][["i", "k", "net_cash"]].itertuples(index=False)
    for row in rows:
        if float(row.net_cash) > float(salvage[row.i]):
            true_kept.add((row.i, int(row.k)))
    outside = 0
    chosen = list(record["levels"].items())
    for i, k in chosen:
        stock = float(problem["products"].at[i, "on_hand"]) + float(record["buy"].get(i, 0))
        if stock > 0 and (i, int(k)) not in true_kept:
            outside += 1
    record["levels_outside_true_grid"] = outside
    record["budget"] = float(budget)
    return record


def _value(products, prices, demand_array, record):
    value, profit = common.expected_value(products, prices, demand_array, record["buy"],
                                          record["levels"])
    return value, profit


def paired_se(first, second):
    """Standard error of mean(first - second) over shared scenarios: sd(d, ddof=1) / sqrt(n).

    first and second are per-scenario values of two plans on the SAME scenarios, so the
    difference is paired and the scenario-to-scenario swing cancels. None when n < 2.
    Failure mode: it measures sampling noise of the evaluation set only, not model error.
    """
    differences = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    count = len(differences)
    if count < 2:
        return None
    return float(np.std(differences, ddof=1) / np.sqrt(count))


def _edges(problem, record):
    return ladder.grid_edge_moves(problem["products"], problem["prices"], record["buy"],
                                  record["levels"])


def valuations(problem, eval_demand, plans, marginal, fee_blind, budgets=BUDGETS):
    """Re-value every stored plan in and out of sample (no solves; --check reuses this).

    Also the paired standard errors of the optimizer's out-of-sample leads (same
    evaluation scenarios for every plan) and each plan's grid-edge recount.
    """
    products = problem["products"]
    prices = problem["prices"]
    train = problem["demand"]
    uniform = np.full(eval_demand.shape[2], 1.0 / eval_demand.shape[2])
    out = {"policies": {}, "marginal": [], "fee_blind": {}, "paired": []}
    profits = {}
    for policy in POLICIES:
        rows = []
        profits[policy] = []
        for record in plans[policy]:
            in_value, _ = _value(products, prices, train, record)
            out_value, out_profit = _value(products, prices, eval_demand, record)
            profits[policy].append(out_profit)
            rows.append({"in_sample": in_value, "out_of_sample": out_value,
                         "out_of_sample_p10": _weighted_quantile(out_profit, uniform, 0.10),
                         "out_of_sample_p90": _weighted_quantile(out_profit, uniform, 0.90),
                         "edges": _edges(problem, record)})
        out["policies"][policy] = rows
    positions = list(range(len(plans["optimizer"])))
    for position in positions:
        out["paired"].append(_paired_row(plans, profits, position))
    for row in marginal:
        low_in, _ = _value(products, prices, train, row["low"])
        high_in, _ = _value(products, prices, train, row["high"])
        low_out, _ = _value(products, prices, eval_demand, row["low"])
        high_out, _ = _value(products, prices, eval_demand, row["high"])
        out["marginal"].append({"low_in": low_in, "high_in": high_in,
                                "low_out": low_out, "high_out": high_out})
    true_in, _ = _value(products, prices, train, fee_blind)
    true_out, true_out_profit = _value(products, prices, eval_demand, fee_blind)
    gross = ladder.fee_blind_prices(prices)
    own_in, _ = _value(products, gross, train, fee_blind)
    fee_aware_out = None
    fee_aware_se = None
    if fee_blind["budget"] in budgets:
        fee_position = list(budgets).index(fee_blind["budget"])
        fee_aware_out = out["policies"]["optimizer"][fee_position]["out_of_sample"]
        fee_aware_se = paired_se(true_out_profit, profits["optimizer"][fee_position])
    out["fee_blind"] = {"own_objective": own_in, "true_in_sample": true_in,
                        "true_out_of_sample": true_out, "fee_aware_out_of_sample": fee_aware_out,
                        "minus_fee_aware_out_se": fee_aware_se}
    return out


def _paired_row(plans, profits, position):
    """Paired out-of-sample leads of the optimizer at one budget, with their standard errors.

    lead_vs_greedy_out = mean over scenarios of (optimizer - greedy) profit; its SE is
    paired_se. The RoRD lift's SE uses the per-scenario ratio
    ((p_B - p_0) / spend) of each policy, whose mean is exactly the RoRD; None at zero spend.
    """
    optimizer = profits["optimizer"][position]
    greedy = profits["greedy"][position]
    today = profits["today_prices"][position]
    row = {"lead_vs_greedy_out": float(np.mean(optimizer - greedy)),
           "lead_vs_greedy_out_se": paired_se(optimizer, greedy),
           "lead_vs_today_out": float(np.mean(optimizer - today)),
           "lead_vs_today_out_se": paired_se(optimizer, today),
           "rord_lift_vs_greedy_out_se": None}
    optimizer_spend = plans["optimizer"][position]["spend"]
    greedy_spend = plans["greedy"][position]["spend"]
    if optimizer_spend > 1e-9 and greedy_spend > 1e-9:
        optimizer_ratio = (optimizer - profits["optimizer"][0]) / optimizer_spend
        greedy_ratio = (greedy - profits["greedy"][0]) / greedy_spend
        row["rord_lift_vs_greedy_out_se"] = paired_se(optimizer_ratio, greedy_ratio)
    return row


def derive(budgets, plans, values, marginal, fee_blind):
    """Every reported number from the stored plans and their values (pure arithmetic).

    Per policy p and budget B (table rows):
      rord_in / rord_out = rord(V_p(B), V_p(0), spend_p(B)), baseline the same policy's
          budget-0 plan on the same scenarios (see rord for its failure modes);
      curse_gap = V_in(B) - V_out(B), the same plan graded on its 24 training scenarios
          minus on the fresh evaluation set; baseline: none, it is a plan against itself.
          It has no sign guarantee, so it is reported, never asserted; with 24 scenarios
          it carries seed noise of its own (EXPERIMENTS.md §5);
      price_moves / _at_top_edge / _at_bottom_edge / m2_flags: the plan's recounted price
          moves and how many sit on the grid edge (ladder.grid_edge_moves). A move on
          the edge says the grid, not the data, binds.
    Marginal rows: marginal_in = (V_in(B + step) - V_in(B - step)) / (2 step), the slope of
      the optimizer's re-solves; baseline the average return rord_in at B;
      average_minus_marginal = rord_in(B) - marginal_in(B). Failure mode: each re-solve may
      stop mip_gap x |bound| short, so the slope carries tolerance_per_dollar of noise,
      and a step across a kink averages it away.
    """
    table = []
    for policy in POLICIES:
        base_in = values["policies"][policy][0]["in_sample"]
        base_out = values["policies"][policy][0]["out_of_sample"]
        positions = list(range(len(budgets)))
        for position in positions:
            record = plans[policy][position]
            value = values["policies"][policy][position]
            spend = record["spend"]
            table.append({
                "policy": policy, "budget": float(budgets[position]), "spend": spend,
                "in_sample": value["in_sample"], "out_of_sample": value["out_of_sample"],
                "out_of_sample_p10": value["out_of_sample_p10"],
                "out_of_sample_p90": value["out_of_sample_p90"],
                "base_in_sample": base_in, "base_out_of_sample": base_out,
                "rord_in": rord(value["in_sample"], base_in, spend),
                "rord_out": rord(value["out_of_sample"], base_out, spend),
                "curse_gap": value["in_sample"] - value["out_of_sample"],
                "price_moves": value["edges"]["moved"],
                "price_moves_at_top_edge": value["edges"]["moved_top"],
                "price_moves_at_bottom_edge": value["edges"]["moved_bottom"],
                "m2_flags": value["edges"]["m2"],
            })
    marginal_rows = []
    numbered = list(enumerate(marginal))
    for position, row in numbered:
        value = values["marginal"][position]
        width = 2.0 * row["step"]
        bound = max(abs(row["low"]["objective_bound"]), abs(row["high"]["objective_bound"]))
        average = None
        for entry in table:
            if entry["policy"] == "optimizer" and entry["budget"] == row["budget"]:
                average = entry["rord_in"]
        marginal_in = (value["high_in"] - value["low_in"]) / width
        average_minus_marginal = None
        if average is not None:
            average_minus_marginal = average - marginal_in
        marginal_rows.append({
            "budget": row["budget"], "step": row["step"],
            "marginal_in": marginal_in,
            "marginal_out": (value["high_out"] - value["low_out"]) / width,
            "tolerance_per_dollar": 2.0 * common.EXPERIMENT_GAP * max(1.0, bound) / width,
            "average_in": average,
            "average_minus_marginal": average_minus_marginal,
        })
    fee = values["fee_blind"]
    fee_row = {"budget": fee_blind["budget"], "own_objective": fee["own_objective"],
               "true_in_sample": fee["true_in_sample"],
               "true_out_of_sample": fee["true_out_of_sample"],
               "overstatement": fee["own_objective"] - fee["true_in_sample"],
               "levels_outside_true_grid": fee_blind["levels_outside_true_grid"],
               "spend": fee_blind["spend"],
               "fee_aware_out_of_sample": fee["fee_aware_out_of_sample"],
               "out_of_sample_minus_fee_aware": None,
               "out_of_sample_minus_fee_aware_se": fee["minus_fee_aware_out_se"]}
    if fee["fee_aware_out_of_sample"] is not None:
        fee_row["out_of_sample_minus_fee_aware"] = fee["true_out_of_sample"] - fee["fee_aware_out_of_sample"]
    paired_rows = []
    positions = list(range(len(budgets)))
    for position in positions:
        paired = dict(values["paired"][position])
        paired["budget"] = float(budgets[position])
        paired_rows.append(paired)
    return table, marginal_rows, fee_row, paired_rows


def context_rows(problem, eval_demand, budgets, plans, table, seed_spread, budget=HEADLINE_BUDGET):
    """Context only (never a baseline): the straw man, the common-base alternative, the seed.

    same_buys_today: the optimizer's own buys held at today's prices, the "keep" row
        idea; a straw man, because it keeps buys chosen for other prices.
        straw_man_loss_out = V_out(today's prices, buys re-optimized) - V_out(straw man),
        computed before rounding (the two printed values may differ by a cent from it).
    optimizer_rord_out_today_base: the optimizer's RoRD if its base were the
        today's-prices budget-0 plan instead of its own; it credits repricing the shelf
        to the buys, which is why the headline uses each policy's own base.
    pinned_seed: where the pinned scenario seed's RP sits among the seed-spread RPs
        (reports/numbers.json -> rp_seed_spread): minus the seed mean, in seed sds, and
        its rank from the top. Failure mode: five seeds rank coarsely; it says "mild",
        not "typical".
    """
    position = list(budgets).index(budget)
    record = plans["optimizer"][position]
    now = ladder.today_levels(problem["prices"])
    levels = {}
    priced = list(record["levels"].keys())
    for unit in priced:
        levels[unit] = now[unit]
    held = {"buy": record["buy"], "levels": levels}
    held_in, _ = _value(problem["products"], problem["prices"], problem["demand"], held)
    held_out, _ = _value(problem["products"], problem["prices"], eval_demand, held)
    rows = {}
    for entry in table:
        if entry["budget"] == budget:
            rows[entry["policy"]] = entry
    optimizer = rows["optimizer"]
    today = rows["today_prices"]
    pinned_rp = None
    seeds = list(seed_spread["seeds"])
    if common.SCENARIO_SEED in seeds:
        pinned_rp = seed_spread["rp_values"][seeds.index(common.SCENARIO_SEED)]
    rank = None
    minus_mean = None
    in_sds = None
    if pinned_rp is not None:
        rank = 1
        for value in seed_spread["rp_values"]:
            if value > pinned_rp:
                rank += 1
        minus_mean = pinned_rp - seed_spread["mean"]
        in_sds = minus_mean / seed_spread["std"]
    return {
        "budget": float(budget),
        "straw_man_loss_out": today["out_of_sample"] - held_out,
        "pinned_seed": {"seed": common.SCENARIO_SEED, "rp": pinned_rp, "seed_mean": seed_spread["mean"],
                        "minus_seed_mean": minus_mean, "in_seed_sds": in_sds,
                        "rank_from_top": rank, "n_seeds": len(seeds)},
        "same_buys_today_in_sample": held_in, "same_buys_today_out_of_sample": held_out,
        "same_buys_today_rord_out": rord(held_out, today["base_out_of_sample"], optimizer["spend"]),
        "optimizer_rord_out_today_base": rord(optimizer["out_of_sample"], today["base_out_of_sample"],
                                              optimizer["spend"]),
        "repricing_value_out_of_sample": optimizer["base_out_of_sample"] - today["base_out_of_sample"],
    }


def headline(table, paired_rows, budget=HEADLINE_BUDGET):
    """The headline numbers at the headline budget (pure arithmetic on the table).

    rord_out_lift_vs_greedy = RoRD_out(optimizer) - RoRD_out(greedy), each against its own
        budget-0 plan; rord_lift_vs_greedy_out_se is its paired standard error.
    lead_vs_greedy_out = V_out(optimizer) - V_out(greedy) in dollars, with its paired SE.
    curse_gap = V_in - V_out of the optimizer's plan; curse_gap_pct_of_in_sample divides
        by |V_in|; curse_rord_gap = RoRD_in - RoRD_out, what the in-sample headline would
        overstate per dollar.
    optimizer_price_moves / _at_top_edge: the plan's recounted price moves and how many
        sit on the grid's upper edge (M2).
    Failure modes: the leads are out-of-sample on the model's own generator (not a real
    store); the SEs cover evaluation-set sampling only; a lead built on edge moves is
    bounded by the price grid, not by the data.
    """
    rows = {}
    for entry in table:
        if entry["budget"] == budget:
            rows[entry["policy"]] = entry
    paired = None
    for row in paired_rows:
        if row["budget"] == budget:
            paired = row
    optimizer = rows["optimizer"]
    greedy = rows["greedy"]
    return {
        "budget": float(budget),
        "optimizer_rord_out": optimizer["rord_out"], "greedy_rord_out": greedy["rord_out"],
        "today_prices_rord_out": rows["today_prices"]["rord_out"],
        "optimizer_rord_in": optimizer["rord_in"], "greedy_rord_in": greedy["rord_in"],
        "today_prices_rord_in": rows["today_prices"]["rord_in"],
        "rord_out_lift_vs_greedy": optimizer["rord_out"] - greedy["rord_out"],
        "optimizer_in_sample": optimizer["in_sample"],
        "optimizer_out_of_sample": optimizer["out_of_sample"],
        "curse_gap": optimizer["curse_gap"],
        "curse_gap_pct_of_in_sample": optimizer["curse_gap"] / abs(optimizer["in_sample"]) * 100.0,
        "curse_rord_gap": optimizer["rord_in"] - optimizer["rord_out"],
        "rord_lift_vs_greedy_out_se": paired["rord_lift_vs_greedy_out_se"],
        "lead_vs_greedy_out": paired["lead_vs_greedy_out"],
        "lead_vs_greedy_out_se": paired["lead_vs_greedy_out_se"],
        "lead_vs_today_out": paired["lead_vs_today_out"],
        "lead_vs_today_out_se": paired["lead_vs_today_out_se"],
        "optimizer_price_moves": optimizer["price_moves"],
        "optimizer_price_moves_at_top_edge": optimizer["price_moves_at_top_edge"],
    }


def _inputs(problem):
    eval_demand = common.make_demand(problem, N_EVAL, EVAL_SEED)
    fingerprint = jsonout.fingerprint(problem["products"], problem["prices"],
                                      [problem["demand"], eval_demand])
    return eval_demand, fingerprint


def build(problem=None, log_rows=None):
    """Solve every plan, value it, and return the scorecard dict."""
    if problem is None:
        problem = common.load_problem()
    if log_rows is None:
        log_rows = []
    if EVAL_SEED in training_seeds():
        raise ValueError("eval seed must never be a training seed")
    eval_demand, fingerprint = _inputs(problem)
    plans = policy_plans(problem, log_rows=log_rows)
    marginal = marginal_plans(problem, log_rows=log_rows)
    fee_blind = fee_blind_plan(problem, log_rows=log_rows)
    return assemble(problem, eval_demand, fingerprint, BUDGETS, plans, marginal, fee_blind)


def _seed_spread():
    with open(common.REPORTS_DIR / "numbers.json", encoding="utf-8") as handle:
        return json.load(handle)["rp_seed_spread"]


def derived_sections(problem, eval_demand, budgets, plans, marginal, fee_blind):
    """Every derived section of the file from the plans (no solves; build, reuse and check)."""
    values = valuations(problem, eval_demand, plans, marginal, fee_blind, budgets=budgets)
    table, marginal_rows, fee_row, paired_rows = derive(budgets, plans, values, marginal, fee_blind)
    return {"table": table, "marginal": marginal_rows, "fee_blind": fee_row, "paired": paired_rows,
            "headline": headline(table, paired_rows),
            "context": context_rows(problem, eval_demand, budgets, plans, table, _seed_spread()),
            "why": WHY}


def assemble(problem, eval_demand, fingerprint, budgets, plans, marginal, fee_blind):
    """The whole scorecard dict from the plans (the solves are already done)."""
    card = {
        "generated_by": "experiments/scorecard.py (committed synthetic dataset)",
        "label": ("in-sample = the pinned " + str(common.N_SCENARIOS) + "-scenario set (seed "
                  + str(common.SCENARIO_SEED) + "); out-of-sample = " + format(N_EVAL, ",") + " fresh "
                  "scenarios (eval seed " + str(EVAL_SEED) + ") from the model's own generator"),
        "settings": {"budgets": list(budgets), "headline_budget": HEADLINE_BUDGET,
                     "marginal_step": MARGINAL_STEP, "n_eval": N_EVAL, "eval_seed": EVAL_SEED,
                     "n_scenarios": common.N_SCENARIOS, "scenario_seed": common.SCENARIO_SEED,
                     "mip_gap": common.EXPERIMENT_GAP, "time_limit": common.EXPERIMENT_TIME_LIMIT,
                     "set_cap_share": 1.0, "price_grid_low": float(PRICE_LEVELS[0]),
                     "price_grid_high": float(PRICE_LEVELS[-1])},
        "input_fingerprint": fingerprint,
        "policy_labels": POLICY_LABELS,
        "plans": plans, "marginal_plans": marginal, "fee_blind_plan": fee_blind,
    }
    sections = derived_sections(problem, eval_demand, budgets, plans, marginal, fee_blind)
    names = list(sections.keys())
    for name in names:
        card[name] = sections[name]
    return card


def reuse_plans(stored=None, problem=None):
    """The scorecard re-derived from the committed plans (no solves), for a code or text change."""
    if stored is None:
        with open(SCORECARD_PATH, encoding="utf-8") as handle:
            stored = json.load(handle)
    if problem is None:
        problem = common.load_problem()
    eval_demand, fingerprint = _inputs(problem)
    if stored["input_fingerprint"] != fingerprint:
        raise ValueError("inputs changed since the committed run; run the full build")
    return assemble(problem, eval_demand, fingerprint, tuple(stored["settings"]["budgets"]),
                    stored["plans"], stored["marginal_plans"], stored["fee_blind_plan"])


def _close(a, b):
    """Equal at the JSON's 10-significant-digit rounding (None only equals None)."""
    if a is None or b is None:
        return a is None and b is None
    return jsonout.round_sig(a) == jsonout.round_sig(b) or abs(a - b) <= 1e-9 * max(1.0, abs(a))


def _compare(stored, fresh, path, problems):
    """Append "missing", "is extra", "length differs" or "is stale" for every difference."""
    if isinstance(fresh, dict):
        if not isinstance(stored, dict):
            problems.append(path + " is stale")
            return
        for key in fresh:
            if key not in stored:
                problems.append(path + "." + str(key) + " missing")
                continue
            _compare(stored[key], fresh[key], path + "." + str(key), problems)
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
            _compare(stored[position], fresh[position], path + "[" + str(position) + "]", problems)
    elif isinstance(fresh, float) or fresh is None:
        if not _close(stored, fresh):
            problems.append(path + " is stale")
    elif stored != fresh:
        problems.append(path + " is stale")


def compare_sections(stored, fresh, root, problems):
    """_compare for each top-level section of fresh (a file also holds sections not re-derived)."""
    names = list(fresh.keys())
    for name in names:
        if name not in stored:
            problems.append(root + "." + name + " missing")
            continue
        _compare(stored[name], fresh[name], root + "." + name, problems)


def greedy_problems(problem, budgets, stored_plans, label):
    """Re-run the deterministic greedy list at every stored budget; "greedy plan stale" on a change.

    greedy_plan takes well under a second for every budget, so --check can afford it; a
    change to experiments/greedy_baseline.py would otherwise leave the stored lists stale.
    stored_plans holds one record per budget with full-key "buy" and "levels" maps (zero
    buys may be omitted).
    """
    problems = []
    positions = list(range(len(budgets)))
    for position in positions:
        budget = budgets[position]
        buy, levels, _ = greedy_baseline.greedy_plan(problem["products"], problem["prices"],
                                                     problem["demand"], budget)
        fresh_buy = {}
        pairs = list(buy.items())
        for unit, units in pairs:
            if int(units) > 0:
                fresh_buy[unit] = int(units)
        stored_buy = {}
        stored_pairs = list(stored_plans[position]["buy"].items())
        for unit, units in stored_pairs:
            if int(units) > 0:
                stored_buy[unit] = int(units)
        fresh_levels = {}
        level_pairs = list(levels.items())
        for unit, level in level_pairs:
            fresh_levels[unit] = int(level)
        stored_levels = {}
        stored_level_pairs = list(stored_plans[position]["levels"].items())
        for unit, level in stored_level_pairs:
            stored_levels[unit] = int(level)
        if fresh_buy != stored_buy or fresh_levels != stored_levels:
            problems.append(label + " greedy plan stale at $" + format(budget, ",.0f"))
    return problems


def check(stored=None, problem=None):
    """Re-value the committed plans and re-derive every number; returns a list of problems.

    No MILP is solved: the inputs fingerprint must match, the deterministic greedy list
    is re-run at every budget and must equal the stored one, every stored plan is
    re-valued in and out of sample, and every derived section is recomputed from those
    values and compared to the file (missing and extra keys included). Failure mode: a
    change that would make the solver pick a DIFFERENT plan on the same inputs (a HiGHS
    upgrade) is not caught here; rerun the build to see it.
    """
    if stored is None:
        with open(SCORECARD_PATH, encoding="utf-8") as handle:
            stored = json.load(handle)
    if problem is None:
        problem = common.load_problem()
    problems = []
    eval_demand, fingerprint = _inputs(problem)
    if stored["input_fingerprint"] != fingerprint:
        problems.append("input_fingerprint differs: inputs changed since the committed run")
    budgets = tuple(stored["settings"]["budgets"])
    problems.extend(greedy_problems(problem, budgets, stored["plans"]["greedy"], "scorecard"))
    fresh = derived_sections(problem, eval_demand, budgets, stored["plans"], stored["marginal_plans"],
                             stored["fee_blind_plan"])
    compare_sections(stored, jsonout.rounded(fresh), "scorecard", problems)
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
        print("reports/scorecard.json is current")
        return 0
    started = time.perf_counter()
    log_rows = []
    if args.reuse_plans:
        card = reuse_plans()
    else:
        card = build(log_rows=log_rows)
    common.REPORTS_DIR.mkdir(exist_ok=True)
    jsonout.dump(card, SCORECARD_PATH)
    if log_rows:
        common.save_solve_log(log_rows)
    head = card["headline"]
    print("headline at $" + format(head["budget"], ",.0f") + ": RoRD out-of-sample optimizer "
          + format(head["optimizer_rord_out"], ".3f") + " vs greedy "
          + format(head["greedy_rord_out"], ".3f") + " vs today's prices "
          + format(head["today_prices_rord_out"], ".3f"))
    print("wrote", SCORECARD_PATH)
    print("total wall clock: " + format(time.perf_counter() - started, ".1f") + " s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
