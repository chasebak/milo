"""Budgeted greedy baseline vs the exact MILP, with the gap measured per budget.

The greedy heuristic:
  1. For each product independently, pick the price level with the best STANDALONE
     expected profit (own on-hand units plus its own best unconstrained buy) - no
     budget, no interaction with other products.
  2. With levels fixed, buy units in marginal-expected-profit-per-dollar order.
     A product with min_order > 1 enters as one all-or-nothing block of min_order
     units (block profit / block cost); after the block, one unit at a time.
     Marginal profits are non-increasing, so the per-dollar ordering is well defined.

The MILP (solve_plan) optimizes buy units and price levels JOINTLY under the budget.
Everything else is held identical - the same scenario set, the same fee model, the
same keep-a-level-only-if-net-cash-beats-salvage filter, the same evaluate_plan
accounting - so the measured gap is attributable to joint optimization alone.
No (1-1/e)-style guarantee is claimed anywhere: the gap is measured, not bounded.

Run from the repo root:  python -m experiments.greedy_baseline
"""
import numpy as np
import pandas as pd

from experiments import common
from milo.optimize import budget_curve

GAP_BUDGETS = (250.0, 500.0, 1000.0, 1500.0, 2500.0)


def _level_tables(prices):
    """Dicts (i, k) -> net_cash and the k -> demand-column position map."""
    net_cash = {}
    rows = prices[["i", "k", "net_cash"]].itertuples(index=False)
    for row in rows:
        net_cash[(row.i, int(row.k))] = float(row.net_cash)
    levels = sorted(prices["k"].astype(int).unique())
    level_pos = {}
    numbered = list(enumerate(levels))
    for pos, k in numbered:
        level_pos[k] = pos
    return net_cash, level_pos


def _kept_levels(products, prices, net_cash):
    """The optimizer's M1 rule: keep level k only when net cash beats salvage value."""
    kept = {}
    product_ids = list(products.index)
    levels = sorted(prices["k"].astype(int).unique())
    for i in product_ids:
        salvage_value = float(products.at[i, "salvage_frac"]) * float(products.at[i, "cost"])
        good = []
        for k in levels:
            key = (i, k)
            if key not in net_cash:
                continue
            if net_cash[key] > salvage_value:
                good.append(k)
        if good:
            kept[i] = good
    return kept


def _increments(demand_row, probs, on_hand, net, salvage_value, cost, max_units):
    """Expected profit of adding the 1st..max_units-th bought unit on top of on_hand.

    Adding one unit at stock s raises sold units by E[clip(D - s, 0, 1)] and leftovers
    by the complement, exactly matching evaluate_plan's sold = min(D, stock).
    """
    increments = []
    unit_numbers = list(range(int(max_units)))
    for u in unit_numbers:
        s = float(on_hand) + u
        delta_sold = float(np.dot(probs, np.clip(demand_row - s, 0.0, 1.0)))
        gain = net * delta_sold + salvage_value * (1.0 - delta_sold) - cost
        increments.append(gain)
    return increments


def _base_profit(demand_row, probs, on_hand, net, salvage_value):
    """Expected profit of the on-hand units alone (x = 0, no buy cost)."""
    sold = np.minimum(demand_row, float(on_hand))
    left = float(on_hand) - sold
    per_scenario = net * sold + salvage_value * left
    return float(np.dot(probs, per_scenario))


def _best_standalone_level(kept, demand_array, row_pos, level_pos, probs,
                           products, net_cash, i):
    """The product's best single-product level and its purchase increments at it."""
    cost = float(products.at[i, "cost"])
    on_hand = float(products.at[i, "on_hand"])
    salvage_value = float(products.at[i, "salvage_frac"]) * cost
    min_order = int(products.at[i, "min_order"])
    max_units = int(products.at[i, "max_units"])
    best = None
    for k in kept[i]:
        net = net_cash[(i, k)]
        demand_row = demand_array[row_pos[i], level_pos[k], :]
        base = _base_profit(demand_row, probs, on_hand, net, salvage_value)
        increments = _increments(demand_row, probs, on_hand, net, salvage_value, cost, max_units)
        running = np.cumsum(increments)
        standalone = base
        if max_units >= min_order:
            x_options = list(range(min_order, max_units + 1))
            for x in x_options:
                candidate = base + float(running[x - 1])
                if candidate > standalone:
                    standalone = candidate
        if best is None or standalone > best["standalone"]:
            best = {"level": k, "standalone": standalone, "increments": increments}
    return best


def greedy_plan(products, prices, demand_array, budget, probs=None):
    """Greedy buy units + price levels under the budget. Returns (buy, levels, spend)."""
    demand_array = np.asarray(demand_array, dtype=float)
    n_scen = demand_array.shape[2]
    if probs is None:
        probs = np.full(n_scen, 1.0 / n_scen)
    probs = np.asarray(probs, dtype=float)
    net_cash, level_pos = _level_tables(prices)
    kept = _kept_levels(products, prices, net_cash)
    row_pos = {}
    numbered = list(enumerate(products.index))
    for pos, i in numbered:
        row_pos[i] = pos

    levels = {}
    state = {}
    kept_ids = list(kept.keys())
    for i in kept_ids:
        best = _best_standalone_level(kept, demand_array, row_pos, level_pos, probs,
                                      products, net_cash, i)
        levels[i] = best["level"]
        state[i] = {"bought": 0, "increments": best["increments"],
                    "cost": float(products.at[i, "cost"]),
                    "min_order": int(products.at[i, "min_order"]),
                    "max_units": int(products.at[i, "max_units"]), "done": False}

    buy = {}
    for i in kept_ids:
        buy[i] = 0
    remaining = float(budget)
    while True:
        best_id = None
        best_ratio = 0.0
        best_step = None
        for i in kept_ids:
            slot = state[i]
            if slot["done"] or slot["bought"] >= slot["max_units"]:
                continue
            if slot["bought"] == 0 and slot["min_order"] > 1:
                block = slot["min_order"]
                if block > slot["max_units"]:
                    slot["done"] = True
                    continue
                gain = float(np.sum(slot["increments"][0:block]))
                spend = block * slot["cost"]
            else:
                block = 1
                gain = float(slot["increments"][slot["bought"]])
                spend = slot["cost"]
            if gain <= 0.0:
                slot["done"] = True
                continue
            if spend > remaining + 1e-9:
                continue
            ratio = gain / spend
            if ratio > best_ratio:
                best_ratio = ratio
                best_id = i
                best_step = block
        if best_id is None:
            break
        buy[best_id] += best_step
        remaining -= best_step * state[best_id]["cost"]
        state[best_id]["bought"] += best_step

    spend = float(budget) - remaining
    return buy, levels, spend


def gap_table(problem, budgets=GAP_BUDGETS, mip_gap=common.EXPERIMENT_GAP,
              time_limit=common.EXPERIMENT_TIME_LIMIT):
    """Greedy vs exact MILP across budgets; gap_pct = (MILP - greedy) / MILP * 100."""
    products = problem["products"]
    prices = problem["prices"]
    demand_array = problem["demand"]
    budget_list = list(budgets)
    milp = budget_curve(products, prices, demand_array, budget_list,
                        mip_gap=mip_gap, time_limit=time_limit)
    rows = []
    numbered_budgets = list(enumerate(budget_list))
    for position, budget in numbered_budgets:
        buy, levels, spend = greedy_plan(products, prices, demand_array, budget)
        greedy_value, _ = common.expected_value(products, prices, demand_array, buy, levels)
        milp_value = float(milp["objective"].iloc[position])
        gap = milp_value - greedy_value
        rows.append({"budget": float(budget), "milp_objective": milp_value,
                     "greedy_objective": greedy_value, "gap": gap,
                     "gap_pct": gap / abs(milp_value) * 100.0,
                     "milp_spend": float(milp["spend"].iloc[position]),
                     "greedy_spend": spend})
    return pd.DataFrame(rows)


def main():
    problem = common.load_problem()
    table = gap_table(problem)
    print("Greedy vs exact MILP (in-sample, conditional on the scenario set):")
    print(table.round(2).to_string(index=False))
    worst = table["gap_pct"].max()
    print("largest measured gap: " + format(worst, ".2f") + "% of the MILP objective")


if __name__ == "__main__":
    main()
