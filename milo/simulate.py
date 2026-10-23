"""Scenario evaluation of a fixed plan, and ordered demand draws."""

import numpy as np
import pandas as pd
from scipy.stats import poisson


def _net_cash_lookup(prices):
    """Dict (i, k) -> net cash per unit."""
    lookup = {}
    rows = prices[["i", "k", "net_cash"]].itertuples(index=False)
    for row in rows:
        lookup[(row.i, int(row.k))] = float(row.net_cash)
    return lookup


def evaluate_plan(products, prices, demand, buy_units, price_level, probs=None):
    """Profit per scenario for a fixed buy + price plan.

    sold = min(D_ikω, q_i + x_i); profit = Σ n·sold + ρ·c·leftover − c·x.
    Returns (profit array of length n_scenarios, summary dict).
    """
    demand = np.asarray(demand, dtype=float)
    n_scen = demand.shape[2]
    if probs is None:
        probs = np.full(n_scen, 1.0 / n_scen)
    probs = np.asarray(probs, dtype=float)

    levels = sorted(prices["k"].astype(int).unique())
    level_pos = {}
    level_list = list(enumerate(levels))
    for pos, k in level_list:
        level_pos[int(k)] = pos
    net_cash = _net_cash_lookup(prices)

    profit = np.zeros(n_scen)
    total_sold = np.zeros(n_scen)
    total_stock = 0.0
    leftover_value = np.zeros(n_scen)

    product_rows = list(enumerate(products.index))
    for row_pos, i in product_rows:
        cost = float(products.at[i, "cost"])
        on_hand = float(products.at[i, "on_hand"])
        salvage = float(products.at[i, "salvage_frac"]) * cost
        units = float(buy_units.get(i, 0))
        stock = on_hand + units
        total_stock += stock
        profit -= cost * units
        if stock <= 0:
            continue
        if i not in price_level:
            raise ValueError(f"product {i} has stock but no price level")
        k = int(price_level[i])
        d = demand[row_pos, level_pos[k], :]
        sold = np.minimum(d, stock)
        left = stock - sold
        profit += net_cash[(i, k)] * sold + salvage * left
        total_sold += sold
        leftover_value += salvage * left

    expected = float(np.dot(probs, profit))
    if total_stock > 0:
        sell_through = float(np.dot(probs, total_sold)) / total_stock
    else:
        sell_through = float("nan")
    summary = {
        "expected": expected,
        "p10": _weighted_quantile(profit, probs, 0.10),
        "p50": _weighted_quantile(profit, probs, 0.50),
        "p90": _weighted_quantile(profit, probs, 0.90),
        "sell_through": sell_through,
        "leftover_value": float(np.dot(probs, leftover_value)),
    }
    return profit, summary


def _weighted_quantile(values, weights, q):
    """Smallest value whose cumulative probability reaches q."""
    order = np.argsort(values)
    sorted_values = values[order]
    cum = np.cumsum(weights[order])
    pos = int(np.searchsorted(cum, q - 1e-12))
    pos = min(pos, len(sorted_values) - 1)
    return float(sorted_values[pos])


def strategy_table(products, prices, demand, strategies, probs=None):
    """Summary row per named strategy: name -> (buy_units, price_level)."""
    rows = []
    names = list(strategies.keys())
    for name in names:
        buy_units, price_level = strategies[name]
        _, summary = evaluate_plan(products, prices, demand, buy_units, price_level, probs)
        row = {"strategy": name}
        row.update(summary)
        rows.append(row)
    return pd.DataFrame(rows)


def ordered_demand(lam, ratio_multipliers, n_scenarios, rng):
    """Poisson demand (I, K, Ω), non-increasing in k by construction.

    One shared uniform per (i, ω) goes through the Poisson inverse CDF at
    every price level, so a lower mean can never give more units.
    """
    lam = np.asarray(lam, dtype=float)
    mult = np.asarray(ratio_multipliers, dtype=float)
    steps = np.diff(mult, axis=1)
    if np.any(steps > 1e-12):
        raise ValueError("ratio_multipliers must be non-increasing in k")
    u = rng.uniform(size=(lam.shape[0], n_scenarios))
    u = np.clip(u, 1e-9, 1.0 - 1e-9)
    mean = lam[:, None] * mult
    draws = poisson.ppf(u[:, None, :], mean[:, :, None])
    return draws.astype(int)
