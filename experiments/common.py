"""Shared assembly for the experiments: the pinned scenario problem.

Every experiment works on the same problem: the committed synthetic dataset run through
the pipeline's data steps with the pinned end-to-end settings (24 scenarios, 8 bootstrap
draws, main budget 1500 - the same values as scripts/regen_goldens.E2E_SETTINGS), and no
set-spend cap (set_cap_share=1.0), so the numbers answer the pure budgeted two-stage
question. All experiment solves use rel gap 1e-4 - never the operating 0.005, whose gap
noise could violate the EVPI/VSS inequalities - and a 120 s per-solve time limit.

Every result is in-sample, conditional on the pinned scenario set.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from milo import demand, pipeline, simulate

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "out"
REPORTS_DIR = REPO / "reports"

# Mirrors scripts/regen_goldens.E2E_SETTINGS (n_scenarios, n_boot, main budget).
N_SCENARIOS = 24
N_BOOT = 8
MAIN_BUDGET = 1500.0
SCENARIO_SEED = 11          # build_scenarios seed, as used by pipeline.run
HORIZON_WEEKS = 8           # planning horizon of every scenario set (pipeline.run's default)

EXPERIMENT_GAP = 1e-4       # rel gap for every experiment solve
EXPERIMENT_TIME_LIMIT = 120  # seconds per solve

SOLVE_LOG_COLUMNS = ["experiment", "label", "transform", "n_scenarios", "budget",
                     "status", "termination", "optimal", "mip_gap", "objective",
                     "build_seconds", "solve_seconds"]


def load_problem(data_dir=pipeline.DEFAULT_DATA, today=pipeline.DEFAULT_TODAY,
                 n_scenarios=N_SCENARIOS, n_boot=N_BOOT, seed=SCENARIO_SEED):
    """Data -> demand units -> elasticities -> price grid -> scenario set.

    The same steps as pipeline.run up to the optimizer inputs, with everything needed
    to redraw scenario sets later (make_demand) kept on the returned dict.
    """
    data = pipeline.load_all(data_dir=data_dir, today=today)
    rates = demand.p2_shrink(demand.p1_raw_rates(data["fam"]))
    online = pipeline.online_vs_inperson(data)
    unit_index = pipeline.unit_online_index(data, online["online_index_by_key"])
    sens = pipeline.price_sensitivity(data, online["products"], n_boot=n_boot)
    units = pipeline.restock_units(data, rates, online["products"])
    grid, demand_array = pipeline.build_scenarios(
        units, sens["elasticities"], horizon_weeks=HORIZON_WEEKS, lead_weeks=1, traffic_mult=1.0,
        online_index=unit_index, season_uplift=1.3, n_scenarios=n_scenarios, seed=seed,
        card_share=data["card_share"], units_per_order=data["units_per_order"],
        config=data["config"])
    products, prices = pipeline.optimizer_inputs(units, grid)
    prices = pipeline.fill_missing_levels(products, prices, demand_array)
    problem = {
        "products": products, "prices": prices, "demand": demand_array,
        "units": units, "elasticities": sens["elasticities"], "unit_index": unit_index,
        "card_share": data["card_share"], "units_per_order": data["units_per_order"],
        "config": data["config"], "n_scenarios": n_scenarios, "seed": seed,
    }
    return problem


def make_demand(problem, n_scenarios, seed):
    """A fresh scenario set for the same units/elasticities (used by SAA and RP spread)."""
    grid, demand_array = pipeline.build_scenarios(
        problem["units"], problem["elasticities"], horizon_weeks=HORIZON_WEEKS, lead_weeks=1,
        traffic_mult=1.0, online_index=problem["unit_index"], season_uplift=1.3,
        n_scenarios=n_scenarios, seed=seed, card_share=problem["card_share"],
        units_per_order=problem["units_per_order"], config=problem["config"])
    return demand_array


def mean_demand(demand_array, probs=None):
    """Probability-weighted scenario mean, shape (I, K, 1).

    Averaging over scenarios preserves non-increasing-in-k: each scenario slice is
    non-increasing in k, and a convex combination of non-increasing arrays is too.
    """
    demand_array = np.asarray(demand_array, dtype=float)
    n_scen = demand_array.shape[2]
    if probs is None:
        probs = np.full(n_scen, 1.0 / n_scen)
    probs = np.asarray(probs, dtype=float)
    mean = np.tensordot(demand_array, probs, axes=([2], [0]))
    return mean[:, :, None]


def plan_maps(result):
    """(buy_units dict, price_level dict) from a solve_plan result."""
    buy_units = {}
    for _, row in result["buy"].iterrows():
        buy_units[row["i"]] = int(row["units"])
    price_level = {}
    for _, row in result["prices"].iterrows():
        if row["level"] is not None and not pd.isna(row["level"]):
            price_level[row["i"]] = int(row["level"])
    return buy_units, price_level


def expected_value(products, prices, demand_array, buy_units, price_level, probs=None):
    """Expected profit of a fixed plan, restricted to the products that have a level.

    Products whose every price level fails the optimizer's M1 filter are outside the
    model (the solver's objective never counts them), so the evaluation drops them too:
    both sides of every comparison then count exactly the same products.
    """
    kept_ids = []
    positions = []
    all_ids = list(products.index)
    for position, unit_id in enumerate(all_ids):
        if unit_id in price_level:
            kept_ids.append(unit_id)
            positions.append(position)
    kept_products = products.loc[kept_ids]
    kept_demand = np.asarray(demand_array, dtype=float)[positions, :, :]
    profit, summary = simulate.evaluate_plan(kept_products, prices, kept_demand,
                                             buy_units, price_level, probs)
    return float(summary["expected"]), profit


def log_solve(log_rows, experiment, label, result, n_scenarios, budget, transform="hull"):
    """Append one solve's timings to a list of solve-log rows."""
    log_rows.append({
        "experiment": experiment, "label": label, "transform": transform,
        "n_scenarios": int(n_scenarios), "budget": float(budget),
        "status": result["status"], "termination": result["termination"],
        "optimal": bool(result["optimal"]), "mip_gap": float(result["mip_gap"]),
        "objective": float(result["objective"]),
        "build_seconds": float(result["build_seconds"]),
        "solve_seconds": float(result["solve_seconds"]),
    })


def save_solve_log(log_rows, path=None):
    """Append solve-log rows to out/solve_log.csv (header written once)."""
    if path is None:
        path = OUT_DIR / "solve_log.csv"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(log_rows, columns=SOLVE_LOG_COLUMNS)
    write_header = not path.exists()
    table.to_csv(path, mode="a", header=write_header, index=False)
    return path
