"""EVPI / VSS on the committed dataset's pinned scenario set.

Four values, all in-sample and conditional on the scenario set:

RP   the two-stage stochastic optimum: solve_plan on the full scenario set, then the
     incumbent plan re-valued with evaluate_plan (identical accounting; the module
     checks the two agree).
EV   the mean-value problem: solve_plan on the probability-weighted scenario-MEAN
     demand array. Averaging preserves non-increasing-in-k (tested).
EEV  the EV solution's buy units AND price levels held fixed, valued over the FULL
     scenario set with evaluate_plan.
WS   wait-and-see: one solve per scenario on a capped subset (default cap 30; a
     multiple of 3 keeps the low/base/high elasticity mix). The pinned problem has
     24 scenarios, so the cap does not bind and WS covers the full set - which is
     what makes the inequalities below exact rather than subset approximations.

Maximization inequalities: EEV <= RP <= WS, so EVPI = WS - RP >= 0 and
VSS = RP - EEV >= 0. All solves run at rel gap 1e-4 (the operating 0.005 would let
gap noise violate the inequalities) with a 120 s per-solve limit; solve times are
appended to out/solve_log.csv.

Run from the repo root:  python -m experiments.evpi_vss
"""
import json

import numpy as np

from experiments import common
from milo.optimize import solve_plan


def evpi_vss(problem, budget=common.MAIN_BUDGET, ws_cap=30,
             mip_gap=common.EXPERIMENT_GAP, time_limit=common.EXPERIMENT_TIME_LIMIT,
             log_rows=None):
    """Compute RP, EV, EEV, WS, EVPI and VSS; returns a plain dict of floats."""
    if log_rows is None:
        log_rows = []
    products = problem["products"]
    prices = problem["prices"]
    demand_array = problem["demand"]
    n_scen = demand_array.shape[2]

    # RP: here-and-now optimum over the full scenario set.
    rp_result = solve_plan(products, prices, demand_array, budget=budget,
                           time_limit=time_limit, mip_gap=mip_gap, lp_relaxation=False)
    common.log_solve(log_rows, "evpi_vss", "RP", rp_result, n_scen, budget)
    rp_buy, rp_levels = common.plan_maps(rp_result)
    rp, _ = common.expected_value(products, prices, demand_array, rp_buy, rp_levels)

    # EV: the mean-value problem on the scenario-mean demand array.
    mean_array = common.mean_demand(demand_array)
    ev_result = solve_plan(products, prices, mean_array, budget=budget,
                           time_limit=time_limit, mip_gap=mip_gap, lp_relaxation=False)
    common.log_solve(log_rows, "evpi_vss", "EV", ev_result, 1, budget)
    ev_buy, ev_levels = common.plan_maps(ev_result)

    # EEV: the EV plan held fixed, valued over the full scenario set.
    eev, _ = common.expected_value(products, prices, demand_array, ev_buy, ev_levels)

    # WS: per-scenario optima on a capped subset (uniform probabilities).
    ws_count = min(int(ws_cap), n_scen)
    ws_values = []
    scenario_numbers = list(range(ws_count))
    for w in scenario_numbers:
        single = demand_array[:, :, w:w + 1]
        ws_result = solve_plan(products, prices, single, budget=budget,
                               time_limit=time_limit, mip_gap=mip_gap, lp_relaxation=False)
        common.log_solve(log_rows, "evpi_vss", "WS scenario " + str(w), ws_result, 1, budget)
        ws_values.append(float(ws_result["objective"]))
    ws = float(np.mean(ws_values))

    evpi = ws - rp
    vss = rp - eev
    results = {
        "label": "in-sample, conditional on the scenario set",
        "budget": float(budget),
        "n_scenarios": int(n_scen),
        "ws_scenarios": int(ws_count),
        "mip_gap": float(mip_gap),
        "rp": rp,
        "rp_solver_objective": float(rp_result["objective"]),
        "ev_objective": float(ev_result["objective"]),
        "eev": eev,
        "ws": ws,
        "evpi": evpi,
        "vss": vss,
        "vss_pct_of_rp": vss / abs(rp) * 100.0,
        "evpi_pct_of_rp": evpi / abs(rp) * 100.0,
    }
    return results


def main():
    problem = common.load_problem()
    log_rows = []
    results = evpi_vss(problem, log_rows=log_rows)
    log_path = common.save_solve_log(log_rows)
    print(json.dumps(results, indent=1))
    print("EEV <= RP <= WS:",
          results["eev"] <= results["rp"] <= results["ws"])
    print("solve log appended to", log_path)


if __name__ == "__main__":
    main()
