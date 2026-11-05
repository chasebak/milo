"""gdp.hull vs gdp.bigm on the same instances: LP bound, times, objective check.

For each budget, the pinned scenario problem is solved twice - once per GDP-to-MILP
transformation - recording the MILP objective, the LP-relaxation bound (the M5
diagnostic path inside solve_plan), and the build/solve times. The two MILPs describe
the same feasible set, so WHEN BOTH solve to the 1e-4 rel gap their objectives must
agree to within it; when a transform instead hits the per-solve time limit, its
incumbent is a feasible plan of that same set, so it can trail but never beat the
hull optimum (optimal within the 1e-4 gap) by more than the 2 x gap slack checked
below. The hull relaxation is at least as tight, so its LP bound is never above
big-M's (maximization). On the committed dataset big-M times out at every
budget - that measured difference is the point of the table.

Run from the repo root:  python -m experiments.hull_vs_bigm
"""
import pandas as pd

from experiments import common
from milo.optimize import solve_plan

TABLE_BUDGETS = (500.0, 1500.0, 2500.0)


def hull_vs_bigm(problem, budgets=TABLE_BUDGETS, mip_gap=common.EXPERIMENT_GAP,
                 time_limit=common.EXPERIMENT_TIME_LIMIT, log_rows=None):
    """One row per (budget, transform) with objective, LP bound and times."""
    if log_rows is None:
        log_rows = []
    products = problem["products"]
    prices = problem["prices"]
    demand_array = problem["demand"]
    n_scen = demand_array.shape[2]
    rows = []
    budget_list = list(budgets)
    for budget in budget_list:
        objectives = {}
        optimal = {}
        for transform in ("hull", "bigm"):
            result = solve_plan(products, prices, demand_array, budget=budget,
                                transform=transform, time_limit=time_limit,
                                mip_gap=mip_gap, lp_relaxation=True)
            label = transform + " budget " + str(int(budget))
            common.log_solve(log_rows, "hull_vs_bigm", label, result, n_scen, budget,
                             transform=transform)
            objectives[transform] = float(result["objective"])
            optimal[transform] = bool(result["optimal"])
            rows.append({"budget": float(budget), "transform": transform,
                         "objective": float(result["objective"]),
                         "lp_relaxation_objective": float(result["lp_relaxation_objective"]),
                         "build_seconds": float(result["build_seconds"]),
                         "solve_seconds": float(result["solve_seconds"]),
                         "optimal": bool(result["optimal"])})
        scale = max(abs(objectives["hull"]), 1.0)
        slack = 2.0 * mip_gap * scale
        both_optimal = optimal["hull"] and optimal["bigm"]
        agree = abs(objectives["hull"] - objectives["bigm"]) <= slack
        if both_optimal and not agree:
            raise AssertionError("both transforms reached the rel gap but disagree at budget "
                                 + str(budget))
        if optimal["hull"] and not optimal["bigm"]:
            if objectives["bigm"] > objectives["hull"] + slack:
                raise AssertionError("big-M incumbent beats the hull optimum by more than the gap at budget "
                                     + str(budget))
        rows[-1]["both_optimal"] = both_optimal
        rows[-2]["both_optimal"] = both_optimal
        rows[-1]["objectives_agree"] = agree
        rows[-2]["objectives_agree"] = agree
    return pd.DataFrame(rows)


def main():
    problem = common.load_problem()
    log_rows = []
    table = hull_vs_bigm(problem, log_rows=log_rows)
    log_path = common.save_solve_log(log_rows)
    print("hull vs bigm (same instances, rel gap 1e-4):")
    print(table.round(4).to_string(index=False))
    print("solve log appended to", log_path)


if __name__ == "__main__":
    main()
