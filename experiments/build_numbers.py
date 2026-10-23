"""Write reports/numbers.json: every headline number, printed by code.

This is the single source the docs stage consumes verbatim - no number in any
README or doc is typed by hand. It gathers, on the committed frozen dataset:

  - EVPI / VSS (RP, EV, EEV, WS) from experiments/evpi_vss.py
  - greedy-vs-MILP gaps per budget from experiments/greedy_baseline.py
  - the hull-vs-bigm table from experiments/hull_vs_bigm.py
  - the demo strategy table and budget curve (pipeline.run with the pinned
    end-to-end settings - the same numbers the demo prints)
  - the seed-to-seed spread of RP across 5 scenario-set seeds
  - the SAA convergence summary, read from reports/saa_convergence.csv when the
    heavy run has been made (experiments/saa_convergence.py)

Run from the repo root:  python -m experiments.build_numbers
"""
import json
import sys

import pandas as pd

from experiments import common, evpi_vss, greedy_baseline, hull_vs_bigm, saa_convergence
from milo import pipeline
from milo.optimize import solve_plan

sys.path.insert(0, str(common.REPO / "scripts"))
from regen_goldens import E2E_SETTINGS  # noqa: E402

RP_SPREAD_SEEDS = (11, 12, 13, 14, 15)


def rp_seed_spread(problem, seeds=RP_SPREAD_SEEDS, log_rows=None):
    """RP re-solved on 5 scenario sets that differ only in the generator seed."""
    if log_rows is None:
        log_rows = []
    values = []
    seed_list = list(seeds)
    for seed in seed_list:
        demand_array = common.make_demand(problem, problem["n_scenarios"], seed)
        result = solve_plan(problem["products"], problem["prices"], demand_array,
                            budget=common.MAIN_BUDGET, mip_gap=common.EXPERIMENT_GAP,
                            time_limit=common.EXPERIMENT_TIME_LIMIT, lp_relaxation=False)
        common.log_solve(log_rows, "rp_seed_spread", "seed " + str(seed), result,
                         problem["n_scenarios"], common.MAIN_BUDGET)
        buy, levels = common.plan_maps(result)
        value, _ = common.expected_value(problem["products"], problem["prices"],
                                         demand_array, buy, levels)
        values.append(value)
    series = pd.Series(values, dtype=float)
    return {"seeds": seed_list, "rp_values": values,
            "mean": float(series.mean()), "std": float(series.std()),
            "min": float(series.min()), "max": float(series.max()),
            "spread_pct_of_mean": float((series.max() - series.min()) / abs(series.mean()) * 100.0)}


def saa_summary():
    """Per-N summary of the committed SAA run, or None if the heavy run is absent."""
    csv_path = common.REPORTS_DIR / "saa_convergence.csv"
    if not csv_path.exists():
        return None
    table = pd.read_csv(csv_path)
    summary = saa_convergence.summarize(table)
    return summary.to_dict("records")


def demo_numbers():
    """The demo's headline numbers: pinned end-to-end settings through pipeline.run."""
    out = pipeline.run(**E2E_SETTINGS)
    curve = out["budget_curve"][["budget", "objective", "spend"]]
    return {
        "settings": E2E_SETTINGS,
        "plan_objective": float(out["plan"]["objective"]),
        "budget_curve": curve.to_dict("records"),
        "strategies": out["strategies"].to_dict("records"),
    }


def build():
    with open(pipeline.DEFAULT_DATA / "MANIFEST.json", encoding="utf-8") as handle:
        manifest = json.load(handle)
    problem = common.load_problem()
    log_rows = []
    evpi_results = evpi_vss.evpi_vss(problem, log_rows=log_rows)
    greedy_table = greedy_baseline.gap_table(problem)
    hull_table = hull_vs_bigm.hull_vs_bigm(problem, log_rows=log_rows)
    spread = rp_seed_spread(problem, log_rows=log_rows)
    numbers = {
        "generated_by": "experiments/build_numbers.py (committed synthetic dataset)",
        "dataset_manifest": manifest,
        "experiment_settings": {
            "n_scenarios": common.N_SCENARIOS, "n_boot": common.N_BOOT,
            "main_budget": common.MAIN_BUDGET, "scenario_seed": common.SCENARIO_SEED,
            "mip_gap": common.EXPERIMENT_GAP, "time_limit": common.EXPERIMENT_TIME_LIMIT,
            "set_cap_share": 1.0,
            "label": "in-sample, conditional on the scenario set",
        },
        "evpi_vss": evpi_results,
        "greedy_vs_milp": {
            "budgets": greedy_table.to_dict("records"),
            "largest_gap_pct": float(greedy_table["gap_pct"].max()),
        },
        "hull_vs_bigm": hull_table.to_dict("records"),
        "demo": demo_numbers(),
        "rp_seed_spread": spread,
        "saa_convergence": saa_summary(),
    }
    common.save_solve_log(log_rows)
    return numbers


def main():
    numbers = build()
    common.REPORTS_DIR.mkdir(exist_ok=True)
    target = common.REPORTS_DIR / "numbers.json"
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(numbers, handle, indent=1)
    print(json.dumps(numbers, indent=1))
    print("wrote", target)


if __name__ == "__main__":
    main()
