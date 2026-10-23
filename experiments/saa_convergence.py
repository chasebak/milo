"""SAA convergence experiment (HEAVY: run on request, never in CI).

For each N in {20, 40, 80, 160} and each of 10 seeds, a fresh N-scenario set is
drawn, the two-stage problem is solved on it, and the chosen plan is then valued
on 1,000 fresh evaluation scenarios that no training draw used. The in-sample
objective overstates the plan's true value (the optimizer's curse); the gap
between the in-sample and out-of-sample curves shrinks as N grows (not
monotonically seed to seed), which is the validation that earns the "SAA"
label. Per-solve time limit: 120 s.

Committed outputs (written by main): reports/saa_convergence.csv (one row per
N x seed) and reports/saa_convergence.png. Solve times go to out/solve_log.csv.

Run from the repo root:  python -m experiments.saa_convergence
The pytest smoke test runs the same code on a tiny injected problem.
"""
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import ticker  # noqa: E402

from experiments import common  # noqa: E402
from milo.optimize import solve_plan  # noqa: E402

NS = (20, 40, 80, 160)
SEEDS = (101, 102, 103, 104, 105, 106, 107, 108, 109, 110)
N_EVAL = 1000
EVAL_SEED = 999                  # never used by a training draw
IN_COLOR = "#2a78d6"             # in-sample (blue)
OUT_COLOR = "#eb6834"            # out-of-sample (orange)


def saa_table(products, prices, make_demand_fn, ns=NS, seeds=SEEDS, n_eval=N_EVAL,
              eval_seed=EVAL_SEED, budget=common.MAIN_BUDGET,
              mip_gap=common.EXPERIMENT_GAP, time_limit=common.EXPERIMENT_TIME_LIMIT,
              log_rows=None):
    """One row per (N, seed): in-sample and out-of-sample value of the chosen plan.

    make_demand_fn(n_scenarios, seed) returns a demand array for the same products
    and prices; the evaluation set is drawn once with eval_seed.
    """
    if log_rows is None:
        log_rows = []
    eval_demand = make_demand_fn(n_eval, eval_seed)
    rows = []
    n_list = list(ns)
    seed_list = list(seeds)
    for n in n_list:
        for seed in seed_list:
            train_demand = make_demand_fn(n, seed)
            result = solve_plan(products, prices, train_demand, budget=budget,
                                time_limit=time_limit, mip_gap=mip_gap,
                                lp_relaxation=False)
            label = "N=" + str(n) + " seed=" + str(seed)
            common.log_solve(log_rows, "saa_convergence", label, result, n, budget)
            buy, levels = common.plan_maps(result)
            in_sample, _ = common.expected_value(products, prices, train_demand, buy, levels)
            out_of_sample, _ = common.expected_value(products, prices, eval_demand, buy, levels)
            rows.append({"n_scenarios": int(n), "seed": int(seed),
                         "in_sample": in_sample, "out_of_sample": out_of_sample,
                         "solver_objective": float(result["objective"]),
                         "optimal": bool(result["optimal"]),
                         "mip_gap": float(result["mip_gap"]),
                         "build_seconds": float(result["build_seconds"]),
                         "solve_seconds": float(result["solve_seconds"])})
    return pd.DataFrame(rows)


def summarize(table):
    """Per-N means and seed spread for both curves."""
    grouped = table.groupby("n_scenarios")
    summary = grouped.agg(
        in_sample_mean=("in_sample", "mean"), in_sample_std=("in_sample", "std"),
        out_of_sample_mean=("out_of_sample", "mean"), out_of_sample_std=("out_of_sample", "std"),
        solve_seconds_mean=("solve_seconds", "mean"), n_seeds=("seed", "count"))
    summary = summary.reset_index()
    summary["curse_gap"] = summary["in_sample_mean"] - summary["out_of_sample_mean"]
    return summary


def plot(table, png_path, n_eval=N_EVAL):
    """In-sample vs out-of-sample value by N, per-seed points plus mean lines."""
    summary = summarize(table)
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=110)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.03, 0.03, size=len(table))
    x_points = table["n_scenarios"].to_numpy(dtype=float) * (1.0 + jitter)
    ax.scatter(x_points, table["in_sample"], s=22, color=IN_COLOR, alpha=0.45, linewidths=0)
    ax.scatter(x_points, table["out_of_sample"], s=22, color=OUT_COLOR, alpha=0.45, linewidths=0)
    ax.plot(summary["n_scenarios"], summary["in_sample_mean"], color=IN_COLOR,
            linewidth=2, marker="o", markersize=6, label="in-sample objective (mean over seeds)")
    ax.plot(summary["n_scenarios"], summary["out_of_sample_mean"], color=OUT_COLOR,
            linewidth=2, marker="o", markersize=6,
            label="out-of-sample value, " + format(n_eval, ",") + " fresh scenarios (mean)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(summary["n_scenarios"]))
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("training scenarios N (log scale); dots = individual seeds")
    ax.set_ylabel("expected profit ($)")
    ax.set_title("SAA convergence: the optimizer's-curse gap shrinks with N (non-monotonically)")
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)
    return png_path


def main():
    started = time.perf_counter()
    problem = common.load_problem()

    def make_demand_fn(n_scenarios, seed):
        return common.make_demand(problem, n_scenarios, seed)

    log_rows = []
    table = saa_table(problem["products"], problem["prices"], make_demand_fn,
                      log_rows=log_rows)
    common.REPORTS_DIR.mkdir(exist_ok=True)
    csv_path = common.REPORTS_DIR / "saa_convergence.csv"
    table.to_csv(csv_path, index=False)
    png_path = plot(table, common.REPORTS_DIR / "saa_convergence.png")
    log_path = common.save_solve_log(log_rows)
    print("SAA convergence (budget " + format(common.MAIN_BUDGET, ".0f")
          + ", per-solve limit " + str(common.EXPERIMENT_TIME_LIMIT) + " s):")
    print(summarize(table).round(2).to_string(index=False))
    print("wrote", csv_path)
    print("wrote", png_path)
    print("solve log appended to", log_path)
    elapsed = time.perf_counter() - started
    print("total wall clock: " + format(elapsed / 60.0, ".1f") + " min")


if __name__ == "__main__":
    main()
