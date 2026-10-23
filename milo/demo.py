"""Demo entrypoint: python -m milo.demo

Runs the full pipeline on the committed synthetic dataset with CI-tier settings
(fewer scenarios, bootstrap draws and budget points than the notebook defaults),
prints the strategy-comparison table and writes the Excel exports to out/ only.
"""
import time

from milo import pipeline


def main():
    started = time.perf_counter()
    out = pipeline.run(
        budgets=(0, 500, 1500),
        main_budget=1500,
        n_scenarios=24,
        n_boot=8,
        curve_scenarios=12,
        time_limit=30,
    )
    files = pipeline.export(out)

    print("\nData summary:")
    print(out["summary"].to_string(index=False))
    plan = out["plan"]
    print("\nSolver: status=" + plan["status"] + " termination=" + plan["termination"]
          + " optimal=" + str(plan["optimal"]))
    print("Expected profit at the main budget: " + format(plan["objective"], ".2f"))
    print("\nBudget curve:")
    print(out["budget_curve"].round(2).to_string(index=False))
    print("\nStrategy comparison (same buy list, different pricing rules):")
    print(out["strategies"].round(2).to_string(index=False))
    print("\nWrote:")
    for path in files:
        print("  " + str(path))
    elapsed = time.perf_counter() - started
    print(f"\nDemo finished in {elapsed:.1f} s")


if __name__ == "__main__":
    main()
