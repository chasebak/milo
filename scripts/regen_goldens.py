"""Regenerate tests/goldens.json: run the pipeline on the committed synthetic dataset
with the pinned end-to-end settings and record the golden values the e2e test asserts.

Run from the repo root:  python scripts/regen_goldens.py
The committed dataset and these goldens were frozen together in one pass; neither
regenerates again afterwards (the docs consume reports/numbers.json verbatim).
"""
import json
from pathlib import Path

from milo import pipeline

REPO = Path(__file__).resolve().parents[1]

# Pinned settings shared by scripts/regen_goldens.py and tests/test_end_to_end.py.
E2E_SETTINGS = {
    "budgets": [0, 500, 1500],
    "main_budget": 1500,
    "n_scenarios": 24,
    "n_boot": 8,
    "curve_scenarios": 12,
    "time_limit": 30,
}


def compute_goldens():
    out = pipeline.run(**E2E_SETTINGS)
    curve = out["budget_curve"]
    strategies = out["strategies"].set_index("strategy")
    goldens = {
        "n_units": int(len(out["units"])),
        "n_sold_products": int(len(out["data"]["sold"])),
        "demand_shape": list(out["demand"].shape),
        "card_share": float(out["data"]["card_share"]),
        "units_per_order": float(out["data"]["units_per_order"]),
        "traffic_test_vs_fit": float(out["traffic_test_vs_fit"]),
        "plan_objective": float(out["plan"]["objective"]),
        "budget_curve_objectives": [float(value) for value in curve["objective"]],
        "optimal_strategy_expected": float(strategies.at["optimal (Pyomo)", "expected"]),
        "keep_strategy_expected": float(strategies.at["keep", "expected"]),
    }
    return goldens


def main():
    goldens = compute_goldens()
    payload = {"settings": E2E_SETTINGS, "golden": goldens}
    target = REPO / "tests" / "goldens.json"
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    print("wrote", target)
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
