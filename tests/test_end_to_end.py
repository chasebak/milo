"""End-to-end run on the committed dataset at data/synthetic/ (seed and analysis
dates pinned in milo.synth.DEFAULTS / milo.pipeline defaults).

The dataset and the golden values below were frozen together in one pass
(scripts/regen_dataset.py then scripts/regen_goldens.py); see SCHEMA.md for the knobs.
"""
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from milo import optimize, pipeline

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from regen_goldens import E2E_SETTINGS  # noqa: E402

with open(REPO / "tests" / "goldens.json", encoding="utf-8") as handle:
    GOLDENS = json.load(handle)

# The plan is solved to a relative MIP gap, so two correct solves (another platform,
# another solver build) may stop at different incumbents inside that gap. Values that
# come out of the optimizer are compared at the gap; data values are compared exactly.
SOLVER_GAP = inspect.signature(optimize.solve_plan).parameters["mip_gap"].default


@pytest.fixture(scope="module")
def result():
    assert GOLDENS["settings"] == E2E_SETTINGS, "goldens.json is stale: rerun scripts/regen_goldens.py"
    return pipeline.run(**E2E_SETTINGS)


def test_pipeline_completes_with_expected_shapes(result):
    golden = GOLDENS["golden"]
    assert len(result["units"]) == golden["n_units"]
    assert len(result["data"]["sold"]) == golden["n_sold_products"]
    assert list(result["demand"].shape) == golden["demand_shape"]
    # demand is non-increasing in the price level in every scenario (M11)
    assert np.all(np.diff(result["demand"], axis=1) <= 1e-9)
    assert len(result["strategies"]) == 5


def test_golden_data_values(result):
    golden = GOLDENS["golden"]
    assert result["data"]["card_share"] == pytest.approx(golden["card_share"], abs=1e-9)
    assert result["data"]["units_per_order"] == pytest.approx(golden["units_per_order"], abs=1e-9)
    assert result["traffic_test_vs_fit"] == pytest.approx(golden["traffic_test_vs_fit"], rel=1e-9)


def test_golden_optimizer_values(result):
    golden = GOLDENS["golden"]
    assert result["plan"]["objective"] == pytest.approx(golden["plan_objective"], rel=SOLVER_GAP)
    curve_objectives = [float(value) for value in result["budget_curve"]["objective"]]
    assert curve_objectives == pytest.approx(golden["budget_curve_objectives"], rel=SOLVER_GAP)
    assert result["plan"]["buy"]["spend"].sum() <= E2E_SETTINGS["main_budget"] + 1e-6


def test_optimal_prices_beat_heuristics_on_same_buy_list(result):
    strategies = result["strategies"].set_index("strategy")
    optimal = float(strategies.at["optimal (Pyomo)", "expected"])
    golden = GOLDENS["golden"]
    assert optimal == pytest.approx(golden["optimal_strategy_expected"], rel=SOLVER_GAP)
    assert float(strategies.at["keep", "expected"]) == pytest.approx(golden["keep_strategy_expected"], rel=SOLVER_GAP)
    for name in ["keep", "match_market", "market_minus_5", "market_plus_10"]:
        assert optimal >= float(strategies.at[name, "expected"]) * 0.99
