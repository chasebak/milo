"""EVPI/VSS inequality tests on the committed dataset (the dataset acceptance test).

All solves run at rel gap 1e-4, so the maximization inequalities EEV <= RP <= WS
hold up to that gap; the asserts allow exactly that much slack and no more.
"""
import numpy as np
import pytest

from experiments import common
from experiments.evpi_vss import evpi_vss


@pytest.fixture(scope="module")
def problem():
    return common.load_problem()


@pytest.fixture(scope="module")
def results(problem):
    return evpi_vss(problem)


def test_mean_demand_preserves_non_increasing_small():
    demand = np.array([[[5.0, 1.0], [3.0, 1.0], [2.0, 0.0]],
                       [[9.0, 4.0], [6.0, 4.0], [6.0, 2.0]]])
    mean = common.mean_demand(demand)
    assert mean.shape == (2, 3, 1)
    assert np.all(np.diff(mean, axis=1) <= 1e-9)
    weighted = common.mean_demand(demand, probs=np.array([0.25, 0.75]))
    assert np.all(np.diff(weighted, axis=1) <= 1e-9)
    assert weighted[0, 0, 0] == pytest.approx(0.25 * 5.0 + 0.75 * 1.0)


def test_mean_demand_preserves_non_increasing_on_committed_set(problem):
    mean = common.mean_demand(problem["demand"])
    assert mean.shape == (problem["demand"].shape[0], problem["demand"].shape[1], 1)
    assert np.all(np.diff(mean, axis=1) <= 1e-9)


def test_rp_accounting_is_consistent(results):
    # evaluate_plan re-values the solver's incumbent plan with identical accounting
    scale = max(1.0, abs(results["rp"]))
    assert abs(results["rp"] - results["rp_solver_objective"]) <= 1e-6 * scale


def test_ws_covers_the_full_committed_scenario_set(results):
    # the cap (30) must not bind on the pinned 24-scenario problem, or the
    # inequalities below would only be subset approximations
    assert results["ws_scenarios"] == results["n_scenarios"]


def test_maximization_inequalities(results):
    gap_slack = results["mip_gap"] * max(1.0, abs(results["rp"]))
    assert results["eev"] <= results["rp"] + gap_slack
    assert results["rp"] <= results["ws"] + gap_slack
    assert results["evpi"] >= -gap_slack
    assert results["vss"] >= -gap_slack


def test_evpi_and_vss_are_positive_on_committed_dataset(results):
    # the frozen dataset is tuned so both have real margin (see SCHEMA.md knobs)
    assert results["evpi"] > 0.0
    assert results["vss"] > 0.0
