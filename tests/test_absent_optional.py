"""Optional inputs may be absent: loaders return their documented empty forms and the
pipeline still runs end to end without them."""
import pytest

from milo import load, pipeline, synth
from milo.config import DEFAULT_SELLERS


@pytest.fixture(scope="module")
def bare_dataset(tmp_path_factory):
    """A dataset with every optional file removed."""
    out_dir = tmp_path_factory.mktemp("synth_bare")
    synth.generate_dataset(out_dir, seed=3, n_products=18, weeks=10)
    raw = out_dir / "raw"
    (raw / "store_daily.csv").unlink()
    (raw / "store_weekday.csv").unlink()
    (raw / "set_release_dates.csv").unlink()
    for name in ["restock_events", "sold_out", "restock_soon", "shared_products", "restock_days"]:
        (raw / "restock" / (name + ".csv")).unlink()
    (out_dir / "inputs" / "restock_costs.csv").unlink()
    return out_dir


def test_loaders_return_documented_empty_forms(bare_dataset):
    raw = bare_dataset / "raw"
    assert load.load_store_daily(raw) is None
    assert load.load_weekday_shares(raw) is None
    assert load.load_set_release_dates(raw) is None
    assert load.load_restock_tables(raw) == {}
    events = load.load_store_events(raw, sellers=DEFAULT_SELLERS)
    # without restock_events.csv every sell_date falls back to the entry date
    assert (events["sell_date"] == events["date"]).all()


def test_pipeline_runs_without_optional_inputs(bare_dataset):
    out = pipeline.run(budgets=(0, 300), main_budget=300, n_scenarios=9, n_boot=2,
                       curve_scenarios=3, time_limit=10,
                       data_dir=bare_dataset, today="2026-09-15",
                       split_date="2026-08-23", test_end="2026-09-13", fit_start="2026-07-13")
    assert out["weekday"] is None
    assert out["restock_tables"] == {}
    assert out["weekend"].empty                      # no weekday table -> no weekend check
    assert len(out["units"]) > 0
    assert out["plan"]["buy"]["spend"].sum() <= 300 + 1e-6
    assert len(out["strategies"]) == 5
