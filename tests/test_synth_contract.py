"""Contract check: every file the generator writes loads through milo.load / milo.market,
and the interesting NON-FALLBACK code paths fire (no silent fallback defaults)."""
import pandas as pd
import pytest

from milo import catalog, load, market, synth, tcgmap
from milo.config import DEFAULT_SELLERS

SEED = 5


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("synth_contract")
    manifest = synth.generate_dataset(out_dir, seed=SEED, n_products=20, weeks=12)
    return {"dir": out_dir, "raw": out_dir / "raw", "inputs": out_dir / "inputs", "manifest": manifest}


@pytest.fixture(scope="module")
def events(dataset):
    return load.load_store_events(dataset["raw"], sellers=DEFAULT_SELLERS)


def test_events_load_and_kinds(dataset, events):
    assert len(events) > 200
    assert set(events["kind"].unique()) <= {"sale", "stock_in", "stock_out"}
    assert set(events["seller"].unique()) == {"seller_a", "seller_b"}
    assert events["ts"].notna().all()


def test_first_selling_day_matched(events):
    # some restocks were entered after hours, so their sell_date moved off the entry date
    stock_ins = events[events["kind"] == "stock_in"]
    moved = stock_ins[stock_ins["sell_date"] != stock_ins["date"]]
    assert len(moved) > 0
    # and a moved sell_date is always on or after the entry date
    assert (moved["sell_date"] >= moved["date"]).all()


def test_bulk_order_flagged(events):
    bulk = events[events["bulk"]]
    assert len(bulk) > 0
    assert bulk["units"].max() >= 15


def test_card_share_not_fallback(dataset):
    commission = load.load_commission_lines(dataset["raw"])
    share = load.card_share(commission)
    # 1.0 would mean the 'Order paid by' column was silently absorbed; 0.8 is the no-data fallback
    assert share != 1.0
    assert share != 0.8
    assert 0.5 < share < 1.0
    # the oldest workbook predates the paid-by column; its lines default to Card
    assert commission["paid_by"].notna().all()


def test_unit_costs_dollar_cleaning(dataset):
    raw_text = (dataset["raw"] / "unit_costs.csv").read_text()
    assert "$1,234.56" in raw_text        # the cleaning path has something to clean
    ledger = load.load_unit_costs(dataset["raw"])
    assert ledger["cost"].notna().all()
    assert (ledger["cost"] == 1234.56).any()
    assert ledger["market"].notna().all()


def test_catalog_overrides_applied(dataset, events):
    ledger = load.load_unit_costs(dataset["raw"])
    cat = catalog.build_catalog(events, overrides_path=dataset["inputs"] / "catalog_overrides.csv",
                                ledger=ledger)
    fixed = cat[cat["product"].str.startswith("Collector Crate")]
    assert len(fixed) == 1
    assert fixed.iloc[0]["product_type"] == "collection"       # parses to other_sealed without the override
    assert fixed.iloc[0]["set"] == "shadow summit"
    # and the cost sheet supplied a set for the singles that lack one in the title
    from_ledger = cat[cat["set_source"] == "ledger"]
    assert len(from_ledger) > 0


def test_restock_tables_loaded(dataset):
    tables = load.load_restock_tables(dataset["raw"])
    assert set(tables.keys()) == {"sold_out", "restock_soon", "shared_products", "restock_days"}
    for name, table in tables.items():
        assert len(table) > 0, name


def test_store_daily_and_weekday(dataset):
    daily = load.load_store_daily(dataset["raw"])
    assert daily is not None
    assert list(daily.columns) == ["date", "weekday", "day_kind", "store_orders"]
    weekday = load.load_weekday_shares(dataset["raw"], closed_weekdays=(2,))
    assert weekday is not None
    closed = weekday[weekday["weekday"] == "Wed"]
    assert float(closed["share_of_week"].iloc[0]) == 0.0
    assert abs(weekday["share_of_week"].sum() - 1.0) < 1e-9


def test_order_ledger_sequential(dataset):
    orders = load.load_order_ledger(dataset["raw"])
    numbers = orders["Number"].to_numpy()
    assert (numbers[1:] > numbers[:-1]).all()
    assert str(orders["Number"].dtype).startswith("int")


def test_market_feed_stitches(dataset):
    today = pd.Timestamp(dataset["manifest"]["today"])
    daily = market.daily_market(dataset["raw"], today=today)
    assert not daily.empty
    resolutions = set(daily["resolution"].unique())
    assert "sales" in resolutions                   # individual sales present
    assert len(resolutions & {"daily", "3day", "weekly"}) > 0   # chart buckets fill the history
    assert daily["date"].max() <= today             # the running bucket was dropped (M17b)
    assert daily["avg_price"].notna().all()


def test_sold_products_consistent_with_events(dataset, events):
    saved = pd.read_csv(dataset["raw"] / "sold_products.csv", parse_dates=["first_sale", "last_sale"])
    ledger = load.load_unit_costs(dataset["raw"])
    cat = catalog.build_catalog(events, overrides_path=dataset["inputs"] / "catalog_overrides.csv", ledger=ledger)
    refreshed = tcgmap.sold_products(events, cat)
    joined = saved[["key", "product", "variant"]].merge(refreshed, on=["product", "variant"], how="inner")
    assert len(joined) == len(saved)
    mapping = pd.read_csv(dataset["inputs"] / "tcg_map.csv")
    ok_rows = mapping[mapping["status"] == "ok"]
    assert len(ok_rows) > 10
    assert ok_rows["key"].isin(saved["key"]).all()
