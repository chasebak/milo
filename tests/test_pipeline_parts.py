import numpy as np
import pandas as pd

from milo import compare, demand, market, pipeline, tcgmap


def test_weekend_check_flags_thin_stock():
    units = pd.DataFrame({"unit": ["a", "b"], "on_hand": [1.0, 50.0], "shrunk_rate": [10.0, 10.0]})
    plan = {"buy": pd.DataFrame({"i": ["a"], "units": [0], "spend": [0.0]})}
    weekday = pd.DataFrame({"weekday": ["Fri", "Sat", "Sun", "Mon"], "share_of_week": [0.2, 0.2, 0.2, 0.4]})
    table, share = pipeline.weekend_check(units, plan, weekday, None, 1.0, 8)
    assert abs(share - 0.6) < 1e-9
    flagged = table.set_index("unit")["flag"]
    assert flagged["a"] != "" and flagged["b"] == ""


def test_traffic_ratio_uses_store_daily_and_skips_partial_days():
    store = pd.DataFrame({"date": pd.to_datetime(["2026-09-10", "2026-09-11", "2026-10-01", "2026-10-02"]),
                          "weekday": ["Thu", "Fri", "Thu", "Fri"], "day_kind": ["full", "full", "full", "partial"],
                          "store_orders": [50.0, 50.0, 75.0, 5.0]})
    ratio = demand.traffic_ratio(None, "2026-09-30", "2026-10-02", "2026-09-01", store)
    assert abs(ratio - 1.5) < 1e-9


def test_numbers_match_ignores_leading_zeros():
    assert tcgmap.numbers_match("63/132", "063/132")
    assert not tcgmap.numbers_match("215/091", "217/091")


def test_tcg_map_review_rows_filtered(tmp_path):
    from milo import load
    path = tmp_path / "tcg_map.csv"
    table = pd.DataFrame({
        "key": ["P000", "P001", "P002"],
        "status": ["ok", "review", "ok"],
        "tcg_id": [100001, 100002, 100003],
        "product": ["Booster Box", "Blister Pack", "Mini Tin"],
        "variant": ["Default Title", "Default Title", "Default Title"],
        "note": ["auto", "auto", "auto"],
    })
    table.to_csv(path, index=False)
    mapping = load.load_tcg_map(path)
    assert list(mapping["key"]) == ["P000", "P002"]
    assert (mapping["status"] == "ok").all()
    assert "P001" not in set(mapping["key"])


def test_chart_spreads_units_over_bucket_and_drops_running_bucket(tmp_path):
    folder = tmp_path / "tcg_history"
    folder.mkdir()
    record = {"id": 1, "range": "quarter", "chart": {"result": [{"condition": "Unopened", "variant": "Normal", "buckets": [
        {"bucketStartDate": "2026-10-01", "marketPrice": "10", "quantitySold": "6", "transactionCount": "3"},
        {"bucketStartDate": "2026-10-19", "marketPrice": "11", "quantitySold": "3", "transactionCount": "3"}]}]}}
    (folder / "1_2026-10-20.jsonl").write_text(__import__("json").dumps(record) + "\n")
    table = market.chart_table(tmp_path, today=pd.Timestamp("2026-10-20"))
    assert len(table) == 3                       # 3-day bucket spread to 3 days; the running bucket is dropped
    assert np.isclose(table["units"].sum(), 6.0)


def test_online_daily_matches_condition():
    daily = pd.DataFrame({"tcg_id": [5, 5], "condition": ["Near Mint", "Lightly Played"], "variant": ["Normal", "Normal"],
                          "date": pd.to_datetime(["2026-10-01", "2026-10-01"]), "units": [2.0, 7.0],
                          "avg_price": [10.0, 6.0], "tcg_market": [9.0, 6.0]})
    out = compare.online_daily_for(daily, 5, "NM")
    assert float(out["online_units"].iloc[0]) == 2.0


def _events(rows):
    table = pd.DataFrame(rows, columns=["ts", "kind", "product", "units", "order"])
    table["ts"] = pd.to_datetime(table["ts"])
    table["date"] = table["ts"].dt.normalize()
    table["variant"] = "Default Title"
    return table


def test_closed_wednesday_not_counted_unless_sold():
    # 2026-10-12 is a Monday; stock sits through Wed 10-14 with no sale.
    events = _events([("2026-10-12 13:00", "stock_in", "A", 5, None), ("2026-10-16 13:00", "sale", "A", 1, "#1")])
    events["sell_date"] = events["date"]
    panel = demand.stock_panel(events, end_date=pd.Timestamp("2026-10-16"))
    wednesday = panel[panel["date"] == pd.Timestamp("2026-10-14")].iloc[0]
    assert not wednesday["in_stock"]
    assert int(panel["in_stock"].sum()) == 4


def test_restock_counts_from_first_selling_day():
    events = _events([("2026-10-12 23:00", "stock_in", "A", 2, None)])
    events["sell_date"] = pd.Timestamp("2026-10-13")
    panel = demand.stock_panel(events, end_date=pd.Timestamp("2026-10-13"))
    entered_day = panel[panel["date"] == pd.Timestamp("2026-10-12")]
    assert entered_day.empty or not bool(entered_day.iloc[0]["in_stock"])
    assert panel[panel["date"] == pd.Timestamp("2026-10-13")].iloc[0]["in_stock"]


def test_bulk_order_left_out_of_demand_but_leaves_shelf():
    from milo import load
    events = _events([("2026-10-12 13:00", "stock_in", "A", 40, None), ("2026-10-12 14:00", "sale", "A", 33, "#9"),
                      ("2026-10-13 14:00", "sale", "A", 1, "#10")])
    events["sell_date"] = events["date"]
    events = load.flag_bulk_orders(events)
    panel = demand.stock_panel(events, end_date=pd.Timestamp("2026-10-13"))
    assert int(panel["sold"].sum()) == 1
    assert float(panel["stock_end"].iloc[-1]) == 6.0
