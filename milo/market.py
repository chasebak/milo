"""Market-feed data -> one daily table per (tcg_id, condition) (plan: M17, M17b-d).

Inputs: jsonl exports from a licensed third-party market-data feed:
  <raw_dir>/tcg_sales/<id>_<date>.jsonl    one line per sales page:  {"id", "page": {...sales json}}
  <raw_dir>/tcg_history/<id>_<date>.jsonl  one line per chart range: {"id", "range", "chart": {...}}

Output columns: tcg_id, condition, variant, date, units, transactions, avg_price (incl. shipping, qty-weighted),
min_price, max_price, tcg_market (chart market price), resolution (sales | daily | 3day | weekly).
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from milo import load

RANGE_RESOLUTION = {"month": ("daily", 1), "quarter": ("3day", 3), "semi-annual": ("weekly", 7), "annual": ("weekly", 7)}
RESOLUTION_RANK = {"sales": 0, "daily": 1, "3day": 2, "weekly": 3}


def read_jsonl(folder):
    records = []
    files = sorted(glob.glob(str(Path(folder) / "*.jsonl")))
    for path in files:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def sales_table(raw_dir=load.RAW):
    """Every individual market-feed sale, de-duplicated (M17c)."""
    rows = []
    records = read_jsonl(Path(raw_dir) / "tcg_sales")
    for record in records:
        tcg_id = int(record["id"])
        page = record.get("page") or {}
        for sale in page.get("data") or []:
            rows.append({
                "tcg_id": tcg_id,
                "order_ts": sale.get("orderDate"),
                "condition": sale.get("condition"),
                "variant": sale.get("variant"),
                "language": sale.get("language"),
                "quantity": sale.get("quantity"),
                "purchase_price": sale.get("purchasePrice"),
                "shipping_price": sale.get("shippingPrice"),
                "listing_type": sale.get("listingType"),
                "title": sale.get("title"),
            })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["order_ts"] = pd.to_datetime(table["order_ts"], utc=True, format="ISO8601")
    table = table.drop_duplicates(["tcg_id", "order_ts", "purchase_price", "quantity", "condition", "variant"])
    table["date"] = table["order_ts"].dt.tz_convert("America/Los_Angeles").dt.tz_localize(None).dt.normalize()
    table["price"] = table["purchase_price"].fillna(0) + table["shipping_price"].fillna(0)
    return table


def drop_outliers(sales, low=0.5, high=2.0, window_days=7):
    """M17d: drop sales outside low..high x the rolling 7-day median price of the same item/condition."""
    if sales.empty:
        return sales
    kept = []
    groups = sales.sort_values("order_ts").groupby(["tcg_id", "condition", "variant"])
    for _, group in groups:
        indexed = group.set_index("order_ts")
        median = indexed["price"].rolling(str(window_days) + "D", min_periods=1).median()
        ratio = indexed["price"] / median.to_numpy()
        mask = (ratio >= low) & (ratio <= high)
        kept.append(group[mask.to_numpy()])
    return pd.concat(kept, ignore_index=True)


def daily_from_sales(sales):
    """Daily aggregates from individual sales: units, transactions, quantity-weighted average price."""
    if sales.empty:
        return pd.DataFrame()
    work = sales.copy()
    work["value"] = work["price"] * work["quantity"]
    daily = work.groupby(["tcg_id", "condition", "variant", "date"]).agg(
        units=("quantity", "sum"), transactions=("quantity", "size"), value=("value", "sum"),
        min_price=("price", "min"), max_price=("price", "max")).reset_index()
    daily["avg_price"] = daily["value"] / daily["units"]
    daily["resolution"] = "sales"
    return daily.drop(columns=["value"])


def chart_table(raw_dir=load.RAW, today=None):
    """Chart buckets, spread to days: price held as a step, units divided across the bucket's days (M17).

    The newest bucket of each range is dropped when it is still running (M17b).
    """
    rows = []
    records = read_jsonl(Path(raw_dir) / "tcg_history")
    if today is None:
        today = pd.Timestamp.today().normalize()
    for record in records:
        tcg_id = int(record["id"])
        range_name = record.get("range")
        if range_name not in RANGE_RESOLUTION:
            continue
        resolution, bucket_days = RANGE_RESOLUTION[range_name]
        chart = record.get("chart") or {}
        for sku in chart.get("result") or []:
            buckets = sku.get("buckets") or []
            for bucket in buckets:
                start = pd.Timestamp(bucket["bucketStartDate"])
                end = start + pd.Timedelta(days=bucket_days)
                if end > today:
                    continue
                quantity = float(bucket.get("quantitySold") or 0)
                transactions = float(bucket.get("transactionCount") or 0)
                for offset in range(bucket_days):
                    rows.append({
                        "tcg_id": tcg_id, "condition": sku.get("condition"), "variant": sku.get("variant"),
                        "date": start + pd.Timedelta(days=offset), "resolution": resolution,
                        "tcg_market": float(bucket.get("marketPrice") or np.nan),
                        "units": quantity / bucket_days, "transactions": transactions / bucket_days,
                        "min_price": float(bucket.get("lowSalePriceWithShipping") or np.nan),
                        "max_price": float(bucket.get("highSalePriceWithShipping") or np.nan),
                    })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["rank"] = table["resolution"].map(RESOLUTION_RANK)
    table = table.sort_values("rank").drop_duplicates(["tcg_id", "condition", "variant", "date"]).drop(columns="rank")
    return table


def daily_market(raw_dir=load.RAW, today=None):
    """Stitch finest first (M17): individual sales where present, else daily, else 3-day, else weekly chart.

    Keeps both price views: avg_price (from sales, or chart market when no sales) and tcg_market (chart).
    """
    sales = drop_outliers(sales_table(raw_dir))
    from_sales = daily_from_sales(sales)
    chart = chart_table(raw_dir, today)
    if chart.empty and from_sales.empty:
        return pd.DataFrame()
    if chart.empty:
        from_sales["tcg_market"] = np.nan
        return from_sales
    keys = ["tcg_id", "condition", "variant", "date"]
    market_price = chart[keys + ["tcg_market"]]
    if from_sales.empty:
        stitched = chart.copy()
        stitched["avg_price"] = stitched["tcg_market"]
        return stitched
    sales_days = from_sales.merge(market_price, on=keys, how="left")
    sales_keys = sales_days[keys].assign(has_sales=True)
    chart_only = chart.merge(sales_keys, on=keys, how="left")
    chart_only = chart_only[chart_only["has_sales"] != True].drop(columns="has_sales")  # noqa: E712
    # Sales history reaches back only so far: before the oldest sale date, use the chart even on zero-sale days.
    oldest = sales.groupby(["tcg_id", "condition", "variant"])["date"].min().rename("oldest_sale").reset_index()
    chart_only = chart_only.merge(oldest, on=["tcg_id", "condition", "variant"], how="left")
    covered = chart_only["oldest_sale"].notna() & (chart_only["date"] >= chart_only["oldest_sale"])
    chart_only.loc[covered, "units"] = 0.0
    chart_only.loc[covered, "transactions"] = 0.0
    chart_only.loc[covered, "resolution"] = "sales"
    chart_only["avg_price"] = chart_only["tcg_market"]
    chart_only = chart_only.drop(columns="oldest_sale")
    stitched = pd.concat([sales_days, chart_only], ignore_index=True)
    return stitched.sort_values(keys).reset_index(drop=True)


def overlap_check(raw_dir=load.RAW):
    """Weekly units from sales vs the weekly chart where both exist; flags >10% mismatch (M17)."""
    sales = daily_from_sales(drop_outliers(sales_table(raw_dir)))
    chart = chart_table(raw_dir)
    if sales.empty or chart.empty:
        return pd.DataFrame()
    weekly_chart = chart[chart["resolution"] == "weekly"].copy()
    weekly_chart["week"] = weekly_chart["date"].dt.to_period("W-SUN")
    sales["week"] = sales["date"].dt.to_period("W-SUN")
    a = sales.groupby(["tcg_id", "condition", "week"])["units"].sum().rename("sales_units")
    b = weekly_chart.groupby(["tcg_id", "condition", "week"])["units"].sum().rename("chart_units")
    joined = pd.concat([a, b], axis=1).dropna().reset_index()
    joined["ratio"] = joined["sales_units"] / joined["chart_units"].replace(0, np.nan)
    joined["flag"] = (joined["ratio"] - 1).abs() > 0.10
    return joined
