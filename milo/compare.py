"""Online (market feed) vs in-person (the store), per product and product type (notebook section 3c).

Measures: in-person premium (our price / market price), capture rate (store units / online units),
weekly co-movement and a 0/1/2-week lag test, and the online demand index used by P3.
"""
import numpy as np
import pandas as pd

CONDITION_TO_TCG = {"Sealed": "Unopened", "NM": "Near Mint", "LP": "Lightly Played",
                    "MP": "Moderately Played", "HP": "Heavily Played", "DMG": "Damaged"}


def online_daily_for(daily_market, tcg_id, condition=None):
    """Daily online units + price for one market-feed product, condition matched (M18), variants summed."""
    rows = daily_market[daily_market["tcg_id"] == int(tcg_id)]
    if condition is not None and condition in CONDITION_TO_TCG:
        wanted = CONDITION_TO_TCG[condition]
        matched = rows[rows["condition"] == wanted]
        if not matched.empty:
            rows = matched
    if rows.empty:
        return pd.DataFrame(columns=["date", "online_units", "online_price", "tcg_market"])
    rows = rows.copy()
    rows["value"] = rows["avg_price"] * rows["units"]
    daily = rows.groupby("date").agg(online_units=("units", "sum"), value=("value", "sum"),
                                     tcg_market=("tcg_market", "mean"), any_price=("avg_price", "mean"))
    daily["online_price"] = np.where(daily["online_units"] > 0, daily["value"] / daily["online_units"].replace(0, np.nan),
                                     daily["any_price"])
    # Smooth price over 7 days so a single odd sale doesn't move m (P4).
    daily["online_price"] = daily["online_price"].rolling(7, min_periods=1).median()
    return daily.drop(columns=["value", "any_price"]).reset_index()


def product_comparison(sold, mapping, store_panel, daily_market):
    """One row per mapped product: store units/price vs online units/price over the product's in-stock days."""
    rows = []
    mapped = sold.merge(mapping[["key", "tcg_id"]], on="key", how="inner")
    mapped = mapped[mapped["tcg_id"].notna()]
    for _, product in mapped.iterrows():
        online = online_daily_for(daily_market, product["tcg_id"], product.get("condition"))
        if online.empty:
            continue
        store = store_panel[(store_panel["product"] == product["product"]) & (store_panel["variant"] == product["variant"])]
        store = store[store["in_stock"]]
        joined = store.merge(online, on="date", how="left")
        days = int(len(joined))
        store_units = float(joined["sold"].sum())
        online_units = float(joined["online_units"].fillna(0).sum())
        online_price = float(joined["online_price"].median()) if joined["online_price"].notna().any() else np.nan
        our_price = float(product["list_price"])
        rows.append({
            "key": product["key"], "product": product["product"], "variant": product["variant"],
            "seller": product["seller"], "product_type": product["product_type"], "set": product["set"],
            "tcg_id": int(product["tcg_id"]), "in_stock_days": days,
            "store_units": store_units, "online_units": online_units,
            "store_per_week": store_units / max(days, 1) * 7, "online_per_week": online_units / max(days, 1) * 7,
            "our_price": our_price, "online_price": online_price,
            "premium": our_price / online_price if online_price and online_price > 0 else np.nan,
            "capture_rate": store_units / online_units if online_units > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def type_summary(comparison):
    """Premium and capture rate by product type (medians, so one odd product doesn't dominate)."""
    if comparison.empty:
        return comparison
    grouped = comparison.groupby("product_type").agg(
        products=("key", "count"), store_units=("store_units", "sum"), online_units=("online_units", "sum"),
        median_premium=("premium", "median"), p25_premium=("premium", lambda s: s.quantile(0.25)),
        p75_premium=("premium", lambda s: s.quantile(0.75)), median_capture=("capture_rate", "median")).reset_index()
    grouped["pooled_capture"] = grouped["store_units"] / grouped["online_units"].replace(0, np.nan)
    return grouped


def weekly_pairs(sold, mapping, store_panel, daily_market):
    """Weekly store units vs online units per product (in-stock weeks only)."""
    frames = []
    mapped = sold.merge(mapping[["key", "tcg_id"]], on="key", how="inner")
    mapped = mapped[mapped["tcg_id"].notna()]
    for _, product in mapped.iterrows():
        online = online_daily_for(daily_market, product["tcg_id"], product.get("condition"))
        store = store_panel[(store_panel["product"] == product["product"]) & (store_panel["variant"] == product["variant"])]
        joined = store.merge(online, on="date", how="left")
        joined["week"] = joined["date"].dt.to_period("W-SUN")
        joined["online_units"] = pd.to_numeric(joined["online_units"], errors="coerce").fillna(0.0)
        weekly = joined.groupby("week").agg(store_units=("sold", "sum"), in_stock_days=("in_stock", "sum"),
                                            online_units=("online_units", "sum")).reset_index()
        weekly = weekly[weekly["in_stock_days"] >= 4]
        weekly["key"] = product["key"]
        weekly["product_type"] = product["product_type"]
        frames.append(weekly)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def lag_test(weekly, max_lag=2):
    """Correlation between online weekly units (lagged 0..max_lag weeks) and store weekly units, within product.

    Both series are demeaned per product, then pooled, so it measures co-movement, not product size.
    """
    rows = []
    if weekly.empty:
        return pd.DataFrame()
    for lag in range(max_lag + 1):
        pieces = []
        for key, group in weekly.sort_values("week").groupby("key"):
            group = group.copy()
            group["online_lagged"] = group["online_units"].astype(float).shift(lag)
            group["store_units"] = group["store_units"].astype(float)
            group = group.dropna(subset=["online_lagged"])
            if len(group) < 3:
                continue
            group["store_dm"] = group["store_units"] - group["store_units"].mean()
            group["online_dm"] = np.log1p(group["online_lagged"]) - np.log1p(group["online_lagged"]).mean()
            pieces.append(group)
        if not pieces:
            rows.append({"lag_weeks": lag, "corr": np.nan, "pairs": 0})
            continue
        pooled = pd.concat(pieces)
        corr = float(np.corrcoef(pooled["store_dm"], pooled["online_dm"])[0, 1]) if len(pooled) > 2 else np.nan
        rows.append({"lag_weeks": lag, "corr": corr, "pairs": int(len(pooled))})
    return pd.DataFrame(rows)


def online_index(weekly, recent_weeks=4):
    """P3 outside-market multiplier per product: recent online units/week vs its whole-window average."""
    index = {}
    if weekly.empty:
        return index
    for key, group in weekly.sort_values("week").groupby("key"):
        overall = group["online_units"].mean()
        recent = group["online_units"].tail(recent_weeks).mean()
        if overall and overall > 0:
            index[key] = float(np.clip(recent / overall, 0.5, 2.0))
    return index
