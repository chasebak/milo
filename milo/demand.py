"""Demand steps P1-P3 (plan section 4).

P1  raw rate     units per in-stock week, per family (demand unit)
P2  shrunk rate  gamma-Poisson posterior, pulled toward the product type's average
P3  time effects store traffic trend x outside-market index x holiday uplift (scenario input)
"""
import numpy as np
import pandas as pd


def stock_panel(events, end_date=None, closed_weekdays=(2,)):
    """Daily panel per (product, variant): units sold, stock at start of day, in_stock flag.

    Stock starts at 0 before the first event (every product's history starts with a stock_in).
    A day counts as in stock if stock at the start of the day > 0 or something sold or arrived that day.
    closed_weekdays (0=Monday): the store's closed days, from StoreConfig.store_closed_weekdays.
    """
    if end_date is None:
        end_date = events["date"].max()
    signed = events.copy()
    if "sell_date" in signed:
        # Restocks count from the day they can first sell (shelf rule), not the day they were entered.
        stock_in_rows = signed["kind"] == "stock_in"
        signed.loc[stock_in_rows, "date"] = signed.loc[stock_in_rows, "sell_date"]
    if "bulk" not in signed:
        signed["bulk"] = False
    signed["delta"] = 0
    signed.loc[signed["kind"] == "stock_in", "delta"] = signed["units"]
    signed.loc[signed["kind"] == "stock_out", "delta"] = -signed["units"]
    signed.loc[signed["kind"] == "sale", "delta"] = -signed["units"]
    signed["sold"] = 0
    # Demand counts leave bulk orders out; the stock math above still removes them from the shelf.
    normal_sale = (signed["kind"] == "sale") & (~signed["bulk"].astype(bool))
    signed.loc[normal_sale, "sold"] = signed["units"]
    signed["bulk_units"] = 0
    bulk_sale = (signed["kind"] == "sale") & signed["bulk"].astype(bool)
    signed.loc[bulk_sale, "bulk_units"] = signed["units"]
    signed["arrived"] = 0
    signed.loc[signed["kind"] == "stock_in", "arrived"] = signed["units"]

    daily = signed.groupby(["product", "variant", "date"]).agg(
        delta=("delta", "sum"), sold=("sold", "sum"), arrived=("arrived", "sum"),
        bulk_units=("bulk_units", "sum")).reset_index()

    frames = []
    groups = daily.groupby(["product", "variant"])
    for (product, variant), group in groups:
        start = group["date"].min()
        days = pd.date_range(start, end_date, freq="D")
        frame = pd.DataFrame({"date": days})
        frame = frame.merge(group[["date", "delta", "sold", "arrived", "bulk_units"]], on="date", how="left").fillna(0)
        end_stock = frame["delta"].cumsum()
        frame["stock_start"] = end_stock.shift(1).fillna(0).clip(lower=0)
        frame["stock_end"] = end_stock.clip(lower=0)
        # A closed day only counts if something actually sold (online orders).
        open_day = ~frame["date"].dt.day_of_week.isin(list(closed_weekdays))
        has_stock = (frame["stock_start"] > 0) | (frame["arrived"] > 0)
        frame["in_stock"] = (has_stock & open_day) | (frame["sold"] > 0)
        frame["product"] = product
        frame["variant"] = variant
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    return panel


def family_panel(panel, catalog_table):
    """Roll the product panel up to the demand unit (family). A family is in stock if any member is."""
    keyed = panel.merge(catalog_table[["product", "variant", "family", "product_type", "seller", "in_scope"]],
                        on=["product", "variant"], how="left")
    keyed = keyed[keyed["in_scope"] == True]  # noqa: E712  (column may hold NaN)
    # The demand unit is seller + family: two sellers' packs of the same set are separate units on one shelf.
    keyed = keyed.rename(columns={"family": "product_family"})
    keyed["family"] = keyed["seller"] + "|" + keyed["product_family"]
    rolled = keyed.groupby(["family", "date"]).agg(
        sold=("sold", "sum"), in_stock=("in_stock", "max"), stock_end=("stock_end", "sum"),
        product_type=("product_type", "first"), seller=("seller", "first"),
        product_family=("product_family", "first")).reset_index()
    return rolled


def p1_raw_rates(fam_panel, start=None, end=None):
    """P1: units per in-stock week for each family over [start, end]."""
    window = fam_panel
    if start is not None:
        window = window[window["date"] >= pd.Timestamp(start)]
    if end is not None:
        window = window[window["date"] <= pd.Timestamp(end)]
    in_stock_rows = window[window["in_stock"]]
    rates = in_stock_rows.groupby("family").agg(
        units=("sold", "sum"), in_stock_days=("date", "nunique"),
        product_type=("product_type", "first"), seller=("seller", "first")).reset_index()
    rates["exposure_weeks"] = rates["in_stock_days"] / 7.0
    rates["raw_rate"] = rates["units"] / rates["exposure_weeks"]
    return rates


def gamma_prior(units, exposure):
    """Method-of-moments gamma prior (shape a, rate b) for weekly rates in one product type.

    Mean m = total units / total exposure. Between-family variance = variance of raw rates minus the
    Poisson noise part (m / mean exposure); floored so the prior never becomes a spike.
    """
    total_exposure = float(np.sum(exposure))
    if total_exposure <= 0:
        return 1.0, 1.0
    mean_rate = float(np.sum(units)) / total_exposure
    if mean_rate <= 0:
        return 0.5, 1.0
    raw = np.asarray(units, dtype=float) / np.maximum(np.asarray(exposure, dtype=float), 1e-9)
    noise = mean_rate / max(float(np.mean(exposure)), 1e-9)
    between = float(np.var(raw)) - noise
    floor = (0.5 * mean_rate) ** 2
    between = max(between, floor)
    shape = mean_rate ** 2 / between
    rate = mean_rate / between
    return shape, rate


def p2_shrink(rates, min_types=1):
    """P2: posterior mean rate per family = (a + units) / (b + exposure weeks), prior per product type."""
    out = rates.copy()
    out["prior_shape"] = np.nan
    out["prior_rate"] = np.nan
    types = list(out["product_type"].dropna().unique())
    for product_type in types:
        mask = out["product_type"] == product_type
        shape, rate = gamma_prior(out.loc[mask, "units"].to_numpy(), out.loc[mask, "exposure_weeks"].to_numpy())
        out.loc[mask, "prior_shape"] = shape
        out.loc[mask, "prior_rate"] = rate
    out["post_shape"] = out["prior_shape"] + out["units"]
    out["post_rate"] = out["prior_rate"] + out["exposure_weeks"]
    out["shrunk_rate"] = out["post_shape"] / out["post_rate"]
    out["p10"] = gamma_quantile(out["post_shape"], out["post_rate"], 0.10)
    out["p50"] = gamma_quantile(out["post_shape"], out["post_rate"], 0.50)
    out["p90"] = gamma_quantile(out["post_shape"], out["post_rate"], 0.90)
    flags = []
    for _, row in out.iterrows():
        reasons = []
        if row["units"] <= 2:
            reasons.append("thin data (<=2 sales)")
        if row["exposure_weeks"] < 2.5:
            reasons.append("short history (<2.5 in-stock weeks)")
        flags.append("; ".join(reasons))
    out["flag"] = flags
    return out


def gamma_quantile(shape, rate, q):
    from scipy.stats import gamma
    return gamma.ppf(q, a=np.asarray(shape, dtype=float), scale=1.0 / np.asarray(rate, dtype=float))


def traffic_index(orders, weeks=4):
    """Store traffic: orders per week, last `weeks` weeks vs the whole history (P3 multiplier)."""
    weekly = orders.set_index("date").resample("W-SUN")["Order"].count()
    if len(weekly) == 0:
        return 1.0, weekly
    recent = float(weekly.tail(weeks).mean())
    overall = float(weekly.mean())
    if overall <= 0:
        return 1.0, weekly
    return recent / overall, weekly


def p3_forecast(shrunk, horizon_weeks, traffic_mult=1.0, online_index=None, season_uplift=1.0):
    """P3: expected horizon demand at today's price = shrunk weekly rate x weeks x multipliers.

    online_index: optional Series family -> multiplier from online market demand (1.0 when missing).
    """
    out = shrunk.copy()
    out["traffic_mult"] = traffic_mult
    out["online_mult"] = 1.0
    if online_index is not None:
        mapped = out["family"].map(online_index)
        out["online_mult"] = mapped.fillna(1.0)
    out["season_uplift"] = season_uplift
    combined = out["traffic_mult"] * out["online_mult"] * out["season_uplift"]
    out["weekly_forecast"] = out["shrunk_rate"] * combined
    out["horizon_demand"] = out["weekly_forecast"] * horizon_weeks
    return out


def sample_rates(shrunk, n_scenarios, rng):
    """Draw weekly rates from each family's gamma posterior: array (families, scenarios)."""
    shape = shrunk["post_shape"].to_numpy(dtype=float)
    scale = 1.0 / shrunk["post_rate"].to_numpy(dtype=float)
    draws = rng.gamma(shape[:, None], scale[:, None], size=(len(shape), n_scenarios))
    return draws


def store_orders_per_day(orders):
    """Whole-store orders per day from the store-wide sequential order numbers.

    The ledger only lists orders holding tracked sellers' items, but order numbers count every store
    order, so the order number at each timestamp, interpolated to midnight, gives total store orders
    between days.
    """
    ordered = orders.sort_values("ts")
    times = ordered["ts"].astype("int64").to_numpy(dtype=float)
    numbers = ordered["Number"].to_numpy(dtype=float)
    start = ordered["date"].min()
    end = ordered["date"].max() + pd.Timedelta(days=1)
    midnights = pd.date_range(start, end, freq="D")
    at_midnight = np.interp(midnights.astype("int64").to_numpy(dtype=float), times, numbers)
    per_day = pd.Series(np.diff(at_midnight), index=midnights[:-1], name="store_orders")
    return per_day


def traffic_ratio(orders, fit_end, test_end=None, fit_start=None, store_daily=None):
    """Store orders per day in the test window divided by the fit window (P3 traffic step).

    Uses the store's daily estimates when given (store_daily), else interpolates order numbers here.
    Partial days (first/last) are left out.
    """
    if store_daily is not None:
        full = store_daily[store_daily["day_kind"] != "partial"]
        per_day = full.set_index("date")["store_orders"]
    else:
        per_day = store_orders_per_day(orders)
    fit_part = per_day[per_day.index <= pd.Timestamp(fit_end)]
    if fit_start is not None:
        fit_part = fit_part[fit_part.index >= pd.Timestamp(fit_start)]
    test_part = per_day[per_day.index > pd.Timestamp(fit_end)]
    if test_end is not None:
        test_part = test_part[test_part.index <= pd.Timestamp(test_end)]
    fit_rate = float(fit_part.mean()) if len(fit_part) else 0.0
    test_rate = float(test_part.mean()) if len(test_part) else 0.0
    if fit_rate <= 0:
        return 1.0
    return test_rate / fit_rate


def backtest(fam_panel, split_date, end_date=None, traffic_mult=1.0):
    """Fit P1+P2 up to split_date, predict units after it using the actual in-stock days. Error by type.

    traffic_mult: P3 store-traffic change between the fit window and the test window (1.0 = none).
    """
    fit = p2_shrink(p1_raw_rates(fam_panel, end=split_date))
    test_window = fam_panel[fam_panel["date"] > pd.Timestamp(split_date)]
    if end_date is not None:
        test_window = test_window[test_window["date"] <= pd.Timestamp(end_date)]
    test = test_window[test_window["in_stock"]].groupby("family").agg(
        actual=("sold", "sum"), test_days=("date", "nunique")).reset_index()
    joined = test.merge(fit[["family", "product_type", "shrunk_rate", "raw_rate"]], on="family", how="inner")
    joined["pred_shrunk"] = joined["shrunk_rate"] * joined["test_days"] / 7.0 * traffic_mult
    joined["pred_raw"] = joined["raw_rate"] * joined["test_days"] / 7.0 * traffic_mult
    joined["err_shrunk"] = joined["pred_shrunk"] - joined["actual"]
    joined["err_raw"] = joined["pred_raw"] - joined["actual"]
    by_type = joined.groupby("product_type").agg(
        families=("family", "count"), actual=("actual", "sum"), pred_shrunk=("pred_shrunk", "sum"),
        pred_raw=("pred_raw", "sum"),
        mae_shrunk=("err_shrunk", lambda s: float(np.mean(np.abs(s)))),
        mae_raw=("err_raw", lambda s: float(np.mean(np.abs(s))))).reset_index()
    return joined, by_type
