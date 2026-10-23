"""End-to-end pipeline used by the notebook: data -> baseline -> online vs in-person -> price response ->
restock + pricing optimization -> strategy comparison -> Excel exports. Every number shown comes from here.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from milo import catalog, compare, demand, fees, load, market, optimize, pricing, simulate, tcgmap
from milo.config import EXAMPLE_CONFIG

PROJECT = load.PROJECT
DEFAULT_DATA = PROJECT / "data" / "synthetic"
DEFAULT_OUT = PROJECT / "out"
DEFAULT_COST_SHARE = {"sealed": 0.75, "single": 0.50}   # buy cost as a share of market price when unknown
DEFAULT_SALVAGE = {"sealed": 0.85, "single": 0.60}

# Analysis window defaults, pinned to the committed synthetic dataset (synth.DEFAULTS).
DEFAULT_TODAY = "2026-10-20"
DEFAULT_SPLIT = "2026-09-27"
DEFAULT_TEST_END = "2026-10-18"
DEFAULT_FIT_START = "2026-07-13"


# ---------------------------------------------------------------- 1. data
def load_all(data_dir=DEFAULT_DATA, today=DEFAULT_TODAY, config=EXAMPLE_CONFIG):
    raw_dir = Path(data_dir) / "raw"
    inputs_dir = Path(data_dir) / "inputs"
    events = load.load_store_events(raw_dir, sellers=config.sellers)
    ledger = load.load_unit_costs(raw_dir)
    cat = catalog.build_catalog(events, overrides_path=inputs_dir / "catalog_overrides.csv", ledger=ledger,
                                sellers=config.sellers)
    panel = demand.stock_panel(events, closed_weekdays=config.store_closed_weekdays)
    fam = demand.family_panel(panel, cat)
    commission = load.load_commission_lines(raw_dir)
    orders = load.load_order_ledger(raw_dir)
    # Keys (P000...) come from the saved list so they stay matched to inputs/tcg_map.csv; the parsed columns
    # (set, family, type, condition) are refreshed from today's catalog so parser fixes always apply.
    saved = pd.read_csv(raw_dir / "sold_products.csv", parse_dates=["first_sale", "last_sale"])
    refreshed = tcgmap.sold_products(events, cat)
    sold = saved[["key", "product", "variant"]].merge(refreshed, on=["product", "variant"], how="inner")
    mapping = load.load_tcg_map(inputs_dir / "tcg_map.csv")
    daily = market.daily_market(raw_dir, today=pd.Timestamp(today))
    return {"events": events, "ledger": ledger, "catalog": cat, "panel": panel, "fam": fam,
            "commission": commission, "orders": orders, "sold": sold, "mapping": mapping, "daily": daily,
            "store_daily": load.load_store_daily(raw_dir),
            "weekday": load.load_weekday_shares(raw_dir, closed_weekdays=config.store_closed_weekdays),
            "restock_tables": load.load_restock_tables(raw_dir),
            "card_share": load.card_share(commission), "units_per_order": load.units_per_order(events),
            "data_dir": Path(data_dir), "inputs_dir": inputs_dir, "today": pd.Timestamp(today), "config": config}


def data_summary(data):
    events = data["events"]
    rows = []
    sellers = sorted(events["seller"].unique())
    for seller in sellers:
        part = events[(events["seller"] == seller) & (events["kind"] == "sale")]
        rows.append({"source": "store sales (" + seller + ")", "rows": len(part), "units": int(part["units"].sum()),
                     "from": part["date"].min().date(), "to": part["date"].max().date()})
    daily = data["daily"]
    if not daily.empty:
        sales_days = daily[daily["resolution"] == "sales"]
        rows.append({"source": "market feed daily (from individual sales)", "rows": len(sales_days),
                     "units": float(sales_days["units"].sum()), "from": sales_days["date"].min().date(),
                     "to": sales_days["date"].max().date()})
        rows.append({"source": "market feed products pulled", "rows": int(daily["tcg_id"].nunique()),
                     "units": np.nan, "from": daily["date"].min().date(), "to": daily["date"].max().date()})
    rows.append({"source": "Commission workbook lines", "rows": len(data["commission"]), "units": np.nan,
                 "from": data["commission"]["date"].min().date(), "to": data["commission"]["date"].max().date()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 3c. online vs in-person
def online_vs_inperson(data):
    sold = data["sold"].copy()
    comparison = compare.product_comparison(sold, data["mapping"], data["panel"], data["daily"])
    weekly = compare.weekly_pairs(sold, data["mapping"], data["panel"], data["daily"])
    return {"products": comparison, "by_type": compare.type_summary(comparison),
            "lags": compare.lag_test(weekly), "weekly": weekly,
            "online_index_by_key": compare.online_index(weekly)}


def unit_online_index(data, online_by_key):
    """Average the per-product online index up to the demand unit (seller|family)."""
    sold = data["sold"]
    rows = []
    for _, row in sold.iterrows():
        if row["key"] in online_by_key:
            unit = row["seller"] + "|" + row["family"]
            rows.append({"family": unit, "index": online_by_key[row["key"]]})
    if not rows:
        return pd.Series(dtype=float)
    return pd.DataFrame(rows).groupby("family")["index"].mean()


# ---------------------------------------------------------------- P4/P5
def fixed_effects_panel(data):
    """Product-day rows (in-stock days only) with our price, market price and online units (P4)."""
    frames = []
    mapped = data["sold"].merge(data["mapping"][["key", "tcg_id"]], on="key", how="inner")
    for _, product in mapped.iterrows():
        online = compare.online_daily_for(data["daily"], product["tcg_id"], product.get("condition"))
        if online.empty:
            continue
        store = data["panel"][(data["panel"]["product"] == product["product"]) &
                              (data["panel"]["variant"] == product["variant"]) & data["panel"]["in_stock"]]
        joined = store.merge(online, on="date", how="inner")
        joined = joined[joined["online_price"] > 0]
        if joined.empty:
            continue
        joined["unit"] = product["key"]
        joined["group"] = pricing.type_group(product["product_type"])
        joined["our_price"] = float(product["list_price"])
        joined["market_price"] = joined["online_price"]
        joined["units"] = joined["sold"]
        smoothed = joined["online_units"].fillna(0).rolling(7, min_periods=1).mean()
        joined["log_market_units"] = np.log1p(smoothed)
        joined["week"] = joined["date"].dt.to_period("W-SUN")
        joined["dow"] = joined["date"].dt.day_name()
        joined["weight"] = 1.0
        frames.append(joined)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    return pricing.relative_price(panel)


def price_sensitivity(data, comparison, n_boot=60):
    fe_panel = fixed_effects_panel(data)
    fe_fits = {}
    fe_info = []
    if not fe_panel.empty:
        groups = list(fe_panel["group"].unique())
        for group in groups:
            part = fe_panel[fe_panel["group"] == group]
            movement = part.groupby("unit")["m"].agg(lambda s: float(s.max() / s.min()) if s.min() > 0 else 1.0)
            moving_units = movement[movement > 1.05].index
            usable = part[part["unit"].isin(moving_units)]
            fe_info.append({"group": group, "units": int(part["unit"].nunique()),
                            "units_with_price_movement": int(len(moving_units)), "product_days": int(len(usable))})
            if len(moving_units) >= 5 and len(usable) >= 25:
                fe_fits[group] = pricing.fit_fixed_effects(usable, n_boot=n_boot)
    cross = {}
    if comparison is not None and not comparison.empty:
        for group in ["sealed", "single"]:
            part = comparison[comparison["product_type"].map(pricing.type_group) == group].copy()
            part = part.rename(columns={"store_per_week": "rate", "premium": "m"})
            part["set_popularity"] = part["online_per_week"]
            cross[group] = pricing.fit_cross_section(part)
    table = pricing.combine_elasticities(fe_fit=fe_fits, cross_fit=cross)
    return {"elasticities": table, "fe_fits": fe_fits, "fe_info": pd.DataFrame(fe_info), "cross": cross,
            "fe_panel": fe_panel}


# ---------------------------------------------------------------- restock units
def restock_units(data, rates, comparison):
    """The primary seller's in-scope demand units with price now, market price, cost, on hand and limits."""
    config = data["config"]
    primary = config.primary_seller
    sold = data["sold"]
    own = sold[sold["seller"] == primary].copy()
    own["unit"] = primary + "|" + own["family"]
    if not comparison.empty:
        comp_prices = comparison[["key", "online_price"]]
    else:
        comp_prices = pd.DataFrame(columns=["key", "online_price"])
    own = own.merge(comp_prices, on="key", how="left")
    ledger = data["ledger"][["sku", "cost"]].dropna()
    own = own.merge(ledger.drop_duplicates("sku"), on="sku", how="left")
    stock_now = data["panel"].sort_values("date").groupby(["product", "variant"])["stock_end"].last().rename("on_hand")
    own = own.merge(stock_now.reset_index(), on=["product", "variant"], how="left")
    grouped = own.groupby("unit").agg(
        product_type=("product_type", "first"), set_name=("set", "first"),
        price_now=("list_price", "median"), market_price=("online_price", "median"),
        ledger_cost=("cost", "median"), on_hand=("on_hand", "sum"), products=("key", "count")).reset_index()
    grouped = grouped.merge(rates[["family", "shrunk_rate", "post_shape", "post_rate", "flag"]],
                            left_on="unit", right_on="family", how="inner").drop(columns="family")
    costs_file = data["inputs_dir"] / "restock_costs.csv"
    overrides = pd.read_csv(costs_file) if costs_file.exists() else pd.DataFrame(columns=["unit"])
    rows = []
    for _, unit in grouped.iterrows():
        group = pricing.type_group(unit["product_type"])
        reference = unit["market_price"] if pd.notna(unit["market_price"]) else unit["price_now"]
        cost = unit["ledger_cost"]
        cost_source = "unit cost sheet"
        if pd.isna(cost):
            cost = DEFAULT_COST_SHARE[group] * reference
            cost_source = "default " + str(int(DEFAULT_COST_SHARE[group] * 100)) + "% of market price"
        row = {"unit": unit["unit"], "product_type": unit["product_type"], "set_code": unit["set_name"] or "unknown",
               "price_now": float(unit["price_now"]), "market_price": unit["market_price"], "cost": float(cost),
               "cost_source": cost_source, "on_hand": float(max(unit["on_hand"], 0) if pd.notna(unit["on_hand"]) else 0),
               "min_order": 1, "max_units": 24, "salvage_frac": DEFAULT_SALVAGE[group],
               "shrunk_rate": unit["shrunk_rate"], "post_shape": unit["post_shape"], "post_rate": unit["post_rate"],
               "data_flag": unit["flag"]}
        match = overrides[overrides["unit"] == unit["unit"]] if "unit" in overrides else pd.DataFrame()
        if not match.empty:
            fix = match.iloc[0]
            for column in ["cost", "min_order", "max_units", "salvage_frac"]:
                if column in fix and pd.notna(fix[column]):
                    row[column] = float(fix[column])
                    if column == "cost":
                        row["cost_source"] = "restock_costs.csv"
        rows.append(row)
    table = pd.DataFrame(rows)
    # Products another seller also carries: the primary seller's demand there can depend on the other
    # seller's stock. v1 treats demand as independent, so these are marked for a manual look.
    other_families = set(sold[sold["seller"] != primary]["family"])
    shared = []
    for _, row in table.iterrows():
        product_family = row["unit"].split("|", 1)[1]
        shared.append(product_family in other_families)
    table["shared_with_other_seller"] = shared
    return table


def write_cost_template(units, path):
    """Write a one-row-per-unit cost template with the default cost, for hand editing (kept if it exists)."""
    path = Path(path)
    if path.exists():
        return path
    template = units[["unit", "product_type", "price_now", "market_price", "cost", "cost_source", "min_order", "max_units",
                      "salvage_frac"]].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------- P6/P7 scenarios
def build_scenarios(units, elasticities, horizon_weeks, lead_weeks, traffic_mult, online_index, season_uplift,
                    n_scenarios=120, seed=11, card_share=0.8, units_per_order=2.3, config=EXAMPLE_CONFIG):
    """Price grid, net cash per level and ordered demand D[i, k, w] (P6 + P7).

    Scenarios mix three elasticity sets (low / base / high, equal weight) with gamma rate draws; ONE uniform per
    (unit, scenario) is shared across price levels, so demand never rises with price (M11).
    """
    rng = np.random.default_rng(seed)
    units = units.reset_index(drop=True)
    grid = pricing.price_grid(units.rename(columns={"unit": "unit"}))
    grid["net_cash"] = fees.net_cash_per_unit(grid["price"].to_numpy(), config, card_share, units_per_order)
    levels = sorted(grid["k"].unique())
    n_units = len(units)
    n_levels = len(pricing.PRICE_LEVELS)
    weeks_new = max(horizon_weeks - lead_weeks, 0)
    online = units["unit"].map(online_index).fillna(1.0).to_numpy() if online_index is not None else np.ones(n_units)
    multiplier_time = traffic_mult * online * season_uplift
    shape = units["post_shape"].to_numpy(dtype=float)
    scale = 1.0 / units["post_rate"].to_numpy(dtype=float)
    rates = rng.gamma(shape[:, None], scale[:, None], size=(n_units, n_scenarios)) * multiplier_time[:, None]
    uniforms = np.clip(rng.uniform(size=(n_units, n_scenarios)), 1e-9, 1 - 1e-9)
    elasticity_pick = np.array(["low", "base", "high"])[np.arange(n_scenarios) % 3]
    demand_array = np.zeros((n_units, n_levels, n_scenarios))
    for i in range(n_units):
        unit = units.iloc[i]
        group = pricing.type_group(unit["product_type"])
        m_now = 1.0
        if pd.notna(unit["market_price"]) and unit["market_price"] > 0:
            m_now = unit["price_now"] / unit["market_price"]
        for name in ("low", "base", "high"):
            below = elasticities[(elasticities["group"] == group) & (elasticities["side"] == "below")].iloc[0][name]
            above = elasticities[(elasticities["group"] == group) & (elasticities["side"] == "above")].iloc[0][name]
            mult = pricing.level_multipliers(m_now, below, above)
            columns = np.where(elasticity_pick == name)[0]
            for w in columns:
                # Lead-time rule (M9) — per PRODUCT, not per unit: products already in stock sell over the
                # whole horizon, including newly bought units; only out-of-stock products wait out the lead time.
                weeks = horizon_weeks if unit["on_hand"] > 0 else weeks_new
                mean = rates[i, w] * weeks * mult
                demand_array[i, :, w] = poisson.ppf(uniforms[i, w], np.maximum(mean, 1e-9))
    keep_levels = levels
    demand_array = demand_array[:, keep_levels, :]
    return grid, demand_array


def optimizer_inputs(units, grid):
    products = units.set_index("unit")[["cost", "on_hand", "min_order", "max_units", "set_code", "salvage_frac"]]
    prices = grid.rename(columns={})[["i", "k", "ratio", "price", "net_cash"]]
    return products, prices


def fill_missing_levels(products, prices, demand_array):
    """Units whose grid lost levels (1.6x market cap) get those levels copied from the nearest kept level with
    zero demand, so every unit has the same K for the optimizer; the optimizer drops them (M1 filter) anyway."""
    complete = []
    all_levels = list(range(demand_array.shape[1]))
    for unit_id in products.index:
        rows = prices[prices["i"] == unit_id]
        for k in all_levels:
            if k in set(rows["k"]):
                complete.append(rows[rows["k"] == k].iloc[0].to_dict())
            else:
                ratio = float(pricing.PRICE_LEVELS[k])
                complete.append({"i": unit_id, "k": k, "ratio": ratio, "price": np.nan, "net_cash": -1e6})
    return pd.DataFrame(complete)


# ---------------------------------------------------------------- run everything
def run(horizon_weeks=8, lead_weeks=1, budgets=(0, 1000, 2500, 5000, 10000), main_budget=5000,
        season_uplift=1.3, n_scenarios=60, set_cap_share=0.35, n_boot=60, curve_scenarios=30, time_limit=120,
        data_dir=DEFAULT_DATA, today=DEFAULT_TODAY, config=EXAMPLE_CONFIG,
        split_date=DEFAULT_SPLIT, test_end=DEFAULT_TEST_END, fit_start=DEFAULT_FIT_START,
        cost_template_path=None):
    data = load_all(data_dir=data_dir, today=today, config=config)
    out = {"data": data, "summary": data_summary(data), "weekday": data["weekday"],
           "restock_tables": data["restock_tables"]}
    rates = demand.p2_shrink(demand.p1_raw_rates(data["fam"]))
    out["p1p2"] = rates
    online = online_vs_inperson(data)
    out["online"] = online
    unit_index = unit_online_index(data, online["online_index_by_key"])
    out["unit_online_index"] = unit_index
    traffic = demand.traffic_ratio(data["orders"], split_date, test_end, fit_start, data["store_daily"])
    out["traffic_test_vs_fit"] = traffic
    out["backtest_plain"] = demand.backtest(data["fam"], split_date, test_end, 1.0)
    out["backtest_traffic"] = demand.backtest(data["fam"], split_date, test_end, traffic)
    sens = price_sensitivity(data, online["products"], n_boot=n_boot)
    out["sensitivity"] = sens
    units = restock_units(data, rates, online["products"])
    if cost_template_path is not None:
        write_cost_template(units, cost_template_path)
    out["units"] = units
    grid, demand_array = build_scenarios(units, sens["elasticities"], horizon_weeks, lead_weeks, 1.0, unit_index,
                                         season_uplift, n_scenarios, card_share=data["card_share"],
                                         units_per_order=data["units_per_order"], config=config)
    products, prices = optimizer_inputs(units, grid)
    prices = fill_missing_levels(products, prices, demand_array)
    out["products"], out["prices"], out["demand"] = products, prices, demand_array
    plan = optimize.solve_plan(products, prices, demand_array, budget=main_budget, set_cap_share=set_cap_share,
                               time_limit=time_limit)
    out["plan"] = plan
    # The budget curve re-solves once per budget; it uses the first `curve_scenarios` scenarios (the low/base/high
    # elasticity mix repeats every 3, so the subset keeps the same mix) to keep each solve short.
    small = demand_array[:, :, :curve_scenarios]
    curve = optimize.budget_curve(products, prices, small, list(budgets), set_cap_share=set_cap_share,
                                  time_limit=60, lp_relaxation=False)
    # Profit at budget 0 is what the shelf you already own earns; the gain over it is what the restock adds.
    base_row = curve[curve["budget"] == 0]
    base_profit = float(base_row["objective"].iloc[0]) if not base_row.empty else np.nan
    curve["profit_from_restock"] = curve["objective"] - base_profit
    curve["return_per_dollar"] = curve["profit_from_restock"] / curve["spend"].replace(0, np.nan)
    out["budget_curve"] = curve
    out["edge_reasons"] = edge_reasons(units, plan, demand_array, prices)
    out["strategies"] = strategy_comparison(units, products, prices, demand_array, plan)
    out["weekend"], out["weekend_share"] = weekend_check(units, plan, data["weekday"], unit_index, season_uplift,
                                                         horizon_weeks)
    return out


def edge_reasons(units, plan, demand_array, prices):
    """For products priced at the top level (M2/M3): is it scarcity (stock below demand) or low sensitivity?"""
    chosen = plan["prices"]
    if chosen.empty:
        return pd.DataFrame()
    top_level = int(prices["k"].max())
    buy = plan["buy"].set_index("i")["units"] if not plan["buy"].empty else pd.Series(dtype=float)
    index_of = {}
    unit_ids = list(units["unit"])
    for position, unit_id in enumerate(unit_ids):
        index_of[unit_id] = position
    rows = []
    for _, row in chosen.iterrows():
        if int(row["level"]) != top_level or row["i"] not in index_of:
            continue
        position = index_of[row["i"]]
        stock = float(units.iloc[position]["on_hand"]) + float(buy.get(row["i"], 0.0))
        if stock <= 0:
            continue  # nothing to sell, so the price level is irrelevant
        demand_at_top = float(np.median(demand_array[position, top_level, :]))
        reason = "scarce: median demand at the top price still exceeds stock" if demand_at_top >= stock else \
            "low price sensitivity: revenue still rises at the top price"
        rows.append({"unit": row["i"], "stock": stock, "median_demand_at_top_price": demand_at_top, "reason": reason})
    return pd.DataFrame(rows)


def weekend_check(units, plan, weekday, online_index, season_uplift, horizon_weeks):
    """Will the stock cover a busy Fri-Sun? Shelf target = P90 of weekend demand at the chosen price.

    Fri-Sun share of an open week comes from the store's weekday table (store-wide orders).
    """
    if weekday is None:
        return pd.DataFrame(), np.nan
    weekend_share = float(weekday[weekday["weekday"].isin(["Fri", "Sat", "Sun"])]["share_of_week"].sum())
    buy = plan["buy"].set_index("i")["units"] if not plan["buy"].empty else pd.Series(dtype=float)
    rows = []
    for _, unit in units.iterrows():
        bought = float(buy.get(unit["unit"], 0.0))
        stock = unit["on_hand"] + bought
        if stock <= 0:
            continue
        online = float(online_index.get(unit["unit"], 1.0)) if online_index is not None else 1.0
        weekly = unit["shrunk_rate"] * online * season_uplift
        weekend_mean = weekly * weekend_share
        target = float(poisson.ppf(0.9, max(weekend_mean, 1e-9)))
        rows.append({"unit": unit["unit"], "stock_after_buy": stock, "weekly_forecast": weekly,
                     "weekend_mean": weekend_mean, "friday_shelf_target_p90": target,
                     "weekends_covered": stock / max(weekend_mean, 1e-9) if weekend_mean > 0 else np.inf,
                     "flag": "may sell out in one weekend - restock before Friday" if stock < target else ""})
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("weekend_mean", ascending=False)
    return table, weekend_share


def strategy_comparison(units, products, prices, demand_array, plan):
    """Same buy list as the optimal plan, different pricing rules, plus the optimal prices (fair comparison)."""
    buy = {}
    for _, row in plan["buy"].iterrows():
        buy[row["i"]] = int(row["units"])
    strategies = {}
    unit_table = units.copy()
    for rule in ["keep", "match_market", "market_minus_5", "market_plus_10"]:
        chosen = pricing.strategy_levels(unit_table, prices.dropna(subset=["price"]), rule)
        strategies[rule] = (buy, chosen)
    optimal_levels = {}
    for _, row in plan["prices"].iterrows():
        optimal_levels[row["i"]] = int(row["level"])
    # Products the optimizer left out (every level failed M1) but that still have stock keep today's price.
    keep_levels = strategies["keep"][1]
    for unit_id, level in keep_levels.items():
        if unit_id not in optimal_levels:
            optimal_levels[unit_id] = level
    strategies["optimal (Pyomo)"] = (buy, optimal_levels)
    return simulate.strategy_table(products, prices, demand_array, strategies)


def export(out, out_dir=DEFAULT_OUT):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    with pd.ExcelWriter(out_dir / "restock_plan.xlsx") as writer:
        plan = out["plan"]
        buy = plan["buy"].merge(plan["prices"], on="i", how="left")
        buy.to_excel(writer, sheet_name="Buy list + prices", index=False)
        out["budget_curve"].to_excel(writer, sheet_name="Profit vs budget", index=False)
        out["strategies"].to_excel(writer, sheet_name="Strategies", index=False)
        plan["flags"].to_excel(writer, sheet_name="Flags", index=False)
        out["sensitivity"]["elasticities"].to_excel(writer, sheet_name="Price sensitivity", index=False)
        if not out["weekend"].empty:
            out["weekend"].to_excel(writer, sheet_name="Weekend stock check", index=False)
    with pd.ExcelWriter(out_dir / "online_vs_inperson.xlsx") as writer:
        out["online"]["products"].to_excel(writer, sheet_name="By product", index=False)
        out["online"]["by_type"].to_excel(writer, sheet_name="By type", index=False)
        out["online"]["lags"].to_excel(writer, sheet_name="Lag test", index=False)
    return [out_dir / "restock_plan.xlsx", out_dir / "online_vs_inperson.xlsx"]
