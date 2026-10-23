"""Synthetic raw-dataset generator.

generate_dataset(out_dir, ...) writes a complete raw data directory in the exact shapes
milo.load / milo.market consume (see SCHEMA.md for every file and column):

    <out_dir>/raw/        seller_a_sales.csv, seller_b_sales.csv, order_ledger.csv,
                          store_daily.csv, store_weekday.csv, sold_products.csv,
                          unit_costs.csv, set_release_dates.csv,
                          restock/*.csv, commission/*.xlsx, tcg_sales/*.jsonl, tcg_history/*.jsonl
    <out_dir>/inputs/     tcg_map.csv, catalog_overrides.csv, restock_costs.csv

Everything is drawn from one numpy default_rng(seed), so the same knobs always produce
the same dataset. Knobs:
    seed                 rng seed
    n_products           number of store items (floor; the fixed archetypes are always included)
    weeks                length of the store history window
    start                first day of the window (a Monday keeps the weekly tables tidy)
    demand_spread        sigma of the lognormal spread of weekly demand rates across items
    elasticity_strength  scales every item's price elasticity (0 = price never matters)
    budget               price scale anchor: all prices/costs scale by budget/5000, so the
                         default restock budget stays binding at any scale
"""
import json
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from milo import catalog, load, tcgmap
from milo.config import DEFAULT_SELLERS
from milo.pricing import h as demand_curve

# Knobs of the COMMITTED dataset (frozen; scripts/regen_dataset.py uses these).
# seed/demand_spread/elasticity_strength were retuned from provisional 11/1.0/1.0
# until the acceptance bands held: VSS >= 2% of RP, greedy-vs-MILP gap >= 2% at >= 1
# budget, EVPI > 0 (measured values live in reports/numbers.json; see SCHEMA.md).
DEFAULTS = {"seed": 19, "n_products": 24, "weeks": 15, "start": "2026-07-06",
            "demand_spread": 1.4, "elasticity_strength": 1.6, "budget": 5000.0}

SETS = [
    {"name": "Shadow Summit", "code": "SS01", "release": "2026-05-30", "popularity": 1.6},
    {"name": "Ember Crown", "code": "EC02", "release": "2026-03-14", "popularity": 1.2},
    {"name": "Crystal Tide", "code": "CT03", "release": "2026-01-17", "popularity": 1.0},
    {"name": "Iron Verdict", "code": "IV04", "release": "2025-10-04", "popularity": 0.8},
    {"name": "Lunar Bloom", "code": "LB05", "release": "2025-07-12", "popularity": 0.6},
]

CARD_NAMES = ["Emberfox", "Tidalon", "Galeheart", "Duskmaw", "Thornelk", "Mirevyn", "Cindershell",
              "Bramblyn", "Frostwren", "Smoldra", "Pebblit", "Glimmoth", "Riptalon", "Vexbloom"]

STAFF = ["Alex Rowe", "Sam Field", "Jamie Park"]
WEB_IMPORT = "Web Import - Alex Rowe"
LOCATION = "Main St"
CLOSED_WEEKDAY = 2            # Wednesday, matching EXAMPLE_CONFIG.store_closed_weekdays
OPEN_DOW_FACTOR = {0: 0.80, 1: 0.85, 3: 0.95, 4: 1.30, 5: 1.50, 6: 1.25}   # demand by weekday
STORE_ORDER_MEANS = {0: 34, 1: 38, 2: 6, 3: 40, 4: 55, 5: 66, 6: 50}       # store-wide orders/day
FEED_CONDITION = {"Sealed": "Unopened", "NM": "Near Mint", "LP": "Lightly Played",
                  "MP": "Moderately Played", "HP": "Heavily Played"}


# ---------------------------------------------------------------- item specs
def _base_items(scale):
    """The fixed archetypes: every parser path, a family product, a shared product, an
    override target, out-of-scope items and two singles whose set only the cost sheet knows."""
    items = [
        # seller A family product: one title, the set lives in the variant
        {"seller": "seller_a", "kind": "booster_pack", "set": "Shadow Summit", "sku": "SKU-0101",
         "product": "Game A Booster Pack (Seller A)", "variant": "Shadow Summit",
         "price": 6.5, "market": 4.9, "rate": 11.0, "condition": "Sealed", "mapped": True},
        {"seller": "seller_a", "kind": "booster_pack", "set": "Ember Crown", "sku": "SKU-0102",
         "product": "Game A Booster Pack (Seller A)", "variant": "Ember Crown",
         "price": 6.0, "market": 4.5, "rate": 7.0, "condition": "Sealed", "mapped": True},
        {"seller": "seller_a", "kind": "booster_pack", "set": "Crystal Tide", "sku": "SKU-0103",
         "product": "Game A Booster Pack (Seller A)", "variant": "Crystal Tide",
         "price": 5.5, "market": 4.2, "rate": 4.5, "condition": "Sealed", "mapped": True},
        {"seller": "seller_a", "kind": "etb", "set": "Shadow Summit", "sku": "SKU-0104",
         "product": "Game A Elite Trainer Box (Seller A) (ETB)", "variant": "Shadow Summit",
         "price": 60.0, "market": 47.0, "rate": 2.2, "condition": "Sealed", "mapped": True},
        {"seller": "seller_a", "kind": "etb", "set": "Ember Crown", "sku": "SKU-0105",
         "product": "Game A Elite Trainer Box (Seller A) (ETB)", "variant": "Ember Crown",
         "price": 55.0, "market": 46.0, "rate": 1.4, "condition": "Sealed", "mapped": True},
        {"seller": "seller_a", "kind": "booster_bundle", "set": "Crystal Tide", "sku": "SKU-0106",
         "product": "Booster Bundle (Crystal Tide) · Sealed (Seller A)", "variant": "Default Title",
         "price": 32.0, "market": 26.0, "rate": 2.0, "condition": "Sealed", "mapped": True},
        {"seller": "seller_a", "kind": "booster_box", "set": "Shadow Summit", "sku": "SKU-0107",
         "product": "Booster Box (Shadow Summit) · Sealed (Seller A)", "variant": "Default Title",
         "price": 165.0, "market": 140.0, "rate": 0.7, "condition": "Sealed", "mapped": True},
        {"seller": "seller_a", "kind": "tin", "set": "Lunar Bloom", "sku": "SKU-0108",
         "product": "Mini Tin (Lunar Bloom) · Sealed (Seller A)", "variant": "Default Title",
         "price": 12.0, "market": 10.0, "rate": 1.6, "condition": "Sealed", "mapped": True},
        {"seller": "seller_a", "kind": "collection", "set": "Ember Crown", "sku": "SKU-0109",
         "product": "Poster Collection (Ember Crown) · Sealed (Seller A)", "variant": "Default Title",
         "price": 42.0, "market": 36.0, "rate": 0.9, "condition": "Sealed", "mapped": True},
        # singles with the set in the title
        {"seller": "seller_a", "kind": "single", "set": "Shadow Summit", "sku": "SKU-0133",
         "product": "Emberfox 026/128 · SS01: Shadow Summit · NM (Seller A)", "variant": "Near Mint",
         "price": 12.0, "market": 9.5, "rate": 1.8, "condition": "NM", "mapped": True},
        {"seller": "seller_a", "kind": "single", "set": "Ember Crown", "sku": "SKU-0134",
         "product": "Tidalon ex 091/128 · Ember Crown · NM (Seller A)", "variant": "Near Mint",
         "price": 18.0, "market": 15.0, "rate": 1.2, "condition": "NM", "mapped": True},
        {"seller": "seller_a", "kind": "single", "set": "Crystal Tide", "sku": "SKU-0135",
         "product": "Galeheart 004/102 · Crystal Tide · LP (Seller A)", "variant": "Lightly Played",
         "price": 4.0, "market": 3.2, "rate": 2.4, "condition": "LP", "mapped": True},
        # singles whose set only the unit-cost sheet knows (two-segment titles)
        {"seller": "seller_a", "kind": "single", "set": "Iron Verdict", "sku": "SKU-0136",
         "product": "Duskmaw 066/094 · NM (Seller A)", "variant": "Near Mint",
         "price": 8.0, "market": 6.8, "rate": 1.1, "condition": "NM", "mapped": True, "set_from_ledger": True},
        {"seller": "seller_a", "kind": "single", "set": "Lunar Bloom", "sku": "SKU-0137",
         "product": "Thornelk 012/078 · NM (Seller A)", "variant": "Near Mint",
         "price": 6.0, "market": 5.1, "rate": 0.9, "condition": "NM", "mapped": True, "set_from_ledger": True},
        # the override target: parses to other_sealed until catalog_overrides.csv fixes it
        {"seller": "seller_a", "kind": "collection", "set": "Shadow Summit", "sku": "SKU-0138",
         "product": "Collector Crate (Shadow Summit) · Sealed (Seller A)", "variant": "Default Title",
         "price": 48.0, "market": 40.0, "rate": 0.8, "condition": "Sealed", "mapped": True, "override": True},
        # out of scope: plush and a graded slab
        {"seller": "seller_a", "kind": "plush", "set": "", "sku": "SKU-0139",
         "product": "Clip-On Plushie Charm (3.5 in) (Seller A)", "variant": "New",
         "price": 20.0, "market": 18.0, "rate": 0.8, "condition": "New", "mapped": False},
        {"seller": "seller_a", "kind": "slab", "set": "", "sku": "SKU-0140",
         "product": "Tidalon (Illustration Rare) 232/217 · Graded 10 (Seller A)", "variant": "Graded 10",
         "price": 85.0, "market": 80.0, "rate": 0.15, "condition": "Graded", "mapped": False},
        # seller B (prefix grammar); the Shadow Summit pack and ETB are shared with seller A
        {"seller": "seller_b", "kind": "booster_pack", "set": "Shadow Summit", "sku": "",
         "product": "Booster Pack - Shadow Summit (Seller B)", "variant": "Default Title",
         "price": 7.0, "market": 4.9, "rate": 8.0, "condition": "Sealed", "mapped": True},
        {"seller": "seller_b", "kind": "etb", "set": "Shadow Summit", "sku": "",
         "product": "ETB - Shadow Summit (Seller B)", "variant": "Default Title",
         "price": 62.0, "market": 47.0, "rate": 1.8, "condition": "Sealed", "mapped": True},
        {"seller": "seller_b", "kind": "booster_box", "set": "Ember Crown", "sku": "",
         "product": "Booster Box - Ember Crown (Seller B)", "variant": "Default Title",
         "price": 150.0, "market": 128.0, "rate": 0.5, "condition": "Sealed", "mapped": True},
        {"seller": "seller_b", "kind": "booster_bundle", "set": "Lunar Bloom", "sku": "",
         "product": "Specialty - Lunar Bloom Booster Bundle (Seller B)", "variant": "Default Title",
         "price": 28.0, "market": 24.0, "rate": 0.8, "condition": "Sealed", "mapped": True},
    ]
    for item in items:
        item["price"] = round(item["price"] * scale, 2)
        item["market"] = round(item["market"] * scale, 2)
    return items


def _extra_items(rng, how_many, scale):
    """Random extra singles and sealed items to reach n_products."""
    items = []
    for extra_index in range(how_many):
        set_info = SETS[int(rng.integers(0, len(SETS)))]
        card = CARD_NAMES[int(rng.integers(0, len(CARD_NAMES)))]
        number = int(rng.integers(2, 140))
        denom = int(rng.integers(number, 220))
        sku = "SKU-" + str(200 + extra_index)
        if rng.uniform() < 0.6:
            market = round(float(rng.uniform(2.5, 30.0)) * scale, 2)
            price = round(market * float(rng.uniform(1.0, 1.35)), 2)
            title = (card + " " + str(number).zfill(3) + "/" + str(denom).zfill(3) + " · "
                     + set_info["name"] + " · NM (Seller A)")
            items.append({"seller": "seller_a", "kind": "single", "set": set_info["name"], "sku": sku,
                          "product": title, "variant": "Near Mint", "price": price, "market": market,
                          "rate": float(rng.uniform(0.3, 2.5)), "condition": "NM", "mapped": True})
        else:
            market = round(float(rng.uniform(9.0, 55.0)) * scale, 2)
            price = round(market * float(rng.uniform(1.05, 1.3)), 2)
            if extra_index % 2 == 0:
                title = "Mini Tin (" + set_info["name"] + ") · Sealed (Seller A)"
                kind = "tin"
            else:
                title = "Blister Pack (" + set_info["name"] + ") · Sealed (Seller A)"
                kind = "blister"
            items.append({"seller": "seller_a", "kind": kind, "set": set_info["name"], "sku": sku,
                          "product": title, "variant": "Default Title", "price": price, "market": market,
                          "rate": float(rng.uniform(0.4, 2.0)), "condition": "Sealed", "mapped": True})
    return items


def _finish_specs(rng, items, demand_spread, elasticity_strength):
    """Attach rates, elasticities and feed ids; drop duplicate (product, variant) pairs."""
    seen = set()
    specs = []
    next_feed_id = 100001
    for item in items:
        key = (item["product"], item["variant"])
        if key in seen:
            continue
        seen.add(key)
        spread_draw = float(rng.normal(0.0, 0.35 * demand_spread))
        item["rate"] = float(item["rate"]) * float(np.exp(spread_draw))
        strength = elasticity_strength * float(rng.uniform(0.7, 1.3))
        if item["kind"] == "single":
            item["e_below"] = -0.6 * strength
            item["e_above"] = -1.2 * strength
        else:
            item["e_below"] = -1.0 * strength
            item["e_above"] = -2.0 * strength
        if item["mapped"]:
            item["tcg_id"] = next_feed_id
            next_feed_id += 7
        else:
            item["tcg_id"] = None
        specs.append(item)
    return specs


# ---------------------------------------------------------------- market feed series
def _market_series(rng, specs, today, history_days=370):
    """Per mapped item: daily market price (random walk around the base) and online units."""
    dates = pd.date_range(today - pd.Timedelta(days=history_days), today, freq="D")
    series = {}
    for item in specs:
        if item["tcg_id"] is None:
            continue
        steps = rng.normal(0.0, 0.012, size=len(dates))
        walk = np.cumsum(steps)
        walk = np.clip(walk - walk.mean(), -0.2, 0.2)
        prices = item["market"] * np.exp(walk)
        online_base = item["rate"] * float(rng.uniform(2.5, 8.0))
        online_units = rng.poisson(max(online_base / 7.0, 0.05), size=len(dates))
        series[item["tcg_id"]] = {"dates": dates, "price": prices, "units": online_units}
    return series


# ---------------------------------------------------------------- store simulation
def _sale_minute(rng, closed_day):
    if closed_day:
        hour = int(rng.integers(8, 23))
    else:
        hour = int(rng.integers(10, 20))
    minute = int(rng.integers(0, 60))
    return hour, minute


def _simulate_store(rng, specs, start, weeks, feed_series, today):
    """Walk the calendar day by day; returns event rows, restock rows and per-day bookkeeping."""
    n_days = weeks * 7
    days = pd.date_range(start, periods=n_days, freq="D")
    state = {}
    for index, item in enumerate(specs):
        target = max(2, int(round(item["rate"] * 2.5)))
        late_start = index % 9 == 7            # a couple of items arrive mid-window
        state[index] = {"stock": 0, "pending": [], "sold": 0, "bulk_sold": 0, "days_in_stock": 0,
                        "days_out": 0, "restocks": 0, "start_day": 42 if late_start else index % 3,
                        "target": target, "out_since": None}
    event_rows = []
    restock_rows = []
    daily_rows = []
    order_counter = 4200
    order_number_by_ts = {}
    bulk_done = False

    for day_index, day in enumerate(days):
        dow = int(day.day_of_week)
        closed = dow == CLOSED_WEEKDAY
        day_sales = []

        for index, item in enumerate(specs):
            slot = state[index]
            # pending restocks land on their first selling day
            still_pending = []
            for (sell_day, units) in slot["pending"]:
                if sell_day <= day:
                    slot["stock"] += units
                else:
                    still_pending.append((sell_day, units))
            slot["pending"] = still_pending

            # initial stocking
            if day_index == slot["start_day"]:
                units = slot["target"] * 2
                hour, minute = 10 + index % 7, int(rng.integers(0, 60))
                ts = day + pd.Timedelta(hours=hour, minutes=minute)
                event_rows.append({"ts": ts, "kind": "stock_in", "order": "", "index": index, "units": units,
                                   "by": WEB_IMPORT, "reason": "Initial inventory", "sell_date": day})
                restock_rows.append({"index": index, "ts": ts, "first_selling_day": day, "units": units,
                                     "reason": "Initial inventory"})
                slot["stock"] += units
                slot["restocks"] += 1

            if slot["stock"] > 0:
                slot["days_in_stock"] += 1
                slot["out_since"] = None
            elif slot["restocks"] > 0:
                slot["days_out"] += 1
                if slot["out_since"] is None:
                    slot["out_since"] = day

            # demand today
            if closed or slot["stock"] <= 0:
                demand_today = 0
            else:
                feed = feed_series.get(item["tcg_id"])
                price_mult = 1.0
                if feed is not None:
                    position = (day - feed["dates"][0]).days
                    market_today = float(feed["price"][position])
                    m_now = item["price"] / market_today
                    m_base = item["price"] / item["market"]
                    numer = demand_curve(m_now, item["e_below"], item["e_above"])
                    denom = demand_curve(m_base, item["e_below"], item["e_above"])
                    price_mult = float(numer / denom)
                lam = item["rate"] / 6.0 * OPEN_DOW_FACTOR[dow] * price_mult
                demand_today = int(rng.poisson(max(lam, 1e-9)))
            sold_today = min(demand_today, slot["stock"])

            for _ in range(sold_today):
                hour, minute = _sale_minute(rng, closed)
                ts = day + pd.Timedelta(hours=hour, minutes=minute)
                units = 1
                day_sales.append({"ts": ts, "index": index, "units": units})
                slot["stock"] -= units
                slot["sold"] += units

            # one bulk order mid-window on the fastest pack (left out of demand, still leaves the shelf)
            if not bulk_done and day_index == 50 and index == 0:
                bulk_units = 20
                top_ts = day + pd.Timedelta(hours=11, minutes=5)
                event_rows.append({"ts": top_ts - pd.Timedelta(hours=1), "kind": "stock_in", "order": "",
                                   "index": index, "units": bulk_units, "by": WEB_IMPORT,
                                   "reason": "Received restock", "sell_date": day})
                restock_rows.append({"index": index, "ts": top_ts - pd.Timedelta(hours=1),
                                     "first_selling_day": day, "units": bulk_units, "reason": "Received restock"})
                day_sales.append({"ts": top_ts, "index": index, "units": bulk_units})
                slot["sold"] += bulk_units
                slot["bulk_sold"] += bulk_units
                bulk_done = True

            # restock when sold out for a few days
            if slot["stock"] == 0 and slot["restocks"] > 0 and slot["out_since"] is not None:
                gap = (day - slot["out_since"]).days
                wait = 2 + index % 4
                if gap >= wait and day_index < n_days - 3:
                    units = slot["target"]
                    # every third restock lands after hours, so some 'First selling day'
                    # values differ from the entry date (deterministic, not seed-dependent)
                    evening = slot["restocks"] % 3 == 1
                    if evening and not closed:
                        hour, minute = 21 + int(rng.integers(0, 2)), int(rng.integers(0, 60))
                        sell_day = day + pd.Timedelta(days=1)
                        while int(sell_day.day_of_week) == CLOSED_WEEKDAY:
                            sell_day = sell_day + pd.Timedelta(days=1)
                    else:
                        hour, minute = 10 + int(rng.integers(0, 8)), int(rng.integers(0, 60))
                        sell_day = day
                    ts = day + pd.Timedelta(hours=hour, minutes=minute)
                    event_rows.append({"ts": ts, "kind": "stock_in", "order": "", "index": index, "units": units,
                                       "by": WEB_IMPORT, "reason": "Received restock", "sell_date": sell_day})
                    restock_rows.append({"index": index, "ts": ts, "first_selling_day": sell_day, "units": units,
                                         "reason": "Received restock"})
                    if sell_day <= day:
                        slot["stock"] += units
                    else:
                        slot["pending"].append((sell_day, units))
                    slot["restocks"] += 1
                    slot["out_since"] = None

            # the odd damaged unit
            if day_index == 30 and index == 5 and slot["stock"] > 0:
                ts = day + pd.Timedelta(hours=18, minutes=45)
                event_rows.append({"ts": ts, "kind": "stock_out", "order": "", "index": index, "units": 1,
                                   "by": "Register - " + STAFF[1], "reason": "Damaged", "sell_date": day})
                slot["stock"] -= 1

        # assign store-wide order numbers for today's sales
        background = int(rng.poisson(STORE_ORDER_MEANS[dow]))
        total_orders = background + len(day_sales)
        if len(day_sales) > 0:
            positions = rng.choice(total_orders, size=len(day_sales), replace=False)
            positions = np.sort(positions)
            ordered_sales = sorted(day_sales, key=lambda sale: sale["ts"])
            for sale, position in zip(ordered_sales, positions):
                number = order_counter + int(position)
                sale["order_number"] = number
                staff = STAFF[int(rng.integers(0, len(STAFF)))]
                item = specs[sale["index"]]
                event_rows.append({"ts": sale["ts"], "kind": "sale", "order": "ORD-" + str(number),
                                   "index": sale["index"], "units": sale["units"],
                                   "by": "Register - " + staff, "reason": "", "sell_date": None})
                order_number_by_ts[(number,)] = {"ts": sale["ts"], "staff": staff, "closed": closed}
        order_counter += total_orders

        kind = "partial" if day_index in (0, n_days - 1) else "full"
        daily_rows.append({"date": day, "dow": dow, "kind": kind, "store_orders": float(total_orders)})

    # carry the per-item bookkeeping onto the specs for the derived reporting tables
    for index, item in enumerate(specs):
        slot = state[index]
        item["days_in_stock"] = slot["days_in_stock"]
        item["days_out"] = slot["days_out"]
        item["bulk_sold"] = slot["bulk_sold"]
        item["sold_units"] = slot["sold"]
    return event_rows, restock_rows, daily_rows, order_number_by_ts


# ---------------------------------------------------------------- writers
def _write_sales_csvs(raw_dir, specs, event_rows):
    frames = {"seller_a": [], "seller_b": []}
    for row in event_rows:
        item = specs[row["index"]]
        price = item["price"]
        csv_row = {
            "Date (PT)": row["ts"].strftime("%Y-%m-%d %H:%M"),
            "Kind": row["kind"],
            "Order": row["order"],
            "Product": item["product"],
            "Variant": item["variant"],
            "SKU": item["sku"],
            "Units": int(row["units"]),
            "List price": f"{price:.2f}",
            "Revenue at list": f"{price * int(row['units']):.2f}",
            "By": row["by"],
            "Location": LOCATION,
            "Reason": row["reason"],
        }
        frames[item["seller"]].append(csv_row)
    for seller in DEFAULT_SELLERS:
        table = pd.DataFrame(frames[seller.seller_id])
        table = table.sort_values("Date (PT)").reset_index(drop=True)
        table.to_csv(raw_dir / seller.sales_file, index=False)


def _write_restock_events(raw_dir, specs, restock_rows):
    rows = []
    for row in restock_rows:
        item = specs[row["index"]]
        display = "Seller A" if item["seller"] == "seller_a" else "Seller B"
        rows.append({
            "Vendor": display,
            "Entered (PT)": row["ts"].strftime("%Y-%m-%d %H:%M"),
            "First selling day": row["first_selling_day"].strftime("%Y-%m-%d"),
            "Product": item["product"],
            "Variant": item["variant"],
            "SKU": item["sku"],
            "Units": int(row["units"]),
            "Catch-up units": 0,
            "List price": item["price"],
            "Reason": row["reason"],
            "By": WEB_IMPORT,
        })
    folder = raw_dir / "restock"
    folder.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / "restock_events.csv", index=False)


def _write_restock_tables(raw_dir, specs, events, daily_rows):
    """The four copied-as-is reporting tables, derived from the simulated history."""
    folder = raw_dir / "restock"
    sales = events[events["kind"] == "sale"]
    sold_by_item = sales.groupby(["product", "variant"])["units"].sum()
    sold_out_rows = []
    soon_rows = []
    for item in specs:
        key = (item["product"], item["variant"])
        sold = int(sold_by_item.get(key, 0))
        display = "Seller A" if item["seller"] == "seller_a" else "Seller B"
        rate = round(item["rate"] / 6.0, 2)
        if sold == 0:
            continue
        if item.get("days_out", 0) > 0:
            missed = round(item["days_out"] * rate, 1)
            sold_out_rows.append({
                "Vendor": display, "Product": item["product"], "Variant": item["variant"], "SKU": item["sku"],
                "Price": item["price"], "Sold": sold, "Days in stock": item.get("days_in_stock", 0),
                "Steady days": item.get("days_in_stock", 0), "Steady sale days": max(sold - item.get("bulk_sold", 0), 0),
                "Bulk units left out": item.get("bulk_sold", 0), "Days out": item["days_out"],
                "Sold per steady day": rate, "Other vendor units while out": 0, "Other vendor $ while out": 0.0,
                "Days store had none": item["days_out"], "Est missed units": missed,
                "Est missed $": round(missed * item["price"], 2), "Out now": "no"})
        if item.get("end_stock", 0) > 0 and rate > 0:
            soon_rows.append({
                "Vendor": display, "Product": item["product"], "Variant": item["variant"], "SKU": item["sku"],
                "On shelf now": item["end_stock"], "Arriving": 0, "Lands": "",
                "Sold per in-stock day": rate, "In-stock days": item.get("days_in_stock", 0),
                "Sale days": max(sold - item.get("bulk_sold", 0), 0), "Bulk units left out": item.get("bulk_sold", 0),
                "Days of cover": round(item["end_stock"] / max(rate, 0.01), 1),
                "Days of cover with arriving": round(item["end_stock"] / max(rate, 0.01), 1)})
    pd.DataFrame(sold_out_rows).to_csv(folder / "sold_out.csv", index=False)
    pd.DataFrame(soon_rows).to_csv(folder / "restock_soon.csv", index=False)

    shared_rows = []
    families = {}
    for item in specs:
        family = (item["kind"], item["set"])
        families.setdefault(family, set()).add(item["seller"])
    for (kind, set_name), sellers in sorted(families.items()):
        if len(sellers) < 2:
            continue
        shared_rows.append({"Type": kind, "Set": set_name, "Seller A prices": "", "Seller B prices": "",
                            "Seller A sold": "", "Seller B sold": "", "Days Seller A in stock": "",
                            "Days Seller B in stock": "", "Days both out": 0})
    pd.DataFrame(shared_rows).to_csv(folder / "shared_products.csv", index=False)

    day_rows = []
    sales_by_day = sales.groupby(["date", "seller"]).agg(orders=("order", "nunique"), items=("units", "sum"),
                                                         revenue=("revenue", "sum"))
    for daily in daily_rows:
        for seller_id, display in [("seller_a", "Seller A"), ("seller_b", "Seller B")]:
            key = (daily["date"], seller_id)
            if key in sales_by_day.index:
                stats = sales_by_day.loc[key]
            else:
                stats = {"orders": 0, "items": 0, "revenue": 0.0}
            day_rows.append({
                "Date": daily["date"].strftime("%Y-%m-%d"),
                "Day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][daily["dow"]],
                "Kind": daily["kind"], "Vendor": display,
                "Store orders (est)": daily["store_orders"], "Orders": int(stats["orders"]),
                "Items": int(stats["items"]), "$": round(float(stats["revenue"]), 2),
                "Restock new units": 0, "Restock new $": 0.0, "Restock day": "no",
                "Open days since restock": "", "Shelf units at open": "", "Items in stock at open": "",
                "Shelf $ at open": "", "Shared products in stock": ""})
    pd.DataFrame(day_rows).to_csv(folder / "restock_days.csv", index=False)


def _write_order_ledger(raw_dir, order_numbers):
    rows = []
    for (number,), info in sorted(order_numbers.items()):
        ts = info["ts"]
        rows.append({
            "Order": "ORD-" + str(number),
            "Number": number,
            "Time UTC": (ts + pd.Timedelta(hours=7)).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "Time PT": ts.strftime("%Y-%m-%d %H:%M"),
            "Register sale": "no" if info["closed"] else "yes",
            "Staff": info["staff"],
        })
    pd.DataFrame(rows).to_csv(raw_dir / "order_ledger.csv", index=False)


def _write_store_tables(raw_dir, daily_rows):
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    daily_table = []
    for row in daily_rows:
        daily_table.append({"Date": row["date"].strftime("%Y-%m-%d"), "Day": day_names[row["dow"]],
                            "Kind": row["kind"], "Store orders (est)": row["store_orders"]})
    pd.DataFrame(daily_table).to_csv(raw_dir / "store_daily.csv", index=False)

    weekday_rows = []
    for dow, name in enumerate(day_names):
        full_days = []
        for row in daily_rows:
            if row["dow"] == dow and row["kind"] == "full":
                full_days.append(row["store_orders"])
        average = float(np.mean(full_days)) if full_days else 0.0
        weekday_rows.append({"Weekday": name, "Full days": len(full_days),
                             "Avg store orders": round(average, 1)})
    pd.DataFrame(weekday_rows).to_csv(raw_dir / "store_weekday.csv", index=False)


def _write_unit_costs(raw_dir, rng, specs):
    rows = []
    for index, item in enumerate(specs):
        if item["seller"] != "seller_a" or not item["sku"]:
            continue
        if item.get("skip_cost"):
            continue
        cost = item["market"] * float(rng.uniform(0.55, 0.8))
        if item["kind"] == "slab":
            cost = 1234.56           # exercises the "$1,234.56" cleaning path
        if index % 2 == 0:
            cost_text = "$" + f"{cost:,.2f}"
        else:
            cost_text = f"{cost:.2f}"
        rows.append({
            "SKU": item["sku"],
            "Cost (ea)": cost_text,
            "Ask": f"{item['price']:.2f}",
            "Market (ea)": "$" + f"{item['market']:,.2f}",
            "Set": item["set"],
            "Format": "Standard",
            "Product type": "Single" if item["kind"] == "single" else "Sealed",
            "Date In": "2026-07-01",
        })
    pd.DataFrame(rows).to_csv(raw_dir / "unit_costs.csv", index=False)


def _write_commission_workbooks(raw_dir, rng, specs, events, window_end):
    """Three weekly workbooks over the last three Mon-Sun weeks; the oldest one predates
    the 'Order paid by' column (its lines count as card, the workbook's own default)."""
    folder = raw_dir / "commission"
    folder.mkdir(exist_ok=True)
    sku_by_key = {}
    for item in specs:
        sku_by_key[(item["product"], item["variant"])] = item
    sales = events[(events["kind"] == "sale") & (events["seller"] == "seller_a")]
    for week_index in range(3):
        week_end = window_end - pd.Timedelta(days=7 * (2 - week_index))
        week_start = week_end - pd.Timedelta(days=6)
        with_paid_by = week_index > 0
        part = sales[(sales["date"] >= week_start) & (sales["date"] <= week_end)]
        book = openpyxl.Workbook()
        sheet = book.active
        sheet.title = "Sales " + week_start.strftime("%m-%d") + " to " + week_end.strftime("%m-%d")
        sheet.append(["Seller A consignment sales: " + week_start.strftime("%a %b %d") + " to "
                      + week_end.strftime("%a %b %d, %Y") + " (synthetic data)"])
        sheet.append(["Generated by milo.synth for the public demo dataset"])
        sheet.append([])
        header = ["Order #", "Date", "Time (PT)", "Line # in order", "Lines in this order", "SKU", "Item",
                  "Qty", "Unit price", "Sold", "Store commission", "Card fee", "Fixed fee per order",
                  "Commission + fees", "Net to you"]
        if with_paid_by:
            header = header + ["Paid by (pick)", "Order paid by", "Note"]
        sheet.append(header)
        total_sold = 0.0
        for _, sale in part.sort_values("ts").iterrows():
            item = sku_by_key.get((sale["product"], sale["variant"]))
            if item is None:
                continue
            qty = int(sale["units"])
            sold = qty * item["price"]
            commission = round(0.10 * sold, 2)
            card_fee = round(0.029 * sold, 2)
            row = [sale["order"], sale["ts"].normalize().to_pydatetime(), sale["ts"].strftime("%H:%M"),
                   1, 1, item["sku"], item["product"], qty, item["price"], sold, commission, card_fee,
                   0.30, round(commission + card_fee + 0.30, 2), round(sold - commission - card_fee - 0.30, 2)]
            if with_paid_by:
                paid = "Cash" if rng.uniform() < 0.2 else "Card"
                row = row + [paid, paid, ""]
            sheet.append(row)
            total_sold += sold
        # one custom register line per workbook (no SKU in the catalog)
        custom = ["ORD-0000", week_start.to_pydatetime(), "12:00", 1, 1, "custom-item",
                  "Custom register item", 1, 5.0, 5.0, 0.5, 0.15, 0.30, 0.95, 4.05]
        if with_paid_by:
            custom = custom + ["Cash", "Cash", "custom sale rung up at the register"]
        sheet.append(custom)
        sheet.append(["TOTAL", None, None, None, None, None, None, None, None, round(total_sold + 5.0, 2)])
        name = week_start.strftime("%Y-%m-%d") + " to " + week_end.strftime("%Y-%m-%d") + ".xlsx"
        book.save(folder / name)


def _write_feed_files(raw_dir, rng, specs, feed_series, today):
    """tcg_sales + tcg_history jsonl files in the market feed's nested schema."""
    sales_dir = raw_dir / "tcg_sales"
    history_dir = raw_dir / "tcg_history"
    sales_dir.mkdir(exist_ok=True)
    history_dir.mkdir(exist_ok=True)
    stamp = today.strftime("%Y-%m-%d")
    for item in specs:
        feed_id = item["tcg_id"]
        if feed_id is None:
            continue
        feed = feed_series[feed_id]
        condition = FEED_CONDITION.get(item["condition"], "Near Mint")
        dates = feed["dates"]
        # ---- individual sales: the last 21 days
        sale_records = []
        for position in range(len(dates) - 21, len(dates)):
            day = dates[position]
            count = int(feed["units"][position])
            for _ in range(count):
                price = float(feed["price"][position]) * float(rng.uniform(0.93, 1.07))
                hour = int(rng.integers(0, 24))
                minute = int(rng.integers(0, 60))
                sale_records.append({
                    "condition": condition, "variant": "Normal", "language": "English",
                    "quantity": int(rng.choice([1, 1, 1, 2])), "title": item["product"].split(" ·")[0],
                    "listingType": "ListingWithoutPhotos", "customListingId": "0",
                    "purchasePrice": round(price, 2), "shippingPrice": float(rng.choice([0.0, 1.99])),
                    "orderDate": day.strftime("%Y-%m-%d") + "T" + f"{hour:02d}:{minute:02d}:00+00:00",
                })
        page = {"previousPage": "", "nextPage": "", "resultCount": len(sale_records),
                "totalResults": len(sale_records), "data": sale_records}
        record = {"id": feed_id, "offset": 0, "fetched_at": stamp, "page": page}
        with open(sales_dir / (str(feed_id) + "_" + stamp + ".jsonl"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        # ---- chart ranges: month (daily), quarter (3-day), semi-annual + annual (weekly)
        lines = []
        range_specs = [("month", 1, 31), ("quarter", 3, 31), ("semi-annual", 7, 27), ("annual", 7, 53)]
        for range_name, bucket_days, n_buckets in range_specs:
            buckets = []
            for bucket_index in range(n_buckets, 0, -1):
                start = today - pd.Timedelta(days=bucket_days * bucket_index)
                positions = []
                for offset in range(bucket_days):
                    day = start + pd.Timedelta(days=offset)
                    day_position = (day - dates[0]).days
                    if 0 <= day_position < len(dates):
                        positions.append(day_position)
                if not positions:
                    continue
                quantity = 0
                for day_position in positions:
                    quantity += int(feed["units"][day_position])
                price = float(np.mean(feed["price"][positions]))
                buckets.append({
                    "marketPrice": str(round(price, 2)), "quantitySold": str(quantity),
                    "lowSalePrice": str(round(price * 0.9, 2)),
                    "lowSalePriceWithShipping": str(round(price * 0.92, 2)),
                    "highSalePrice": str(round(price * 1.1, 2)),
                    "highSalePriceWithShipping": str(round(price * 1.12, 2)),
                    "transactionCount": str(max(quantity, 0)),
                    "bucketStartDate": start.strftime("%Y-%m-%d"),
                })
            # a still-running bucket, which the loader must drop (M17b)
            buckets.append({
                "marketPrice": str(round(float(feed["price"][-1]), 2)), "quantitySold": "1",
                "lowSalePrice": "0", "lowSalePriceWithShipping": "0", "highSalePrice": "0",
                "highSalePriceWithShipping": "0", "transactionCount": "1",
                "bucketStartDate": today.strftime("%Y-%m-%d"),
            })
            sku = {"skuId": feed_id * 10, "variant": "Normal", "language": "English",
                   "condition": condition, "buckets": buckets}
            chart = {"count": 1, "result": [sku]}
            lines.append(json.dumps({"id": feed_id, "range": range_name, "fetched_at": stamp, "chart": chart}))
        with open(history_dir / (str(feed_id) + "_" + stamp + ".jsonl"), "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")


def _write_inputs(inputs_dir, specs, sold):
    """tcg_map.csv, catalog_overrides.csv, restock_costs.csv.

    In tcg_map.csv a 'review' row appears only when a blister sells; with the committed
    knobs (seed 19) none does, so the committed file is all-'ok' rows.
    """
    spec_by_key = {}
    for item in specs:
        spec_by_key[(item["product"], item["variant"])] = item
    map_rows = []
    review_done = False
    for _, row in sold.iterrows():
        item = spec_by_key.get((row["product"], row["variant"]))
        if item is None or item["tcg_id"] is None:
            continue
        status = "ok"
        if not review_done and item["kind"] == "blister":
            status = "review"
            review_done = True
        map_rows.append({"key": row["key"], "status": status, "tcg_id": item["tcg_id"],
                         "product": row["product"], "variant": row["variant"], "note": "auto"})
    pd.DataFrame(map_rows).to_csv(inputs_dir / "tcg_map.csv", index=False)

    overrides = pd.DataFrame([{
        "product": "Collector Crate (Shadow Summit) · Sealed (Seller A)", "variant": "Default Title",
        "product_type": "collection", "set": "shadow summit", "condition": "", "game": "",
    }])
    overrides.to_csv(inputs_dir / "catalog_overrides.csv", index=False)

    # restock_costs.csv: hand-edited limits; kept as-is when it exists (keep-if-exists semantics)
    cost_rows = []
    for item in specs:
        if item["kind"] == "booster_box" and item["seller"] == "seller_a":
            unit = "seller_a|" + catalog.family_key(catalog.parse_title(item["product"], item["variant"],
                                                                        item["seller"]), item["price"])
            cost_rows.append({"unit": unit, "product_type": "booster_box", "price_now": item["price"],
                              "market_price": item["market"], "cost": round(item["market"] * 0.78, 2),
                              "cost_source": "restock_costs.csv", "min_order": 6, "max_units": 12,
                              "salvage_frac": 0.85})
    pd.DataFrame(cost_rows).to_csv(inputs_dir / "restock_costs.csv", index=False)


def _write_side_files(raw_dir):
    set_rows = []
    for set_info in SETS:
        set_rows.append({"set": set_info["name"].lower(), "release_date": set_info["release"]})
    pd.DataFrame(set_rows).to_csv(raw_dir / "set_release_dates.csv", index=False)


# ---------------------------------------------------------------- entry point
def generate_dataset(out_dir, seed=11, n_products=24, weeks=15, start="2026-07-06",
                     demand_spread=1.0, elasticity_strength=1.0, budget=5000.0):
    """Write a complete synthetic raw dataset under out_dir; returns a manifest dict."""
    out_dir = Path(out_dir)
    raw_dir = out_dir / "raw"
    inputs_dir = out_dir / "inputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    scale = budget / 5000.0
    items = _base_items(scale)
    extra_count = max(n_products - len(items), 0)
    items = items + _extra_items(rng, extra_count, scale)
    specs = _finish_specs(rng, items, demand_spread, elasticity_strength)
    # skip the cost sheet for two mapped items so the default-cost path stays live
    skip_count = 0
    for item in specs:
        if skip_count < 2 and item["kind"] == "tin" and item["seller"] == "seller_a":
            item["skip_cost"] = True
            skip_count += 1

    window_start = pd.Timestamp(start)
    window_end = window_start + pd.Timedelta(days=weeks * 7 - 1)
    today = window_end + pd.Timedelta(days=2)

    feed_series = _market_series(rng, specs, today)
    event_rows, restock_rows, daily_rows, order_numbers = _simulate_store(
        rng, specs, window_start, weeks, feed_series, today)

    # carry end-of-window state onto the specs for the derived reporting tables
    stock_left = {}
    for row in event_rows:
        delta = 0
        if row["kind"] == "stock_in":
            delta = int(row["units"])
        else:
            delta = -int(row["units"])
        stock_left[row["index"]] = stock_left.get(row["index"], 0) + delta
    for index, item in enumerate(specs):
        item["end_stock"] = max(stock_left.get(index, 0), 0)

    _write_sales_csvs(raw_dir, specs, event_rows)
    _write_restock_events(raw_dir, specs, restock_rows)
    _write_order_ledger(raw_dir, order_numbers)
    _write_store_tables(raw_dir, daily_rows)
    _write_unit_costs(raw_dir, rng, specs)
    _write_side_files(raw_dir)

    # round-trip the written sales files through the real loaders so sold_products.csv,
    # tcg_map.csv and the commission workbooks are exactly consistent with them
    events = load.load_store_events(raw_dir, sellers=DEFAULT_SELLERS)
    ledger = load.load_unit_costs(raw_dir)
    cat = catalog.build_catalog(events, overrides_path=inputs_dir / "catalog_overrides.csv", ledger=ledger)
    sold = tcgmap.sold_products(events, cat)
    sold = sold.sort_values(["product", "variant"]).reset_index(drop=True)
    keys = []
    for position in range(len(sold)):
        keys.append("P" + str(position).zfill(3))
    sold["key"] = keys
    sold.to_csv(raw_dir / "sold_products.csv", index=False)

    _write_inputs(inputs_dir, specs, sold)
    _write_commission_workbooks(raw_dir, rng, specs, events, window_end)

    events_with_revenue = events.copy()
    events_with_revenue["revenue"] = pd.to_numeric(events_with_revenue["revenue_at_list"], errors="coerce")
    _write_restock_tables(raw_dir, specs, events_with_revenue, daily_rows)

    _write_feed_files(raw_dir, rng, specs, feed_series, today)

    manifest = {"seed": seed, "n_products": len(specs), "weeks": weeks, "start": str(window_start.date()),
                "end": str(window_end.date()), "today": str(today.date()),
                "demand_spread": demand_spread, "elasticity_strength": elasticity_strength, "budget": budget,
                "n_events": len(event_rows), "n_sold_products": len(sold)}
    with open(out_dir / "MANIFEST.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1)
    return manifest
