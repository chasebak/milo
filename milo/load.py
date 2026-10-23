"""Read every raw source into clean pandas tables. Read-only; nothing here writes outside the project."""
import glob
import re
from pathlib import Path

import openpyxl
import pandas as pd

from milo.config import DEFAULT_SELLERS, closed_weekday_names

PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "synthetic" / "raw"


def load_store_events(raw_dir=RAW, sellers=DEFAULT_SELLERS):
    """Store stock history per seller: sale / stock_in / stock_out lines, one row each.

    Each seller's file is named in the SellerConfig registry (seller_a_sales.csv, ...).
    """
    frames = []
    for seller in sellers:
        table = pd.read_csv(Path(raw_dir) / seller.sales_file)
        table["seller"] = seller.seller_id
        frames.append(table)
    events = pd.concat(frames, ignore_index=True)
    events = events.rename(columns={
        "Date (PT)": "ts", "Kind": "kind", "Order": "order", "Product": "product",
        "Variant": "variant", "SKU": "sku", "Units": "units", "List price": "list_price",
        "Revenue at list": "revenue_at_list", "By": "by", "Location": "location", "Reason": "reason",
    })
    events["ts"] = pd.to_datetime(events["ts"])
    events["date"] = events["ts"].dt.normalize()
    events["variant"] = events["variant"].fillna("Default Title")
    events = events.sort_values(["ts", "kind"]).reset_index(drop=True)
    events = attach_sell_dates(events, raw_dir)
    events = flag_bulk_orders(events)
    return events


def attach_sell_dates(events, raw_dir=RAW):
    """Shelf rule: a stock_in sells from its 'First selling day' (an upload entered after hours
    sells the next open day; edits during store hours sell the same day). Default = the entry date."""
    events["sell_date"] = events["date"]
    path = Path(raw_dir) / "restock" / "restock_events.csv"
    if not path.exists():
        return events
    restocks = pd.read_csv(path)
    restocks["seller"] = restocks["Vendor"].str.lower().str.replace(" ", "_")
    restocks["ts_min"] = pd.to_datetime(restocks["Entered (PT)"]).dt.floor("min")
    restocks = restocks.rename(columns={"Product": "product", "Variant": "variant", "Units": "units",
                                        "First selling day": "first_selling_day"})
    restocks = restocks.drop_duplicates(["seller", "ts_min", "product", "variant", "units"])
    events["ts_min"] = events["ts"].dt.floor("min")
    keys = ["seller", "ts_min", "product", "variant", "units"]
    merged = events.merge(restocks[keys + ["first_selling_day"]], on=keys, how="left")
    is_stock_in = (merged["kind"] == "stock_in") & merged["first_selling_day"].notna()
    merged.loc[is_stock_in, "sell_date"] = pd.to_datetime(merged.loc[is_stock_in, "first_selling_day"])
    return merged.drop(columns=["ts_min", "first_selling_day"])


def load_restock_tables(raw_dir=RAW):
    """Restock-timing tables exported by the store's reporting (copied as-is; not recomputed here)."""
    folder = Path(raw_dir) / "restock"
    tables = {}
    for name in ["sold_out", "restock_soon", "shared_products", "restock_days"]:
        path = folder / (name + ".csv")
        if path.exists():
            tables[name] = pd.read_csv(path)
    return tables


def flag_bulk_orders(events, min_units=15, share=0.5):
    """Bulk orders are left out of demand rates: 15+ units of an item in one order, or
    one order that is most of an item's sales. They still leave the shelf (stock math keeps them)."""
    events["bulk"] = False
    sales = events[events["kind"] == "sale"]
    item_totals = sales.groupby(["product", "variant"])["units"].sum()
    per_order = sales.groupby(["product", "variant", "order"])["units"].sum()
    bulk_keys = []
    for key, units in per_order.items():
        product, variant, order = key
        total = item_totals[(product, variant)]
        if units >= min_units or (total >= 4 and units / total > share):
            bulk_keys.append(key)
    for product, variant, order in bulk_keys:
        mask = (events["kind"] == "sale") & (events["product"] == product) & (events["variant"] == variant) & \
               (events["order"] == order)
        events.loc[mask, "bulk"] = True
    return events


def load_commission_lines(raw_dir=RAW):
    """Item rows from the weekly commission workbooks (primary seller only): includes custom register
    items and payment type."""
    files = sorted(glob.glob(str(Path(raw_dir) / "commission" / "*.xlsx")))
    frames = []
    for path in files:
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        sheet = workbook.worksheets[0]
        rows = list(sheet.iter_rows(values_only=True))
        header_index = None
        for index, row in enumerate(rows):
            if row and row[0] == "Order #":
                header_index = index
                break
        if header_index is None:
            continue
        header = list(rows[header_index])
        body = []
        for row in rows[header_index + 1:]:
            first = row[0]
            if first is None or str(first).strip().upper().startswith("TOTAL"):
                continue
            body.append(row)
        table = pd.DataFrame(body, columns=header)
        table["source_file"] = Path(path).name
        frames.append(table)
    lines = pd.concat(frames, ignore_index=True)
    lines = lines.rename(columns={
        "Order #": "order", "Date": "date", "SKU": "sku", "Item": "item", "Qty": "qty",
        "Unit price": "unit_price", "Sold": "sold", "Order paid by": "paid_by", "Note": "note",
    })
    lines["date"] = pd.to_datetime(lines["date"])
    lines["is_custom"] = lines["sku"].astype(str).str.contains("custom", case=False)
    # Weeks before the 'Paid by' column existed count as card (the workbook's own default).
    if "paid_by" not in lines.columns:
        lines["paid_by"] = None
    lines["paid_by"] = lines["paid_by"].fillna("Card")
    return lines


def card_share(commission_lines):
    """Share of sold dollars paid by card (drives the card-fee rate)."""
    sold = pd.to_numeric(commission_lines["sold"], errors="coerce").fillna(0.0)
    by_card = sold[commission_lines["paid_by"].astype(str).str.lower() == "card"].sum()
    total = sold.sum()
    if total <= 0:
        return 0.8
    return float(by_card / total)


def load_order_ledger(raw_dir=RAW):
    """Every store order (all sellers): store traffic."""
    orders = pd.read_csv(Path(raw_dir) / "order_ledger.csv")
    orders["ts"] = pd.to_datetime(orders["Time PT"])
    orders["date"] = orders["ts"].dt.normalize()
    return orders


def load_store_daily(raw_dir=RAW):
    """Whole-store orders per day estimated from order-number gaps. None if not present."""
    path = Path(raw_dir) / "store_daily.csv"
    if not path.exists():
        return None
    daily = pd.read_csv(path)
    daily["date"] = pd.to_datetime(daily["Date"])
    daily = daily.rename(columns={"Store orders (est)": "store_orders", "Day": "weekday", "Kind": "day_kind"})
    return daily[["date", "weekday", "day_kind", "store_orders"]]


def load_weekday_shares(raw_dir=RAW, closed_weekdays=(2,)):
    """Average store orders by weekday and each day's share of an open week. None if not present."""
    path = Path(raw_dir) / "store_weekday.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    table = table.rename(columns={"Weekday": "weekday", "Avg store orders": "avg_store_orders", "Full days": "full_days"})
    closed_names = closed_weekday_names(closed_weekdays)
    open_days = table[~table["weekday"].isin(closed_names)]
    total = open_days["avg_store_orders"].sum()
    table["share_of_week"] = table["avg_store_orders"] / total
    table.loc[table["weekday"].isin(closed_names), "share_of_week"] = 0.0
    return table


def units_per_order(events):
    """Average units per order with at least one sale line (spreads the per-order fee)."""
    sales = events[events["kind"] == "sale"]
    per_order = sales.groupby("order")["units"].sum()
    if len(per_order) == 0:
        return 1.0
    return float(per_order.mean())


def load_unit_costs(raw_dir=RAW):
    """Cost basis per SKU from the unit-cost sheet."""
    ledger = pd.read_csv(Path(raw_dir) / "unit_costs.csv")
    ledger = ledger.rename(columns={"SKU": "sku", "Cost (ea)": "cost", "Ask": "ask",
                                    "Market (ea)": "market", "Set": "ledger_set", "Format": "format",
                                    "Product type": "ledger_type", "Date In": "date_in"})
    for column in ["cost", "ask", "market"]:
        text = ledger[column].astype(str).str.replace(r"[$,]", "", regex=True)
        ledger[column] = pd.to_numeric(text, errors="coerce")
    return ledger


def load_tcg_map(path):
    """Product-key -> feed-id mapping (inputs/tcg_map.csv). Only rows with status == "ok" are used;
    a `review` row is a match still waiting on a human and never reaches the pipeline."""
    mapping = pd.read_csv(path)
    mapping = mapping[mapping["status"] == "ok"]
    return mapping.reset_index(drop=True)


def tcg_id_from_link(link):
    if link is None or pd.isna(link):
        return None
    found = re.search(r"/product/(\d+)", str(link))
    if found:
        return int(found.group(1))
    return None


def load_set_release_dates(raw_dir=RAW):
    """Canonical set names and release dates. Optional input: None if the file is absent."""
    path = Path(raw_dir) / "set_release_dates.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["release_date"])
