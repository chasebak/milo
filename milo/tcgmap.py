"""Sold-product roster for the market-feed mapping.

The product-key -> feed-id mapping itself (inputs/tcg_map.csv) is hand-maintained; this
module builds the sold-product table those keys attach to (see SCHEMA.md) and matches
card numbers across the two numbering styles.
"""
import re


def sold_products(events, catalog_table):
    """In-scope products with at least one sale, with units sold and the latest list price."""
    sales = events[events["kind"] == "sale"]
    grouped = sales.groupby(["product", "variant"]).agg(
        units_sold=("units", "sum"), list_price=("list_price", "last"), seller=("seller", "first"),
        sku=("sku", "last"), first_sale=("date", "min"), last_sale=("date", "max")).reset_index()
    merged = grouped.merge(catalog_table.drop(columns=["seller", "sku"]), on=["product", "variant"], how="left")
    return merged[merged["in_scope"]].reset_index(drop=True)


def numbers_match(ours, theirs):
    """'63/132' matches '063/132'; 'PR 102' matches 'PR102'."""
    def clean(text):
        parts = re.split(r"[/\s-]", str(text).upper())
        kept = []
        for part in parts:
            if part:
                kept.append(part.lstrip("0") or "0")
        return "".join(kept)
    return clean(ours) == clean(theirs) or clean(ours).split("/")[0] == clean(theirs).split("/")[0]
