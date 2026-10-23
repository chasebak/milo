import numpy as np
import pandas as pd
import pytest

from milo import catalog, demand, pricing


@pytest.mark.parametrize("title, variant, seller, kind, set_name", [
    ("Game A Booster Pack (Seller A)", "Shadow Summit", "seller_a", "booster_pack", "shadow summit"),
    ("Game A Elite Trainer Box (Seller A) (ETB)", "Ember Crown", "seller_a", "etb", "ember crown"),
    ("Booster Pack (Crystal Tide) · Sealed · SKU-0133 (Seller A)", "Default Title", "seller_a", "booster_pack",
     "crystal tide"),
    ("Emberfox 026/128 · SS01: Shadow Summit · NM (Seller A)", "Near Mint", "seller_a", "single", "shadow summit"),
    ("ETB - Shadow Summits(Seller B)", "Default Title", "seller_b", "etb", "shadow summit"),
    ("Specialty - Ember Crown Booster Bundle (Seller B)", "Default Title", "seller_b", "booster_bundle", "ember crown"),
    ("Tidalon (Illustration Rare) 232/217 · Graded 10 (Seller A)", "Graded 10", "seller_a", "slab", ""),
    ("Clip-On Plushie Charm (3.5 in) (Seller A)", "New", "seller_a", "plush", "plush"),
])
def test_parse_title(title, variant, seller, kind, set_name):
    parsed = catalog.parse_title(title, variant, seller)
    assert parsed["product_type"] == kind
    assert parsed["set"] == set_name


def test_family_key_bands_singles():
    parsed = catalog.parse_title("Emberfox 026/128 · SS01: Shadow Summit · NM (Seller A)", "Near Mint", "seller_a")
    assert catalog.family_key(parsed, 3.0).endswith("|<5")
    assert catalog.family_key(parsed, 12.0).endswith("|5-20")


def make_events():
    rows = [
        ("2026-09-01 10:00", "stock_in", "A", 2), ("2026-09-02 10:00", "sale", "A", 1),
        ("2026-09-03 10:00", "sale", "A", 1),   # sold out after this
        ("2026-09-10 10:00", "stock_in", "A", 1), ("2026-09-11 10:00", "sale", "A", 1),
    ]
    table = pd.DataFrame(rows, columns=["ts", "kind", "product", "units"])
    table["ts"] = pd.to_datetime(table["ts"])
    table["date"] = table["ts"].dt.normalize()
    table["variant"] = "Default Title"
    return table


def test_in_stock_days_skip_sold_out_gap():
    panel = demand.stock_panel(make_events(), end_date=pd.Timestamp("2026-09-12"))
    in_stock_days = int(panel["in_stock"].sum())
    # Sep 1-3 and Sep 10-11 are in stock; Sep 4-9 and 12 are sold out (D1).
    assert in_stock_days == 5


def test_gamma_shrinkage_pulls_thin_items_toward_mean():
    rates = pd.DataFrame({"family": ["a", "b", "c"], "units": [40.0, 30.0, 1.0],
                          "exposure_weeks": [4.0, 4.0, 0.3], "product_type": ["x", "x", "x"]})
    rates["raw_rate"] = rates["units"] / rates["exposure_weeks"]
    out = demand.p2_shrink(rates)
    thin = out[out["family"] == "c"].iloc[0]
    assert abs(thin["shrunk_rate"] - 8.75) < abs(thin["raw_rate"] - 8.75)
    assert "thin data" in thin["flag"]


def test_kinked_curve_and_monotone_multipliers():
    multipliers = pricing.level_multipliers(1.1, -1.0, -2.5)
    assert np.all(np.diff(multipliers) <= 1e-12)
    index_of_one = int(np.argmin(np.abs(pricing.PRICE_LEVELS - 1.0)))
    assert multipliers[index_of_one] == pytest.approx(1.0)


def test_poisson_irls_recovers_slope():
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.4, 0.4, 4000)
    y = rng.poisson(np.exp(1.0 - 2.0 * x))
    X = np.column_stack([np.ones_like(x), x])
    beta = pricing.poisson_irls(X, y)
    assert beta[1] == pytest.approx(-2.0, abs=0.1)


def test_combine_rejects_positive_fit():
    cross = {"sealed": {"slope": 0.4, "se": 0.1, "n": 40}, "single": None}
    table = pricing.combine_elasticities(cross_fit=cross)
    row = table[(table["group"] == "sealed") & (table["side"] == "above")].iloc[0]
    assert row["source"] == "assumed range"
    assert row["base"] < 0
