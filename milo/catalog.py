"""Turn store product titles into game, product type, set, condition and a family key.

The family key is the demand unit:
- sealed:  "<game>|<type>|<set>"          e.g. "game_a|etb|shadow summit"
- singles: "single|<game>|<set>|<band>"   e.g. "single|game_a|shadow summit|<5"

Each seller lists products under their own title grammar; the grammar is looked up
from the SellerConfig registry (config.DEFAULT_SELLERS), never hardcoded here.
"""
import re
from pathlib import Path

import pandas as pd

from milo.config import DEFAULT_SELLERS

# Set name spellings seen in titles -> one canonical name.
SET_ALIASES = {
    "shadow summits": "shadow summit",
    "shadow summit base set": "shadow summit",
    "ember crown base": "ember crown",
    "crystal tides": "crystal tide",
    "iron verdicts": "iron verdict",
    "lunar blooms": "lunar bloom",
}

CONDITIONS = {
    "NM": "NM", "Near Mint": "NM",
    "LP": "LP", "Lightly Played": "LP",
    "MP": "MP", "Moderately Played": "MP",
    "HP": "HP", "Heavily Played": "HP",
    "DMG": "DMG", "Damaged": "DMG",
}

# (regex on the lowercased title, product type). First match wins.
SEALED_TYPE_RULES = [
    (r"elite trainer box|\betb\b", "etb"),
    (r"booster box|\bdisplay\b", "booster_box"),
    (r"booster bundle", "booster_bundle"),
    (r"sleeved .*booster pack|sleeved booster", "sleeved_pack"),
    (r"blister|checklane", "blister"),
    (r"booster pack|\bpack\b", "booster_pack"),
    (r"mini tin|\btin\b", "tin"),
    (r"poster collection|collection|\bbox\b|premium", "collection"),
    (r"\bplush\b|plushie", "plush"),
    (r"binder|sleeves|deck box|supplies|charm", "supplies"),
]

SINGLE_BANDS = [(5.0, "<5"), (20.0, "5-20"), (float("inf"), "20+")]


TYPE_WORDS = (r"\b(elite trainer box|etb|booster box|booster bundle|booster pack|sleeved|premium checklane"
              r"|checklane|blister|single pack|sealed|collection box)\b")
GRADED = re.compile(r"\bgraded\s*\d+(\.\d)?\b", re.I)


def normalize_set(name):
    """Lowercase, trim, drop product-type words and map known spellings to one set name."""
    if name is None or pd.isna(name):
        return ""
    cleaned = str(name).strip().lower()
    cleaned = re.sub(TYPE_WORDS, " ", cleaned)
    cleaned = re.sub(r"^\s*[a-z]{1,4}\d{0,3}[a-z]?:\s*", "", cleaned)
    cleaned = cleaned.replace("—", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(" -·")
    if cleaned in SET_ALIASES:
        return SET_ALIASES[cleaned]
    return cleaned


def price_band(price):
    """Singles price band used for the singles demand unit."""
    if price is None or pd.isna(price):
        return "unknown"
    for upper, label in SINGLE_BANDS:
        if price < upper:
            return label
    return "20+"


def detect_game(title):
    low = title.lower()
    if "game b" in low or re.search(r"\bgb\d{2}-\d{3}\b", low):
        return "game_b"
    if "game c" in low:
        return "game_c"
    return "game_a"


def sealed_type(text):
    low = text.lower()
    for pattern, kind in SEALED_TYPE_RULES:
        if re.search(pattern, low):
            return kind
    return "other_sealed"


def seller_tag_pattern(sellers):
    """Regex alternation of the sellers' display names, e.g. 'seller a|seller b'."""
    names = []
    for seller in sellers:
        names.append(re.escape(seller.display_name.lower()))
    return "|".join(names)


def strip_seller(title, sellers=DEFAULT_SELLERS):
    """Remove the '(Seller A)' / '- Seller A' tags and the '(BB)' style suffixes."""
    text = str(title)
    tags = seller_tag_pattern(sellers)
    text = re.sub(r"\((" + tags + r")\)", "", text, flags=re.I)
    text = re.sub(r"\s*-\s*(" + tags + r")\s*$", "", text, flags=re.I)
    text = re.sub(r"\((bb|etb|pl)\)", "", text, flags=re.I)
    text = re.sub(r"\b(" + tags + r")\b\s*$", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip(" -·")


def parse_prefix_title(title, sellers=DEFAULT_SELLERS):
    """Prefix-grammar titles look like 'ETB - Shadow Summit (Seller B)'."""
    base = strip_seller(title, sellers)
    parts = base.split(" - ", 1)
    prefix = parts[0].strip().lower()
    rest = parts[1].strip() if len(parts) > 1 else ""
    prefix_types = {
        "booster pack": "booster_pack",
        "etb": "etb",
        "booster box": "booster_box",
        "slab": "slab",
        "supplies": "supplies",
    }
    if prefix in prefix_types:
        kind = prefix_types[prefix]
        set_name = rest
    elif prefix == "specialty":
        kind = sealed_type(rest)
        if kind == "other_sealed":
            kind = "collection"
        set_name = rest
    else:
        kind = sealed_type(base)
        set_name = rest or base
    condition = "Graded" if kind == "slab" else "Sealed"
    return {"product_type": kind, "set_raw": set_name, "condition": condition}


FAMILY_TITLES = [
    (r"^game a booster pack$", "booster_pack", "game_a"),
    (r"^game a booster bundle$", "booster_bundle", "game_a"),
    (r"^game a elite trainer box$", "etb", "game_a"),
    (r"^game a center box$", "collection", "game_a"),
    (r"^game b illustration collection( vol \d+)?$", "collection", "game_b"),
    (r"^game b booster pack$", "booster_pack", "game_b"),
    (r"\bplush\b", "plush", "game_a"),
]


def parse_segment_title(title, variant, sellers=DEFAULT_SELLERS):
    """Segment-grammar titles: product families (set in the variant), single sealed items, or singles."""
    base = strip_seller(title, sellers)
    low = base.lower()
    variant_text = "" if variant is None or pd.isna(variant) else str(variant).strip()

    for pattern, kind, game in FAMILY_TITLES:
        if re.search(pattern, low):
            set_name = variant_text if variant_text != "Default Title" else ""
            return {"product_type": kind, "set_raw": set_name, "condition": "Sealed", "game": game}

    if "plushie" in low:
        return {"product_type": "plush", "set_raw": "plush", "condition": "New"}
    if GRADED.search(base) or GRADED.search(variant_text):
        return {"product_type": "slab", "set_raw": "", "condition": "Graded"}

    segments = [seg.strip() for seg in base.split("·")]
    segment_labels = []
    for seg in segments:
        segment_labels.append(seg.lower())

    is_sealed = "sealed" in segment_labels or "(sealed)" in low
    condition = ""
    for seg in segments:
        if seg in CONDITIONS:
            condition = CONDITIONS[seg]
    if not condition and variant_text in CONDITIONS:
        condition = CONDITIONS[variant_text]

    if is_sealed or (not condition and sealed_type(low) != "other_sealed"):
        kind = sealed_type(segments[0])
        bracket = re.search(r"\(([^)]+)\)", segments[0])
        set_name = bracket.group(1) if bracket else segments[0]
        return {"product_type": kind, "set_raw": set_name, "condition": "Sealed"}

    # A single: "Name Number · Set · NM" (set segment is optional).
    set_name = ""
    if len(segments) >= 3:
        set_name = segments[1]
    return {"product_type": "single", "set_raw": set_name, "condition": condition or "NM"}


def seller_from_title(title, sellers):
    """Which seller's tag appears in the title; defaults to the first seller in the registry."""
    low = str(title).lower()
    for seller in sellers:
        if "(" + seller.display_name.lower() + ")" in low:
            return seller
    return sellers[0]


def parse_title(title, variant=None, seller=None, sellers=DEFAULT_SELLERS):
    """Parse one product title. Returns a dict with game, product_type, set, condition."""
    title_text = str(title)
    if seller is None:
        seller_config = seller_from_title(title_text, sellers)
    else:
        seller_config = None
        for candidate in sellers:
            if candidate.seller_id == seller:
                seller_config = candidate
        if seller_config is None:
            raise KeyError(f"unknown seller id {seller!r}")
    if seller_config.grammar == "prefix":
        parsed = parse_prefix_title(title_text, sellers)
    else:
        parsed = parse_segment_title(title_text, variant, sellers)
    if "game" not in parsed:
        parsed["game"] = detect_game(title_text)
    parsed["set"] = normalize_set(parsed["set_raw"])
    parsed["seller"] = seller_config.seller_id
    parsed["is_single"] = parsed["product_type"] == "single"
    return parsed


def family_key(parsed, list_price=None):
    if parsed["is_single"]:
        band = price_band(list_price)
        return "single|" + parsed["game"] + "|" + (parsed["set"] or "unknown set") + "|" + band
    return parsed["game"] + "|" + parsed["product_type"] + "|" + (parsed["set"] or "unknown set")


def load_overrides(path):
    """inputs/catalog_overrides.csv: product, variant, product_type, set, condition (blank = keep parsed)."""
    path = Path(path)
    if not path.exists():
        return {}
    table = pd.read_csv(path, dtype=str).fillna("")
    overrides = {}
    for _, row in table.iterrows():
        key = (row["product"], row.get("variant", ""))
        overrides[key] = row
    return overrides


IN_SCOPE_TYPES = {"single", "booster_pack", "sleeved_pack", "booster_bundle", "etb", "booster_box",
                  "blister", "tin", "collection", "other_sealed"}


def ledger_sets_by_sku(ledger):
    """SKU -> normalized set name from the unit-cost sheet's Set column."""
    lookup = {}
    if ledger is None:
        return lookup
    for _, row in ledger.iterrows():
        sku = row.get("sku")
        set_name = row.get("ledger_set")
        if isinstance(sku, str) and isinstance(set_name, str) and set_name.strip():
            lookup[sku.strip()] = normalize_set(set_name)
    return lookup


def build_catalog(items, overrides_path=None, ledger=None, sellers=DEFAULT_SELLERS):
    """items: DataFrame with product, variant, seller, list_price (+ sku). Returns one row per (product, variant).

    Singles with no set in the title take the unit-cost sheet's Set for their SKU.
    """
    overrides = {}
    if overrides_path is not None:
        overrides = load_overrides(overrides_path)
    sku_sets = ledger_sets_by_sku(ledger)
    columns = ["product", "variant", "seller", "list_price"]
    if "sku" in items.columns:
        columns.append("sku")
    unique_items = items[columns].drop_duplicates(["product", "variant"])
    rows = []
    for _, item in unique_items.iterrows():
        parsed = parse_title(item["product"], item["variant"], item["seller"], sellers)
        sku = item.get("sku")
        if isinstance(sku, str) and sku in sku_sets and (parsed["set"] == "" or parsed["is_single"]):
            parsed["set"] = sku_sets[sku]
            parsed["set_source"] = "ledger"
        else:
            parsed["set_source"] = "title"
        parsed["sku"] = sku if isinstance(sku, str) else ""
        key = (item["product"], "" if pd.isna(item["variant"]) else item["variant"])
        if key in overrides:
            fix = overrides[key]
            for column in ["product_type", "set", "condition", "game"]:
                value = fix.get(column, "")
                if value:
                    parsed[column] = value
            parsed["set"] = normalize_set(parsed["set"])
            parsed["is_single"] = parsed["product_type"] == "single"
        parsed["product"] = item["product"]
        parsed["variant"] = item["variant"]
        parsed["family"] = family_key(parsed, item["list_price"])
        parsed["in_scope"] = parsed["product_type"] in IN_SCOPE_TYPES
        parsed["needs_review"] = parsed["in_scope"] and (
            (parsed["set"] == "" and not parsed["is_single"]) or parsed["product_type"] == "other_sealed")
        rows.append(parsed)
    return pd.DataFrame(rows)
