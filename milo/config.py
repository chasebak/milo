"""Run-time configuration: sellers, fees and the store's closed days.

Everything the pipeline needs to know about a particular store lives here, so the
rest of the code never hardcodes a seller name, a fee rate or a closed weekday.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SellerConfig:
    """One consignment seller whose listings appear in the store's stock history.

    grammar picks the title parser in catalog.py:
      "segment"  titles like "Booster Pack (Crystal Tide) · Sealed · SKU-0133 (Seller A)"
                 or family titles like "Game A Booster Pack (Seller A)" with the set in the variant
      "prefix"   titles like "ETB - Shadow Summit (Seller B)"
    """
    seller_id: str
    display_name: str
    grammar: str
    sales_file: str


DEFAULT_SELLERS = (
    SellerConfig("seller_a", "Seller A", "segment", "seller_a_sales.csv"),
    SellerConfig("seller_b", "Seller B", "prefix", "seller_b_sales.csv"),
)


@dataclass(frozen=True)
class StoreConfig:
    """Fee model and store calendar. `commission` has no default on purpose: it is
    a contractual number that must be set explicitly per store."""
    commission: float                        # REQUIRED: store commission share of each sale
    card_fee_rate: float = 0.029             # standard published card-processing rate
    order_fee: float = 0.30                  # standard published flat per-order processing fee
    store_closed_weekdays: tuple = (2,)      # 0=Monday ... 6=Sunday
    sellers: tuple = DEFAULT_SELLERS
    primary_seller: str = "seller_a"         # the seller whose restock plan is being built


# Illustrative example rate for the synthetic dataset; not any real store's rate.
EXAMPLE_CONFIG = StoreConfig(commission=0.10)

WEEKDAY_ABBREVIATIONS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def closed_weekday_names(closed_weekdays):
    """Map weekday numbers (0=Monday) to the Mon/Tue/... labels used in the weekday table."""
    names = []
    for day_number in closed_weekdays:
        names.append(WEEKDAY_ABBREVIATIONS[int(day_number)])
    return names


def seller_by_id(sellers, seller_id):
    for seller in sellers:
        if seller.seller_id == seller_id:
            return seller
    raise KeyError(f"no seller with id {seller_id!r}")
