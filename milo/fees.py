"""Fee math: store commission, card fee, per-order fee.

The commission comes from StoreConfig (a required key); the card fee (2.9%) and the
flat per-order fee ($0.30) default to standard published card-processing rates.
"""


def blended_fee_rate(config, card_share=0.80):
    """Percentage fee: store commission plus card fee on the card-paid share."""
    return config.commission + config.card_fee_rate * card_share


def net_cash_per_unit(price, config, card_share=0.80, units_per_order=2.3):
    """Cash the seller keeps per unit sold at `price` after all fees."""
    fee_rate = blended_fee_rate(config, card_share)
    order_fee_per_unit = config.order_fee / units_per_order
    return price * (1.0 - fee_rate) - order_fee_per_unit
