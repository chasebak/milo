from milo.config import EXAMPLE_CONFIG
from milo.fees import blended_fee_rate, net_cash_per_unit


def test_net_cash_all_card_single_unit_order():
    got = net_cash_per_unit(100.0, EXAMPLE_CONFIG, card_share=1.0, units_per_order=1)
    assert abs(got - (100 * (1 - 0.129) - 0.30)) < 1e-9


def test_blended_rate_default():
    assert abs(blended_fee_rate(EXAMPLE_CONFIG, 0.80) - (0.10 + 0.029 * 0.80)) < 1e-12
