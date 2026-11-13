from agent.money import format_money, is_indian_locale


def test_en_in_grouping():
    assert format_money(120000, "en-IN") == "₹1,20,000"
    assert format_money(4500, "en-IN") == "₹4,500"
    assert format_money(2500000, "en-IN") == "₹25,00,000"
    assert format_money(500, "en-IN") == "₹500"
    assert format_money(0, "en-IN") == "₹0"


def test_en_us_grouping():
    assert format_money(120000, "en-US") == "₹120,000"
    assert format_money(2500000, "en-US") == "₹2,500,000"
    assert format_money(999, "en-US") == "₹999"


def test_decimals_and_negative():
    assert format_money(666.666, "en-US") == "₹666.67"
    assert format_money(666.666, "en-IN") == "₹666.67"
    assert format_money(-1000, "en-US") == "-₹1,000"
    assert format_money(-120000, "en-IN") == "-₹1,20,000"
    assert format_money(4500.0, "en-US") == "₹4,500"


def test_indian_locale_detection():
    assert is_indian_locale("en-IN")
    assert is_indian_locale("hi")
    assert is_indian_locale("hi-IN")
    assert is_indian_locale("ta-IN")
    assert not is_indian_locale("en-US")
    assert not is_indian_locale("en-GB")
    assert format_money(120000, "en-GB") == "₹120,000"
