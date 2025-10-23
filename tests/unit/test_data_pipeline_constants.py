import re

import pytest

from src.data_pipeline.constants import get_file_prefix, get_filename


@pytest.mark.unit
def test_get_file_prefix_includes_sanitised_field():
    prefix = get_file_prefix("news", "Big Tech Boom")
    assert prefix.startswith("big_tech_boom_news_")
    date_part = prefix.split("_news_")[-1]
    assert re.match(r"\d{4}-\d{2}-\d{2}", date_part)


@pytest.mark.unit
def test_get_file_prefix_without_timestamp():
    prefix = get_file_prefix("fundamentals", "AAPL", timestamp=False)
    assert prefix == "fundamentals_"


@pytest.mark.unit
def test_get_filename_appends_extension():
    filename = get_filename("market", "NASDAQ")
    assert filename.endswith(".json")
    assert filename.startswith("market_")
