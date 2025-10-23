import json

import pytest

import src.data_pipeline.constants as c
from src.data_pipeline.data_fetcher import DataFetcher, NewsFetcher


class DummyLLMProvider:
    pass


@pytest.mark.unit
def test_ensure_dir_creates_directories(isolate_data_paths):
    path = DataFetcher.ensure_dir("ticker", "AAPL")
    assert path.exists()
    assert path == isolate_data_paths / "tickers" / "AAPL"


@pytest.mark.unit
def test_save_daily_log_merges_sector_news(isolate_data_paths):
    existing_path = c.SECTOR_DB_PATH / c.get_filename("news", "Technology")
    existing_payload = {
        "sector": "Technology",
        "news": [{"url": "https://example.com/a", "title": "A"}],
        "total_articles": 1,
        "timestamp": "2025-01-01T00:00:00Z",
    }
    existing_path.write_text(json.dumps(existing_payload))

    updated_payload = {
        "sector": "Technology",
        "news": [
            {"url": "https://example.com/a", "title": "A"},
            {"url": "https://example.com/b", "title": "B"},
        ],
        "total_articles": 2,
        "timestamp": "2025-01-02T00:00:00Z",
    }

    result_path = DataFetcher.save_daily_log(updated_payload, "sector", "Technology", "news")
    data = json.loads(result_path.read_text())

    urls = {article["url"] for article in data["news"]}
    assert urls == {"https://example.com/a", "https://example.com/b"}
    assert data["total_articles"] == 2


@pytest.mark.unit
def test_save_daily_log_market_overwrites_metadata(isolate_data_paths):
    existing_path = c.MARKET_DB_PATH / c.get_filename("news", "US")
    existing_payload = {
        "market": "US",
        "news": [{"url": "https://example.com/old", "title": "Old"}],
        "total_articles": 1,
        "timestamp": "2025-01-01T00:00:00Z",
    }
    existing_path.write_text(json.dumps(existing_payload))

    updated_payload = {
        "market": "US",
        "news": [{"url": "https://example.com/new", "title": "New"}],
        "total_articles": 1,
        "timestamp": "2025-01-03T00:00:00Z",
    }

    result_path = DataFetcher.save_daily_log(updated_payload, "market", "US", "news")
    data = json.loads(result_path.read_text())
    assert data["timestamp"] == "2025-01-03T00:00:00Z"
    assert {article["url"] for article in data["news"]} == {"https://example.com/new"}


@pytest.mark.unit
def test_check_curr_news_count_initializes_limits(isolate_data_paths, stub_sentence_transformer):
    fetcher = NewsFetcher(DummyLLMProvider(), use_themes=False)

    sector_count = fetcher._check_curr_news_count("sector", "Technology")
    market_count = fetcher._check_curr_news_count("market", "US")
    ticker_count = fetcher._check_curr_news_count("ticker", "AAPL")

    assert sector_count == 0
    assert market_count == 0
    assert ticker_count == 0
    assert fetcher.k_sector_dict["Technology"] == fetcher.MAX_NEWS_PER_SECTOR
    assert fetcher.k_market_dict["US"] == fetcher.MAX_NEWS_PER_MARKET
