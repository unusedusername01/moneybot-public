import json
from datetime import datetime, timedelta

import pytest

import src.data_pipeline.constants as c
from src.data_pipeline.data_collector import DataCollector
from src.data_pipeline.data_fetcher import DataFetcher


@pytest.mark.unit
def test_collect_fundamentals_returns_latest(isolate_data_paths):
    ticker_dir = c.TICKERS_PATH / "AAPL"
    ticker_dir.mkdir(parents=True, exist_ok=True)

    DataFetcher.save_daily_log(
        {"sector": "Technology", "trailingPE": 18.2},
        "ticker",
        "AAPL",
        "fundamentals",
    )

    assert DataCollector.collect_fundamentals("AAPL", field="sector") == "Technology"


@pytest.mark.unit
def test_collect_fundamentals_skips_stale(isolate_data_paths):
    ticker_dir = c.TICKERS_PATH / "MSFT"
    ticker_dir.mkdir(parents=True, exist_ok=True)

    stale_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    stale = ticker_dir / f"fundamentals_{stale_date}.json"
    stale.write_text(json.dumps({"sector": "Technology"}))

    assert DataCollector.collect_fundamentals("MSFT", field="sector") is None


@pytest.mark.unit
def test_collect_predictions_prefers_strongest(isolate_data_paths):
    ticker_dir = c.TICKERS_PATH / "NVDA"
    ticker_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = c.PREDICTIONS_PATH("NVDA")
    pred_dir.mkdir(parents=True, exist_ok=True)

    weak_file = pred_dir / "predictions_weak_2025-01-01.json"
    weak_file.write_text(json.dumps([{"prediction": {"ticker": "NVDA", "score": 0.2}}]))

    strong_file = pred_dir / f"predictions_strong_{datetime.now().strftime('%Y-%m-%d')}.json"
    strong_file.write_text(json.dumps([{"prediction": {"ticker": "NVDA", "score": 0.9}}]))

    data = DataCollector.collect_predictions("NVDA")
    assert data == [{"ticker": "NVDA", "score": 0.9}]


@pytest.mark.unit
def test_collect_news_uses_latest(isolate_data_paths):
    ticker_dir = c.TICKERS_PATH / "GOOGL"
    ticker_dir.mkdir(parents=True, exist_ok=True)

    old_file = ticker_dir / f"{c.get_file_prefix('news', 'GOOGL', timestamp=False)}2024-01-01.json"
    old_file.write_text(
        json.dumps({"news": [{"title": "Old"}], "total_articles": 1, "ticker": "GOOGL"})
    )

    DataFetcher.save_daily_log(
        {
            "ticker": "GOOGL",
            "news": [{"title": "Fresh", "url": "https://example.com"}],
            "total_articles": 1,
        },
        "ticker",
        "GOOGL",
        "news",
    )

    data = DataCollector.collect_news("GOOGL")
    assert data["news"][0]["title"] == "Fresh"
