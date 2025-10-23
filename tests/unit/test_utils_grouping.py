# Copyright 2025 unusedusername01
# SPDX-License-Identifier: Apache-2.0

import pytest

from src.langgraph_workflow.utils import group_by_criteria, order_ticker_by_score


def _make_ticker(symbol: str, sector: str, market: str, prediction_sum: float) -> dict:
    return {
        "ticker": symbol,
        "fundamentals": {
            "sector": sector,
            "market": market,
            "return_on_equity": 10.0,
            "debt_to_equity": 1.0,
            "profit_margin": 0.2,
            "trailing_pe": 15.0,
            "price_to_book": 2.0,
            "revenue_growth": 0.05,
            "short_name": symbol,
        },
        "news": {
            "ticker": symbol,
            "timestamp": "2025-01-01",
            "news": [],
            "total_articles": 0,
        },
        "predictions": [
            {
                "prediction": prediction_sum,
                "prediction_score": prediction_sum,
                "ticker": symbol,
                "timestamp": "2025-01-01",
                "prediction_mode": "weak",
                "expected_price": 0.0,
                "current_price": 0.0,
                "days_ahead": 1,
                "prediction_date": "2025-01-01",
                "model_details": {},
            }
        ],
    }


@pytest.mark.unit
def test_group_by_sector_balances_batches():
    data = [
        _make_ticker("AAA", "Tech", "US", 0.5),
        _make_ticker("BBB", "Tech", "US", 0.4),
        _make_ticker("CCC", "Finance", "US", 0.3),
        _make_ticker("DDD", "Finance", "US", 0.2),
    ]

    batches = group_by_criteria(data, criteria="sector", max_batch_size=1)

    assert len(batches) == 4
    assert all(len(batch) == 1 for batch in batches)


@pytest.mark.unit
def test_group_by_score_orders_descending():
    data = [
        _make_ticker("AAA", "Tech", "US", 0.1),
        _make_ticker("BBB", "Tech", "US", 0.5),
        _make_ticker("CCC", "Tech", "US", 0.3),
    ]

    ordered = order_ticker_by_score(data)
    symbols = [item["ticker"] for item in ordered]

    assert symbols == ["BBB", "CCC", "AAA"]


@pytest.mark.unit
def test_group_by_market_cap_splits_absolute():
    data = [_make_ticker(f"SYM{i}", "Tech", "US", 0.1 * i) for i in range(5)]

    batches = group_by_criteria(data, criteria="market_cap", max_batch_size=2, split_absolute=True)

    assert len(batches) == 3
    assert all(len(batch) <= 2 for batch in batches)
