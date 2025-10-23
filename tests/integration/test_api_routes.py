# Copyright 2025 unusedusername01
# SPDX-License-Identifier: Apache-2.0

import json
from uuid import uuid4

import pytest

from src.data_pipeline.constants import PORTFOLIO_PATH, PORTFOLIOS_PATH


@pytest.mark.integration
def test_load_portfolio_data_success(client, isolate_data_paths):
    payload = {
        "budget": 1500,
        "target_date": "2025-12-31",
        "holdings": {"AAPL": 1.0},
        "currency": "USD",
        "risk_tolerance": "medium",
        "criteria": "sector",
        "prediction_strength": "weak",
    }
    path = PORTFOLIO_PATH("portfolio1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))

    response = client.post("/utils/load_portfolio_data", json={"portfolio_id": "portfolio1"})
    assert response.status_code == 200
    data = response.json()
    assert data["err_code"] == 200
    assert data["portfolio_data"]["budget"] == 1500


@pytest.mark.integration
def test_load_portfolio_data_missing_returns_404(client, isolate_data_paths):
    response = client.post("/utils/load_portfolio_data", json={"portfolio_id": "missing"})
    assert response.status_code == 200
    data = response.json()
    assert data["err_code"] == 404


@pytest.mark.integration
def test_start_analysis_reports_run_failure(client, monkeypatch):
    async def fake_run(portfolio_id: str):
        raise ValueError("WebSocket connection required")

    monkeypatch.setattr("src.langgraph_workflow.app.run", fake_run)

    response = client.post("/analysis/portfolio/demo/start")
    assert response.status_code == 200
    data = response.json()
    assert data["err_code"] == 400
    assert "WebSocket connection" in data["details"]


@pytest.mark.integration
def test_create_and_delete_portfolio(client, isolate_data_paths):
    portfolio_id = f"ci_{uuid4().hex[:8]}"

    create = client.post(
        "/utils/create_portfolio",
        json={
            "portfolio_id": portfolio_id,
            "budget": 2500,
            "target_date": "2025-12-31",
            "holdings": {"MSFT": 2.0},
            "currency": "USD",
            "risk_tolerance": "low",
            "criteria": "market",
            "prediction_strength": "weak",
        },
    )
    assert create.status_code == 200
    assert create.json()["err_code"] == 200
    assert PORTFOLIO_PATH(portfolio_id).exists()

    delete = client.request("DELETE", "/utils/delete_portfolio", json={"portfolio_id": portfolio_id})
    assert delete.status_code == 200
    assert delete.json()["err_code"] == 200
    assert not PORTFOLIO_PATH(portfolio_id).exists()


@pytest.mark.integration
def test_get_shares_price_supports_list(client, monkeypatch):
    monkeypatch.setattr("src.langgraph_workflow.app.get_shares_price", lambda **kwargs: 123.45)

    response = client.post(
        "/utils/get_shares_price",
        json={"ticker": ["AAPL", "MSFT"], "amount": [1, 2], "currency": "USD"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["err_code"] == 200
    assert payload["price"] == [123.45, 123.45]


@pytest.mark.integration
def test_get_shares_price_validates_payload(client):
    response = client.post(
        "/utils/get_shares_price",
        json={"ticker": [], "amount": [], "currency": "USD"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["err_code"] == 400


@pytest.mark.integration
def test_validate_currency_success_and_failure(client):
    ok = client.post("/utils/validate_currency", json={"currency": "USD"})
    assert ok.status_code == 200
    assert ok.json()["err_code"] == 200

    bad = client.post("/utils/validate_currency", json={"currency": "INVALID"})
    assert bad.status_code == 200
    assert bad.json()["err_code"] == 400


@pytest.mark.integration
def test_list_portfolios_handles_empty(client, isolate_data_paths):
    PORTFOLIOS_PATH.mkdir(parents=True, exist_ok=True)

    response = client.get("/utils/list_portfolios")
    assert response.status_code == 200
    assert response.json()["err_code"] == 404
