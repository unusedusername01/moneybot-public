import pytest

from src.langgraph_workflow.custom_types import (
    AllocationResponse,
    ChoiceResponse,
    MessagePayload,
    MessageResponse,
    MessageType,
)


def _make_state(portfolio_id: str, await_schema: MessageType, version: int = 0):
    return {
        "app_data": {
            "selected_portfolio": portfolio_id,
            "messages": [],
            "budget_allocation": {},
            "awaiting": True,
            "await_schema": await_schema,
            "version": version,
            "next_node": None,
        }
    }


@pytest.fixture
def stubbed_push(monkeypatch, app_module):
    sent_messages = []

    async def fake_push(portfolio_id, message):
        sent_messages.append((portfolio_id, message))

    monkeypatch.setattr(app_module, "push_message_to_client", fake_push)
    return sent_messages


@pytest.fixture
def stubbed_invoke(monkeypatch, app_module):
    calls = {}

    async def fake_invoke(state):
        calls["state"] = state

    monkeypatch.setattr(app_module.app, "ainvoke", fake_invoke)
    return calls


@pytest.mark.e2e
def test_e2e_confirm_flow(client, app_module, stubbed_invoke):
    portfolio_id = "portfolio_test"
    app_module.PERSISTENT_USER_STATE[portfolio_id] = _make_state(
        portfolio_id, MessageType.AWAITING_CHOICE
    )

    response = client.post(
        f"/analysis/portfolio/{portfolio_id}/respond",
        json=MessagePayload(data=ChoiceResponse(selection="confirm")).model_dump(mode="json"),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["err_code"] == 200
    state = app_module.PERSISTENT_USER_STATE[portfolio_id]
    assert state["app_data"]["awaiting"] is False
    assert state["app_data"]["next_node"] == "confirm"
    assert stubbed_invoke["state"]["app_data"]["next_node"] == "confirm"


@pytest.mark.e2e
def test_e2e_deny_rerun_flow(client, app_module, stubbed_push, stubbed_invoke):
    portfolio_id = "portfolio_test"
    app_module.PERSISTENT_USER_STATE[portfolio_id] = _make_state(
        portfolio_id, MessageType.AWAITING_CHOICE
    )

    response = client.post(
        f"/analysis/portfolio/{portfolio_id}/respond",
        json=MessagePayload(data=ChoiceResponse(selection="deny")).model_dump(mode="json"),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["err_code"] == 200
    state = app_module.PERSISTENT_USER_STATE[portfolio_id]
    assert state["app_data"]["awaiting"] is True
    assert state["app_data"]["await_schema"] == MessageType.MESSAGE_RESPONSE
    assert state["app_data"]["next_node"] is None
    assert stubbed_push, "Expected follow-up prompt for deny flow"
    assert "state" in stubbed_invoke


@pytest.mark.e2e
def test_e2e_followup_message_sets_rerun(client, app_module, stubbed_push, stubbed_invoke):
    portfolio_id = "portfolio_test"
    state = _make_state(portfolio_id, MessageType.MESSAGE_RESPONSE, version=1)
    state["app_data"]["messages"] = []
    state["app_data"]["awaiting"] = True
    state["app_data"]["budget_allocation"] = {"AAPL": 1.0}
    app_module.PERSISTENT_USER_STATE[portfolio_id] = state

    response_payload = MessagePayload(
        data=MessageResponse(content="Please emphasize ESG factors"),
    ).model_dump(mode="json")

    response = client.post(
        f"/analysis/portfolio/{portfolio_id}/respond",
        json=response_payload,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["err_code"] == 200

    state = app_module.PERSISTENT_USER_STATE[portfolio_id]
    assert state["app_data"]["awaiting"] is False
    assert state["app_data"]["await_schema"] is None
    assert state["app_data"]["next_node"] == "rerun"
    assert state["app_data"]["version"] == 2
    assert state["app_data"]["budget_allocation"] == {}
    assert state["app_data"]["messages"]
    assert state["app_data"]["messages"][-1].content == "Please emphasize ESG factors"
    assert any(
        message.data.prompt == "Analysis restarted"
        for _, message in stubbed_push
        if hasattr(message, "data") and getattr(message.data, "type", None) == MessageType.STATE_UPDATE
    ), "Expected Analysis restarted state update"
    assert "state" in stubbed_invoke


@pytest.mark.e2e
def test_e2e_cancel_flow(client, app_module, stubbed_push, stubbed_invoke):
    portfolio_id = "portfolio_test"
    app_module.PERSISTENT_USER_STATE[portfolio_id] = _make_state(
        portfolio_id, MessageType.AWAITING_CHOICE
    )

    response = client.post(
        f"/analysis/portfolio/{portfolio_id}/respond",
        json=MessagePayload(data=ChoiceResponse(selection="cancel")).model_dump(mode="json"),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["err_code"] == 200
    state = app_module.PERSISTENT_USER_STATE[portfolio_id]
    assert state["app_data"]["awaiting"] is False
    assert state["app_data"]["next_node"] == "cancel"
    assert "state" in stubbed_invoke


@pytest.mark.e2e
def test_e2e_edit_flow(client, app_module, stubbed_push, stubbed_invoke):
    portfolio_id = "portfolio_test"
    app_module.PERSISTENT_USER_STATE[portfolio_id] = _make_state(
        portfolio_id, MessageType.AWAITING_CHOICE
    )

    response = client.post(
        f"/analysis/portfolio/{portfolio_id}/respond",
        json=MessagePayload(data=ChoiceResponse(selection="edit")).model_dump(mode="json"),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["err_code"] == 200
    state = app_module.PERSISTENT_USER_STATE[portfolio_id]
    assert state["app_data"]["awaiting"] is True
    assert state["app_data"]["await_schema"] == MessageType.ALLOCATION_RESPONSE
    assert state["app_data"]["next_node"] is None
    assert stubbed_push, "Edit flow should request allocation data"


@pytest.mark.e2e
def test_e2e_edge_case_unknown_ticker_allocation(client, app_module):
    portfolio_id = "portfolio_test"
    state = _make_state(portfolio_id, MessageType.ALLOCATION_RESPONSE)
    state["app_data"]["budget_allocation"] = {"AAPL": 5.0}
    app_module.PERSISTENT_USER_STATE[portfolio_id] = state

    response = client.post(
        f"/analysis/portfolio/{portfolio_id}/respond",
        json=MessagePayload(
            data=AllocationResponse(content={"MSFT": 1.0})
        ).model_dump(mode="json"),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["err_code"] == 400
    assert "unknown tickers" in payload["details"].lower()


@pytest.mark.e2e
def test_e2e_edge_case_negative_allocation_rejected(client, app_module):
    portfolio_id = "portfolio_test"
    state = _make_state(portfolio_id, MessageType.ALLOCATION_RESPONSE)
    state["app_data"]["budget_allocation"] = {"AAPL": 5.0}
    app_module.PERSISTENT_USER_STATE[portfolio_id] = state

    response = client.post(
        f"/analysis/portfolio/{portfolio_id}/respond",
        json=MessagePayload(
            data=AllocationResponse(content={"AAPL": -1.0})
        ).model_dump(mode="json"),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["err_code"] == 400
    assert "non-negative" in payload["details"].lower()
