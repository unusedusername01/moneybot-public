import asyncio

import pytest

from src.langgraph_workflow.custom_types import AllocationResponse, MessagePayload, MessageResponse


@pytest.mark.unit
def test_allocate_budget_proportional(monkeypatch, app_module):
    monkeypatch.setattr(
        app_module,
        "get_shares_number",
        lambda ticker, amount, currency, allow_fractional: round(amount, 2),
    )

    scores = {"AAA": (1, "ok"), "BBB": (3, "better")}
    result = app_module.allocate_budget(scores, budget=400, currency="USD")

    assert result == {"AAA": pytest.approx(100.0), "BBB": pytest.approx(300.0)}


@pytest.mark.unit
def test_allocate_budget_handles_zero_total(app_module):
    result = app_module.allocate_budget({"AAA": (0, "nope")}, budget=500, currency="USD")
    assert result == {"AAA": 0.0}


@pytest.mark.unit
def test_decide_next_node_resume(app_module):
    state = {"app_data": {"version": 1, "next_node": "call_logger"}}
    assert app_module.decide_next_node(state) == "call_logger"


@pytest.mark.unit
def test_decide_next_node_default(app_module):
    state = {"app_data": {"version": 0, "next_node": None}}
    assert app_module.decide_next_node(state) == "default"


@pytest.mark.unit
def test_get_state_roundtrip(app_module):
    app_module.PERSISTENT_USER_STATE["demo"] = {"app_data": {"selected_portfolio": "demo"}}
    try:
        state = app_module.get_state("demo")
        assert state["app_data"]["selected_portfolio"] == "demo"
    finally:
        app_module.PERSISTENT_USER_STATE.pop("demo", None)


@pytest.mark.unit
def test_call_synthesizer_merges_batches(monkeypatch, app_module):
    calls = {}

    class DummyLLM:
        def merge_batches(self, prompt: str):
            calls.setdefault("prompts", []).append(prompt)
            return {"AAA": (5, "great")}

    monkeypatch.setattr(app_module, "llm", DummyLLM())
    monkeypatch.setattr(app_module, "allocate_budget", lambda scores, budget, currency, allow_fractional=True: {"AAA": 42.0})
    monkeypatch.setattr(app_module, "TOP_K_INVESTMENTS", 1, raising=False)

    pushed_messages = []

    async def fake_push(portfolio_id, message: MessagePayload):
        pushed_messages.append(message)
        return None

    monkeypatch.setattr(app_module, "push_message_to_client", fake_push)

    class DummyLoop:
        def run_in_executor(self, executor, func, *args):
            async def runner():
                return func(*args)

            return runner()

    monkeypatch.setattr(app_module.asyncio, "get_event_loop", lambda: DummyLoop())

    state = {
        "app_data": {
            "selected_portfolio": "demo",
            "messages": [],
            "budget_allocation": {},
        },
        "portfolio_data": {
            "budget": 1000,
            "currency": "USD",
            "holdings": {},
            "target_date": "2025-12-31",
            "risk_tolerance": "medium",
        },
        "ranked_batches": [
            {"tickers": ["AAA"], "scores": {"AAA": (5, "solid")}},
            {"tickers": ["BBB"], "scores": {"BBB": (3, "ok")}},
        ],
    }

    async def _run():
        return await app_module.call_synthesizer(state)

    result = asyncio.run(_run())

    assert result["app_data"]["budget_allocation"] == {"AAA": 42.0}
    assert calls.get("prompts"), "Expected merge_batches to be invoked"
    assert len(pushed_messages) == 3
    assert isinstance(pushed_messages[1].data, AllocationResponse)
    assert pushed_messages[1].data.currency == "USD"
    assert isinstance(pushed_messages[2].data, MessageResponse)
    assert pushed_messages[2].data.content.startswith("Rationale behind the allocation")

