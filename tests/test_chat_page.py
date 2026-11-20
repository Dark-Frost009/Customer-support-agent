"""The browser chat page, its demo-login endpoint, and the per-tool events it displays."""

import pytest
from fastapi.testclient import TestClient

import main
from conftest import FakeChatClient, latest_token

PRIYA_AUTH = {"Authorization": "Bearer demo-priya"}


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


def test_root_serves_the_chat_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "<title>Customer support agent</title>" in r.text


def test_api_docs_still_available(client):
    assert client.get("/docs").status_code == 200


def test_demo_customers_lists_demo_logins_with_names(client):
    people = client.get("/demo/customers").json()
    assert {p["name"] for p in people} == {"Priya Sharma", "Rahul Verma", "Ananya Iyer"}
    assert all(p["token"].startswith("demo-") for p in people)


def test_demo_customers_hidden_when_demo_mode_off(client, monkeypatch):
    monkeypatch.setattr(main, "DEMO_MODE", False)
    assert client.get("/demo/customers").status_code == 404


def _chat(client, message, session_id=None):
    body = {"message": message, **({"session_id": session_id} if session_id else {})}
    return client.post("/chat", json=body, headers=PRIYA_AUTH).json()


def test_events_show_preview_then_completed_cancellation(client):
    main.agent._client = FakeChatClient([[("cancel_order", {"order_id": 5})], "Shall I cancel it?"])
    first = _chat(client, "Can I cancel order 5?")
    assert first["events"] == [{"tool": "cancel_order", "ok": True, "preview": True,
                                "error_code": None, "order_id": 5, "escalation_id": None}]

    main.agent._client = FakeChatClient([
        lambda m: [("cancel_order", {"order_id": 5, "confirmation_token": latest_token(m)})],
        "Cancelled.",
    ])
    second = _chat(client, "Yes", first["session_id"])
    assert second["events"][0]["ok"] and not second["events"][0]["preview"]


def test_events_show_guardrail_blocks(client):
    main.agent._client = FakeChatClient([[("get_order", {"order_id": 8})], "I can't find that order."])
    event = _chat(client, "Show me order 8")["events"][0]
    assert event["ok"] is False and event["error_code"] == "ORDER_NOT_FOUND" and event["order_id"] == 8


def test_events_carry_escalation_ticket(client):
    main.agent._client = FakeChatClient([[("escalate_to_human", {"reason": "Customer asked for a person"})],
                                         "A colleague will be in touch."])
    event = _chat(client, "I'd like a person")["events"][0]
    assert event["tool"] == "escalate_to_human" and event["escalation_id"] is not None


def test_events_contain_no_tool_output(client):
    # Events are status only: order details stay in the reply text, so the
    # waybill can't leak anything the model didn't say.
    main.agent._client = FakeChatClient([[("get_order", {"order_id": 4})], "It has shipped."])
    event = _chat(client, "Where is order 4?")["events"][0]
    assert set(event) == {"tool", "ok", "preview", "error_code", "order_id", "escalation_id"}
