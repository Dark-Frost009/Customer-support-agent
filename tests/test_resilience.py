"""
Resilience: OpenAI timeouts, retries and fallback replies.

The key property: when the model fails mid-turn, the customer is told about
anything that already happened (a cancellation, a return), so they don't retry
and duplicate it. That summary is built by Python from the tool results.
"""

from types import SimpleNamespace as NS

import httpx2
import openai
import pytest
from fastapi.testclient import TestClient

import agent as agent_module
import main
import rag
from agent import UNAVAILABLE_REPLY, SessionStore, SupportAgent
from conftest import ITEM_SHOES, ORDER_PROCESSING, ORDER_SHOES, PRIYA, FakeChatClient, latest_token
from db import get_db

_REQUEST = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")


def _timeout(_messages=None):
    raise openai.APITimeoutError(request=_REQUEST)


class FailingAfter(FakeChatClient):
    """Plays the script, then raises a timeout on every later call."""

    def _create(self, **kwargs):
        if not self.script:
            self.calls.append(kwargs)
            _timeout()
        return super()._create(**kwargs)


def _status(order_id: int) -> str:
    with get_db() as conn:
        return conn.execute("SELECT status FROM orders WHERE id = ?", (order_id,)).fetchone()[0]


# --------------------------------------------------------------------------- #
# Client configuration and real SDK retry behaviour
# --------------------------------------------------------------------------- #
def test_default_client_has_timeout_and_retries():
    client = agent_module.default_client()
    assert client.max_retries == agent_module.OPENAI_MAX_RETRIES >= 1
    assert client.timeout == agent_module.OPENAI_TIMEOUT_SECONDS


def test_sdk_retries_transient_server_errors():
    """Two 503s then success: the SDK's back-off should absorb them."""
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx2.Response(503, json={"error": {"message": "overloaded"}},
                                   headers={"retry-after-ms": "1"})
        return httpx2.Response(200, json={
            "id": "c1", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "Hello!"}}],
        })

    client = openai.OpenAI(api_key="test", max_retries=3, timeout=5,
                           http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    out = SupportAgent(client=client).respond(SessionStore().get_or_create(None, PRIYA), "hi")
    assert out.reply == "Hello!" and not out.degraded
    assert attempts["n"] == 3


# --------------------------------------------------------------------------- #
# Fallback replies
# --------------------------------------------------------------------------- #
def test_failure_before_anything_happens_gives_polite_reply_and_clean_history():
    session = SessionStore().get_or_create(None, PRIYA)
    out = SupportAgent(client=FailingAfter([])).respond(session, "Where is order 4?")
    assert out.degraded and out.reply == UNAVAILABLE_REPLY
    assert session.messages == []  # unanswered message dropped, so a retry isn't duplicated


def test_failure_after_cancellation_tells_customer_it_was_cancelled():
    session = SessionStore().get_or_create(None, PRIYA)
    preview = FakeChatClient([[("cancel_order", {"order_id": ORDER_PROCESSING})], "Confirm?"])
    agent = SupportAgent(client=preview)
    agent.respond(session, "Cancel order 5")

    # Turn 2: the model confirms the cancellation, then OpenAI times out
    # before it can write its reply.
    agent._client = FailingAfter([
        lambda m: [("cancel_order", {"order_id": ORDER_PROCESSING,
                                     "confirmation_token": latest_token(m)})],
    ])
    out = agent.respond(session, "Yes")

    assert _status(ORDER_PROCESSING) == "cancelled"
    assert out.degraded
    assert "Order #5 has been cancelled" in out.reply and "Rs. 2,097.00" in out.reply


def test_failure_after_escalated_return_says_it_went_for_review():
    session = SessionStore().get_or_create(None, PRIYA)
    args = {"order_id": ORDER_SHOES, "order_item_id": ITEM_SHOES, "reason": "doesn't fit"}
    agent = SupportAgent(client=FakeChatClient([[("request_return", args)], "Confirm?"]))
    agent.respond(session, "Refund my shoes")
    agent._client = FailingAfter([lambda m: [("request_return", {**args, "confirmation_token": latest_token(m)})]])
    out = agent.respond(session, "Yes")
    assert "sent to our team for review" in out.reply
    assert "approved" not in out.reply.lower()


def test_conversation_continues_normally_after_a_fallback():
    session = SessionStore().get_or_create(None, PRIYA)
    agent = SupportAgent(client=FailingAfter([[("list_orders", {})]]))
    agent.respond(session, "What are my orders?")  # tool runs, then the model fails

    # History must still be valid for the API: every tool result is followed
    # by an assistant message, so the next call works.
    fake = FakeChatClient(["Here are your orders."])
    agent._client = fake
    out = agent.respond(session, "Try again please")
    assert out.reply == "Here are your orders." and not out.degraded
    roles = [m["role"] for m in fake.last_messages[1:]]
    assert roles == ["user", "assistant", "tool", "assistant", "user"]


def test_policy_search_failure_is_a_tool_error_not_a_crash(call, monkeypatch, policy_index):
    def broken_embedder(model=None):
        return lambda texts: _timeout()

    monkeypatch.setattr(rag, "openai_embedder", broken_embedder)
    r = call("search_policy", {"query": "return window"})
    assert not r.ok and r.error_code == "POLICY_SEARCH_UNAVAILABLE"


def test_api_returns_degraded_reply_instead_of_error():
    main.agent._client = NS(chat=NS(completions=NS(create=lambda **_: _timeout())))
    with TestClient(main.app) as client:
        r = client.post("/chat", json={"message": "hi"},
                        headers={"Authorization": "Bearer demo-priya"})
    assert r.status_code == 200
    assert r.json()["degraded"] is True and r.json()["reply"] == UNAVAILABLE_REPLY
