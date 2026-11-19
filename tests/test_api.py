"""HTTP-level tests for main.py: authentication, sessions, and failure handling."""

from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

import main
from conftest import FakeChatClient

PRIYA_AUTH = {"Authorization": "Bearer demo-priya"}
RAHUL_AUTH = {"Authorization": "Bearer demo-rahul"}


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


def _use_model(script):
    fake = FakeChatClient(script)
    main.agent._client = fake
    return fake


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong-token"}])
def test_chat_requires_valid_token(client, headers):
    _use_model(["hi"])
    assert client.post("/chat", json={"message": "hi"}, headers=headers).status_code == 401


def test_chat_returns_reply_and_session(client):
    _use_model([[("get_order", {"order_id": 4})], "Your order has shipped."])
    r = client.post("/chat", json={"message": "Where is order 4?"}, headers=PRIYA_AUTH)
    body = r.json()
    assert r.status_code == 200
    assert body["reply"] == "Your order has shipped."
    assert body["tools_called"] == ["get_order"]
    assert body["session_id"]


def test_customer_id_in_body_is_rejected(client):
    _use_model(["hi"])
    r = client.post("/chat", json={"message": "hi", "customer_id": 2}, headers=PRIYA_AUTH)
    assert r.status_code == 422


def test_another_customer_cannot_use_your_session(client):
    _use_model(["hi"])
    sid = client.post("/chat", json={"message": "hi"}, headers=PRIYA_AUTH).json()["session_id"]
    _use_model(["hi"])
    r = client.post("/chat", json={"message": "show my chat", "session_id": sid}, headers=RAHUL_AUTH)
    assert r.status_code == 403


def test_model_outage_returns_503_and_keeps_history_clean(client):
    _use_model(["hi"])
    sid = client.post("/chat", json={"message": "hi"}, headers=PRIYA_AUTH).json()["session_id"]
    session = main.sessions._sessions[sid]
    before = len(session.messages)

    def boom(**_):
        raise RuntimeError("OpenAI is down")

    main.agent._client = NS(chat=NS(completions=NS(create=boom)))
    r = client.post("/chat", json={"message": "hello?", "session_id": sid}, headers=PRIYA_AUTH)
    assert r.status_code == 503
    assert len(session.messages) == before  # failed message not left dangling