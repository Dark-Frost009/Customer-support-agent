"""
main.py — FastAPI entry point.

Run from the project root:
    uvicorn main:app --app-dir app --reload

Then open http://127.0.0.1:8000/ for the chat page, or /docs for the raw API
(click "Authorize", enter a demo token such as `demo-priya`, try POST /chat).

Authentication
--------------
The customer is identified ONLY from the bearer token, never from the request
body. That's the anchor for the whole ownership guardrail: the token resolves
to a customer_id here, the Session is bound to it, and every tool call gets it
from the Session.

The token table below is a development stand-in. Replace `get_current_customer`
with real auth (JWT / session cookie from your login system) before deploying;
nothing else needs to change.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import rag
from agent import SessionError, SessionStore, SupportAgent
from db import get_db, init_db
from schemas import ChatRequest, ChatResponse, ToolEvent

log = logging.getLogger("support_agent")

# Dev-only tokens -> customer ids. Override with env DEMO_TOKENS='{"tok": 1}'.
DEMO_TOKENS: dict[str, int] = json.loads(
    os.getenv("DEMO_TOKENS", '{"demo-priya": 1, "demo-rahul": 2, "demo-ananya": 3}')
)

# The chat page needs to know the demo logins. Exposing tokens over HTTP is only
# acceptable because they're throwaway demo credentials; set DEMO_MODE=0 (and
# replace the token table with real auth) for any real deployment.
DEMO_MODE: bool = os.getenv("DEMO_MODE", "1") != "0"
CHAT_PAGE = Path(__file__).resolve().parent / "static" / "index.html"

bearer = HTTPBearer(auto_error=False)
sessions = SessionStore()
agent = SupportAgent()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    init_db()  # creates the schema if missing; never wipes data
    if not os.getenv("OPENAI_API_KEY"):
        log.warning("OPENAI_API_KEY is not set; chat requests will fail until it is.")
    if not rag.INDEX_PATH.exists():
        log.warning("Policy index not found; run `python app/rag.py build` before chatting.")
    yield


app = FastAPI(title="Customer Support Agent", version="0.1.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Auth dependency
# --------------------------------------------------------------------------- #
def get_current_customer(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> int:
    if creds is None or creds.credentials not in DEMO_TOKENS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    customer_id = DEMO_TOKENS[creds.credentials]
    with get_db() as conn:
        exists = conn.execute("SELECT 1 FROM customers WHERE id = ?", (customer_id,)).fetchone()
    if not exists:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unknown customer.")
    return customer_id


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "policy_index": rag.INDEX_PATH.exists(),
        "openai_key_configured": bool(os.getenv("OPENAI_API_KEY")),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, customer_id: int = Depends(get_current_customer)) -> ChatResponse:
    # Sync def on purpose: the OpenAI client and sqlite3 are blocking, so
    # FastAPI runs this in its thread pool instead of blocking the event loop.
    try:
        session = sessions.get_or_create(req.session_id, customer_id)
    except SessionError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Session not found.")

    try:
        out = agent.respond(session, req.message)
    except Exception as exc:  # unexpected bug; OpenAI failures are handled inside the agent
        log.exception("Agent failure for session %s", session.session_id)
        # Drop the unanswered user message so a retry doesn't send it twice.
        if session.messages and session.messages[-1].get("role") == "user":
            session.messages.pop()
        # The most common setup mistake gets a specific message instead of a
        # generic one. (OpenAI outages never reach here: the agent handles them.)
        detail = (
            "No OpenAI API key is configured on the server. Set OPENAI_API_KEY and restart it."
            if not os.getenv("OPENAI_API_KEY")
            else "The assistant is temporarily unavailable. Please try again shortly."
        )
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail) from exc

    return ChatResponse(
        session_id=session.session_id,
        reply=out.reply,
        escalated=out.escalation_id is not None,
        escalation_id=out.escalation_id,
        tools_called=out.tools_called,
        degraded=out.degraded,
        events=[ToolEvent(**e) for e in out.events],
    )


@app.get("/", include_in_schema=False)
def chat_page() -> FileResponse:
    """The browser chat page (a single self-contained HTML file)."""
    return FileResponse(CHAT_PAGE, media_type="text/html")


@app.get("/demo/customers", include_in_schema=False)
def demo_customers() -> list[dict[str, object]]:
    """Demo logins for the chat page's customer switcher. Disabled when DEMO_MODE=0."""
    if not DEMO_MODE:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")
    with get_db() as conn:
        out = []
        for token, cid in DEMO_TOKENS.items():
            row = conn.execute("SELECT name, city FROM customers WHERE id = ?", (cid,)).fetchone()
            if row:
                out.append({"token": token, "customer_id": cid, "name": row["name"], "city": row["city"]})
    return out
