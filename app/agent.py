"""
agent.py — the OpenAI function-calling loop and per-session state.

Flow for one customer message
-----------------------------
1. Append the user message to the session history.
2. Call the model with the tool specs from tools.py.
3. If it asks for tools, run each through `tools.dispatch_tool()` with the
   session's authenticated customer_id, append the results, and go to 2.
4. When it replies with plain text, return that to the customer.

Where safety lives
------------------
* The system prompt explains the rules so the model can talk about them, but
  enforcement is in Python: tools.py / guardrails.py decide ownership, refund
  limits, eligibility and escalation. The prompt is a guide, not the guard.
* `customer_id` is fixed on the Session when it's created from the
  authenticated request. A session can't be reused by another customer, so
  history containing one customer's orders never reaches another.
* Cancelling and returning need a preview in one customer turn and a token
  redeemed in a later one (guardrails.ConfirmationGate), so a question like
  "can I cancel?" can't cancel anything on its own.
* The tool loop is capped (MAX_TOOL_ROUNDS). A model that keeps calling tools
  without answering is escalated rather than allowed to spin and burn tokens.
* "Out of scope" is a judgement call only the model can make from free text,
  so it is handled by the prompt + the escalate_to_human tool. Everything that
  *can* be checked mechanically is checked in Python.
"""

from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from db import DB_PATH, get_db
from guardrails import REFUND_LIMIT_PAISE, ConfirmationGate, ErrorTracker
from schemas import EscalateArgs
from tools import dispatch_tool, escalate_to_human, openai_tool_specs

MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE: float = 0.2          # support answers should be consistent, not creative
MAX_TOOL_ROUNDS: int = 6          # model calls per customer message before we give up
MAX_HISTORY_MESSAGES: int = 40    # keep context (and cost) bounded on long chats

SYSTEM_PROMPT_TEMPLATE = """You are the customer support assistant for an Indian online store.
You are talking to {customer_name}, who is already signed in. You can only see and act on
their orders; the tools enforce this, so never ask for or accept another customer's details.

How to work:
- Use tools for every fact about orders, returns and policy. Never guess order status,
  dates, amounts or rules. For any policy question, call search_policy first and answer
  from the passages it returns.
- Quote money exactly as the tools give it in the "display" field (e.g. "Rs. 1,299.00").
  Do not calculate amounts yourself.
- cancel_order and request_return work in two steps. The first call only returns a preview
  and a confirmation_token; nothing has happened yet. Show the preview (items, refund amount,
  whether it is approved or sent for review) and ask the customer to confirm. Only when they
  clearly say yes in their next message, call the same tool again with the same arguments
  plus the token. If they say no or change anything, do not use the token.
- A question like "Can I cancel order 5?" is not a yes. Preview it, explain, and ask.
- request_return decides on its own whether a refund is approved or sent to a human. Report
  its decision as-is. Refunds above Rs. {limit_rupees:,} are never approved by you; if the
  tool escalated, say a team member will review it within 1-2 business days.
- If a tool returns ok=false, explain the reason plainly. Do not retry the same failing call
  with the same arguments. If the result says the case was escalated, stop and tell the
  customer a human will follow up.
- Call escalate_to_human when the request is outside what these tools cover (e.g. account
  changes, payment disputes, complaints about staff, legal or safety issues), when the
  customer disputes a policy decision, or when they ask for a human.
- Never ask for card numbers, bank details, OTPs or passwords in chat. For cash-on-delivery
  refunds, tell them the team will share a secure form.
- Be concise, warm and plain-spoken. Use the customer's first name occasionally.
"""


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
@dataclass
class Session:
    session_id: str
    customer_id: int
    messages: list[dict[str, Any]] = field(default_factory=list)
    tracker: ErrorTracker = field(default_factory=ErrorTracker)
    gate: ConfirmationGate = field(default_factory=ConfirmationGate)
    user_turn: int = 0  # incremented per customer message; used by the confirmation gate
    escalation_ids: list[int] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


class SessionError(Exception):
    """Raised when a session id doesn't belong to the requesting customer."""


class SessionStore:
    """
    In-memory session store. Fine for a single process; swap for Redis or a
    DB table before running multiple workers.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str | None, customer_id: int) -> Session:
        with self._lock:
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                # A leaked or guessed session id must not hand one customer
                # another customer's chat history (which contains their orders).
                if session.customer_id != customer_id:
                    raise SessionError("Session does not belong to this customer.")
                return session
            # Unknown ids get a fresh server-generated id, so clients can't
            # choose ids that collide with someone else's.
            new_id = uuid.uuid4().hex
            session = Session(session_id=new_id, customer_id=customer_id)
            self._sessions[new_id] = session
            return session


# --------------------------------------------------------------------------- #
# OpenAI client protocol (lets tests pass a fake client)
# --------------------------------------------------------------------------- #
class ChatClient(Protocol):
    @property
    def chat(self) -> Any: ...


def default_client() -> ChatClient:
    from openai import OpenAI  # reads OPENAI_API_KEY from the environment
    return OpenAI()


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
@dataclass
class AgentReply:
    reply: str
    tools_called: list[str]
    escalation_id: int | None


class SupportAgent:
    def __init__(
        self,
        client: ChatClient | None = None,
        *,
        model: str = MODEL,
        db_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self._client = client
        self.model = model
        self.db_path = db_path or DB_PATH
        self.tool_specs = openai_tool_specs()

    @property
    def client(self) -> ChatClient:
        # Created lazily so importing this module never needs an API key.
        if self._client is None:
            self._client = default_client()
        return self._client

    # ------------------------------------------------------------------ #
    def _system_prompt(self, customer_id: int) -> str:
        with get_db(self.db_path) as conn:
            row = conn.execute("SELECT name FROM customers WHERE id = ?", (customer_id,)).fetchone()
        name = row["name"] if row else "the customer"
        return SYSTEM_PROMPT_TEMPLATE.format(
            customer_name=name, limit_rupees=REFUND_LIMIT_PAISE // 100
        )

    @staticmethod
    def _trim(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Keep the most recent messages, starting at a user turn.

        Cutting in the middle of a tool exchange would leave a 'tool' message
        without the assistant message that requested it, which the API rejects.
        """
        if len(messages) <= MAX_HISTORY_MESSAGES:
            return messages
        tail = messages[-MAX_HISTORY_MESSAGES:]
        for i, m in enumerate(tail):
            if m["role"] == "user":
                return tail[i:]
        return tail[-1:]

    @staticmethod
    def _assistant_to_dict(msg: Any) -> dict[str, Any]:
        """Convert an SDK message object into the plain dict we store in history."""
        out: dict[str, Any] = {"role": "assistant", "content": msg.content}
        if getattr(msg, "tool_calls", None):
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        return out

    # ------------------------------------------------------------------ #
    def respond(self, session: Session, user_message: str) -> AgentReply:
        # One request at a time per session: concurrent messages would
        # interleave tool calls in the history.
        with session.lock:
            return self._respond_locked(session, user_message)

    def _respond_locked(self, session: Session, user_message: str) -> AgentReply:
        session.user_turn += 1
        session.messages.append({"role": "user", "content": user_message})
        session.messages = self._trim(session.messages)

        system = {"role": "system", "content": self._system_prompt(session.customer_id)}
        tools_called: list[str] = []
        new_escalation: int | None = None

        for _ in range(MAX_TOOL_ROUNDS):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[system, *session.messages],
                tools=self.tool_specs,
                tool_choice="auto",
                temperature=TEMPERATURE,
            )
            msg = response.choices[0].message
            session.messages.append(self._assistant_to_dict(msg))

            if not getattr(msg, "tool_calls", None):
                return AgentReply(
                    reply=msg.content or "",
                    tools_called=tools_called,
                    escalation_id=new_escalation,
                )

            for tc in msg.tool_calls:
                name = tc.function.name
                tools_called.append(name)
                # customer_id comes from the Session (set from authentication),
                # never from tc.function.arguments.
                result = dispatch_tool(
                    name,
                    tc.function.arguments,
                    customer_id=session.customer_id,
                    tracker=session.tracker,
                    gate=session.gate,
                    user_turn=session.user_turn,
                    db_path=self.db_path,
                )
                if result.escalation_id is not None:
                    new_escalation = result.escalation_id
                    session.escalation_ids.append(result.escalation_id)
                elif name == "escalate_to_human" and result.ok and isinstance(result.data, dict):
                    new_escalation = result.data.get("escalation_id")
                    if new_escalation:
                        session.escalation_ids.append(new_escalation)

                session.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result.model_dump_json(exclude_none=True),
                    }
                )

        # Ran out of rounds: hand over to a human instead of looping forever.
        data, esc_id = escalate_to_human(
            session.customer_id,
            EscalateArgs(
                reason=f"Automatic escalation: agent exceeded {MAX_TOOL_ROUNDS} tool rounds "
                       f"without resolving. Last message: {user_message[:300]}",
            ),
            db_path=self.db_path,
        )
        session.escalation_ids.append(esc_id)
        reply = (
            "I'm sorry, I wasn't able to resolve this myself. I've passed your request to our "
            "support team, and someone will get back to you within 1-2 business days."
        )
        session.messages.append({"role": "assistant", "content": reply})
        return AgentReply(reply=reply, tools_called=tools_called, escalation_id=esc_id)


if __name__ == "__main__":
    # Quick terminal chat: python agent.py --customer 1
    import argparse

    parser = argparse.ArgumentParser(description="Chat with the support agent in the terminal")
    parser.add_argument("--customer", type=int, default=1, help="Customer id to act as (demo)")
    args = parser.parse_args()

    agent = SupportAgent()
    session = SessionStore().get_or_create(None, args.customer)
    print(f"Chatting as customer {args.customer}. Type 'quit' to exit.\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit"}:
            break
        if not text:
            continue
        out = agent.respond(session, text)
        print(f"\nAgent: {out.reply}")
        if out.tools_called:
            print(f"  [tools: {', '.join(out.tools_called)}]")
        if out.escalation_id:
            print(f"  [escalation #{out.escalation_id} created]")
        print()