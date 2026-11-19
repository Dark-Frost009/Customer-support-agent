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
* OpenAI failures (after the SDK's retries) never strand the customer: they
  get a fallback reply, and if a cancel/return/escalation already happened in
  this turn, the fallback says so, built from the tool results by Python.
* "Out of scope" is a judgement call only the model can make from free text,
  so it is handled by the prompt + the escalate_to_human tool. Everything that
  *can* be checked mechanically is checked in Python.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

import openai

from db import DB_PATH, format_paise, get_db
from guardrails import REFUND_LIMIT_PAISE, ConfirmationGate, ErrorTracker
from schemas import EscalateArgs
from tools import dispatch_tool, escalate_to_human, openai_tool_specs

MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE: float = 0.2          # support answers should be consistent, not creative
MAX_TOOL_ROUNDS: int = 6          # model calls per customer message before we give up
MAX_HISTORY_MESSAGES: int = 40    # keep context (and cost) bounded on long chats

# Resilience. The SDK retries connection errors, timeouts, 429s and 5xx with
# exponential back-off; these bound how long a customer can be kept waiting.
OPENAI_TIMEOUT_SECONDS: float = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))
OPENAI_MAX_RETRIES: int = int(os.getenv("OPENAI_MAX_RETRIES", "3"))

WRITE_TOOLS = frozenset({"cancel_order", "request_return", "escalate_to_human"})

log = logging.getLogger("support_agent.agent")

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
- As soon as the customer asks to cancel or return something, or asks whether they can, call
  the tool straight away without a token to get the preview. Don't build your own summary
  with get_order first: that makes the customer confirm twice.
- A question like "Can I cancel order 5?" is not a yes. Preview it, explain, and ask.
- Never describe an item as eligible for return if get_order shows return_window_open is false
  or request_return refuses it. Say plainly that it isn't eligible and why (for example, the
  30-day window from delivery has passed).
- request_return decides on its own whether a refund is approved or sent to a human. Report
  its decision as-is. Refunds above Rs. {limit_rupees:,} are never approved by you; if the
  tool escalated, say a team member will review it within 1-2 business days.
- If a tool returns ok=false, explain the reason plainly. Do not retry the same failing call
  with the same arguments. If the result says the case was escalated, stop and tell the
  customer a human will follow up.
- Call escalate_to_human when the request is outside what these tools cover (e.g. account
  changes, payment disputes, complaints about staff, legal or safety issues), or when the
  customer disputes a policy decision.
- If the customer asks for a human, call escalate_to_human immediately. Don't ask them why
  first; use whatever reason they gave, or "Customer asked to speak to a person".
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
    # Reads OPENAI_API_KEY from the environment. timeout bounds each attempt;
    # max_retries uses the SDK's exponential back-off for transient errors.
    return openai.OpenAI(timeout=OPENAI_TIMEOUT_SECONDS, max_retries=OPENAI_MAX_RETRIES)


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
@dataclass
class AgentReply:
    reply: str
    tools_called: list[str]
    escalation_id: int | None
    degraded: bool = False   # True when the model failed and Python wrote the reply


UNAVAILABLE_REPLY = (
    "I'm sorry, I'm having trouble responding right now. Please try again in a minute. "
    "Nothing has been changed on your account."
)


def _describe_action(name: str, data: dict[str, Any]) -> str | None:
    """Plain-language summary of a completed action, written by Python, not the model."""
    if name == "cancel_order":
        return (f"Order #{data['order_id']} has been cancelled. Your refund of "
                f"{data['refund']['display']} will reach your original payment method "
                "within 5-7 business days.")
    if name == "request_return":
        if data.get("status") == "approved":
            return (f"Your return for {data['product_name']} (return #{data['return_id']}) is "
                    f"approved. The refund of {data['refund']['display']} is issued within "
                    "5-7 business days after the item passes inspection.")
        return (f"Your return request for {data['product_name']} (return #{data['return_id']}) "
                "has been sent to our team for review. They'll respond within 1-2 business days.")
    if name == "escalate_to_human":
        return (f"Your request has been passed to our support team (ticket "
                f"#{data['escalation_id']}). Someone will respond within 1-2 business days.")
    return None


def fallback_reply(actions: list[tuple[str, dict[str, Any]]]) -> str:
    """
    Reply used when the model fails mid-turn.

    If an action already went through, the customer must be told, or they may
    retry and, for example, file a second return. So the summary is built
    from the tool results themselves rather than left to the model.
    """
    lines = [d for name, data in actions if (d := _describe_action(name, data))]
    if not lines:
        return UNAVAILABLE_REPLY
    return ("I'm sorry, I'm having trouble finishing my reply, but here's what has been done:\n"
            + "\n".join(f"- {line}" for line in lines)
            + "\nPlease check back in a minute if you need anything else.")


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

    def _degraded(
        self,
        session: Session,
        completed: list[tuple[str, dict[str, Any]]],
        tools_called: list[str],
        escalation_id: int | None,
    ) -> AgentReply:
        if not tools_called and session.messages and session.messages[-1]["role"] == "user":
            # Nothing happened this turn: drop the unanswered message so a
            # retry doesn't send it twice.
            session.messages.pop()
        else:
            # Tool calls already in history need a closing assistant message,
            # or the next request to the API would be malformed.
            session.messages.append({"role": "assistant", "content": fallback_reply(completed)})
        return AgentReply(
            reply=fallback_reply(completed),
            tools_called=tools_called,
            escalation_id=escalation_id,
            degraded=True,
        )

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
        completed: list[tuple[str, dict[str, Any]]] = []  # write actions done this turn

        for _ in range(MAX_TOOL_ROUNDS):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[system, *session.messages],
                    tools=self.tool_specs,
                    tool_choice="auto",
                    temperature=TEMPERATURE,
                )
            except openai.APIError:
                # Reached only after the SDK's own retries are exhausted
                # (timeout, rate limit, outage, bad key...).
                log.exception("OpenAI call failed for session %s", session.session_id)
                return self._degraded(session, completed, tools_called, new_escalation)
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
                if (result.ok and name in WRITE_TOOLS and isinstance(result.data, dict)
                        and not result.data.get("requires_confirmation")):
                    completed.append((name, result.data))
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