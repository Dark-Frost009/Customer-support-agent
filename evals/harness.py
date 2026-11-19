"""
Eval harness: run scripted conversations against a chat model and score them.

Unit tests (tests/) prove the Python guardrails hold against a fake model.
Evals answer a different question: does the *real* model behave well inside
those guardrails? Does it pick the right tools, wait for confirmation, refuse
prompt injection, quote the policy correctly?

Each case is scored on facts, not vibes:
  * database state after the conversation (orders, returns, escalations)
  * which tools were called
  * what the replies contain, or must never contain (e.g. another customer's data)

The harness takes any OpenAI-compatible client, so tests/test_evals_harness.py
can exercise it offline with a scripted fake.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import seed as seed_module
from agent import AgentReply, SessionStore, SupportAgent
from db import DB_PATH, get_db


# --------------------------------------------------------------------------- #
# Context passed to checks
# --------------------------------------------------------------------------- #
@dataclass
class EvalContext:
    customer_id: int
    replies: list[AgentReply]
    baseline: dict[str, Any]

    @property
    def texts(self) -> list[str]:
        return [r.reply for r in self.replies]

    @property
    def tools(self) -> list[str]:
        return [t for r in self.replies for t in r.tools_called]

    def query(self, sql: str, *params: Any) -> Any:
        with get_db() as conn:
            return conn.execute(sql, params).fetchone()


CheckFn = Callable[[EvalContext], tuple[bool, str]]


@dataclass
class Check:
    name: str
    fn: CheckFn


@dataclass
class EvalCase:
    id: str
    description: str
    turns: list[str]
    checks: list[Check]
    customer_id: int = 1


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0
    error: str | None = None


# --------------------------------------------------------------------------- #
# Check builders
# --------------------------------------------------------------------------- #
def tool_called(name: str) -> Check:
    return Check(f"called {name}", lambda c: (name in c.tools, f"tools used: {c.tools}"))


def order_status(order_id: int, expected: str) -> Check:
    def fn(c: EvalContext) -> tuple[bool, str]:
        actual = c.query("SELECT status FROM orders WHERE id = ?", order_id)[0]
        return actual == expected, f"order #{order_id} is '{actual}'"
    return Check(f"order #{order_id} is {expected}", fn)


def return_status(order_item_id: int, expected: str | None) -> Check:
    """expected=None means: no new return was filed for this item."""
    def fn(c: EvalContext) -> tuple[bool, str]:
        rows = c.query(
            "SELECT COUNT(*), MAX(status) FROM returns WHERE order_item_id = ? AND id > ?",
            order_item_id, c.baseline["max_return_id"],
        )
        count, status = rows[0], rows[1]
        if expected is None:
            return count == 0, f"{count} new return(s), status {status}"
        return count == 1 and status == expected, f"{count} new return(s), status {status}"
    label = "no return filed" if expected is None else f"return is {expected}"
    return Check(f"item {order_item_id}: {label}", fn)


def return_not_approved(order_item_id: int) -> Check:
    def fn(c: EvalContext) -> tuple[bool, str]:
        n = c.query(
            "SELECT COUNT(*) FROM returns WHERE order_item_id = ? AND id > ? AND status = 'approved'",
            order_item_id, c.baseline["max_return_id"],
        )[0]
        return n == 0, f"{n} approved return(s)"
    return Check(f"item {order_item_id}: never approved", fn)


def escalation_created() -> Check:
    def fn(c: EvalContext) -> tuple[bool, str]:
        n = c.query("SELECT COUNT(*) FROM escalations WHERE id > ?", c.baseline["max_escalation_id"])[0]
        return n >= 1, f"{n} new escalation(s)"
    return Check("escalation created", fn)


def reply_matches(pattern: str, turn: int | None = -1) -> Check:
    """Check one reply (default: the last), or any reply when turn=None."""
    rx = re.compile(pattern, re.IGNORECASE)
    def fn(c: EvalContext) -> tuple[bool, str]:
        texts = c.texts if turn is None else [c.texts[turn]]
        return any(rx.search(t) for t in texts), f"reply: {texts[-1][:160]!r}"
    where = "any reply" if turn is None else "reply"
    return Check(f"{where} matches /{pattern}/", fn)


def never_says(pattern: str, why: str) -> Check:
    rx = re.compile(pattern, re.IGNORECASE)
    def fn(c: EvalContext) -> tuple[bool, str]:
        hits = [t for t in c.texts if rx.search(t)]
        return not hits, f"found in: {hits[0][:160]!r}" if hits else "ok"
    return Check(f"never says {why}", fn)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _baseline() -> dict[str, Any]:
    with get_db() as conn:
        return {
            "max_return_id": conn.execute("SELECT COALESCE(MAX(id), 0) FROM returns").fetchone()[0],
            "max_escalation_id": conn.execute("SELECT COALESCE(MAX(id), 0) FROM escalations").fetchone()[0],
        }


def run_case(case: EvalCase, client: Any, model: str) -> CaseResult:
    """Run one case on a freshly seeded database and score it."""
    seed_module.seed(DB_PATH)
    baseline = _baseline()
    agent = SupportAgent(client=client, model=model)
    session = SessionStore().get_or_create(None, case.customer_id)

    started = time.perf_counter()
    replies: list[AgentReply] = []
    transcript: list[dict[str, Any]] = []
    try:
        for turn in case.turns:
            out = agent.respond(session, turn)
            replies.append(out)
            transcript.append({"user": turn, "agent": out.reply, "tools": out.tools_called,
                               "degraded": out.degraded})
    except Exception as exc:  # a crash is a failed case, not a crashed run
        return CaseResult(case.id, False, [f"crashed: {exc!r}"], transcript,
                          time.perf_counter() - started, error=repr(exc))

    ctx = EvalContext(case.customer_id, replies, baseline)
    failures = []
    for check in case.checks:
        try:
            ok, detail = check.fn(ctx)
        except Exception as exc:
            ok, detail = False, f"check errored: {exc!r}"
        if not ok:
            failures.append(f"{check.name} — {detail}")
    if any(r.degraded for r in replies):
        failures.append("model call failed (degraded fallback reply)")

    return CaseResult(case.id, not failures, failures, transcript, time.perf_counter() - started)
