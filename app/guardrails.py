"""
guardrails.py — business rules enforced in Python, never delegated to the prompt.

The system prompt *describes* these rules so the LLM can explain them to the
customer, but the prompt is not what enforces them. A jailbroken or confused
model can say anything; it still can't get past these checks, because every
state-changing tool calls them before touching the database.

Rules implemented here
----------------------
1. Ownership     — orders are loaded with `WHERE id = ? AND customer_id = ?`,
                   where customer_id comes from the authenticated session.
2. Refund limit  — refunds strictly above Rs. 2,000 (200,000 paise) are never
                   approved by the agent; they're filed as 'requested' and
                   escalated in the same transaction.
3. Return rules  — delivered status, 30-day window, returnable flag, quantity
                   not already returned, 7-day window for defect reports.
4. Confirmation  — cancel/return run only after a preview and a later
                   customer turn, with a token tied to the exact action.
5. Error budget  — a tool that fails twice in one session forces escalation.
"""

from __future__ import annotations

import secrets
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from db import from_db_ts, to_db_ts

# --------------------------------------------------------------------------- #
# Limits (single source of truth — seed.py mirrors REFUND_LIMIT_PAISE)
# --------------------------------------------------------------------------- #
REFUND_LIMIT_PAISE: int = 200_000      # Rs. 2,000. Policy: "ABOVE Rs. 2,000" escalates.
RETURN_WINDOW_DAYS: int = 30
DEFECT_REPORT_DAYS: int = 7
MAX_ERRORS_PER_TOOL: int = 2           # "a tool returns an error twice" -> escalate
REPEAT_RETURN_THRESHOLD: int = 3       # this many non-rejected returns ...
REPEAT_RETURN_LOOKBACK_DAYS: int = 90  # ... in this many days = "repeated refund requests"


class GuardrailViolation(Exception):
    """
    A request broke a business rule.

    `code` is machine-readable (the agent loop and tests branch on it);
    `message` is safe to show the LLM and, through it, the customer.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# --------------------------------------------------------------------------- #
# 1. Ownership
# --------------------------------------------------------------------------- #
def load_owned_order(conn: sqlite3.Connection, order_id: int, customer_id: int) -> sqlite3.Row:
    """
    Fetch an order only if it belongs to the authenticated customer.

    Ownership is part of the WHERE clause rather than a check after loading,
    so another customer's row never enters Python memory and can't leak into a
    tool result by mistake.

    "Doesn't exist" and "belongs to someone else" deliberately return the same
    error. Distinct messages would let a user discover which order IDs are
    real by probing.
    """
    row = conn.execute(
        "SELECT * FROM orders WHERE id = ? AND customer_id = ?", (order_id, customer_id)
    ).fetchone()
    if row is None:
        raise GuardrailViolation(
            "ORDER_NOT_FOUND",
            f"No order #{order_id} was found on this account. Ask the customer to "
            "double-check the order number.",
        )
    return row


# --------------------------------------------------------------------------- #
# 2. Refund limit
# --------------------------------------------------------------------------- #
def refund_requires_escalation(refund_paise: int) -> bool:
    """Strictly greater than: exactly Rs. 2,000 is still within the agent's authority."""
    return refund_paise > REFUND_LIMIT_PAISE


# --------------------------------------------------------------------------- #
# 3. Order / return eligibility
# --------------------------------------------------------------------------- #
def check_cancellable(order: sqlite3.Row) -> None:
    if order["status"] == "processing":
        return
    if order["status"] == "cancelled":
        raise GuardrailViolation("ALREADY_CANCELLED", "This order is already cancelled.")
    raise GuardrailViolation(
        "NOT_CANCELLABLE",
        f"Order is '{order['status']}'. Only orders still 'processing' can be cancelled. "
        "Once shipped, the customer can refuse delivery or request a return after delivery.",
    )


def days_since(ts: str | None, now: datetime) -> int | None:
    return None if ts is None else (now - from_db_ts(ts)).days


def check_return_eligible(
    *,
    order: sqlite3.Row,
    item: sqlite3.Row,
    is_returnable: bool,
    quantity: int,
    already_returned: int,
    is_defective: bool,
    now: datetime,
) -> None:
    """Raise GuardrailViolation if this return is not allowed by policy §3."""
    if order["status"] != "delivered" or order["delivered_at"] is None:
        raise GuardrailViolation(
            "NOT_DELIVERED",
            f"Order is '{order['status']}'. Only delivered orders can be returned.",
        )

    age = days_since(order["delivered_at"], now)
    assert age is not None
    if age > RETURN_WINDOW_DAYS:
        raise GuardrailViolation(
            "RETURN_WINDOW_EXPIRED",
            f"Delivered {age} days ago; the return window is {RETURN_WINDOW_DAYS} days. "
            "The item is not eligible. If the customer disputes this, call escalate_to_human.",
        )

    if is_defective and age > DEFECT_REPORT_DAYS:
        raise GuardrailViolation(
            "DEFECT_REPORT_LATE",
            f"Defects must be reported within {DEFECT_REPORT_DAYS} days of delivery "
            f"(this was {age} days). A standard return may still be possible if the item "
            "is returnable and unused.",
        )

    # Non-returnable items are only accepted when they arrived defective.
    if not is_returnable and not is_defective:
        raise GuardrailViolation(
            "NOT_RETURNABLE",
            "This item is marked non-returnable (e.g. innerwear, personal care, gift cards) "
            "and can only be returned if it arrived defective or damaged.",
        )

    remaining = item["quantity"] - already_returned
    if quantity > remaining:
        raise GuardrailViolation(
            "QUANTITY_EXCEEDED",
            f"Only {remaining} unit(s) of this item can still be returned "
            f"({item['quantity']} bought, {already_returned} already returned or pending).",
        )


def is_repeat_returner(conn: sqlite3.Connection, customer_id: int, now: datetime) -> bool:
    """Policy §4 lists 'repeated refund requests' as an escalation trigger."""
    cutoff = to_db_ts(now - timedelta(days=REPEAT_RETURN_LOOKBACK_DAYS))
    n = conn.execute(
        """SELECT COUNT(*) FROM returns
           WHERE customer_id = ? AND status != 'rejected' AND requested_at >= ?""",
        (customer_id, cutoff),
    ).fetchone()[0]
    return n >= REPEAT_RETURN_THRESHOLD


# --------------------------------------------------------------------------- #
# 4. Customer confirmation for state-changing actions
# --------------------------------------------------------------------------- #
# Tools that change orders or money. They run in two steps: a preview that
# issues a token, then the real action, which needs that token.
CONFIRMATION_REQUIRED_TOOLS: frozenset[str] = frozenset({"cancel_order", "request_return"})


@dataclass
class PendingAction:
    tool: str
    args_key: str    # canonical JSON of the previewed arguments
    issued_turn: int


@dataclass
class ConfirmationGate:
    """
    Makes "ask the customer before acting" a rule Python enforces, not a
    prompt instruction the model can skip (in testing, "Can I cancel order 5?"
    was enough for the model to cancel it immediately).

    Guarantees:
      * The action can't run in the same turn it was previewed. At least one
        new customer message must arrive between the preview and the action.
      * The action that runs is exactly the one previewed. Changing the
        order, item, quantity or defect flag invalidates the token.
      * Tokens are single-use.

    What it can't guarantee: that the customer's reply actually meant "yes".
    Reading that from free text is the model's job. Python ensures there was a
    reply to read, and that nothing different from what was shown gets done.
    """

    pending: dict[str, PendingAction] = field(default_factory=dict)

    def issue(self, tool: str, args_key: str, turn: int) -> str:
        token = secrets.token_hex(4)
        self.pending[token] = PendingAction(tool, args_key, turn)
        return token

    def redeem(self, token: str, tool: str, args_key: str, turn: int) -> None:
        action = self.pending.get(token)
        if action is None or action.tool != tool or action.args_key != args_key:
            raise GuardrailViolation(
                "CONFIRMATION_INVALID",
                "That confirmation token doesn't match this action. Call the tool again "
                "without a token to get a fresh preview, and confirm it with the customer.",
            )
        if turn <= action.issued_turn:
            raise GuardrailViolation(
                "CONFIRMATION_TOO_EARLY",
                "You must show the preview to the customer and wait for their reply before "
                "confirming. Ask them now; do not call this tool again in this turn.",
            )
        del self.pending[token]


# --------------------------------------------------------------------------- #
# 5. Error budget
# --------------------------------------------------------------------------- #
@dataclass
class ErrorTracker:
    """
    Counts tool failures per session.

    Interpretation of "a tool returns an error twice": the *same* tool failing
    twice in one conversation. That catches a model stuck retrying (e.g. the
    customer keeps giving an order ID that isn't theirs) without escalating
    just because two unrelated lookups each hit one typo.

    Lives in Python session state, not in the prompt, so the model can't
    "forget" how many times it has failed.
    """

    max_errors: int = MAX_ERRORS_PER_TOOL
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    escalated: bool = False

    def record_error(self, tool_name: str) -> bool:
        """Record a failure; return True exactly once, when the budget is exhausted."""
        self.counts[tool_name] += 1
        if self.counts[tool_name] >= self.max_errors and not self.escalated:
            self.escalated = True
            return True
        return False