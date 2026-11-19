"""
tools.py — the functions the LLM can call, plus the dispatcher that calls them.

Contract
--------
* Every tool is a plain function: `tool(customer_id, args, *, db_path=None)`.
  `customer_id` is the FIRST parameter and is supplied by `dispatch_tool()`
  from the authenticated session. It is not in any args model, so the LLM
  cannot set it (and `extra="forbid"` rejects it if the model tries).
* The LLM never writes SQL. Every query below is a fixed string with `?`
  placeholders; the LLM only supplies typed, validated values.
* Tools raise `GuardrailViolation` for rule breaks. `dispatch_tool()` turns
  that into a `ToolResult.failure`, counts it against the error budget, and
  escalates automatically when the budget runs out.
* Escalations that Python decides on (refund over limit, defect claims,
  repeat returners, error budget) are written in the same DB transaction as
  the action that triggered them, so the LLM cannot skip them.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

import rag
from db import format_paise, get_db, to_db_ts, utc_now
from guardrails import (
    CONFIRMATION_REQUIRED_TOOLS,
    RETURN_WINDOW_DAYS,
    ConfirmationGate,
    ErrorTracker,
    GuardrailViolation,
    check_cancellable,
    check_return_eligible,
    days_since,
    is_repeat_returner,
    load_owned_order,
    refund_requires_escalation,
)
from schemas import (
    CancelOrderArgs,
    EscalateArgs,
    GetOrderArgs,
    ListOrdersArgs,
    Money,
    OrderDetail,
    OrderItemOut,
    OrderSummary,
    RequestReturnArgs,
    ReturnOut,
    SearchPolicyArgs,
    ToolResult,
)

DbPath = str | Path | None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _money(paise: int) -> Money:
    return Money(paise=paise, display=format_paise(paise))


def _returned_qty(conn: sqlite3.Connection, order_item_id: int) -> int:
    """Units already returned or pending. Rejected returns don't count."""
    return conn.execute(
        "SELECT COALESCE(SUM(quantity), 0) FROM returns "
        "WHERE order_item_id = ? AND status != 'rejected'",
        (order_item_id,),
    ).fetchone()[0]


def _insert_escalation(
    conn: sqlite3.Connection,
    *,
    customer_id: int,
    reason: str,
    order_id: int | None = None,
    return_id: int | None = None,
    priority: str = "normal",
) -> int:
    cur = conn.execute(
        """INSERT INTO escalations (customer_id, order_id, return_id, reason, priority, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (customer_id, order_id, return_id, reason, priority, to_db_ts(utc_now())),
    )
    return int(cur.lastrowid)


# --------------------------------------------------------------------------- #
# Read tools
# --------------------------------------------------------------------------- #
def get_order(customer_id: int, args: GetOrderArgs, *, db_path: DbPath = None) -> dict[str, Any]:
    now = utc_now()
    with get_db(db_path) as conn:
        order = load_owned_order(conn, args.order_id, customer_id)
        items = conn.execute(
            """SELECT oi.id, oi.quantity, oi.unit_price_paise,
                      p.sku, p.name, p.category, p.is_returnable
               FROM order_items oi JOIN products p ON p.id = oi.product_id
               WHERE oi.order_id = ? ORDER BY oi.id""",
            (order["id"],),
        ).fetchall()
        returns = conn.execute(
            """SELECT id, order_item_id, quantity, status, refund_amount_paise,
                      requires_escalation, requested_at
               FROM returns WHERE order_id = ? ORDER BY id""",
            (order["id"],),
        ).fetchall()
        item_out = [
            OrderItemOut(
                order_item_id=i["id"], sku=i["sku"], product_name=i["name"],
                category=i["category"], quantity=i["quantity"],
                unit_price=_money(i["unit_price_paise"]),
                is_returnable=bool(i["is_returnable"]),
                quantity_already_returned=_returned_qty(conn, i["id"]),
            )
            for i in items
        ]

    age = days_since(order["delivered_at"], now)
    detail = OrderDetail(
        order_id=order["id"], status=order["status"], total=_money(order["total_paise"]),
        subtotal=_money(order["subtotal_paise"]), shipping=_money(order["shipping_paise"]),
        placed_at=order["placed_at"], delivered_at=order["delivered_at"],
        shipped_at=order["shipped_at"], cancelled_at=order["cancelled_at"],
        shipping_address=order["shipping_address"], tracking_number=order["tracking_number"],
        days_since_delivery=age,
        item_names=[i.product_name for i in item_out],
        return_window_open=order["status"] == "delivered" and age is not None
        and age <= RETURN_WINDOW_DAYS,
        items=item_out,
        returns=[
            ReturnOut(
                return_id=r["id"], order_item_id=r["order_item_id"], quantity=r["quantity"],
                status=r["status"], refund=_money(r["refund_amount_paise"]),
                requires_escalation=bool(r["requires_escalation"]),
                requested_at=r["requested_at"],
            )
            for r in returns
        ],
    )
    return detail.model_dump()


def list_orders(customer_id: int, args: ListOrdersArgs, *, db_path: DbPath = None) -> dict[str, Any]:
    # customer_id is always in the WHERE clause; the optional status filter is
    # a validated Literal, bound as a parameter (never string-formatted).
    sql = "SELECT * FROM orders WHERE customer_id = ?"
    params: list[Any] = [customer_id]
    if args.status:
        sql += " AND status = ?"
        params.append(args.status)
    sql += " ORDER BY placed_at DESC LIMIT ?"
    params.append(args.limit)

    with get_db(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
        names: dict[int, list[str]] = {r["id"]: [] for r in rows}
        if names:
            marks = ",".join("?" * len(names))  # placeholders only; ids come from our own query
            for n in conn.execute(
                f"""SELECT oi.order_id, p.name FROM order_items oi
                    JOIN products p ON p.id = oi.product_id
                    WHERE oi.order_id IN ({marks}) ORDER BY oi.id""",
                list(names),
            ):
                names[n["order_id"]].append(n["name"])
    orders = [
        OrderSummary(
            order_id=r["id"], status=r["status"], total=_money(r["total_paise"]),
            placed_at=r["placed_at"], delivered_at=r["delivered_at"],
            item_names=names[r["id"]],
        ).model_dump()
        for r in rows
    ]
    return {"orders": orders, "count": len(orders)}


def search_policy(customer_id: int, args: SearchPolicyArgs, *, db_path: DbPath = None) -> dict[str, Any]:
    # customer_id is unused but kept so every tool has the same signature.
    try:
        passages = rag.search_policy(args.query, args.top_k)
    except rag.PolicyIndexError as exc:
        raise GuardrailViolation("POLICY_INDEX_UNAVAILABLE", str(exc)) from exc
    return {"passages": passages}


# --------------------------------------------------------------------------- #
# Write tools
# --------------------------------------------------------------------------- #
def cancel_order(
    customer_id: int, args: CancelOrderArgs, *, db_path: DbPath = None, dry_run: bool = False
) -> dict[str, Any]:
    now_ts = to_db_ts(utc_now())
    with get_db(db_path) as conn:
        order = load_owned_order(conn, args.order_id, customer_id)
        check_cancellable(order)

        if dry_run:
            # Same checks as the real run, so the customer is never asked to
            # confirm something that would then fail.
            items = [r["name"] for r in conn.execute(
                "SELECT p.name FROM order_items oi JOIN products p ON p.id = oi.product_id "
                "WHERE oi.order_id = ?", (order["id"],))]
            return {
                "action": "cancel_order",
                "order_id": order["id"],
                "items": items,
                "refund": _money(order["total_paise"]).model_dump(),
                "refund_timeline": "5-7 business days to the original payment method",
            }

        # The status condition is repeated in the UPDATE so that if a warehouse
        # job ships the order between our SELECT and here, nothing changes.
        cur = conn.execute(
            """UPDATE orders SET status = 'cancelled', cancelled_at = ?
               WHERE id = ? AND customer_id = ? AND status = 'processing'""",
            (now_ts, order["id"], customer_id),
        )
        if cur.rowcount != 1:
            raise GuardrailViolation("NOT_CANCELLABLE", "The order changed status and can no longer be cancelled.")

        # Put the stock back.
        conn.execute(
            """UPDATE products SET stock_quantity = stock_quantity + (
                   SELECT COALESCE(SUM(oi.quantity), 0) FROM order_items oi
                   WHERE oi.order_id = ? AND oi.product_id = products.id)
               WHERE id IN (SELECT product_id FROM order_items WHERE order_id = ?)""",
            (order["id"], order["id"]),
        )
    return {
        "order_id": order["id"],
        "status": "cancelled",
        "refund": _money(order["total_paise"]).model_dump(),
        "refund_timeline": "5-7 business days to the original payment method",
    }


def request_return(
    customer_id: int, args: RequestReturnArgs, *, db_path: DbPath = None, dry_run: bool = False
) -> tuple[dict[str, Any], int | None]:
    """
    File a return. Returns (data, escalation_id).

    Outcome is decided entirely in Python:
      * approved  — eligible and refund <= Rs. 2,000
      * requested + escalation — refund > Rs. 2,000, a defect claim (needs
        photo review a chat agent can't do), or a repeat returner
    """
    now = utc_now()
    with get_db(db_path) as conn:
        order = load_owned_order(conn, args.order_id, customer_id)

        # The item must belong to *this* order — otherwise a customer could
        # pair their own order_id with an order_item_id from someone else's.
        item = conn.execute(
            """SELECT oi.*, p.name AS product_name, p.is_returnable, p.category
               FROM order_items oi JOIN products p ON p.id = oi.product_id
               WHERE oi.id = ? AND oi.order_id = ?""",
            (args.order_item_id, order["id"]),
        ).fetchone()
        if item is None:
            raise GuardrailViolation(
                "ITEM_NOT_IN_ORDER",
                f"Item {args.order_item_id} is not part of order #{order['id']}. "
                "Use get_order to find the correct order_item_id.",
            )

        check_return_eligible(
            order=order, item=item, is_returnable=bool(item["is_returnable"]),
            quantity=args.quantity, already_returned=_returned_qty(conn, item["id"]),
            is_defective=args.is_defective, now=now,
        )

        # Refund = price actually paid (snapshotted unit price), integer paise.
        refund = item["unit_price_paise"] * args.quantity
        # Policy §4: shipping is refunded only when it's our error, and only once per order.
        if args.is_defective:
            prior = conn.execute(
                "SELECT COUNT(*) FROM returns WHERE order_id = ? AND status != 'rejected'",
                (order["id"],),
            ).fetchone()[0]
            if prior == 0:
                refund += order["shipping_paise"]

        over_limit = refund_requires_escalation(refund)
        escalation_reasons: list[str] = []
        if over_limit:
            escalation_reasons.append(
                f"Refund of {format_paise(refund)} exceeds the agent approval limit of Rs. 2,000."
            )
        if args.is_defective:
            escalation_reasons.append("Defect/damage claim needs photo verification by the team.")
        if is_repeat_returner(conn, customer_id, now):
            escalation_reasons.append("Customer has multiple recent returns (repeat-refund review).")

        needs_human = bool(escalation_reasons)
        status = "requested" if needs_human else "approved"

        if dry_run:
            # Nothing has been written yet; this is everything the customer
            # needs to decide, computed by the same code that will act.
            return {
                "action": "request_return",
                "order_id": order["id"],
                "product_name": item["product_name"],
                "quantity": args.quantity,
                "refund": _money(refund).model_dump(),
                "outcome_if_confirmed": (
                    "Sent to our team for review (not approved by the assistant)"
                    if needs_human else "Approved"
                ),
            }, None

        cur = conn.execute(
            """INSERT INTO returns (order_id, order_item_id, customer_id, quantity, reason,
                                    status, refund_amount_paise, requires_escalation, requested_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (order["id"], item["id"], customer_id, args.quantity, args.reason, status,
             refund, int(over_limit), to_db_ts(now)),
        )
        return_id = int(cur.lastrowid)

        escalation_id: int | None = None
        if needs_human:
            escalation_id = _insert_escalation(
                conn, customer_id=customer_id, order_id=order["id"], return_id=return_id,
                reason=" ".join(escalation_reasons),
                priority="high" if over_limit else "normal",
            )

    data: dict[str, Any] = {
        "return_id": return_id,
        "status": status,
        "product_name": item["product_name"],
        "quantity": args.quantity,
        "refund": _money(refund).model_dump(),
    }
    if needs_human:
        data["next_step"] = (
            "Do NOT promise or approve this refund. Tell the customer a team member will "
            "review the request and respond within 1-2 business days."
        )
        data["escalation_reasons"] = escalation_reasons
    else:
        data["next_step"] = (
            "Return approved. Refund is issued to the original payment method within "
            "5-7 business days after the item passes inspection."
        )
    if item["category"] == "footwear":
        data["condition_note"] = "Shoes must be unworn and in the original box, or the return will be rejected."
    else:
        data["condition_note"] = "Item must be unused, undamaged, with original packaging and tags."
    return data, escalation_id


def escalate_to_human(customer_id: int, args: EscalateArgs, *, db_path: DbPath = None) -> tuple[dict[str, Any], int]:
    with get_db(db_path) as conn:
        # Even escalations respect ownership: the model can't attach another
        # customer's order to a ticket (that would leak it to whoever reads it
        # under this customer's name).
        order_id = None
        if args.order_id is not None:
            order_id = load_owned_order(conn, args.order_id, customer_id)["id"]
        esc_id = _insert_escalation(
            conn, customer_id=customer_id, order_id=order_id,
            reason=args.reason, priority=args.priority,
        )
    return {
        "escalation_id": esc_id,
        "message": "A member of our team will review this and respond within 1-2 business days.",
    }, esc_id


# --------------------------------------------------------------------------- #
# Registry + dispatcher
# --------------------------------------------------------------------------- #
ToolFn = Callable[..., Any]

# name -> (args model, function, description shown to the LLM)
TOOLS: dict[str, tuple[type[BaseModel], ToolFn, str]] = {
    "get_order": (GetOrderArgs, get_order,
                  "Get full details of one of the customer's orders: status, items "
                  "(with order_item_id), tracking, return window, existing returns."),
    "list_orders": (ListOrdersArgs, list_orders,
                    "List the customer's recent orders, optionally filtered by status."),
    "cancel_order": (CancelOrderArgs, cancel_order,
                     "Cancel an order (only while 'processing'). Two steps: first call returns a "
                     "preview and confirmation_token; call again with the token only after the "
                     "customer replies yes."),
    "request_return": (RequestReturnArgs, request_return,
                       "File a return/refund for one item in a delivered order. Two steps: first call "
                       "returns a preview (refund amount, approved vs. sent for review) and a "
                       "confirmation_token; call again with the token only after the customer "
                       "replies yes. Report the system's decision as-is."),
    "search_policy": (SearchPolicyArgs, search_policy,
                      "Search the store policy (shipping, cancellation, returns, refunds, "
                      "exchanges, contact). Use before answering any policy question."),
    "escalate_to_human": (EscalateArgs, escalate_to_human,
                          "Hand the case to a human. Use for out-of-scope requests, disputes, "
                          "suspected fraud, or anything the other tools can't resolve."),
}


def openai_tool_specs() -> list[dict[str, Any]]:
    """JSON schemas for the OpenAI `tools=` parameter, generated from the Pydantic models."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": model.model_json_schema(),
            },
        }
        for name, (model, _fn, desc) in TOOLS.items()
    ]


def dispatch_tool(
    name: str,
    raw_arguments: str | dict[str, Any],
    *,
    customer_id: int,
    tracker: ErrorTracker,
    gate: ConfirmationGate | None = None,
    user_turn: int = 0,
    db_path: DbPath = None,
) -> ToolResult:
    """
    Validate, run, and wrap one tool call.

    `customer_id` comes from the authenticated session (the API layer), never
    from `raw_arguments`. Any failure — unknown tool, bad JSON, validation
    error, guardrail violation — counts toward the error budget; when the
    same tool fails MAX_ERRORS_PER_TOOL times, Python files an escalation
    itself instead of trusting the model to do it.

    `gate` and `user_turn` enforce two-step confirmation for tools in
    CONFIRMATION_REQUIRED_TOOLS. The agent always passes them; they are
    optional only so scripts and tests can call tools directly.
    """
    result = _run(name, raw_arguments, customer_id=customer_id, db_path=db_path,
                  gate=gate, user_turn=user_turn)

    if not result.ok and tracker.record_error(name):
        try:
            _, esc_id = escalate_to_human(
                customer_id,
                EscalateArgs(
                    reason=f"Automatic escalation: tool '{name}' failed "
                           f"{tracker.max_errors} times. Last error: {result.error_code}.",
                    priority="normal",
                ),
                db_path=db_path,
            )
            result.escalation_id = esc_id
            result.error = (
                f"{result.error} This request has now been escalated to a human "
                "(ticket created). Tell the customer a team member will follow up "
                "within 1-2 business days, and stop retrying."
            )
        except Exception:  # escalation failure must not mask the original error
            pass
    return result


def _action_key(args: BaseModel) -> str:
    """
    Canonical form of what an action will do. The free-text `reason` is left
    out so the model rephrasing it doesn't invalidate a confirmed action;
    everything that changes the outcome (order, item, quantity, defect flag)
    is included.
    """
    return json.dumps(args.model_dump(exclude={"confirmation_token", "reason"}), sort_keys=True)


def _run(
    name: str,
    raw_arguments: str | dict[str, Any],
    *,
    customer_id: int,
    db_path: DbPath,
    gate: ConfirmationGate | None,
    user_turn: int,
) -> ToolResult:
    if name not in TOOLS:
        return ToolResult.failure("UNKNOWN_TOOL", f"There is no tool named '{name}'.")
    model, fn, _ = TOOLS[name]

    try:
        payload = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        args = model.model_validate(payload or {})
    except json.JSONDecodeError:
        return ToolResult.failure("BAD_ARGUMENTS", "Tool arguments were not valid JSON.")
    except ValidationError as exc:
        # Pass field-level messages back so the model can correct itself.
        msgs = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'args'}: {e['msg']}" for e in exc.errors())
        return ToolResult.failure("BAD_ARGUMENTS", msgs)

    needs_confirmation = gate is not None and name in CONFIRMATION_REQUIRED_TOOLS
    try:
        if needs_confirmation:
            token = getattr(args, "confirmation_token", None)
            if not token:
                # Step 1: preview. Runs every eligibility check but writes nothing.
                preview = fn(customer_id, args, db_path=db_path, dry_run=True)
                if isinstance(preview, tuple):
                    preview = preview[0]
                return ToolResult.success({
                    "requires_confirmation": True,
                    "confirmation_token": gate.issue(name, _action_key(args), user_turn),
                    "preview": preview,
                    "instruction": "Nothing has been done yet. Show this preview to the customer "
                                   "and ask them to confirm. Only after they reply yes, call this "
                                   "tool again with the same arguments plus confirmation_token.",
                })
            # Step 2: the token must match this exact action and predate this turn.
            gate.redeem(token, name, _action_key(args), user_turn)
        out = fn(customer_id, args, db_path=db_path)
    except GuardrailViolation as exc:
        return ToolResult.failure(exc.code, exc.message)
    except sqlite3.Error:
        # Don't leak DB internals to the model or customer.
        return ToolResult.failure("INTERNAL_ERROR", "A system error occurred. Please try again shortly.")

    # Write tools that may escalate return (data, escalation_id).
    if isinstance(out, tuple):
        data, esc_id = out
        return ToolResult.success(data, escalation_id=esc_id)
    return ToolResult.success(out)