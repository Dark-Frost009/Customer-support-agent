"""
Tool-level tests: ownership, refund limit, return/cancel rules, error budget.

These call tools through `dispatch_tool`, the same entry point the agent uses,
so validation and error handling are covered too.
"""

import json

from conftest import (
    ITEM_BRIEFS,
    ITEM_OLD,
    ITEM_RAHUL,
    ITEM_SHOES,
    ITEM_TSHIRT,
    ORDER_NONRETURN,
    ORDER_OLD,
    ORDER_PROCESSING,
    ORDER_RAHUL,
    ORDER_SHIPPED,
    ORDER_SHOES,
    ORDER_SMALL,
    PRIYA,
    RAHUL,
)
from db import get_db
from guardrails import ErrorTracker
from tools import openai_tool_specs


def _row(sql, *params):
    with get_db() as conn:
        return conn.execute(sql, params).fetchone()


# --------------------------------------------------------------------------- #
# Ownership
# --------------------------------------------------------------------------- #
class TestOwnership:
    def test_customer_can_read_own_order(self, call):
        r = call("get_order", {"order_id": ORDER_SMALL})
        assert r.ok and r.data["order_id"] == ORDER_SMALL

    def test_cannot_read_another_customers_order(self, call):
        r = call("get_order", {"order_id": ORDER_RAHUL}, customer_id=PRIYA)
        assert not r.ok and r.error_code == "ORDER_NOT_FOUND"

    def test_not_yours_and_nonexistent_look_identical(self, call):
        # Different messages would let a user probe which order ids exist.
        theirs = call("get_order", {"order_id": ORDER_RAHUL})
        missing = call("get_order", {"order_id": 9999})
        assert theirs.error_code == missing.error_code
        assert theirs.error.replace(str(ORDER_RAHUL), "N") == missing.error.replace("9999", "N")

    def test_model_cannot_pass_customer_id(self, call):
        r = call("get_order", {"order_id": ORDER_RAHUL, "customer_id": RAHUL})
        assert not r.ok and r.error_code == "BAD_ARGUMENTS"

    def test_customer_id_is_not_in_any_tool_schema(self):
        assert "customer_id" not in json.dumps(openai_tool_specs())

    def test_list_orders_returns_only_own_orders(self, call):
        orders = call("list_orders", {"limit": 20}).data["orders"]
        ids = {o["order_id"] for o in orders}
        with get_db() as conn:
            own = {r[0] for r in conn.execute("SELECT id FROM orders WHERE customer_id = ?", (PRIYA,))}
        assert ids == own

    def test_cannot_return_another_customers_item_via_own_order(self, call):
        r = call("request_return", {"order_id": ORDER_SMALL, "order_item_id": ITEM_RAHUL,
                                    "reason": "mine now"})
        assert not r.ok and r.error_code == "ITEM_NOT_IN_ORDER"

    def test_cannot_attach_another_customers_order_to_escalation(self, call):
        r = call("escalate_to_human", {"reason": "please help", "order_id": ORDER_RAHUL})
        assert not r.ok and r.error_code == "ORDER_NOT_FOUND"

    def test_sql_injection_style_input_is_rejected_by_validation(self, call):
        r = call("get_order", {"order_id": "1 OR 1=1"})
        assert not r.ok and r.error_code == "BAD_ARGUMENTS"


# --------------------------------------------------------------------------- #
# Refunds and returns
# --------------------------------------------------------------------------- #
class TestReturns:
    def test_refund_within_limit_is_approved(self, call):
        r = call("request_return", {"order_id": ORDER_SMALL, "order_item_id": ITEM_TSHIRT,
                                    "reason": "too big"})
        assert r.ok and r.data["status"] == "approved"
        assert r.data["refund"] == {"paise": 59_900, "display": "Rs. 599.00"}
        assert r.escalation_id is None

    def test_refund_above_limit_is_escalated_not_approved(self, call):
        r = call("request_return", {"order_id": ORDER_SHOES, "order_item_id": ITEM_SHOES,
                                    "reason": "doesn't fit"})
        assert r.ok and r.data["status"] == "requested"
        assert r.escalation_id is not None
        ret = _row("SELECT status, refund_amount_paise, requires_escalation FROM returns "
                   "WHERE order_item_id = ?", ITEM_SHOES)
        assert tuple(ret) == ("requested", 449_900, 1)
        esc = _row("SELECT priority, return_id FROM escalations WHERE id = ?", r.escalation_id)
        assert esc["priority"] == "high" and esc["return_id"] is not None

    def test_refund_amount_is_integer_paise(self, call):
        r = call("request_return", {"order_id": ORDER_SMALL, "order_item_id": ITEM_TSHIRT,
                                    "reason": "too big"})
        assert isinstance(r.data["refund"]["paise"], int)

    def test_return_after_30_days_is_rejected(self, call):
        r = call("request_return", {"order_id": ORDER_OLD, "order_item_id": ITEM_OLD,
                                    "reason": "changed mind"})
        assert not r.ok and r.error_code == "RETURN_WINDOW_EXPIRED"

    def test_undelivered_order_cannot_be_returned(self, call):
        item = call("get_order", {"order_id": ORDER_SHIPPED}).data["items"][0]["order_item_id"]
        r = call("request_return", {"order_id": ORDER_SHIPPED, "order_item_id": item,
                                    "reason": "changed mind"})
        assert not r.ok and r.error_code == "NOT_DELIVERED"

    def test_non_returnable_item_is_rejected(self, call):
        r = call("request_return", {"order_id": ORDER_NONRETURN, "order_item_id": ITEM_BRIEFS,
                                    "reason": "don't like"})
        assert not r.ok and r.error_code == "NOT_RETURNABLE"

    def test_defective_non_returnable_item_goes_to_human_review(self, call):
        r = call("request_return", {"order_id": ORDER_NONRETURN, "order_item_id": ITEM_BRIEFS,
                                    "reason": "arrived torn", "is_defective": True})
        assert r.ok and r.data["status"] == "requested" and r.escalation_id is not None

    def test_defect_reported_after_7_days_is_rejected(self, call):
        r = call("request_return", {"order_id": ORDER_SHOES, "order_item_id": ITEM_SHOES,
                                    "reason": "sole came off", "is_defective": True})
        assert not r.ok and r.error_code == "DEFECT_REPORT_LATE"

    def test_cannot_return_more_than_was_bought(self, call):
        args = {"order_id": ORDER_SMALL, "order_item_id": ITEM_TSHIRT, "reason": "too big"}
        assert call("request_return", {**args, "quantity": 2}).ok
        r = call("request_return", {**args, "quantity": 1})
        assert not r.ok and r.error_code == "QUANTITY_EXCEEDED"


# --------------------------------------------------------------------------- #
# Cancellation
# --------------------------------------------------------------------------- #
class TestCancellation:
    def test_processing_order_can_be_cancelled_and_stock_restored(self, call):
        before = _row("SELECT stock_quantity FROM products WHERE sku = 'EL-PWR-002'")[0]
        r = call("cancel_order", {"order_id": ORDER_PROCESSING})
        assert r.ok and r.data["status"] == "cancelled"
        assert _row("SELECT status FROM orders WHERE id = ?", ORDER_PROCESSING)[0] == "cancelled"
        assert _row("SELECT stock_quantity FROM products WHERE sku = 'EL-PWR-002'")[0] == before + 1

    def test_shipped_order_cannot_be_cancelled(self, call):
        r = call("cancel_order", {"order_id": ORDER_SHIPPED})
        assert not r.ok and r.error_code == "NOT_CANCELLABLE"

    def test_cannot_cancel_twice(self, call):
        call("cancel_order", {"order_id": ORDER_PROCESSING})
        r = call("cancel_order", {"order_id": ORDER_PROCESSING})
        assert not r.ok and r.error_code == "ALREADY_CANCELLED"


# --------------------------------------------------------------------------- #
# Error budget and read tools
# --------------------------------------------------------------------------- #
class TestErrorBudget:
    def test_same_tool_failing_twice_creates_one_escalation(self, call):
        tracker = ErrorTracker()
        first = call("get_order", {"order_id": 9999}, tracker=tracker)
        second = call("get_order", {"order_id": 9998}, tracker=tracker)
        third = call("get_order", {"order_id": 9997}, tracker=tracker)
        assert first.escalation_id is None
        assert second.escalation_id is not None
        assert third.escalation_id is None  # no duplicate tickets
        assert _row("SELECT COUNT(*) FROM escalations WHERE customer_id = ?", PRIYA)[0] == 1


class TestReadTools:
    def test_list_orders_includes_product_names(self, call):
        orders = {o["order_id"]: o for o in call("list_orders", {}).data["orders"]}
        assert orders[ORDER_SHOES]["item_names"] == ["Stride Runner Running Shoes"]

    def test_search_policy_finds_refund_limit(self, call, policy_index):
        r = call("search_policy", {"query": "refund above 2000 rupees escalation"})
        assert r.ok
        assert r.data["passages"][0]["subsection"] == "Refund approval limits"

    def test_search_policy_caps_top_k(self, call, policy_index):
        r = call("search_policy", {"query": "returns", "top_k": 50})
        assert not r.ok and r.error_code == "BAD_ARGUMENTS"