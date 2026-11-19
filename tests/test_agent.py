"""
Agent-loop tests with a scripted fake model.

The fake "model" deliberately misbehaves (acts before the customer confirms,
loops forever) to show that Python stops it, whatever the model decides.
"""

import pytest

from agent import MAX_TOOL_ROUNDS, SessionError, SessionStore, SupportAgent
from conftest import (
    ITEM_SHOES,
    ORDER_PROCESSING,
    ORDER_SHOES,
    PRIYA,
    RAHUL,
    FakeChatClient,
    latest_token,
    tool_results,
)
from db import get_db


def _order_status(order_id: int) -> str:
    with get_db() as conn:
        return conn.execute("SELECT status FROM orders WHERE id = ?", (order_id,)).fetchone()[0]


@pytest.fixture
def setup():
    fake = FakeChatClient()
    agent = SupportAgent(client=fake)
    session = SessionStore().get_or_create(None, PRIYA)
    return fake, agent, session


class TestConfirmation:
    def test_question_alone_cannot_cancel_even_if_model_tries(self, setup):
        fake, agent, session = setup
        # The model previews, then immediately tries to confirm in the same turn.
        fake.script = [
            [("cancel_order", {"order_id": ORDER_PROCESSING})],
            lambda m: [("cancel_order", {"order_id": ORDER_PROCESSING,
                                         "confirmation_token": latest_token(m)})],
            "Would you like me to cancel it?",
        ]
        agent.respond(session, "Can I cancel order 5?")
        assert tool_results(fake.last_messages)[-1]["error_code"] == "CONFIRMATION_TOO_EARLY"
        assert _order_status(ORDER_PROCESSING) == "processing"

    def test_cancels_after_customer_confirms_next_turn(self, setup):
        fake, agent, session = setup
        fake.script = [[("cancel_order", {"order_id": ORDER_PROCESSING})], "Confirm?"]
        agent.respond(session, "Can I cancel order 5?")
        assert _order_status(ORDER_PROCESSING) == "processing"

        fake.script = [
            lambda m: [("cancel_order", {"order_id": ORDER_PROCESSING,
                                         "confirmation_token": latest_token(m)})],
            "Cancelled.",
        ]
        agent.respond(session, "Yes, cancel it")
        assert _order_status(ORDER_PROCESSING) == "cancelled"

    def test_preview_shows_escalation_outcome_before_confirming(self, setup):
        fake, agent, session = setup
        fake.script = [[("request_return", {"order_id": ORDER_SHOES, "order_item_id": ITEM_SHOES,
                                            "reason": "doesn't fit"})], "Confirm?"]
        agent.respond(session, "Refund my shoes")
        preview = tool_results(fake.last_messages)[-1]["data"]["preview"]
        assert preview["refund"]["display"] == "Rs. 4,499.00"
        assert "review" in preview["outcome_if_confirmed"].lower()


class TestLoopAndSessions:
    def test_runaway_tool_loop_is_escalated(self, setup):
        fake, agent, session = setup
        fake.script = [[("list_orders", {})]] * MAX_TOOL_ROUNDS
        out = agent.respond(session, "loop forever")
        assert out.escalation_id is not None
        assert len(out.tools_called) == MAX_TOOL_ROUNDS

    def test_system_prompt_uses_authenticated_customers_name(self, setup):
        fake, agent, session = setup
        fake.script = ["Hello!"]
        agent.respond(session, "hi")
        assert "Priya Sharma" in fake.calls[0]["messages"][0]["content"]

    def test_session_cannot_be_reused_by_another_customer(self):
        store = SessionStore()
        session = store.get_or_create(None, PRIYA)
        with pytest.raises(SessionError):
            store.get_or_create(session.session_id, RAHUL)

    def test_client_cannot_choose_its_own_session_id(self):
        session = SessionStore().get_or_create("my-chosen-id", PRIYA)
        assert session.session_id != "my-chosen-id"