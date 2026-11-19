"""Unit tests for the pure rule logic in guardrails.py."""

import pytest

from guardrails import (
    REFUND_LIMIT_PAISE,
    ConfirmationGate,
    ErrorTracker,
    GuardrailViolation,
    refund_requires_escalation,
)


class TestRefundLimit:
    def test_limit_is_two_thousand_rupees_in_paise(self):
        assert REFUND_LIMIT_PAISE == 200_000

    def test_exactly_two_thousand_is_within_agent_authority(self):
        # Policy: refunds of Rs. 2,000 "or less" may be approved by the agent.
        assert refund_requires_escalation(200_000) is False

    def test_one_paisa_over_escalates(self):
        assert refund_requires_escalation(200_001) is True


class TestErrorTracker:
    def test_second_failure_of_same_tool_triggers_escalation(self):
        t = ErrorTracker()
        assert t.record_error("get_order") is False
        assert t.record_error("get_order") is True

    def test_escalates_only_once_per_session(self):
        t = ErrorTracker()
        t.record_error("get_order")
        t.record_error("get_order")
        assert t.record_error("get_order") is False

    def test_failures_are_counted_per_tool(self):
        t = ErrorTracker()
        assert t.record_error("get_order") is False
        assert t.record_error("cancel_order") is False


class TestConfirmationGate:
    def test_cannot_confirm_in_the_same_turn_as_the_preview(self):
        gate = ConfirmationGate()
        token = gate.issue("cancel_order", "key", turn=1)
        with pytest.raises(GuardrailViolation) as exc:
            gate.redeem(token, "cancel_order", "key", turn=1)
        assert exc.value.code == "CONFIRMATION_TOO_EARLY"

    def test_confirms_on_a_later_turn(self):
        gate = ConfirmationGate()
        token = gate.issue("cancel_order", "key", turn=1)
        gate.redeem(token, "cancel_order", "key", turn=2)  # no exception

    def test_token_is_single_use(self):
        gate = ConfirmationGate()
        token = gate.issue("cancel_order", "key", turn=1)
        gate.redeem(token, "cancel_order", "key", turn=2)
        with pytest.raises(GuardrailViolation) as exc:
            gate.redeem(token, "cancel_order", "key", turn=3)
        assert exc.value.code == "CONFIRMATION_INVALID"

    @pytest.mark.parametrize("tool,key", [("request_return", "key"), ("cancel_order", "other")])
    def test_token_is_bound_to_the_exact_action(self, tool, key):
        gate = ConfirmationGate()
        token = gate.issue("cancel_order", "key", turn=1)
        with pytest.raises(GuardrailViolation) as exc:
            gate.redeem(token, tool, key, turn=2)
        assert exc.value.code == "CONFIRMATION_INVALID"