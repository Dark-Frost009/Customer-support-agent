"""
Offline tests for the eval harness itself (evals/harness.py).

The real evals need an API key; these make sure the scoring logic is right,
so a "PASS" from the real run can be trusted. A scripted model that behaves
well must pass; one that leaks data or acts without confirmation must fail.
"""

from cases import CASES, RAHUL_DATA
from conftest import FakeChatClient, latest_token
from harness import EvalCase, never_says, order_status, run_case

BY_ID = {c.id: c for c in CASES}


def test_case_ids_are_unique_and_every_case_has_checks():
    assert len(BY_ID) == len(CASES)
    assert all(c.checks and c.turns for c in CASES)


def test_well_behaved_model_passes_cancel_case():
    fake = FakeChatClient([
        [("cancel_order", {"order_id": 5})],
        "That cancels order 5 with a refund of Rs. 2,097.00. Shall I go ahead?",
        lambda m: [("cancel_order", {"order_id": 5, "confirmation_token": latest_token(m)})],
        "Done, order 5 is cancelled.",
    ])
    result = run_case(BY_ID["cancel_after_yes"], fake, model="fake")
    assert result.passed, result.failures


def test_model_that_never_confirms_fails_cancel_case():
    fake = FakeChatClient([
        [("cancel_order", {"order_id": 5})], "Shall I cancel it?",
        "Okay!",  # says okay but never uses the token
    ])
    result = run_case(BY_ID["cancel_after_yes"], fake, model="fake")
    assert not result.passed
    assert any("order #5 is cancelled" in f for f in result.failures)


def test_leaking_reply_fails_security_case():
    fake = FakeChatClient(["Order 8 is going to Rahul Verma in Noida."])
    result = run_case(BY_ID["other_customers_order"], fake, model="fake")
    assert not result.passed


def test_large_refund_case_passes_when_system_escalates():
    args = {"order_id": 2, "order_item_id": 3, "reason": "don't fit"}
    fake = FakeChatClient([
        [("request_return", args)], "That's Rs. 4,499.00 and goes to our team for review. Proceed?",
        lambda m: [("request_return", {**args, "confirmation_token": latest_token(m)})],
        "Submitted. Our team will review it within 1-2 business days.",
    ])
    result = run_case(BY_ID["large_refund_escalated"], fake, model="fake")
    assert result.passed, result.failures


def test_each_case_runs_on_a_fresh_database():
    # A case that cancels order 5 must not affect the next case.
    cancel = FakeChatClient([
        [("cancel_order", {"order_id": 5})], "Confirm?",
        lambda m: [("cancel_order", {"order_id": 5, "confirmation_token": latest_token(m)})], "Done.",
    ])
    run_case(BY_ID["cancel_after_yes"], cancel, model="fake")
    check_only = EvalCase("fresh", "", ["hi"], [order_status(5, "processing")])
    assert run_case(check_only, FakeChatClient(["Hello!"]), model="fake").passed


def test_crash_is_reported_as_failure_not_raised():
    def boom(**_):
        raise RuntimeError("bad")
    from types import SimpleNamespace as NS
    case = EvalCase("crash", "", ["hi"], [never_says(RAHUL_DATA, "x")])
    result = run_case(case, NS(chat=NS(completions=NS(create=boom))), model="fake")
    assert not result.passed and "crashed" in result.failures[0]
