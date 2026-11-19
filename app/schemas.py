"""
schemas.py — Pydantic v2 models for tool arguments, tool results and the chat API.

Why tool-argument models matter for safety
------------------------------------------
* Every model the LLM fills in uses `extra="forbid"`. If the model tries to
  sneak in `"customer_id": 2` to read someone else's order, validation fails
  instead of silently ignoring it — and that failure is counted as a tool
  error. The real customer_id is injected by Python (see tools.dispatch_tool).
* None of these models has a `customer_id` field at all, so it never appears
  in the JSON schema we send to OpenAI. The model has no reason to think it
  controls whose data it sees.
* Bounds (gt=0, max_length, top_k <= 5) are enforced here, before any SQL runs.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

OrderStatus = Literal["processing", "shipped", "delivered", "cancelled"]
Priority = Literal["low", "normal", "high", "urgent"]


class _LLMArgs(BaseModel):
    """Base for anything the LLM fills in: unknown keys are an error, not ignored."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# --------------------------------------------------------------------------- #
# Tool arguments (what the LLM is allowed to send)
# --------------------------------------------------------------------------- #
class GetOrderArgs(_LLMArgs):
    order_id: int = Field(gt=0, description="The order number the customer is asking about.")


class ListOrdersArgs(_LLMArgs):
    status: OrderStatus | None = Field(
        default=None, description="Only return orders with this status. Omit for all."
    )
    limit: int = Field(default=10, ge=1, le=20, description="Maximum orders to return.")


_TOKEN_DESC = (
    "Leave empty on the first call: you'll get a preview and a token. Show the preview "
    "to the customer, and only after they reply yes, call again with the same arguments "
    "plus this token."
)


class CancelOrderArgs(_LLMArgs):
    order_id: int = Field(gt=0, description="The order to cancel.")
    reason: str = Field(
        default="Customer requested cancellation", max_length=300,
        description="Short reason in the customer's words.",
    )
    confirmation_token: str | None = Field(default=None, max_length=32, description=_TOKEN_DESC)


class RequestReturnArgs(_LLMArgs):
    order_id: int = Field(gt=0, description="The order containing the item.")
    order_item_id: int = Field(
        gt=0, description="The order_item id from get_order (not the product id)."
    )
    quantity: int = Field(default=1, ge=1, le=20, description="How many units to return.")
    reason: str = Field(min_length=3, max_length=500, description="Why the customer is returning it.")
    is_defective: bool = Field(
        default=False,
        description="True only if the customer says the item arrived defective, damaged, or wrong.",
    )
    confirmation_token: str | None = Field(default=None, max_length=32, description=_TOKEN_DESC)


class SearchPolicyArgs(_LLMArgs):
    query: str = Field(min_length=1, max_length=300, description="Natural-language policy question.")
    top_k: int = Field(default=3, ge=1, le=5, description="Number of passages to return.")


class EscalateArgs(_LLMArgs):
    reason: str = Field(min_length=3, max_length=1000, description="Why a human is needed.")
    order_id: int | None = Field(default=None, gt=0, description="Related order, if any.")
    priority: Priority = Field(default="normal")


# --------------------------------------------------------------------------- #
# Tool outputs (what Python sends back to the LLM)
# --------------------------------------------------------------------------- #
# Money is always given as integer paise *and* a pre-formatted string, so the
# LLM quotes the string instead of doing its own (float) arithmetic.
class Money(BaseModel):
    paise: int
    display: str


class OrderItemOut(BaseModel):
    order_item_id: int
    sku: str
    product_name: str
    category: str
    quantity: int
    unit_price: Money
    is_returnable: bool
    quantity_already_returned: int


class ReturnOut(BaseModel):
    return_id: int
    order_item_id: int
    quantity: int
    status: str
    refund: Money
    requires_escalation: bool
    requested_at: str


class OrderSummary(BaseModel):
    order_id: int
    status: OrderStatus
    total: Money
    placed_at: str
    delivered_at: str | None
    # Product names, so "my running shoes" can be matched to an order
    # without a get_order call per order.
    item_names: list[str] = Field(default_factory=list)


class OrderDetail(OrderSummary):
    subtotal: Money
    shipping: Money
    shipping_address: str
    tracking_number: str | None
    shipped_at: str | None
    cancelled_at: str | None
    days_since_delivery: int | None
    return_window_open: bool
    items: list[OrderItemOut]
    returns: list[ReturnOut]


class ToolResult(BaseModel):
    """Uniform envelope for every tool call, success or failure."""

    ok: bool
    data: Any = None
    error_code: str | None = None
    error: str | None = None
    # Set when Python (not the LLM) decided a human must take over.
    escalation_id: int | None = None

    @classmethod
    def success(cls, data: Any, escalation_id: int | None = None) -> "ToolResult":
        return cls(ok=True, data=data, escalation_id=escalation_id)

    @classmethod
    def failure(cls, code: str, message: str, escalation_id: int | None = None) -> "ToolResult":
        return cls(ok=False, error_code=code, error=message, escalation_id=escalation_id)


# --------------------------------------------------------------------------- #
# Chat API
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=64)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    escalated: bool = False
    escalation_id: int | None = None
    tools_called: list[str] = Field(default_factory=list)
    # True when the model was unavailable and the reply is a safe fallback.
    degraded: bool = False