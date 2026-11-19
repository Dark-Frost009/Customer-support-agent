"""
Shared fixtures.

Every test gets a freshly seeded database in a temp folder, and OpenAI is
never called: embeddings use a deterministic bag-of-words stand-in, and the
chat model is a scripted fake. The suite runs offline, free, in seconds.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Callable

import pytest

# Point the app at throwaway files BEFORE any app module is imported, because
# db.py / rag.py read these environment variables at import time.
_TMP = Path(tempfile.mkdtemp(prefix="support_agent_tests_"))
os.environ["SUPPORT_DB_PATH"] = str(_TMP / "test_support.db")
os.environ["POLICY_INDEX_PATH"] = str(_TMP / "test_policy_index.db")
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")

import rag  # noqa: E402
import seed as seed_module  # noqa: E402
from db import DB_PATH  # noqa: E402
from guardrails import ErrorTracker  # noqa: E402
from schemas import ToolResult  # noqa: E402
from tools import dispatch_tool  # noqa: E402

PRIYA, RAHUL = 1, 2

# Order / item ids produced by seed.py (deterministic: seeded RNG, fixed order).
ORDER_SMALL = 1        # delivered 5d ago: t-shirt x2 (item 1), wallet (item 2)
ORDER_SHOES = 2        # delivered 10d ago: Rs. 4,499 shoes (item 3)
ORDER_OLD = 3          # delivered 45d ago (item 4)
ORDER_SHIPPED = 4      # shipped
ORDER_PROCESSING = 5   # processing: power bank + cables
ORDER_NONRETURN = 6    # delivered 3d ago: briefs (item 8, non-returnable), serum (item 9)
ORDER_CANCELLED = 7
ORDER_RAHUL = 8        # Rahul's: sneakers (item 11)
ITEM_TSHIRT, ITEM_SHOES, ITEM_OLD, ITEM_BRIEFS, ITEM_RAHUL = 1, 3, 4, 8, 11


def fake_embedder(texts: list[str]) -> list[list[float]]:
    """Hashed bag-of-words: crude, but deterministic and good enough to rank sections."""
    out = []
    for t in texts:
        v = [0.0] * 256
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
        out.append(v)
    return out


@pytest.fixture(autouse=True)
def fresh_db() -> Path:
    """Wipe and reseed before every test so tests can't affect each other."""
    seed_module.seed(DB_PATH)
    return DB_PATH


@pytest.fixture
def policy_index(monkeypatch: pytest.MonkeyPatch) -> None:
    rag.build_index(embedder=fake_embedder, force=True)
    monkeypatch.setattr(rag, "openai_embedder", lambda model=None: fake_embedder)


@pytest.fixture
def call() -> Callable[..., ToolResult]:
    """Call a tool the way the agent does, as a given customer."""

    def _call(name: str, args: dict[str, Any], customer_id: int = PRIYA,
              tracker: ErrorTracker | None = None, **kw: Any) -> ToolResult:
        return dispatch_tool(name, json.dumps(args), customer_id=customer_id,
                             tracker=tracker or ErrorTracker(), **kw)

    return _call


class FakeChatClient:
    """
    Stands in for openai.OpenAI(). Each `create()` call pops the next scripted
    step: a string is a final text reply; a list of (tool_name, args) is a
    tool-call turn; a callable receives the messages and returns either.
    """

    def __init__(self, script: list[Any] | None = None) -> None:
        self.script: list[Any] = list(script or [])
        self.calls: list[dict[str, Any]] = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        step = self.script.pop(0)
        if callable(step):
            step = step(kwargs["messages"])
        if isinstance(step, str):
            msg = NS(content=step, tool_calls=None)
        else:
            n = len(self.calls)
            msg = NS(content=None, tool_calls=[
                NS(id=f"call_{n}_{i}", function=NS(name=name, arguments=json.dumps(args)))
                for i, (name, args) in enumerate(step)
            ])
        return NS(choices=[NS(message=msg)])

    @property
    def last_messages(self) -> list[dict[str, Any]]:
        return self.calls[-1]["messages"]


def tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [json.loads(m["content"]) for m in messages if m["role"] == "tool"]


def latest_token(messages: list[dict[str, Any]]) -> str | None:
    """Pull the most recent confirmation_token a tool handed back to the model."""
    for result in reversed(tool_results(messages)):
        token = (result.get("data") or {}).get("confirmation_token")
        if token:
            return token
    return None