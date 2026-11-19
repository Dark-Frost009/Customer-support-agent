"""
rag.py — build and query a SQLite-backed vector store over policy.txt.

Why this shape
--------------
* **Structure-aware chunking.** policy.txt has numbered sections and
  sub-headings ("Footwear:", "Refund approval limits:"). Splitting on those
  boundaries — instead of a fixed character window — keeps each rule intact,
  so the refund-limit paragraph never gets cut in half. Each chunk's embedded
  text is prefixed with its "Section > Subsection" path so a query like
  "can I return worn shoes" matches the Footwear chunk even though the word
  "footwear" doesn't appear in the bullet itself.
* **Plain SQLite, no extension.** Vectors are stored as float32 BLOBs and
  scored with brute-force cosine similarity in Python. The policy is ~15
  chunks, so a linear scan takes microseconds and we avoid shipping a native
  extension. If the corpus grows to thousands of chunks, swap in `sqlite-vec`
  behind the same `search_policy()` signature.
* **Separate index file.** The index lives in policy_index.db, not support.db,
  so `seed.py` (which wipes support.db) never forces a paid re-embed.
* **Rebuild only when needed.** We store the SHA-256 of policy.txt and the
  embedding model name; `build_index()` is a no-op if neither changed.
* **Raw OpenAI SDK** for embeddings (project rule: no LangChain/LlamaIndex).
  The embedder is injectable so tests can run offline.
* **Read-only tool.** `search_policy()` only ever runs a fixed SELECT. The LLM
  supplies a natural-language query string, never SQL.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sqlite3
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from db import BASE_DIR, connect, get_db, to_db_ts, utc_now

POLICY_PATH: Path = Path(os.getenv("POLICY_PATH", str(BASE_DIR / "policy.txt")))
INDEX_PATH: Path = Path(os.getenv("POLICY_INDEX_PATH", str(BASE_DIR / "policy_index.db")))
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

MAX_CHUNK_CHARS: int = 700   # soft cap; a single bullet is never split
MAX_TOP_K: int = 5           # cap what the LLM can pull into context per call

# An embedder takes a batch of strings and returns one vector per string.
Embedder = Callable[[Sequence[str]], list[list[float]]]


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PolicyChunk:
    section: str
    subsection: str | None
    body: str

    @property
    def embed_text(self) -> str:
        """Text sent to the embedding model: heading path + body."""
        path = self.section if not self.subsection else f"{self.section} > {self.subsection}"
        return f"{path}\n{self.body}"


_RULE_RE = re.compile(r"^-{10,}\s*$")                 # ------------ divider
_SECTION_RE = re.compile(r"^\d+\.\s+[A-Z][A-Z &/,'-]+$")  # "3. RETURN POLICY"
_SUBHEAD_RE = re.compile(r"^[A-Z][^-].*:\s*$")         # "Footwear:" (col 0, ends with ':')


def _split_bullets(lines: list[str]) -> list[str]:
    """Group lines into bullets; indented continuation lines join their bullet."""
    bullets: list[str] = []
    for line in lines:
        if line.startswith("- "):
            bullets.append(line.rstrip())
        elif bullets and line.strip():
            bullets[-1] += "\n" + line.rstrip()
        elif line.strip():  # free text with no bullet (e.g. the preamble)
            bullets.append(line.strip())
    return bullets


def _pack(bullets: list[str], max_chars: int) -> list[str]:
    """Greedily pack whole bullets into chunks of up to ~max_chars."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for b in bullets:
        if current and size + len(b) > max_chars:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(b)
        size += len(b) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def chunk_policy(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[PolicyChunk]:
    """Split the policy into (section, subsection) groups, then pack bullets."""
    groups: list[tuple[str, str | None, list[str]]] = []
    section, subsection = "GENERAL", None
    buf: list[str] = []

    def flush() -> None:
        if any(l.strip() for l in buf):
            groups.append((section, subsection, list(buf)))
        buf.clear()

    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if _RULE_RE.match(stripped):
            continue
        if _SECTION_RE.match(stripped):
            flush()
            section, subsection = stripped, None
            continue
        if _SUBHEAD_RE.match(line) and not line.startswith(" "):
            flush()
            subsection = stripped.rstrip(":").strip()
            continue
        # Skip the document title / date lines; they carry no policy content.
        if section == "GENERAL" and (stripped == "STORE POLICY" or stripped.startswith("Last updated")):
            continue
        buf.append(line)
    flush()

    chunks: list[PolicyChunk] = []
    for sec, sub, lines in groups:
        for body in _pack(_split_bullets(lines), max_chars):
            chunks.append(PolicyChunk(sec, sub, body))
    return chunks


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def openai_embedder(model: str = EMBEDDING_MODEL) -> Embedder:
    """Default embedder using the raw OpenAI SDK. Reads OPENAI_API_KEY from env."""
    from openai import OpenAI  # imported lazily so offline tests don't need the package

    # Same resilience settings as the chat client (see agent.py).
    client = OpenAI(
        timeout=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30")),
        max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "3")),
    )

    def embed(texts: Sequence[str]) -> list[list[float]]:
        resp = client.embeddings.create(model=model, input=list(texts))
        # The API returns items with an `index`; sort to be safe about ordering.
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

    return embed


def _normalise(vec: Sequence[float]) -> array:
    """L2-normalise once at write time so search is a plain dot product."""
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return array("f", (x / norm for x in vec))


def _dot(a: array, b: array) -> float:
    return sum(x * y for x, y in zip(a, b))


# --------------------------------------------------------------------------- #
# Index storage
# --------------------------------------------------------------------------- #
_INDEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS policy_chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    section     TEXT    NOT NULL,
    subsection  TEXT,
    body        TEXT    NOT NULL,
    embed_text  TEXT    NOT NULL,
    embedding   BLOB    NOT NULL,   -- float32, L2-normalised
    dim         INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS index_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_INDEX_SCHEMA)


def _get_meta(conn: sqlite3.Connection) -> dict[str, str]:
    return {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM index_meta")}


def build_index(
    policy_path: str | Path = POLICY_PATH,
    index_path: str | Path = INDEX_PATH,
    *,
    embedder: Embedder | None = None,
    model_name: str = EMBEDDING_MODEL,
    force: bool = False,
) -> int:
    """
    (Re)build the vector index. Returns the number of chunks indexed.

    Skips work when policy.txt and the model are unchanged, since every
    rebuild costs an embeddings API call.
    """
    text = Path(policy_path).read_text(encoding="utf-8")
    source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    with get_db(index_path) as conn:
        _ensure_schema(conn)
        meta = _get_meta(conn)
        if (not force and meta.get("source_sha256") == source_hash
                and meta.get("embedding_model") == model_name):
            return int(meta.get("chunk_count", "0"))

    chunks = chunk_policy(text)
    embed = embedder or openai_embedder(model_name)
    # Embed before opening the write transaction: a network failure must not
    # leave us with an emptied table.
    vectors = embed([c.embed_text for c in chunks])
    if len(vectors) != len(chunks):
        raise RuntimeError(f"Embedder returned {len(vectors)} vectors for {len(chunks)} chunks")

    with get_db(index_path) as conn:
        _ensure_schema(conn)
        conn.execute("DELETE FROM policy_chunks")
        for chunk, vec in zip(chunks, vectors):
            normed = _normalise(vec)
            conn.execute(
                """INSERT INTO policy_chunks (section, subsection, body, embed_text,
                                              embedding, dim)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (chunk.section, chunk.subsection, chunk.body, chunk.embed_text,
                 normed.tobytes(), len(normed)),
            )
        conn.executemany(
            "INSERT OR REPLACE INTO index_meta (key, value) VALUES (?, ?)",
            [("source_sha256", source_hash), ("embedding_model", model_name),
             ("chunk_count", str(len(chunks))), ("built_at", to_db_ts(utc_now()))],
        )
    return len(chunks)


# --------------------------------------------------------------------------- #
# Retrieval — this is the function exposed to the LLM as the `search_policy` tool
# --------------------------------------------------------------------------- #
class PolicyIndexError(RuntimeError):
    """Raised when the index is missing or was built with a different model."""


def search_policy(
    query: str,
    top_k: int = 3,
    *,
    index_path: str | Path = INDEX_PATH,
    embedder: Embedder | None = None,
    model_name: str = EMBEDDING_MODEL,
) -> list[dict[str, object]]:
    """
    Return the policy passages most relevant to `query`.

    Each result: {"section", "subsection", "text", "score"}. `top_k` is clamped
    to 1..MAX_TOP_K so the model can't dump the whole policy into its context.
    """
    query = (query or "").strip()
    if not query:
        return []
    top_k = max(1, min(int(top_k), MAX_TOP_K))

    conn = connect(index_path)
    try:
        _ensure_schema(conn)
        meta = _get_meta(conn)
        if not meta.get("chunk_count"):
            raise PolicyIndexError("Policy index is empty. Run `python rag.py build`.")
        # Query and document vectors must come from the same model, or cosine
        # scores are meaningless (and dimensions may not even match).
        if meta.get("embedding_model") != model_name:
            raise PolicyIndexError(
                f"Index built with {meta.get('embedding_model')!r} but querying with "
                f"{model_name!r}. Rebuild with `python rag.py build --force`."
            )
        rows = conn.execute(
            "SELECT section, subsection, body, embedding FROM policy_chunks"
        ).fetchall()
    finally:
        conn.close()

    embed = embedder or openai_embedder(model_name)
    q = _normalise(embed([query])[0])

    scored: list[tuple[float, sqlite3.Row]] = []
    for r in rows:
        vec = array("f")
        vec.frombytes(r["embedding"])
        scored.append((_dot(q, vec), r))
    scored.sort(key=lambda t: t[0], reverse=True)

    return [
        {
            "section": r["section"],
            "subsection": r["subsection"],
            "text": r["body"],
            "score": round(score, 4),
        }
        for score, r in scored[:top_k]
    ]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Policy vector store")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Embed policy.txt into the SQLite index")
    b.add_argument("--force", action="store_true", help="Rebuild even if unchanged")

    s = sub.add_parser("search", help="Query the index")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=3)

    c = sub.add_parser("chunks", help="Print chunks without embedding (no API call)")

    args = parser.parse_args()
    if args.cmd == "build":
        n = build_index(force=args.force)
        print(f"Indexed {n} chunks into {INDEX_PATH}")
    elif args.cmd == "search":
        print(json.dumps(search_policy(args.query, args.k), indent=2, ensure_ascii=False))
    else:
        for i, ch in enumerate(chunk_policy(POLICY_PATH.read_text(encoding="utf-8")), 1):
            print(f"--- [{i}] {ch.section} > {ch.subsection or '-'} ({len(ch.body)} chars)")
            print(ch.body)