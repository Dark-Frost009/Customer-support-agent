"""
db.py — SQLite connection management and schema initialisation.

Design notes
------------
* Raw `sqlite3`, no ORM (project rule). Every query elsewhere in the codebase
  goes through `get_db()` so connection settings live in exactly one place.
* `PRAGMA foreign_keys = ON` is a *per-connection* setting in SQLite. The
  PRAGMA at the top of schema.sql only affects the connection that ran the
  script, so we re-apply it on every connection we open. Without this, a
  return could reference an order that doesn't exist.
* One short-lived connection per unit of work. FastAPI runs sync endpoints in
  a thread pool, and sqlite3 connections must not be shared across threads,
  so "open, do the work, commit/rollback, close" is the safe pattern.
* WAL journal mode lets the API read while a write is in flight, which matters
  once several chat sessions hit the DB at once.
* Money is never handled here as floats — the schema stores paise as INTEGER,
  and the helpers below only format for display.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

BASE_DIR: Path = Path(__file__).resolve().parent
SCHEMA_PATH: Path = BASE_DIR / "schema.sql"
DB_PATH: Path = Path(os.getenv("SUPPORT_DB_PATH", str(BASE_DIR / "support.db")))

# Timestamps are stored as ISO-8601 UTC text, matching SQLite's datetime('now').
TS_FORMAT: str = "%Y-%m-%d %H:%M:%S"


# --------------------------------------------------------------------------- #
# Connections
# --------------------------------------------------------------------------- #
def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open a configured connection. Prefer `get_db()` unless you need manual control."""
    conn = sqlite3.connect(str(db_path or DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row  # rows behave like dicts: row["status"]
    conn.execute("PRAGMA foreign_keys = ON")  # per-connection; see module docstring
    conn.execute("PRAGMA busy_timeout = 5000")  # wait instead of failing on a locked DB
    return conn


@contextmanager
def get_db(db_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    """
    Transactional connection scope.

    Commits if the block finishes, rolls back on any exception. This means a
    tool that does two writes (e.g. insert a return + insert an escalation)
    either does both or neither — no half-applied refunds.
    """
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def _schema_exists(db_path: Path) -> bool:
    if not db_path.exists():
        return False
    with get_db(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'customers'"
        ).fetchone()
    return row is not None


def _delete_db_files(db_path: Path) -> None:
    # WAL mode leaves -wal / -shm side files; remove them too or SQLite may
    # replay stale pages into the fresh database.
    for suffix in ("", "-wal", "-shm"):
        p = Path(f"{db_path}{suffix}")
        if p.exists():
            p.unlink()


def init_db(db_path: str | Path | None = None, *, reset: bool = False) -> Path:
    """
    Create the database from schema.sql.

    schema.sql uses plain `CREATE TABLE` (no IF NOT EXISTS), so running it on an
    existing DB would fail. We therefore either skip (schema already present) or
    wipe and recreate (`reset=True`).
    """
    path = Path(db_path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    if reset:
        _delete_db_files(path)
    elif _schema_exists(path):
        return path

    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    conn = connect(path)
    try:
        conn.execute("PRAGMA journal_mode = WAL")  # persistent, set once per DB file
        conn.executescript(schema_sql)
    finally:
        conn.close()
    return path


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #
def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def to_db_ts(dt: datetime) -> str:
    """Format an aware or naive-UTC datetime the way the schema expects."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime(TS_FORMAT)


def from_db_ts(value: str) -> datetime:
    """Parse a stored timestamp back into an aware UTC datetime."""
    return datetime.strptime(value, TS_FORMAT).replace(tzinfo=timezone.utc)


def format_paise(paise: int) -> str:
    """Integer paise -> 'Rs. 1,299.00' using integer maths only (no float rounding)."""
    sign = "-" if paise < 0 else ""
    rupees, rem = divmod(abs(paise), 100)
    return f"{sign}Rs. {rupees:,}.{rem:02d}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Initialise the support database.")
    parser.add_argument("--db", default=str(DB_PATH), help="Path to the SQLite file")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate")
    args = parser.parse_args()
    created = init_db(args.db, reset=args.reset)
    print(f"Database ready at {created}")