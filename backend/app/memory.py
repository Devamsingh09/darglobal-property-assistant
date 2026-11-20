"""Per-session conversation history in SQLite (ephemeral on a free Space - fine for a demo)."""
from __future__ import annotations

import sqlite3
import threading
import time

from . import config

_lock = threading.Lock()
_con = sqlite3.connect(config.SESSION_DB, check_same_thread=False)
_con.execute("""CREATE TABLE IF NOT EXISTS messages(
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, ts REAL)""")
_con.execute("CREATE INDEX IF NOT EXISTS ix_s ON messages(session_id, id)")
_con.commit()


def recent(session_id: str, n: int = config.HISTORY_MESSAGES) -> list[dict]:
    with _lock:
        rows = _con.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, n)).fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def add(session_id: str, role: str, content: str) -> None:
    with _lock:
        _con.execute("INSERT INTO messages(session_id, role, content, ts) VALUES(?,?,?,?)",
                     (session_id, role, content[:6000], time.time()))
        _con.commit()


def clear(session_id: str) -> None:
    with _lock:
        _con.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        _con.commit()
