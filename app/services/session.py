import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings


class SessionStore:
    def __init__(
        self,
        max_turns: int | None = None,
        db_path: str | Path | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._max_turns = max_turns
        self._db_path = Path(
            db_path
            or os.environ.get("CHAT_SESSION_DB_PATH")
            or Path(__file__).resolve().parents[2] / "sessions.sqlite3"
        )
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (session_id, position)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _title(user_message: dict[str, Any]) -> str:
        content = user_message.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        title = " ".join(str(content).split())
        return title[:60] or "New conversation"

    def _limit(self) -> int:
        if self._max_turns is not None:
            return self._max_turns * 2
        return get_settings().max_history_turns * 2

    def new_session(self, title: str = "New conversation") -> str:
        sid = uuid.uuid4().hex
        now = self._now()
        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (sid, title, now, now),
                )
        return sid

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT payload FROM messages WHERE session_id = ? ORDER BY position",
                    (session_id,),
                ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def list_sessions(self, query: str = "") -> list[dict[str, str]]:
        with self._lock:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
                ).fetchall()
        sessions = [dict(row) for row in rows]
        if query:
            term = query.casefold()
            sessions = [item for item in sessions if term in item["title"].casefold()]
        return sessions

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
                if row is None:
                    return None
                messages = connection.execute(
                    "SELECT payload FROM messages WHERE session_id = ? ORDER BY position",
                    (session_id,),
                ).fetchall()
        return {
            **dict(row),
            "messages": [json.loads(message["payload"]) for message in messages],
        }

    def append(
        self,
        session_id: str,
        user_message: dict[str, Any],
        assistant_reply: str,
    ) -> list[dict[str, Any]]:
        now = self._now()
        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    "INSERT OR IGNORE INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (session_id, self._title(user_message), now, now),
                )
                row = connection.execute(
                    "SELECT title FROM sessions WHERE id = ?", (session_id,)
                ).fetchone()
                if row["title"] == "New conversation":
                    connection.execute(
                        "UPDATE sessions SET title = ? WHERE id = ?",
                        (self._title(user_message), session_id),
                    )
                position = connection.execute(
                    "SELECT COALESCE(MAX(position), -1) + 1 FROM messages WHERE session_id = ?",
                    (session_id,),
                ).fetchone()[0]
                connection.executemany(
                    "INSERT INTO messages (session_id, position, payload) VALUES (?, ?, ?)",
                    [
                        (session_id, position, json.dumps(user_message)),
                        (
                            session_id,
                            position + 1,
                            json.dumps({"role": "assistant", "content": assistant_reply}),
                        ),
                    ],
                )
                connection.execute(
                    "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
                )
                count = connection.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
                ).fetchone()[0]
                excess = count - self._limit()
                if excess > 0:
                    connection.execute(
                        "DELETE FROM messages WHERE session_id = ? AND position IN "
                        "(SELECT position FROM messages WHERE session_id = ? ORDER BY position LIMIT ?)",
                        (session_id, session_id, excess),
                    )
        return self.get_history(session_id)

    def clear(self, session_id: str) -> None:
        with self._lock:
            with self._connect() as connection:
                connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


session_store = SessionStore()
