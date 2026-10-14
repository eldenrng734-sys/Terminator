import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings

MAX_TURNS = 6


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
                    updated_at TEXT NOT NULL,
                    user_id INTEGER
                );
                CREATE TABLE IF NOT EXISTS messages (
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (session_id, position)
                );
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    provider_user_id TEXT NOT NULL,
                    email TEXT,
                    name TEXT,
                    avatar_url TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(provider, provider_user_id)
                );
                """
            )
            session_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(sessions)")
            }
            if "user_id" not in session_columns:
                connection.execute("ALTER TABLE sessions ADD COLUMN user_id INTEGER")

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

    def get_or_create_user(
        self,
        provider: str,
        provider_user_id: str,
        email: str | None,
        name: str | None,
        avatar_url: str | None,
    ) -> dict[str, Any]:
        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    "INSERT OR IGNORE INTO users "
                    "(provider, provider_user_id, email, name, avatar_url, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (provider, provider_user_id, email, name, avatar_url, self._now()),
                )
                row = connection.execute(
                    "SELECT id, provider, provider_user_id, email, name, avatar_url, created_at "
                    "FROM users WHERE provider = ? AND provider_user_id = ?",
                    (provider, provider_user_id),
                ).fetchone()
        return dict(row)

    def get_user_by_id(self, user_id: int) -> dict[str, Any] | None:
        with self._lock:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT id, provider, provider_user_id, email, name, avatar_url, created_at "
                    "FROM users WHERE id = ?",
                    (user_id,),
                ).fetchone()
        return dict(row) if row is not None else None

    def new_session(
        self, title: str = "New conversation", user_id: int | None = None
    ) -> str:
        sid = uuid.uuid4().hex
        now = self._now()
        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO sessions (id, title, created_at, updated_at, user_id) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (sid, title, now, now, user_id),
                )
        return sid

    def get_history(
        self, session_id: str, user_id: int | None = None
    ) -> list[dict[str, Any]]:
        with self._lock:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT messages.payload FROM messages "
                    "JOIN sessions ON sessions.id = messages.session_id "
                    "WHERE messages.session_id = ? AND sessions.user_id IS ? "
                    "ORDER BY messages.position",
                    (session_id, user_id),
                ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def load_recent_turns(
        self,
        session_id: str,
        user_id: int | None = None,
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        if limit < 0:
            raise ValueError("limit cannot be negative.")
        try:
            with self._lock:
                with self._connect() as connection:
                    rows = connection.execute(
                        "SELECT messages.payload FROM messages "
                        "JOIN sessions ON sessions.id = messages.session_id "
                        "WHERE messages.session_id = ? AND sessions.user_id IS ? "
                        "ORDER BY messages.position DESC LIMIT ?",
                        (session_id, user_id, limit),
                    ).fetchall()
        except sqlite3.Error as exc:
            raise RuntimeError("Conversation memory unavailable.") from exc

        turns: list[dict[str, Any]] = []
        for row in reversed(rows):
            try:
                payload = json.loads(row["payload"])
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(payload, dict):
                continue
            role = payload.get("role")
            content = payload.get("content")
            if not isinstance(role, str) or not role.strip():
                continue
            if isinstance(content, str):
                if not content.strip():
                    continue
            elif not isinstance(content, list) or not content:
                continue
            turns.append({"role": role, "content": content})
        return turns

    def save_turn(
        self,
        session_id: str,
        role: str,
        content: str | list[Any],
        user_id: int | None = None,
    ) -> list[dict[str, Any]]:
        if role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'.")
        if isinstance(content, str):
            if not content.strip():
                raise ValueError("content cannot be empty.")
        elif not isinstance(content, list) or not content:
            raise ValueError("content must be a non-empty string or list.")

        now = self._now()
        initial_title = (
            self._title({"content": content})
            if role == "user"
            else "New conversation"
        )
        try:
            with self._lock:
                with self._connect() as connection:
                    connection.execute(
                        "INSERT OR IGNORE INTO sessions "
                        "(id, title, created_at, updated_at, user_id) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (session_id, initial_title, now, now, user_id),
                    )
                    row = connection.execute(
                        "SELECT title FROM sessions WHERE id = ? AND user_id IS ?",
                        (session_id, user_id),
                    ).fetchone()
                    if row is None:
                        raise KeyError("Conversation not found.")
                    if role == "user" and row["title"] == "New conversation":
                        connection.execute(
                            "UPDATE sessions SET title = ? "
                            "WHERE id = ? AND user_id IS ?",
                            (initial_title, session_id, user_id),
                        )

                    position = connection.execute(
                        "SELECT COALESCE(MAX(position), -1) + 1 "
                        "FROM messages WHERE session_id = ?",
                        (session_id,),
                    ).fetchone()[0]
                    connection.execute(
                        "INSERT INTO messages (session_id, position, payload) "
                        "VALUES (?, ?, ?)",
                        (session_id, position, json.dumps({"role": role, "content": content})),
                    )
                    connection.execute(
                        "UPDATE sessions SET updated_at = ? "
                        "WHERE id = ? AND user_id IS ?",
                        (now, session_id, user_id),
                    )
                    excess = connection.execute(
                        "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                        (session_id,),
                    ).fetchone()[0] - (MAX_TURNS * 2)
                    if excess > 0:
                        connection.execute(
                            "DELETE FROM messages WHERE session_id = ? AND position IN "
                            "(SELECT position FROM messages WHERE session_id = ? "
                            "ORDER BY position LIMIT ?)",
                            (session_id, session_id, excess),
                        )
        except sqlite3.Error as exc:
            raise RuntimeError("Conversation memory unavailable.") from exc

        return self.load_recent_turns(session_id, user_id, limit=MAX_TURNS * 2)

    def list_sessions(
        self, query: str = "", user_id: int | None = None
    ) -> list[dict[str, str]]:
        with self._lock:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT id, title, created_at, updated_at FROM sessions "
                    "WHERE user_id IS ? ORDER BY updated_at DESC",
                    (user_id,),
                ).fetchall()
        sessions = [dict(row) for row in rows]
        if query:
            term = query.casefold()
            sessions = [item for item in sessions if term in item["title"].casefold()]
        return sessions

    def get_session(
        self, session_id: str, user_id: int | None = None
    ) -> dict[str, Any] | None:
        with self._lock:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT id, title, created_at, updated_at FROM sessions "
                    "WHERE id = ? AND user_id IS ?",
                    (session_id, user_id),
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
        user_id: int | None = None,
    ) -> list[dict[str, Any]]:
        now = self._now()
        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    "INSERT OR IGNORE INTO sessions "
                    "(id, title, created_at, updated_at, user_id) VALUES (?, ?, ?, ?, ?)",
                    (session_id, self._title(user_message), now, now, user_id),
                )
                row = connection.execute(
                    "SELECT title FROM sessions WHERE id = ? AND user_id IS ?",
                    (session_id, user_id),
                ).fetchone()
                if row is None:
                    raise KeyError("Conversation not found.")
                if row["title"] == "New conversation":
                    connection.execute(
                        "UPDATE sessions SET title = ? WHERE id = ? AND user_id IS ?",
                        (self._title(user_message), session_id, user_id),
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
                    "UPDATE sessions SET updated_at = ? WHERE id = ? AND user_id IS ?",
                    (now, session_id, user_id),
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
        return self.get_history(session_id, user_id)

    def clear(self, session_id: str, user_id: int | None = None) -> None:
        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM sessions WHERE id = ? AND user_id IS ?",
                    (session_id, user_id),
                )


session_store = SessionStore()
