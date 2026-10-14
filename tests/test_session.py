import sqlite3

import pytest

from app.services.session import MAX_TURNS, SessionStore


def test_session_append_and_history(tmp_path):
    store = SessionStore(db_path=tmp_path / "sessions.sqlite3")
    sid = store.new_session()
    store.append(sid, {"role": "user", "content": "hi"}, "hello")
    assert store.get_history(sid) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_session_trims_old_history(tmp_path):
    store = SessionStore(db_path=tmp_path / "sessions.sqlite3")
    sid = store.new_session()
    for i in range(30):
        store.append(
            sid, {"role": "user", "content": f"hi {i}"}, f"hello {i}"
        )
    history = store.get_history(sid)
    assert len(history) <= 24
    assert history[-1] == {"role": "assistant", "content": "hello 29"}


def test_sessions_persist_messages_and_sort_by_activity(tmp_path):
    db_path = tmp_path / "sessions.sqlite3"
    store = SessionStore(db_path=db_path)
    older = store.new_session()
    newer = store.new_session()
    store.append(older, {"role": "user", "content": "A first question"}, "First answer")
    store.append(newer, {"role": "user", "content": "A newer question"}, "Newer answer")

    reloaded = SessionStore(db_path=db_path)

    assert [session["id"] for session in reloaded.list_sessions()][0] == newer
    assert reloaded.get_session(older)["title"] == "A first question"
    assert reloaded.get_history(older) == [
        {"role": "user", "content": "A first question"},
        {"role": "assistant", "content": "First answer"},
    ]


def test_users_table_has_unique_provider_identity(tmp_path):
    db_path = tmp_path / "sessions.sqlite3"
    SessionStore(db_path=db_path)

    with sqlite3.connect(db_path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(users)")
        }
        assert columns == {
            "id",
            "provider",
            "provider_user_id",
            "email",
            "name",
            "avatar_url",
            "created_at",
        }
        connection.execute(
            "INSERT INTO users (provider, provider_user_id, created_at) "
            "VALUES ('google', 'user-1', 'now')"
        )
        try:
            connection.execute(
                "INSERT INTO users (provider, provider_user_id, created_at) "
                "VALUES ('google', 'user-1', 'now')"
            )
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("provider identity must be unique")


def test_get_or_create_user_returns_existing_provider_identity(tmp_path):
    store = SessionStore(db_path=tmp_path / "sessions.sqlite3")

    first = store.get_or_create_user(
        "google", "user-1", "first@example.com", "First Name", None
    )
    second = store.get_or_create_user(
        "google", "user-1", "updated@example.com", "Updated Name", None
    )

    assert first["id"] == second["id"]
    assert second["email"] == "first@example.com"


def test_existing_sessions_table_gets_user_id_column(tmp_path):
    db_path = tmp_path / "sessions.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT NOT NULL, "
            "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )

    SessionStore(db_path=db_path)

    with sqlite3.connect(db_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(sessions)")}
    assert "user_id" in columns


def test_session_operations_are_scoped_to_user_id(tmp_path):
    store = SessionStore(db_path=tmp_path / "sessions.sqlite3")
    session_id = store.new_session(user_id=1)
    store.append(
        session_id, {"role": "user", "content": "private"}, "reply", user_id=1
    )

    assert [item["id"] for item in store.list_sessions(user_id=1)] == [session_id]
    assert store.list_sessions(user_id=2) == []
    assert store.get_session(session_id, user_id=2) is None
    assert store.get_history(session_id, user_id=2) == []

    try:
        store.append(
            session_id, {"role": "user", "content": "intrusion"}, "reply", user_id=2
        )
    except KeyError:
        pass
    else:
        raise AssertionError("a different user must not append to the conversation")

    store.clear(session_id, user_id=2)
    assert store.get_session(session_id, user_id=1) is not None
    store.clear(session_id, user_id=1)
    assert store.get_session(session_id, user_id=1) is None


def test_save_turn_and_load_recent_turns_return_chronological_messages(tmp_path):
    store = SessionStore(db_path=tmp_path / "sessions.sqlite3")
    session_id = store.new_session(user_id=1)

    store.save_turn(session_id, "user", "Recommend a thriller", user_id=1)
    history = store.save_turn(
        session_id, "assistant", "Try this one.", user_id=1
    )

    assert history == [
        {"role": "user", "content": "Recommend a thriller"},
        {"role": "assistant", "content": "Try this one."},
    ]
    assert store.load_recent_turns(session_id, user_id=1) == history


def test_save_turn_trims_to_six_user_assistant_turns(tmp_path):
    store = SessionStore(db_path=tmp_path / "sessions.sqlite3")
    session_id = store.new_session(user_id=1)

    for turn in range(MAX_TURNS + 2):
        store.save_turn(session_id, "user", f"user {turn}", user_id=1)
        store.save_turn(session_id, "assistant", f"assistant {turn}", user_id=1)

    history = store.load_recent_turns(session_id, user_id=1, limit=100)

    assert len(history) == MAX_TURNS * 2
    assert history[0] == {"role": "user", "content": "user 2"}
    assert history[-1] == {"role": "assistant", "content": "assistant 7"}


def test_recent_turns_and_save_are_scoped_and_clear_is_user_scoped(tmp_path):
    store = SessionStore(db_path=tmp_path / "sessions.sqlite3")
    session_id = store.new_session(user_id=1)
    store.save_turn(session_id, "user", "private turn", user_id=1)

    assert store.load_recent_turns(session_id, user_id=2) == []
    with pytest.raises(KeyError, match="Conversation not found"):
        store.save_turn(session_id, "assistant", "intrusion", user_id=2)

    store.clear(session_id, user_id=2)
    assert store.load_recent_turns(session_id, user_id=1) == [
        {"role": "user", "content": "private turn"}
    ]
    store.clear(session_id, user_id=1)
    assert store.load_recent_turns(session_id, user_id=1) == []


def test_loading_missing_session_does_not_create_it(tmp_path):
    store = SessionStore(db_path=tmp_path / "sessions.sqlite3")

    assert store.load_recent_turns("missing", user_id=1) == []
    assert store.get_session("missing", user_id=1) is None


@pytest.mark.parametrize(
    ("role", "content"),
    [
        ("system", "not permitted"),
        ("user", ""),
        ("assistant", "  "),
        ("user", []),
        ("user", 123),
    ],
)
def test_save_turn_rejects_invalid_role_or_empty_content(tmp_path, role, content):
    store = SessionStore(db_path=tmp_path / "sessions.sqlite3")
    session_id = store.new_session(user_id=1)

    with pytest.raises(ValueError):
        store.save_turn(session_id, role, content, user_id=1)


def test_recent_turn_methods_wrap_sqlite_errors(tmp_path, monkeypatch):
    store = SessionStore(db_path=tmp_path / "sessions.sqlite3")

    def fail_to_connect():
        raise sqlite3.OperationalError("private database path")

    monkeypatch.setattr(store, "_connect", fail_to_connect)

    with pytest.raises(RuntimeError, match=r"^Conversation memory unavailable\.$") as error:
        store.load_recent_turns("session-1", user_id=1)
    assert isinstance(error.value.__cause__, sqlite3.OperationalError)

    with pytest.raises(RuntimeError, match=r"^Conversation memory unavailable\.$") as error:
        store.save_turn("session-1", "user", "hello", user_id=1)
    assert isinstance(error.value.__cause__, sqlite3.OperationalError)
