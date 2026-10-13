import sqlite3

from app.services.session import SessionStore


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
