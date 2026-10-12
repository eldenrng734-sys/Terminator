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
