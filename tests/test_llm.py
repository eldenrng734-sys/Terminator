from app.services.llm import SYSTEM_PROMPT, build_user_message, get_chat_reply


def test_build_user_message_image_data_url():
    msg = build_user_message("what is this?", [(b"\x89PNG", "image/png")])
    assert msg["role"] == "user"
    assert msg["content"][1]["image_url"]["url"].startswith("data:image/png;base64,iVBOR")


def test_get_chat_reply_sends_markdown_system_prompt(monkeypatch):
    captured_messages = []

    def fake_chat_completion(messages):
        captured_messages.extend(messages)
        return "**A formatted reply**"

    monkeypatch.setattr("app.services.llm.chat_completion", fake_chat_completion)

    reply = get_chat_reply([{"role": "user", "content": "Explain this"}])

    assert reply == "**A formatted reply**"
    assert captured_messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert captured_messages[1] == {"role": "user", "content": "Explain this"}
