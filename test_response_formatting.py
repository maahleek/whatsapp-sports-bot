from types import SimpleNamespace

from bot import _content_to_text, _current_turn_tool_outputs, sanitize_whatsapp_response


def main():
    sample = "**Premier League**\n*Arsenal* are top.\n# Update\n`code`"
    cleaned = sanitize_whatsapp_response(sample)
    assert "*" not in cleaned
    assert "#" not in cleaned
    assert "`" not in cleaned
    assert "Premier League" in cleaned
    assert "Arsenal" in cleaned

    blocks = [{"type": "text", "text": "**Hello**"}, {"type": "text", "text": "World"}]
    normalized = _content_to_text(blocks)
    assert "Hello" in normalized and "World" in normalized

    messages = [
        SimpleNamespace(type="human", content="Old question"),
        SimpleNamespace(type="tool", content="Old tool output"),
        SimpleNamespace(type="ai", content="Old answer"),
        SimpleNamespace(type="human", content="New question"),
        SimpleNamespace(type="tool", content="Current tool output"),
        SimpleNamespace(type="ai", content="Embellished model answer"),
    ]
    outputs = _current_turn_tool_outputs(messages)
    assert outputs == ["Current tool output"]

    print("WhatsApp response formatting test passed.")


if __name__ == "__main__":
    main()
