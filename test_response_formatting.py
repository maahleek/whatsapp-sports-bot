from bot import _content_to_text, sanitize_whatsapp_response


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

    print("WhatsApp response formatting test passed.")


if __name__ == "__main__":
    main()
