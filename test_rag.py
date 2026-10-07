from rag import search_knowledge


def main():
    rules = search_knowledge(
        "What is the offside rule?",
        k=1,
        category="football_rules",
    )
    assert rules, "Expected at least one football-rules passage."
    assert "offside" in rules[0].casefold()

    betting = search_knowledge(
        "What does Asian handicap mean?",
        k=1,
        category="betting_terms",
    )
    assert betting, "Expected at least one betting-terms passage."
    assert "handicap" in betting[0].casefold()

    print("RAG football-rules and betting-terms tests passed.")


if __name__ == "__main__":
    main()
