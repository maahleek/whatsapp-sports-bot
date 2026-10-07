from rag import lookup_betting_term, search_knowledge


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

    exact_handicap = lookup_betting_term("What does Asian Handicap -1.0 mean?")
    assert exact_handicap is not None
    assert exact_handicap.startswith("Asian Handicap:")

    exact_double_chance = lookup_betting_term("Explain Double Chance X2.")
    assert exact_double_chance is not None
    assert exact_double_chance.startswith("Double Chance:")

    print("RAG football-rules and betting-terms tests passed.")


if __name__ == "__main__":
    main()
