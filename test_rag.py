from rag import search_knowledge


def main():
    results = search_knowledge("What is the offside rule?", k=2)
    assert results, "Expected at least one retrieved knowledge-base passage."
    print("\n\n".join(results))


if __name__ == "__main__":
    main()
