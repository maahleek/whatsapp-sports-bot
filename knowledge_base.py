from rag import rebuild_knowledge_base

if __name__ == "__main__":
    count = rebuild_knowledge_base()
    print(f"Knowledge base created with {count} chunks.")
