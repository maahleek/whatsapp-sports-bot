from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools import tool

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./football_db",
    embedding_function=embeddings
)

@tool
def search_football_knowledge(question: str) -> str:
    """Search football rules and regulations."""
    results = vectorstore.similarity_search(question, k=2)
    if not results:
        return "No relevant information found."
    output = "From football knowledge base:\n"
    for r in results:
        output += f"{r.page_content}\n\n"
    return output

# Test directly
result = search_football_knowledge.invoke("offside rule")
print(result)