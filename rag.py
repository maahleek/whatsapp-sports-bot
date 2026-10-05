import shutil
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

KNOWLEDGE_FILE = Path("football_knowledge.txt")
VECTOR_DIR = Path("football_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _load_knowledge_text() -> str:
    if not KNOWLEDGE_FILE.exists():
        raise FileNotFoundError(
            "football_knowledge.txt is missing. Add the knowledge file before using RAG."
        )

    text = KNOWLEDGE_FILE.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("football_knowledge.txt is empty.")
    return text


def rebuild_knowledge_base() -> int:
    text = _load_knowledge_text()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)
    chunks = splitter.split_documents([Document(page_content=text)])

    if VECTOR_DIR.exists():
        shutil.rmtree(VECTOR_DIR)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=str(VECTOR_DIR),
        collection_name="football_rules",
    )

    get_vectorstore.cache_clear()
    return len(chunks)


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    if not VECTOR_DIR.exists():
        rebuild_knowledge_base()

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=str(VECTOR_DIR),
        embedding_function=embeddings,
        collection_name="football_rules",
    )


def search_knowledge(question: str, k: int = 2) -> list[str]:
    if not question.strip():
        return []

    results = get_vectorstore().similarity_search(question, k=k)
    return [document.page_content for document in results]
