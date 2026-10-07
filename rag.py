import shutil
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

KNOWLEDGE_FILES = {
    "football_rules": Path("football_knowledge.txt"),
    "betting_terms": Path("betting_knowledge.txt"),
}
VECTOR_DIR = Path("football_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _load_sections(path: Path, category: str) -> list[Document]:
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is missing.")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path.name} is empty.")

    sections = [
        section.strip()
        for section in text.split("\n\n")
        if section.strip()
        and section.strip() not in {
            "FOOTBALL RULES AND REGULATIONS",
            "FOOTBALL BETTING TERMS AND MARKETS",
        }
    ]
    return [
        Document(page_content=section, metadata={"category": category})
        for section in sections
    ]


def rebuild_knowledge_base() -> int:
    documents: list[Document] = []
    for category, path in KNOWLEDGE_FILES.items():
        documents.extend(_load_sections(path, category))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=420,
        chunk_overlap=30,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(documents)

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


def search_knowledge(
    question: str,
    k: int = 1,
    category: str | None = None,
) -> list[str]:
    if not question.strip():
        return []

    kwargs = {"k": k}
    if category:
        kwargs["filter"] = {"category": category}

    results = get_vectorstore().similarity_search(question, **kwargs)
    return [document.page_content for document in results]
