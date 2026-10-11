import re
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

def lookup_betting_term(question: str) -> str | None:
    """Prefer exact glossary sections before falling back to semantic retrieval."""
    path = KNOWLEDGE_FILES["betting_terms"]
    if not path.exists():
        return None

    text = path.read_text(encoding="utf-8").strip()
    sections = [
        section.strip()
        for section in text.split("\n\n")
        if section.strip()
        and section.strip() != "FOOTBALL BETTING TERMS AND MARKETS"
    ]

    normalized_question = " ".join(
        re.sub(r"[^a-z0-9+\-./ ]+", " ", question.casefold()).split()
    )

    aliases = {
        "1x2 / match result": ("1x2", "match result"),
        "moneyline": ("moneyline",),
        "double chance": ("double chance", "1x", "x2"),
        "draw no bet (dnb)": ("draw no bet", "dnb"),
        "both teams to score (btts)": ("both teams to score", "btts"),
        "over / under goals": ("over ", "under ", "over/under"),
        "asian total": ("asian total",),
        "asian handicap": ("asian handicap",),
        "european handicap": ("european handicap",),
        "correct score": ("correct score", "exact score"),
        "team total goals": ("team total",),
        "clean sheet": ("clean sheet",),
        "win to nil": ("win to nil",),
        "half-time result": ("half-time result", "half time result"),
        "half-time / full-time": ("half-time/full-time", "half time full time"),
        "accumulator / parlay": ("accumulator", "parlay"),
        "same game parlay / bet builder": ("bet builder", "same game parlay"),
        "push / void": ("push", "void"),
        "half win / half loss": ("half win", "half loss"),
        "odds": ("odds",),
        "implied probability": ("implied probability",),
        "cash out": ("cash out",),
        "stake, return, profit": ("stake", "return", "profit"),
        "corners markets": ("corner", "corners"),
        "cards / booking markets": ("card", "cards", "booking"),
        "anytime goalscorer": ("anytime goalscorer", "goalscorer"),
        "player shots / shots on target": ("shots on target", "player shots"),
        "player assists": ("player assists", "assist"),
        "first-half goals": ("first half goals", "first-half goals"),
        "second-half goals": ("second half goals", "second-half goals"),
        "winning either half": ("winning either half",),
        "win both halves": ("win both halves",),
        "top goalscorer": ("top goalscorer",),
        "to qualify": ("to qualify",),
        "extra time": ("extra time",),
        "penalty shootout": ("penalty shootout", "penalties"),
    }

    by_title = {}
    for section in sections:
        title = section.split(":", 1)[0].strip().casefold()
        by_title[title] = section

    matches: list[tuple[int, str]] = []
    for title, terms in aliases.items():
        for term in terms:
            if term in normalized_question:
                matches.append((len(term), title))
                break

    if matches:
        matches.sort(reverse=True)
        best_title = matches[0][1]
        if best_title in by_title:
            return by_title[best_title]

    return None
