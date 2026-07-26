from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Create some football knowledge documents
football_rules = """
FOOTBALL RULES AND REGULATIONS

Offside Rule:
A player is in an offside position if any part of the head, body or feet is in the opponents' half and closer to the opponents' goal line than both the ball and the second-last opponent. A player is not in an offside position if level with the second-last opponent or the last two opponents.

VAR (Video Assistant Referee):
VAR reviews four categories of incidents: goals, penalty decisions, direct red card incidents, and mistaken identity. The on-field referee can review incidents on a pitchside monitor.

Yellow Card:
A player receives a yellow card for unsporting behavior, dissent, persistent infringement of the laws, delaying restart of play, failing to respect required distance, entering or leaving without permission.

Red Card:
A player is sent off for serious foul play, violent conduct, spitting, denying a goal with handball, denying an obvious goal-scoring opportunity, offensive language, receiving two yellow cards.

Penalty Kick:
Awarded when a player commits a foul inside their own penalty area. Taken from the penalty spot, 11 meters from goal.
"""

# Save to a text file
with open("football_knowledge.txt", "w") as f:
    f.write(football_rules)

# Load and split the document
loader = TextLoader("football_knowledge.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Create embeddings and store in ChromaDB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./football_db"
)

print(f"Knowledge base created with {len(chunks)} chunks!")