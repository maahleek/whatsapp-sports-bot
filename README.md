# AI-Powered WhatsApp Football Assistant

A portfolio-ready agentic AI application that lets users ask football questions directly in WhatsApp.

The assistant combines a large language model with tool calling, live sports APIs, web search, retrieval-augmented generation (RAG), persistent conversational memory, and Twilio WhatsApp messaging.

## What it can do

- Answer natural-language football questions in WhatsApp
- Retrieve live scores and league standings
- Show recent results and upcoming fixtures
- Search player and team information
- Retrieve top scorers and injury updates
- Search recent transfer/news information
- Provide head-to-head and team-form summaries
- Produce a transparent, form-based match outlook
- Answer football-rules questions from a local RAG knowledge base
- Remember conversation context per WhatsApp user

## Architecture

```text
WhatsApp User
     |
     v
Twilio WhatsApp
     |
     v
FastAPI Webhook
     |
     v
LangGraph ReAct Agent
     |
     +--> Groq / Llama 3.3 70B
     |
     +--> Sports API tools
     |      - TheSportsDB
     |      - football-data.org
     |      - RapidAPI
     |
     +--> Tavily web search
     |
     +--> Local RAG
     |      - HuggingFace embeddings
     |      - Chroma vector database
     |
     +--> SQLite conversation memory
     |
     v
Twilio outbound message
     |
     v
WhatsApp User
```

## Tech stack

- Python
- FastAPI
- LangGraph
- LangChain
- Groq / Llama 3.3 70B
- Twilio WhatsApp API
- ChromaDB
- HuggingFace sentence-transformers
- SQLite
- Tavily
- TheSportsDB
- football-data.org
- RapidAPI

## Match outlook

The match-outlook feature is intentionally described as an **estimate**, not a machine-learning betting model.

It compares each team's recent five-match form using points-per-game and goal-difference-per-game, then converts the relative strength difference into simple outcome probabilities.

This makes the logic transparent and avoids presenting scraped predictions as a proprietary AI model.

## RAG knowledge base

Football rules are stored in `football_knowledge.txt`.

The project:

1. Splits the knowledge document into chunks.
2. Generates embeddings with `all-MiniLM-L6-v2`.
3. Stores them in Chroma.
4. Retrieves semantically relevant passages.
5. Exposes retrieval as a tool to the LangGraph agent.

Build or rebuild the vector store with:

```bash
python knowledge_base.py
```

## Local setup

### 1. Clone the repository

```bash
git clone https://github.com/maahleek/whatsapp-sports-bot.git
cd whatsapp-sports-bot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it using your operating system's normal virtual-environment command.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and provide your own credentials.

Never commit your real `.env` file or API keys.

### 5. Build the RAG index

```bash
python knowledge_base.py
```

### 6. Start the API

```bash
python run.py
```

or:

```bash
uvicorn bot:app --host 0.0.0.0 --port 8000
```

Health endpoint:

```text
GET /health
```

## Twilio webhook

Configure your Twilio WhatsApp inbound webhook to:

```text
POST https://YOUR-DOMAIN/webhook
```

For production, set:

```env
VERIFY_TWILIO_SIGNATURE=true
```

If you deploy behind a reverse proxy, make sure the public request URL seen by Twilio matches the URL used during signature validation.

## Environment variables

See `.env.example`.

Required for the full application:

- `GROQ_API_KEY`
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `FOOTBALL_DATA_KEY`
- `RAPIDAPI_KEY`
- `TAVILY_API_KEY`

Optional:

- `GROQ_MODEL`
- `TWILIO_WHATSAPP_FROM`
- `VERIFY_TWILIO_SIGNATURE`
- `SPORTSDB_API_KEY`
- `MEMORY_DB_PATH`
- `HTTP_TIMEOUT_SECONDS`
- `PORT`

## Reliability and security improvements

This version includes:

- HTTP timeouts
- API error handling
- environment-variable validation
- Twilio signature validation support
- fast webhook acknowledgement with background processing
- health-check endpoint
- runtime database files excluded from Git
- explicit user-facing fallback messages
- persistent per-user conversation memory

## Testing

A deterministic prediction-probability test is included:

```bash
python test_api.py
```

A RAG smoke test is included:

```bash
python test_rag.py
```

## Portfolio positioning

This project demonstrates:

- AI agents
- tool calling
- agentic workflows
- REST API integration
- RAG
- vector databases
- conversational memory
- prompt engineering
- WhatsApp automation
- Python backend development

## Disclaimer

Match outlooks are informational estimates based on recent form and are not guarantees, betting advice, or financial advice.
