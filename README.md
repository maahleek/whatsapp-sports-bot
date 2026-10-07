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
- Produce fixture-aware match predictions and Poisson correct-score projections
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
Safety / grounding router
     |
     +--> Direct deterministic tools for guarded intents
     |
     v
LangGraph ReAct Agent
     |
     +--> Anthropic Claude
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
- Anthropic Claude
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

The match predictor primarily uses current-season league performance from structured standings data. When recent-match data is available, it blends that into the season signal with a smaller weight.

Before applying any home advantage, the assistant checks the actual upcoming fixture between the two clubs. If it finds the fixture, the real home team receives a small venue adjustment. If no upcoming head-to-head fixture is found, the model treats the matchup as neutral rather than assuming the first team typed is at home.

For teams with usable scoring and conceding data, the assistant builds a simple Poisson score model. The same model supplies win/draw/loss probabilities and the most likely exact scorelines, keeping the outcome and correct-score projections internally consistent.

The output includes estimated probabilities, a predicted outcome, a confidence label, and—when scoring data is available—a most likely scoreline. These are informational statistical estimates, not guaranteed results or betting advice.

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

- `ANTHROPIC_API_KEY`
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `FOOTBALL_DATA_KEY`
- `RAPIDAPI_KEY`
- `TAVILY_API_KEY`

Optional:

- `ANTHROPIC_MODEL`
- `TWILIO_WHATSAPP_FROM`
- `VERIFY_TWILIO_SIGNATURE`
- `SPORTSDB_API_KEY`
- `MEMORY_DB_PATH`
- `MEMORY_NAMESPACE`
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
- stale-memory protection for current sports facts
- MessageSid deduplication for repeated Twilio webhooks
- fixture-aware home/away adjustment instead of assuming message order
- Poisson-based win/draw/loss and correct-score projections

## Testing

API-helper and prediction-probability tests are included:

```bash
python test_api.py
```

A WhatsApp response-formatting and current-turn grounding regression test is included:

```bash
python test_response_formatting.py
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

Match predictions and score projections are informational statistical estimates and are not guarantees, betting advice, or financial advice.
