# ScholarScout


An AI-powered agent that helps students find scholarship and research opportunities.

---

## Architecture

- **Agent** (`scripts/agent.py`) — LangGraph agent with semantic routing across PostgreSQL, Milvus vector store, and optional Tavily web search
- **PostgreSQL** — structured data (programs, deadlines, tuition fees, funding types)
- **Milvus** — vector store for semantic search over program descriptions and professor opportunities
- **Flask** (`app.py`) — HTTP API with streaming support

---

## Prerequisites

- Python 3.10+
- PostgreSQL
- Milvus (via Docker Compose — see below)
- OpenAI API key
- Tavily API key (optional — only needed for web search)

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd ScholarScout
pip install -r requirements.txt
```

Or with `uv`:

```bash
uv sync
```

### 2. Configure environment variables

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```
OPENAI_KEY=your-openai-api-key
TAVILY_API_KEY=your-tavily-api-key    # optional

DB_HOST=localhost
DB_PORT=5432
DB_NAME=scholarscout
DB_USER=postgres
DB_PASSWORD=your-db-password

MILVUS_URI=http://localhost:19530
```

### 3. Download data files

Download the data files and place them in the `data/` directory:

[Google Drive →](https://drive.google.com/drive/folders/1T2d_gP8iNjqLKnmNGXXqBeOxkRrZ4GT9?usp=sharing)

Expected files:
```
data/
├── universities.csv
├── finances.csv
├── universities_semantic_data.json
└── professor_opportunities.json
```

### 4. Start Milvus with Docker Compose

Follow the official Milvus standalone setup using Docker Compose:

[Milvus Docker Compose setup →](https://milvus.io/docs/install_standalone-docker-compose.md)

Once running, Milvus will be available at `http://localhost:19530` by default.

### 5. Set up PostgreSQL

Create the database and tables using the provided schema file:

```bash
psql -U postgres -d scholarscout -f scripts/scholarscout_schema.sql
```

Then load the data:

```bash
python scripts/setup_db.py
```

Expected output:
```
Tables ready.
Loaded N universities.
Loaded N finance records.
Database setup complete.
```

### 6. Index data into Milvus

Index university program descriptions:

```bash
python scripts/index_universities.py
```

Index professor research opportunities:

```bash
python scripts/index_professors.py
```

Both scripts support resuming — if interrupted, re-running will skip already-indexed records.

### 7. Run the app

```bash
python app.py
```

The server starts at `http://localhost:8100`.

---

## API

### `POST /stream`

Streams the agent's response token by token.

```json
{
  "question": "Find fully funded CS programs with deadlines after March",
  "thread_id": "user-123",
  "web_search": false
}
```

- `thread_id` — used to maintain conversation history across requests
- `web_search` — set to `true` to enable live Tavily web search as a fallback

### `GET /health`

Returns `{"status": "ok"}`.

---

## Configuration

All tunable constants live in `configs/config.py`:

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `gpt-4o` | Model used for response generation |
| `LLM_FAST_MODEL` | `gpt-4o-mini` | Model used for routing and profile extraction |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `EMBED_DIM` | `1536` | Embedding dimension |
| `MILVUS_COLLECTION` | `erasmus_university_programs` | Milvus collection for programs |
| `PROFESSOR_COLLECTION` | `professor_opportunities` | Milvus collection for professors |
| `MILVUS_TOP_K` | `5` | Number of results returned from vector search |
| `CHUNK_SIZE` | `4000` | Token chunk size for indexing |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `MILVUS_URI` | `http://localhost:19530` | Override via `MILVUS_URI` env var |
