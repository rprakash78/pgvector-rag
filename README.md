# PGVector RAG

A FastAPI retrieval-augmented generation example that stores document
embeddings in PostgreSQL with the `pgvector` extension and uses Groq to answer
questions from retrieved context.

## Requirements

- Python 3.12 or newer
- PostgreSQL with the `pgvector` extension
- A Groq API key

The embedding model is `sentence-transformers/all-MiniLM-L6-v2`, which creates
384-dimensional vectors.

## Database setup

Create a database, enable `pgvector`, and create the table used by the app:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    document_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    topic VARCHAR(100),
    source TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding VECTOR(384) NOT NULL
);
```

## Run locally

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `DATABASE_URL` and `GROQ_API_KEY` in `.env`. For a local PostgreSQL
database, the connection string has this form:

```text
postgresql://username:password@localhost:5432/database_name
```

Ingest the sample DSA document:

```bash
python -m app.ingestion
```

Start the API:

```bash
uvicorn app.main:app --reload
```

The API exposes `POST /ask`. Example request:

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the time complexity of binary search?","top_k":3}'
```

Interactive API documentation is available at `http://localhost:8000/docs`.

## Layout

```text
app/          API, embedding, ingestion, and retrieval code
data/         Sample documents
pg-vector-rag/ Bruno requests for manual API testing
requirements.txt
```