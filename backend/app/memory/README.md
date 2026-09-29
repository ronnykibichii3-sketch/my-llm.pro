# Memory Layer Foundation

Backend-only foundation for long-term memory. No frontend exposure, no
chat augmentation, no RAG — only the memory infrastructure.

## Pipeline

```
Message Created
    ↓
Embedding Generated          (EmbeddingService, all-MiniLM-L6-v2)
    ↓
Stored In Qdrant             (VectorStore, payload filtered by user_id)
    ↓
Metadata Stored In PostgreSQL (Memory model, alembic migration 002)
```

Ingestion runs best-effort from `ChatService.chat` and
`MessageService.create_user_message`; a vector store outage is logged and
never breaks chat. `POST /api/v1/memory/reindex` backfills missing
memories from chat history and re-embeds existing ones.

## Structure

- `app/memory/services/` — `MemoryService` (orchestration), `EmbeddingService`
  (swappable protocol, lazy sentence-transformers backend), `VectorStore`
  (swappable protocol, lazy Qdrant backend)
- `app/memory/repositories/` — `MemoryRepository` (PostgreSQL metadata)
- `app/memory/schemas/` — Pydantic request/response models (`MemoryRead`,
  `MemorySearchResult`, `ReindexResult`, `MemoryPayload`)

## Endpoints

| Method | Path                        | Description                          |
|--------|-----------------------------|--------------------------------------|
| GET    | `/api/v1/memory`            | List the user's memories (newest first) |
| GET    | `/api/v1/memory/search?q=`  | Vector search scoped to the user     |
| POST   | `/api/v1/memory/reindex`    | Backfill and re-embed the user's memories |

All endpoints require a bearer token.

## Configuration

- `QDRANT_URL` (default `http://localhost:6333`)
- `QDRANT_API_KEY` (optional)
- `MEMORY_COLLECTION_NAME` (default `myllm_memories`)

## Tests

`backend/tests/test_memory.py` runs against in-memory fakes defined in
`backend/tests/conftest.py`; no Qdrant server or ML model is needed for
pytest.

## Operational notes

- Ingestion order is embed -> Qdrant upsert -> PostgreSQL insert. If the
  metadata insert fails, the vector point stays behind; the next
  `POST /api/v1/memory/reindex` re-embeds and heals the mapping, and search
  only returns points that still have a metadata row.
- `GET /api/v1/memory` reads PostgreSQL only, so listing keeps working during
  a Qdrant outage, while `search` and `reindex` surface the store error rather
  than returning misleadingly empty results.
- Every Qdrant query is filtered on the `user_id` payload field, and hits are
  hydrated from PostgreSQL by `embedding_id`.

## Running locally

- `docker compose up` starts Postgres, Redis, and Qdrant; inside the compose
  network the API reaches Qdrant at `QDRANT_URL=http://qdrant:6333`.
- Host ports are overridable when another process already owns them:
  `API_HOST_PORT`, `POSTGRES_HOST_PORT`, `REDIS_HOST_PORT`,
  `QDRANT_HTTP_HOST_PORT`, `QDRANT_GRPC_HOST_PORT` are provided by a root `.env`.
  PostgreSQL intentionally listens on **5433** (`POSTGRES_PORT` inside the
  container, `POSTGRES_HOST_PORT` on the host) because a native PostgreSQL
  service owns 5432 on this machine — never replace 5433 with 5432. The API and
  Alembic read the connection from `DATABASE_URL` only, so the port appears in
  exactly one place: that URL in the environment file.
- Validated against PostgreSQL 16 (`alembic upgrade head` -> `002 (head)`) and
  Qdrant v1.13.2: memory creation, ranked search, user scoping, deletion, real
  `all-MiniLM-L6-v2` embeddings at 384 dimensions, and a forced Qdrant outage
  where chat requests still succeeded.
