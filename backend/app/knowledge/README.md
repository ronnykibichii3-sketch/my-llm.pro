# Knowledge Base Foundation

Backend-only foundation for retrieval over user-supplied documents. No frontend
exposure and no chat augmentation yet — ingestion and retrieval only.

## Pipeline

```
Document Received            (JSON body or raw request body)
    |
Validated Text               (content type, size ceiling, charset)
    |
Chunked                      (chunker: paragraphs -> sentences -> words)
    |
Embedded                     (EmbeddingService, shared with app.memory)
    |
Stored In Qdrant             (own collection, payload filtered by user_id)
    |
Chunk + Document Rows        (PostgreSQL, alembic migration 003)
```

Ingestion is synchronous and all-or-nothing from the caller's point of view: if
embedding or the vector store fails, the points written so far are deleted and
the document row is left with `status = "failed"` and the reason in `error`,
rather than being half-indexed.

## Structure

- `app/knowledge/config.py` — every tunable, ceiling and payload key in one place
- `app/knowledge/services/chunker.py` — pure, deterministic text splitting
- `app/knowledge/services/text_source.py` — raw upload to validated text
- `app/knowledge/services/knowledge_service.py` — orchestration across Qdrant and PostgreSQL
- `app/knowledge/repositories/` — `KnowledgeRepository` (documents and chunks)
- `app/knowledge/schemas/` — Pydantic request/response models plus the internal
  `DocumentDraft`, `ChunkPlan` and `KnowledgePayload` hand-off types

Knowledge vectors live in their own Qdrant collection
(`KNOWLEDGE_COLLECTION_NAME`), so a knowledge search can never return chat
memories, or vice versa, while both reuse the same `VectorStore` and
`EmbeddingService` implementations.

## Endpoints

| Method | Path                                   | Description                                        |
|--------|----------------------------------------|----------------------------------------------------|
| GET    | `/api/v1/knowledge/documents`          | List the user's documents (newest first)            |
| POST   | `/api/v1/knowledge/documents`          | Ingest a document sent as JSON                      |
| POST   | `/api/v1/knowledge/documents/raw`      | Ingest a document sent as a raw text body           |
| GET    | `/api/v1/knowledge/documents/{id}`     | Fetch one document with its chunks in reading order |
| DELETE | `/api/v1/knowledge/documents/{id}`     | Delete the document, its chunks and its vectors     |
| GET    | `/api/v1/knowledge/search?q=`          | Vector search; optional `document_id` narrowing     |

All endpoints require a bearer token, and every query is scoped to the
authenticated user.

Raw bodies are accepted instead of `multipart/form-data` because the project does
not depend on `python-multipart`; `curl --data-binary @notes.md` works without a
new dependency. Binary formats (PDF, DOCX, images) are rejected with `415`
rather than ingested as mojibake.

## Configuration

- `KNOWLEDGE_COLLECTION_NAME` (default `myllm_knowledge`)
- `KNOWLEDGE_CHUNK_MAX_CHARS` (default `1200`)
- `KNOWLEDGE_CHUNK_OVERLAP` (default `200`)
- `KNOWLEDGE_MAX_DOCUMENT_CHARS` (default `200000`)
- `KNOWLEDGE_MAX_UPLOAD_BYTES` (default `2000000`)

## Tests

`backend/tests/test_knowledge.py` covers the chunker as pure functions and runs
the pipeline against the in-memory fakes in `backend/tests/conftest.py`; no
Qdrant server or ML model is needed for pytest.
