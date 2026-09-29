"""Knowledge base orchestration: chunk, embed, store vectors, persist metadata.

Pipeline (see app/knowledge/README.md):

    document received -> validated text -> chunked -> embedded
                      -> stored in Qdrant -> chunk and document rows in PostgreSQL

Ingestion is all-or-nothing from the caller's point of view: if the vector store
fails midway, the points written so far are removed and the document is left
marked ``failed`` with the reason, rather than half-indexed.
"""

import uuid
from collections.abc import Sequence
from typing import NoReturn

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import logger
from app.database.models.knowledge import KnowledgeChunk, KnowledgeDocument
from app.knowledge.config import (
    DELETED_DOCUMENT_TITLE,
    MAX_ERROR_LENGTH,
    PAYLOAD_DOCUMENT_ID_KEY,
    PAYLOAD_KIND_VALUE,
    KnowledgeConfig,
)
from app.knowledge.repositories.knowledge_repository import KnowledgeRepository
from app.knowledge.schemas.knowledge import (
    ChunkPlan,
    DocumentCreate,
    DocumentDraft,
    KnowledgePayload,
    KnowledgeSearchResult,
)
from app.knowledge.services.chunker import chunk_text
from app.knowledge.services.text_source import decode_text, ensure_supported, media_type_of, title_from_text
from app.memory.services.embedding_service import EmbeddingService
from app.memory.services.vector_store import VectorHit, VectorStore


class KnowledgeService:
    """Coordinates knowledge ingestion and retrieval across Qdrant and PostgreSQL."""

    def __init__(
        self,
        session: AsyncSession,
        embedding: EmbeddingService,
        vectors: VectorStore,
        config: KnowledgeConfig,
    ) -> None:
        self._documents = KnowledgeRepository(session)
        self._embedding = embedding
        self._vectors = vectors
        self._config = config

    async def list_documents(self, user_id: int, limit: int) -> list[KnowledgeDocument]:
        """List a user's documents, newest first."""
        return await self._documents.list_for_user(user_id, limit)

    async def get_document(self, user_id: int, document_id: int) -> KnowledgeDocument:
        """Fetch one document, scoped to its owner."""
        return await self._require_document(document_id, user_id)

    async def list_chunks(self, user_id: int, document_id: int) -> list[KnowledgeChunk]:
        """List a document's chunks in reading order."""
        await self._require_document(document_id, user_id)
        return await self._documents.list_chunks(document_id, user_id)

    async def ingest_document(self, user_id: int, payload: DocumentCreate) -> KnowledgeDocument:
        """Ingest a document supplied as a JSON body."""
        content_type = ensure_supported(payload.content_type)
        draft = DocumentDraft(
            user_id=user_id,
            title=payload.title.strip(),
            content=payload.content,
            source=payload.source,
            content_type=content_type,
        )
        return await self._ingest(draft)

    async def ingest_raw(
        self,
        user_id: int,
        raw: bytes,
        content_type: str | None,
        title: str | None,
    ) -> KnowledgeDocument:
        """Ingest a document supplied as a raw request body."""
        text = decode_text(raw, content_type, self._config.max_upload_bytes)
        draft = DocumentDraft(
            user_id=user_id,
            title=(title or "").strip() or title_from_text(text),
            content=text,
            content_type=media_type_of(content_type),
        )
        return await self._ingest(draft)

    async def delete_document(self, user_id: int, document_id: int) -> None:
        """Remove a document's vector points and its metadata rows."""
        document = await self._require_document(document_id, user_id)
        chunks = await self._documents.list_chunks(document_id, user_id)
        await self._delete_points([chunk.embedding_id for chunk in chunks])
        await self._documents.delete(document, chunks)
        logger.info("Deleted knowledge document %s for user %s", document_id, user_id)

    async def search(
        self,
        user_id: int,
        query: str,
        limit: int,
        document_id: int | None = None,
    ) -> list[KnowledgeSearchResult]:
        """Vector-search a user's knowledge, optionally narrowed to one document."""
        vector = await self._embedding.embed(query)
        await self._vectors.ensure_collection(self._embedding.dimension)
        must_match = {PAYLOAD_DOCUMENT_ID_KEY: document_id} if document_id is not None else None
        hits = await self._vectors.search(vector, limit, user_id, must_match)
        if not hits:
            return []
        chunks = await self._documents.get_chunks_by_embedding_ids([hit.point_id for hit in hits], user_id)
        titles = await self._titles_for(chunks, user_id)
        return _hydrate(hits, chunks, titles)

    async def _ingest(self, draft: DocumentDraft) -> KnowledgeDocument:
        self._validate_size(draft.content)
        plans = self._plan_chunks(draft.content)
        document = await self._documents.create(draft)
        logger.info("Ingesting knowledge document %s for user %s chunks=%s", document.id, draft.user_id, len(plans))
        stored: list[str] = []
        try:
            vectors = await self._embedding.embed_batch([plan.content for plan in plans])
            await self._vectors.ensure_collection(self._embedding.dimension)
            await self._upsert_all(document, plans, vectors, stored)
            await self._documents.add_chunks(document, plans)
        except Exception as error:
            await self._fail(document, stored, error)
        return await self._documents.mark_ready(document, len(draft.content), len(plans))

    async def _upsert_all(
        self,
        document: KnowledgeDocument,
        plans: list[ChunkPlan],
        vectors: list[list[float]],
        stored: list[str],
    ) -> None:
        for plan, vector in zip(plans, vectors, strict=True):
            await self._vectors.upsert(plan.embedding_id, vector, _payload(document, plan))
            stored.append(plan.embedding_id)

    async def _fail(self, document: KnowledgeDocument, stored: list[str], error: Exception) -> NoReturn:
        logger.error("Knowledge ingestion failed for document %s", document.id, exc_info=True)
        await self._delete_points(stored)
        await self._documents.mark_failed(document, f"{type(error).__name__}: {error}"[:MAX_ERROR_LENGTH])
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Knowledge ingestion failed; the document was marked as failed",
        ) from error

    async def _delete_points(self, point_ids: list[str]) -> None:
        for point_id in point_ids:
            try:
                await self._vectors.delete(point_id)
            except Exception:
                logger.warning("Failed to delete knowledge point %s", point_id, exc_info=True)

    async def _titles_for(self, chunks: list[KnowledgeChunk], user_id: int) -> dict[int, str]:
        document_ids = list({chunk.document_id for chunk in chunks})
        documents = await self._documents.get_by_ids(document_ids, user_id)
        return {document.id: document.title for document in documents}

    async def _require_document(self, document_id: int, user_id: int) -> KnowledgeDocument:
        document = await self._documents.get_for_user(document_id, user_id)
        if document is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge document not found")
        return document

    def _validate_size(self, content: str) -> None:
        if len(content) > self._config.max_document_chars:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Document exceeds the {self._config.max_document_chars} character limit",
            )

    def _plan_chunks(self, content: str) -> list[ChunkPlan]:
        pieces = chunk_text(content, self._config.chunk_max_chars, self._config.chunk_overlap)
        if not pieces:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Document contains no indexable text",
            )
        return [
            ChunkPlan(position=index, content=piece, embedding_id=str(uuid.uuid4()))
            for index, piece in enumerate(pieces)
        ]


def _payload(document: KnowledgeDocument, plan: ChunkPlan) -> KnowledgePayload:
    return KnowledgePayload(
        user_id=document.user_id,
        document_id=document.id,
        position=plan.position,
        content=plan.content,
        kind=PAYLOAD_KIND_VALUE,
    )


def _hydrate(
    hits: Sequence[VectorHit],
    chunks: Sequence[KnowledgeChunk],
    titles: dict[int, str],
) -> list[KnowledgeSearchResult]:
    """Attach scores, titles and stable ids to hits, preserving score order."""
    by_embedding_id = {chunk.embedding_id: chunk for chunk in chunks}
    results: list[KnowledgeSearchResult] = []
    for hit in hits:
        chunk = by_embedding_id.get(hit.point_id)
        if chunk is None:
            continue
        results.append(
            KnowledgeSearchResult(
                score=hit.score,
                document_id=chunk.document_id,
                document_title=titles.get(chunk.document_id, DELETED_DOCUMENT_TITLE),
                chunk_id=chunk.id,
                position=chunk.position,
                content=chunk.content,
                char_count=chunk.char_count,
            )
        )
    return results
