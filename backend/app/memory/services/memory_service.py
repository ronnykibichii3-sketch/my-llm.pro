"""Memory orchestration: embedding, vector storage, and metadata persistence.

Pipeline (see app/memory/README.md):
    message created -> embedding generated -> stored in Qdrant
                    -> metadata stored in PostgreSQL
"""

import uuid

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import logger
from app.database.models.memory import DEFAULT_IMPORTANCE_SCORE, Memory
from app.memory.repositories.memory_repository import MemoryRepository
from app.memory.schemas.memory import MemoryPayload, MemoryRead, MemorySearchResult, ReindexResult
from app.memory.services.embedding_service import EmbeddingService
from app.memory.services.vector_store import VectorStore
from app.repositories.message_repository import MessageRepository


class MemoryService:
    """Coordinates the memory pipeline across Qdrant and PostgreSQL."""

    def __init__(self, session: AsyncSession, embedding: EmbeddingService, vectors: VectorStore) -> None:
        self._memories = MemoryRepository(session)
        self._messages = MessageRepository(session)
        self._embedding = embedding
        self._vectors = vectors

    async def list_memories(self, user_id: int, limit: int | None = None) -> list[Memory]:
        """List a user's memories, newest first."""
        return await self._memories.list_by_user(user_id, limit)

    async def create_memory(
        self,
        user_id: int,
        conversation_id: int,
        message_id: int,
        content: str,
        importance_score: float = DEFAULT_IMPORTANCE_SCORE,
    ) -> Memory:
        """Run the full pipeline: embed, store the vector, persist metadata."""
        embedding_id = str(uuid.uuid4())
        vector = await self._embedding.embed(content)
        payload = self._build_payload(user_id, conversation_id, message_id, content, importance_score)
        await self._vectors.ensure_collection(self._embedding.dimension)
        await self._vectors.upsert(embedding_id, vector, payload)
        return await self._memories.create(
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=message_id,
            content=content,
            embedding_id=embedding_id,
            importance_score=importance_score,
        )

    async def search_memory(self, user_id: int, query: str, limit: int) -> list[MemorySearchResult]:
        """Vector-search a user's memories and hydrate results from PostgreSQL."""
        vector = await self._embedding.embed(query)
        await self._vectors.ensure_collection(self._embedding.dimension)
        hits = await self._vectors.search(vector, limit, user_id)
        if not hits:
            return []
        memories = await self._memories.get_by_embedding_ids([hit.point_id for hit in hits])
        by_embedding_id = {memory.embedding_id: memory for memory in memories}
        results: list[MemorySearchResult] = []
        for hit in hits:
            memory = by_embedding_id.get(hit.point_id)
            if memory is None:
                continue
            hydrated = MemoryRead.model_validate(memory)
            results.append(MemorySearchResult(**hydrated.model_dump(), score=hit.score))
        return results

    async def delete_memory(self, user_id: int, memory_id: int) -> None:
        """Delete a memory's vector point and metadata; 404 if not owned by the user."""
        memory = await self._memories.get_for_user(memory_id, user_id)
        if memory is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")
        try:
            await self._vectors.delete(memory.embedding_id)
        except Exception:
            logger.warning("Failed to delete vector point %s", memory.embedding_id, exc_info=True)
        await self._memories.delete(memory)

    async def reindex_memory(self, user_id: int) -> ReindexResult:
        """Backfill memories for messages that have none, and re-embed every existing memory."""
        await self._vectors.ensure_collection(self._embedding.dimension)
        messages = await self._messages.get_messages_for_user(user_id)
        existing = await self._memories.list_by_user(user_id)
        memories_by_message = {memory.message_id: memory for memory in existing}
        missing = [message for message in messages if message.id not in memories_by_message]

        texts = [memory.content for memory in existing] + [message.content for message in missing]
        vectors = await self._embedding.embed_batch(texts)

        reindexed = 0
        for memory, vector in zip(existing, vectors[: len(existing)], strict=True):
            payload = self._build_payload(
                memory.user_id,
                memory.conversation_id,
                memory.message_id,
                memory.content,
                memory.importance_score,
            )
            await self._vectors.upsert(memory.embedding_id, vector, payload)
            reindexed += 1

        created = 0
        offset = len(existing)
        for index, message in enumerate(missing):
            embedding_id = str(uuid.uuid4())
            payload = self._build_payload(user_id, message.conversation_id, message.id, message.content, DEFAULT_IMPORTANCE_SCORE)
            await self._vectors.upsert(embedding_id, vectors[offset + index], payload)
            await self._memories.create(
                user_id=user_id,
                conversation_id=message.conversation_id,
                message_id=message.id,
                content=message.content,
                embedding_id=embedding_id,
                importance_score=DEFAULT_IMPORTANCE_SCORE,
            )
            created += 1

        logger.info("Memory reindex for user %s: created=%s reindexed=%s", user_id, created, reindexed)
        return ReindexResult(created=created, reindexed=reindexed)

    @staticmethod
    def _build_payload(
        user_id: int,
        conversation_id: int,
        message_id: int,
        content: str,
        importance_score: float,
    ) -> MemoryPayload:
        return MemoryPayload(
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=message_id,
            content=content,
            importance_score=importance_score,
        )

    async def try_create_memory(
        self,
        user_id: int,
        conversation_id: int,
        message_id: int,
        content: str,
        importance_score: float = DEFAULT_IMPORTANCE_SCORE,
    ) -> Memory | None:
        """Best-effort ingestion: a vector store outage must never break the caller."""
        try:
            return await self.create_memory(user_id, conversation_id, message_id, content, importance_score)
        except Exception:
            logger.warning("Memory ingestion failed for message %s", message_id, exc_info=True)
            return None
