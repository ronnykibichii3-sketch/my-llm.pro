"""Dependency providers for the memory layer.

Providers are cached so the embedding model and the Qdrant client are
created once per process. Tests override them via FastAPI dependency
overrides to substitute in-memory fakes.
"""

from functools import lru_cache

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.database.session import get_db_session
from app.memory.services.embedding_service import EmbeddingService, SentenceTransformerEmbeddingService
from app.memory.services.memory_service import MemoryService
from app.memory.services.vector_store import QdrantVectorStore, VectorStore


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Return the process-wide embedding backend."""
    return SentenceTransformerEmbeddingService()


@lru_cache
def get_vector_store() -> VectorStore:
    """Return the process-wide Qdrant-backed vector store."""
    settings = get_settings()
    api_key: str | None = None
    if settings.qdrant_api_key is not None:
        api_key = settings.qdrant_api_key.get_secret_value() or None
    return QdrantVectorStore(url=settings.qdrant_url, collection_name=settings.memory_collection_name, api_key=api_key)


def get_memory_service(
    session: AsyncSession = Depends(get_db_session),
    embedding: EmbeddingService = Depends(get_embedding_service),
    vectors: VectorStore = Depends(get_vector_store),
) -> MemoryService:
    """Build a request-scoped memory service wired to the configured backends."""
    return MemoryService(session, embedding, vectors)