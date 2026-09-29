"""Dependency providers for the knowledge base.

The knowledge vectors live in their own Qdrant collection so a knowledge search
can never return chat memories, or vice versa, while both collections reuse the
same swappable ``VectorStore`` and ``EmbeddingService`` implementations.
"""

from functools import lru_cache

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.database.session import get_db_session
from app.dependencies.memory import get_embedding_service
from app.knowledge.config import KnowledgeConfig
from app.knowledge.services.knowledge_service import KnowledgeService
from app.memory.services.embedding_service import EmbeddingService
from app.memory.services.vector_store import QdrantVectorStore, VectorStore


@lru_cache
def get_knowledge_config() -> KnowledgeConfig:
    """Return the process-wide knowledge tunables."""
    return KnowledgeConfig.from_settings(get_settings())


@lru_cache
def get_knowledge_vector_store() -> VectorStore:
    """Return the process-wide vector store bound to the knowledge collection."""
    settings = get_settings()
    api_key: str | None = None
    if settings.qdrant_api_key is not None:
        api_key = settings.qdrant_api_key.get_secret_value() or None
    return QdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=settings.knowledge_collection_name,
        api_key=api_key,
    )


def get_knowledge_service(
    session: AsyncSession = Depends(get_db_session),
    embedding: EmbeddingService = Depends(get_embedding_service),
    vectors: VectorStore = Depends(get_knowledge_vector_store),
    config: KnowledgeConfig = Depends(get_knowledge_config),
) -> KnowledgeService:
    """Build a request-scoped knowledge service wired to the configured backends."""
    return KnowledgeService(session, embedding, vectors, config)
