"""Pytest configuration and fixtures."""

import math
import os
import re
import zlib
from collections.abc import AsyncGenerator, Sequence
from typing import Any

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database.base import Base
from app.database.session import get_db_session
from app.dependencies.memory import get_embedding_service, get_vector_store
from app.main import create_app
from app.memory.services.vector_store import VectorHit

# The suite runs against an ephemeral in-memory SQLite database by default so
# it stays isolated from the PostgreSQL instance configured through
# DATABASE_URL; it therefore carries no host, user, password or port. Point the
# TEST_DATABASE_URL environment variable at a disposable database to run the
# same suite against a real server.
TEST_DATABASE_URL: str = os.environ.get("TEST_DATABASE_URL", "sqlite+aiosqlite:///:memory:")
FAKE_EMBEDDING_DIMENSION = 32


class FakeEmbeddingService:
    """Deterministic bag-of-words embedder; identical texts embed identically."""

    @property
    def dimension(self) -> int:
        return FAKE_EMBEDDING_DIMENSION

    def _encode_one(self, text: str) -> list[float]:
        vector = [0.0] * FAKE_EMBEDDING_DIMENSION
        for token in re.findall(r"\w+", text.lower()):
            vector[zlib.crc32(token.encode("utf-8")) % FAKE_EMBEDDING_DIMENSION] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    async def embed(self, text: str) -> list[float]:
        return self._encode_one(text)

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._encode_one(text) for text in texts]


class FakeVectorStore:
    """In-memory vector store with cosine scoring and user-scoped filtering."""

    def __init__(self) -> None:
        self._points: dict[str, tuple[list[float], dict[str, Any]]] = {}
        self.collection_ready = False

    async def ensure_collection(self, vector_size: int) -> None:
        self.collection_ready = True

    async def upsert(self, point_id: str, vector: list[float], payload: dict[str, Any]) -> None:
        self._points[point_id] = (list(vector), dict(payload))

    async def search(self, vector: list[float], limit: int, user_id: int) -> list[VectorHit]:
        hits: list[VectorHit] = []
        for point_id, (stored, payload) in self._points.items():
            if payload.get("user_id") != user_id:
                continue
            hits.append(VectorHit(point_id=point_id, score=_cosine_similarity(vector, stored), payload=dict(payload)))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]

    async def delete(self, point_id: str) -> None:
        self._points.pop(point_id, None)


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a fresh database session for each test."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create a test client with overridden database and memory backends."""
    app = create_app()

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db
    fake_embedding = FakeEmbeddingService()
    fake_vectors = FakeVectorStore()
    app.dependency_overrides[get_embedding_service] = lambda: fake_embedding
    app.dependency_overrides[get_vector_store] = lambda: fake_vectors

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
