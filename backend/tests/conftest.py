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
from app.dependencies.inference import get_inference_service
from app.dependencies.knowledge import get_knowledge_config, get_knowledge_vector_store
from app.dependencies.memory import get_embedding_service, get_vector_store
from app.inference.schemas.inference import LLMResponse
from app.inference.services.inference_service import InferenceService
from app.knowledge.config import KnowledgeConfig
from app.main import create_app
from app.memory.services.vector_store import VectorHit

# The suite runs against an ephemeral in-memory SQLite database by default so
# it stays isolated from the PostgreSQL instance configured through
# DATABASE_URL; it therefore carries no host, user, password or port. Point the
# TEST_DATABASE_URL environment variable at a disposable database to run the
# same suite against a real server.
TEST_DATABASE_URL: str = os.environ.get("TEST_DATABASE_URL", "sqlite+aiosqlite:///:memory:")
FAKE_EMBEDDING_DIMENSION = 32

# Every endpoint test that chats gets this reply instead of a model call: no test
# in the suite reaches an LLM provider, running or not. See app/inference/README.md
# for the provider contract this stub satisfies.
FAKE_LLM_PROVIDER_NAME = "fake"
FAKE_LLM_MODEL = "fake-model"
FAKE_LLM_REPLY = "fake provider reply"
FAKE_LLM_INPUT_TOKENS = 12
FAKE_LLM_OUTPUT_TOKENS = 4
FAKE_LLM_LATENCY_MS = 5


class FakeLLMProvider:
    """Provider that replays one canned completion and records the prompts it was sent."""

    def __init__(self, content: str = FAKE_LLM_REPLY, error: Exception | None = None) -> None:
        self._content = content
        self._error = error
        self.calls: list[tuple[str, str, str]] = []

    @property
    def name(self) -> str:
        return FAKE_LLM_PROVIDER_NAME

    @property
    def model(self) -> str:
        return FAKE_LLM_MODEL

    async def generate(self, system_prompt: str, user_prompt: str, context: str = "") -> LLMResponse:
        self.calls.append((system_prompt, user_prompt, context))
        if self._error is not None:
            raise self._error
        return LLMResponse(
            content=self._content,
            model=FAKE_LLM_MODEL,
            provider=FAKE_LLM_PROVIDER_NAME,
            input_tokens_estimate=FAKE_LLM_INPUT_TOKENS,
            output_tokens_estimate=FAKE_LLM_OUTPUT_TOKENS,
            latency_ms=FAKE_LLM_LATENCY_MS,
        )


def fake_inference_service(provider: FakeLLMProvider | None = None) -> InferenceService:
    """Wire an inference service around a fake provider, bypassing ``LLM_PROVIDER``."""
    return InferenceService(provider=provider if provider is not None else FakeLLMProvider())


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

    async def search(
        self,
        vector: list[float],
        limit: int,
        user_id: int,
        must_match: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        hits: list[VectorHit] = []
        for point_id, (stored, payload) in self._points.items():
            if payload.get("user_id") != user_id:
                continue
            if not _matches(payload, must_match):
                continue
            hits.append(VectorHit(point_id=point_id, score=_cosine_similarity(vector, stored), payload=dict(payload)))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]

    async def delete(self, point_id: str) -> None:
        self._points.pop(point_id, None)


def _matches(payload: dict[str, Any], must_match: dict[str, Any] | None) -> bool:
    return all(payload.get(key) == value for key, value in (must_match or {}).items())


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
    fake_knowledge_vectors = FakeVectorStore()
    app.dependency_overrides[get_embedding_service] = lambda: fake_embedding
    app.dependency_overrides[get_vector_store] = lambda: fake_vectors
    app.dependency_overrides[get_knowledge_vector_store] = lambda: fake_knowledge_vectors
    app.dependency_overrides[get_knowledge_config] = lambda: KnowledgeConfig(
        chunk_max_chars=200,
        chunk_overlap=40,
        max_document_chars=5000,
        max_upload_bytes=10_000,
    )
    app.dependency_overrides[get_inference_service] = lambda: fake_inference_service()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
