"""Embedding generation encapsulated behind a swappable interface.

The concrete implementation uses sentence-transformers with
all-MiniLM-L6-v2. The model is loaded lazily on first use so that
imports and service construction never pull in the heavy ML stack.
Replacing the model in the future only requires a new implementation
of the ``EmbeddingService`` protocol (and an updated dimension).
"""

import asyncio
from collections.abc import Sequence
from threading import Lock
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from app.core.logging import logger

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@runtime_checkable
class EmbeddingService(Protocol):
    """Contract for turning text into dense vector representations."""

    @property
    def dimension(self) -> int:
        """Number of dimensions produced by this embedder."""
        ...

    async def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        ...

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts, preserving order."""
        ...


class SentenceTransformerEmbeddingService:
    """Default embedding backend backed by sentence-transformers."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, dimension: int = EMBEDDING_DIMENSION) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._model: "SentenceTransformer | None" = None
        self._lock = Lock()

    @property
    def dimension(self) -> int:
        return self._dimension

    def _ensure_loaded(self) -> "SentenceTransformer":
        """Load the model on first use; thread-safe for concurrent workers."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    logger.info("Loading embedding model %s", self._model_name)
                    self._model = SentenceTransformer(self._model_name)
        return self._model

    def _encode_one(self, text: str) -> list[float]:
        model = self._ensure_loaded()
        vector = model.encode(text)
        return [float(value) for value in vector]

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_loaded()
        vectors = model.encode(texts)
        return [[float(value) for value in vector] for vector in vectors]

    async def embed(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._encode_one, text)

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode_batch, list(texts))