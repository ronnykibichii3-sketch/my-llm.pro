"""Vector storage behind a swappable interface, backed by Qdrant.

The concrete store lazily imports ``qdrant-client`` and runs its
synchronous API on a worker thread so application code stays fully
async. All points are filtered by the ``user_id`` payload field to
keep search strictly scoped to a single user.
"""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

FILTER_USER_ID_KEY = "user_id"


@dataclass(frozen=True)
class VectorHit:
    """A single search result returned from the vector store."""

    point_id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class VectorStore(Protocol):
    """Contract for persisting and querying embedding vectors."""

    async def ensure_collection(self, vector_size: int) -> None:
        """Create the target collection if it does not exist yet."""
        ...

    async def upsert(self, point_id: str, vector: list[float], payload: Mapping[str, Any]) -> None:
        """Insert or replace a single vector point with its payload."""
        ...

    async def search(self, vector: list[float], limit: int, user_id: int) -> list[VectorHit]:
        """Return the top-scoring points for a user, best score first."""
        ...

    async def delete(self, point_id: str) -> None:
        """Remove a single vector point; missing points are ignored."""
        ...


class QdrantVectorStore:
    """Qdrant-backed implementation of the vector store contract."""

    def __init__(self, url: str, collection_name: str, api_key: str | None = None) -> None:
        self._url = url
        self._api_key = api_key
        self._collection_name = collection_name
        self._client: "QdrantClient | None" = None
        self._lock = asyncio.Lock()

    def _get_client(self) -> "QdrantClient":
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(url=self._url, api_key=self._api_key)
        return self._client

    async def ensure_collection(self, vector_size: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        async with self._lock:
            client = self._get_client()
            exists = await asyncio.to_thread(client.collection_exists, self._collection_name)
            if not exists:
                await asyncio.to_thread(
                    client.create_collection,
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )

    async def upsert(self, point_id: str, vector: list[float], payload: Mapping[str, Any]) -> None:
        from qdrant_client.models import PointStruct

        client = self._get_client()
        await asyncio.to_thread(
            client.upsert,
            collection_name=self._collection_name,
            points=[PointStruct(id=point_id, vector=vector, payload=dict(payload))],
            wait=True,
        )

    async def search(self, vector: list[float], limit: int, user_id: int) -> list[VectorHit]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        client = self._get_client()
        query_filter = Filter(must=[FieldCondition(key=FILTER_USER_ID_KEY, match=MatchValue(value=user_id))])
        response = await asyncio.to_thread(
            client.query_points,
            collection_name=self._collection_name,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return [
            VectorHit(point_id=str(point.id), score=float(point.score), payload=dict(point.payload or {}))
            for point in response.points
        ]

    async def delete(self, point_id: str) -> None:
        client = self._get_client()
        await asyncio.to_thread(
            client.delete,
            collection_name=self._collection_name,
            points_selector=[point_id],
            wait=True,
        )