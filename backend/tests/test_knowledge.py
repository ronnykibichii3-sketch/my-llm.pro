"""Tests for the knowledge base: chunking, ingestion pipeline and endpoints."""

import pytest
from fastapi import HTTPException, status
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models.knowledge import KnowledgeDocument
from app.database.models.user import User
from app.knowledge.config import KnowledgeConfig
from app.knowledge.schemas.knowledge import DocumentCreate
from app.knowledge.services.chunker import chunk_text
from app.knowledge.services.knowledge_service import KnowledgeService
from tests.conftest import FakeEmbeddingService, FakeVectorStore

TEST_CONFIG = KnowledgeConfig(chunk_max_chars=200, chunk_overlap=40, max_document_chars=5000, max_upload_bytes=10_000)


class FailingVectorStore(FakeVectorStore):
    """Vector store whose upserts fail on the third point, mid-ingest."""

    def __init__(self) -> None:
        super().__init__()
        self.upserts = 0

    async def upsert(self, point_id: str, vector: list[float], payload: dict[str, object]) -> None:
        self.upserts += 1
        if self.upserts >= 3:
            raise RuntimeError("qdrant unavailable")
        await super().upsert(point_id, vector, payload)


async def _register_and_login(
    client: AsyncClient, email: str, username: str, password: str = "securepassword123"
) -> str:
    """Helper to create a user and return an auth token."""
    await client.post("/api/v1/auth/register", json={"email": email, "username": username, "password": password})
    response = await client.post("/api/v1/auth/login", json={"email": email, "password": password})
    return response.json()["access_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _document_payload(title: str = "Gardening notes", **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "title": title,
        "content": "Tomatoes need sun. Basil likes warmth.\n\nCompost feeds the soil.",
        "content_type": "text/plain",
    }
    payload.update(overrides)
    return payload


def _create(title: str = "Service notes", content: str = "Tomatoes need sun.", **overrides: object) -> DocumentCreate:
    """Build the request schema used by the service-level tests."""
    return DocumentCreate.model_validate(
        {"title": title, "content": content, "content_type": "text/plain"} | dict(overrides)
    )


async def _seed_user(session: AsyncSession, email: str = "knowledge-service@example.com") -> int:
    """Create a user row directly for service-level tests."""
    user = User(email=email, username="knowledgeservice", hashed_password="not-a-real-hash")
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user.id


def _long_document(paragraphs: int, sentences: int) -> str:
    sentence = "The quick brown fox jumps over the lazy dog near the river bank."
    return "\n\n".join(f"Paragraph {index}: " + " ".join(sentence for _ in range(sentences)) for index in range(paragraphs))


def test_chunk_text_keeps_short_document_as_single_chunk() -> None:
    chunks = chunk_text("Short and simple.", 200, 40)
    assert chunks == ["Short and simple."]


def test_chunk_text_never_exceeds_max_chars() -> None:
    chunks = chunk_text(_long_document(12, 6), 200, 40)
    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_chunk_text_is_deterministic() -> None:
    document = _long_document(4, 4)
    assert chunk_text(document, 180, 30) == chunk_text(document, 180, 30)


def test_chunk_text_splits_and_keeps_every_line() -> None:
    document = _long_document(8, 5)
    chunks = chunk_text(document, 220, 40)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    for index in range(8):
        assert any(f"Paragraph {index}:" in chunk for chunk in chunks)


def test_chunk_text_carries_overlap_between_chunks() -> None:
    """Each chunk after the first begins with a tail of its predecessor."""
    document = "\n\n".join(f"Item {index} records a distinct gardening fact." for index in range(30))
    chunks = chunk_text(document, 140, 45)
    assert len(chunks) > 1
    for previous, following in zip(chunks, chunks[1:], strict=False):
        assert _shared_span(previous, following) >= 10


def _shared_span(previous: str, following: str) -> int:
    """Length of the longest suffix of ``previous`` that prefixes ``following``."""
    for size in range(min(len(previous), len(following)), 0, -1):
        if previous.endswith(following[:size]):
            return size
    return 0


def test_chunk_text_rejects_non_positive_max_chars() -> None:
    with pytest.raises(ValueError):
        chunk_text("anything", 0, 0)


def test_chunk_text_normalises_line_endings() -> None:
    assert chunk_text("one\r\n\r\ntwo", 200, 0) == chunk_text("one\n\ntwo", 200, 0)



@pytest.mark.asyncio
async def test_knowledge_endpoints_require_auth(client: AsyncClient) -> None:
    assert (await client.get("/api/v1/knowledge/documents")).status_code == status.HTTP_401_UNAUTHORIZED
    assert (await client.post("/api/v1/knowledge/documents", json=_document_payload())).status_code == (
        status.HTTP_401_UNAUTHORIZED
    )
    assert (await client.get("/api/v1/knowledge/search", params={"q": "tomatoes"})).status_code == (
        status.HTTP_401_UNAUTHORIZED
    )


@pytest.mark.asyncio
async def test_ingest_document_creates_ready_record(client: AsyncClient) -> None:
    token = await _register_and_login(client, "knowledge@example.com", "knowledgeuser")
    response = await client.post("/api/v1/knowledge/documents", json=_document_payload(), headers=_auth(token))
    assert response.status_code == status.HTTP_201_CREATED
    document = response.json()
    assert document["title"] == "Gardening notes"
    assert document["status"] == KnowledgeDocument.STATUS_READY
    assert document["chunk_count"] >= 1
    assert document["char_count"] == len(_document_payload()["content"])
    assert document["error"] is None
    assert document["content_type"] == "text/plain"


@pytest.mark.asyncio
async def test_list_documents_newest_first(client: AsyncClient) -> None:
    token = await _register_and_login(client, "knowledge@example.com", "knowledgeuser")
    await client.post("/api/v1/knowledge/documents", json=_document_payload("First note"), headers=_auth(token))
    await client.post("/api/v1/knowledge/documents", json=_document_payload("Second note"), headers=_auth(token))

    response = await client.get("/api/v1/knowledge/documents", headers=_auth(token))
    assert response.status_code == status.HTTP_200_OK
    titles = [document["title"] for document in response.json()]
    assert titles == ["Second note", "First note"]


@pytest.mark.asyncio
async def test_get_document_detail_lists_chunks_in_order(client: AsyncClient) -> None:
    token = await _register_and_login(client, "knowledge@example.com", "knowledgeuser")
    created = await client.post(
        "/api/v1/knowledge/documents",
        json=_document_payload(content=_long_document(6, 6)),
        headers=_auth(token),
    )
    document_id = created.json()["id"]

    response = await client.get(f"/api/v1/knowledge/documents/{document_id}", headers=_auth(token))
    assert response.status_code == status.HTTP_200_OK
    detail = response.json()
    positions = [chunk["position"] for chunk in detail["chunks"]]
    assert positions == list(range(len(positions)))
    assert detail["chunk_count"] == len(positions)
    assert set(detail["chunks"][0]) == {"id", "document_id", "position", "content", "char_count", "created_at"}


@pytest.mark.asyncio
async def test_delete_document_removes_rows_and_vectors(client: AsyncClient) -> None:
    token = await _register_and_login(client, "knowledge@example.com", "knowledgeuser")
    created = await client.post("/api/v1/knowledge/documents", json=_document_payload(), headers=_auth(token))
    document_id = created.json()["id"]

    response = await client.delete(f"/api/v1/knowledge/documents/{document_id}", headers=_auth(token))
    assert response.status_code == status.HTTP_204_NO_CONTENT

    assert (await client.get(f"/api/v1/knowledge/documents/{document_id}", headers=_auth(token))).status_code == (
        status.HTTP_404_NOT_FOUND
    )
    assert (await client.get("/api/v1/knowledge/search", params={"q": "tomatoes"}, headers=_auth(token))).json() == []


@pytest.mark.asyncio
async def test_documents_scoped_per_user(client: AsyncClient) -> None:
    token_a = await _register_and_login(client, "alice@example.com", "alice")
    token_b = await _register_and_login(client, "bob@example.com", "bob")
    created = await client.post("/api/v1/knowledge/documents", json=_document_payload(), headers=_auth(token_a))
    document_id = created.json()["id"]

    assert (await client.get("/api/v1/knowledge/documents", headers=_auth(token_b))).json() == []
    assert (await client.get(f"/api/v1/knowledge/documents/{document_id}", headers=_auth(token_b))).status_code == (
        status.HTTP_404_NOT_FOUND
    )
    assert (await client.delete(f"/api/v1/knowledge/documents/{document_id}", headers=_auth(token_b))).status_code == (
        status.HTTP_404_NOT_FOUND
    )
    assert (await client.get("/api/v1/knowledge/search", params={"q": "tomatoes"}, headers=_auth(token_b))).json() == []


@pytest.mark.asyncio
async def test_search_ranks_relevant_chunk_first(client: AsyncClient) -> None:
    token = await _register_and_login(client, "knowledge@example.com", "knowledgeuser")
    await client.post(
        "/api/v1/knowledge/documents",
        json=_document_payload("Gardening", content="Tomatoes need sun and water. Basil likes warmth."),
        headers=_auth(token),
    )
    await client.post(
        "/api/v1/knowledge/documents",
        json=_document_payload("Finance", content="Quarterly financial reports are filed in April."),
        headers=_auth(token),
    )

    response = await client.get("/api/v1/knowledge/search", params={"q": "tomatoes basil warmth"}, headers=_auth(token))
    assert response.status_code == status.HTTP_200_OK
    results = response.json()
    assert results
    assert "Tomatoes" in results[0]["content"]
    assert results[0]["document_title"] == "Gardening"
    assert results[0]["score"] >= results[-1]["score"]
    assert set(results[0]) == {
        "score",
        "document_id",
        "document_title",
        "chunk_id",
        "position",
        "content",
        "char_count",
    }


@pytest.mark.asyncio
async def test_search_narrowed_to_one_document(client: AsyncClient) -> None:
    token = await _register_and_login(client, "knowledge@example.com", "knowledgeuser")
    garden = await client.post(
        "/api/v1/knowledge/documents",
        json=_document_payload("Gardening", content="Tomatoes need sun and water."),
        headers=_auth(token),
    )
    await client.post(
        "/api/v1/knowledge/documents",
        json=_document_payload("Finance", content="Tomatoes are a budget line item this quarter."),
        headers=_auth(token),
    )
    document_id = garden.json()["id"]

    response = await client.get(
        "/api/v1/knowledge/search",
        params={"q": "tomatoes", "document_id": document_id},
        headers=_auth(token),
    )
    assert response.status_code == status.HTTP_200_OK
    results = response.json()
    assert results
    assert {result["document_id"] for result in results} == {document_id}


@pytest.mark.asyncio
async def test_search_requires_query(client: AsyncClient) -> None:
    token = await _register_and_login(client, "knowledge@example.com", "knowledgeuser")
    assert (await client.get("/api/v1/knowledge/search", headers=_auth(token))).status_code == (
        status.HTTP_422_UNPROCESSABLE_ENTITY
    )


@pytest.mark.asyncio
async def test_ingest_rejects_oversized_document(client: AsyncClient) -> None:
    token = await _register_and_login(client, "knowledge@example.com", "knowledgeuser")
    response = await client.post(
        "/api/v1/knowledge/documents",
        json=_document_payload(content="sunlight " * 900),
        headers=_auth(token),
    )
    assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE


@pytest.mark.asyncio
async def test_ingest_rejects_unsupported_content_type(client: AsyncClient) -> None:
    token = await _register_and_login(client, "knowledge@example.com", "knowledgeuser")
    response = await client.post(
        "/api/v1/knowledge/documents",
        json=_document_payload(content_type="application/pdf"),
        headers=_auth(token),
    )
    assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE


@pytest.mark.asyncio
async def test_ingest_rejects_document_without_text(client: AsyncClient) -> None:
    token = await _register_and_login(client, "knowledge@example.com", "knowledgeuser")
    response = await client.post(
        "/api/v1/knowledge/documents", json=_document_payload(content="   \n\n   "), headers=_auth(token)
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert (await client.get("/api/v1/knowledge/documents", headers=_auth(token))).json() == []


@pytest.mark.asyncio
async def test_raw_body_upload(client: AsyncClient) -> None:
    """Raw request bodies work without python-multipart: curl --data-binary."""
    token = await _register_and_login(client, "knowledge@example.com", "knowledgeuser")
    response = await client.post(
        "/api/v1/knowledge/documents/raw",
        params={"title": "Uploaded notes"},
        content=b"# Notes\n\nTomatoes need sun and water.",
        headers={**_auth(token), "Content-Type": "text/plain"},
    )
    assert response.status_code == status.HTTP_201_CREATED
    document = response.json()
    assert document["title"] == "Uploaded notes"
    assert document["status"] == KnowledgeDocument.STATUS_READY


@pytest.mark.asyncio
async def test_raw_body_upload_derives_title(client: AsyncClient) -> None:
    token = await _register_and_login(client, "knowledge@example.com", "knowledgeuser")
    response = await client.post(
        "/api/v1/knowledge/documents/raw",
        content=b"# Marrow planting plan\n\nSow in April.",
        headers={**_auth(token), "Content-Type": "text/markdown"},
    )
    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["title"] == "Marrow planting plan"


@pytest.mark.asyncio
async def test_raw_body_upload_rejects_binary(client: AsyncClient) -> None:
    token = await _register_and_login(client, "knowledge@example.com", "knowledgeuser")
    response = await client.post(
        "/api/v1/knowledge/documents/raw",
        content=b"\x89PNG\x00\x01\x02",
        headers={**_auth(token), "Content-Type": "image/png"},
    )
    assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE


@pytest.mark.asyncio
async def test_raw_body_upload_rejects_oversized_body(client: AsyncClient) -> None:
    token = await _register_and_login(client, "knowledge@example.com", "knowledgeuser")
    response = await client.post(
        "/api/v1/knowledge/documents/raw",
        content=b"sunlight " * 1100,
        headers={**_auth(token), "Content-Type": "text/plain"},
    )
    assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE


@pytest.mark.asyncio
async def test_service_ingest_writes_vectors_then_metadata(db_session: AsyncSession) -> None:
    """The pipeline embeds, indexes every chunk, then marks the document ready."""
    vectors = FakeVectorStore()
    service = KnowledgeService(db_session, FakeEmbeddingService(), vectors, TEST_CONFIG)
    user_id = await _seed_user(db_session)
    content = _long_document(4, 4)

    document = await service.ingest_document(
        user_id,
        _create(title="Pipeline notes", content=content),
    )

    chunks = await service.list_chunks(user_id, document.id)
    assert document.status == KnowledgeDocument.STATUS_READY
    assert document.char_count == len(content)
    assert document.chunk_count == len(chunks)
    assert len(vectors._points) == len(chunks)
    assert vectors.collection_ready is True

    first_chunk = chunks[0]
    _, stored_payload = vectors._points[first_chunk.embedding_id]
    assert stored_payload["user_id"] == user_id
    assert stored_payload["document_id"] == document.id
    assert stored_payload["position"] == 0
    assert stored_payload["kind"] == "knowledge_chunk"
    assert stored_payload["content"] == first_chunk.content


@pytest.mark.asyncio
async def test_service_rolls_back_vectors_and_marks_failed(db_session: AsyncSession) -> None:
    """A vector-store failure leaves no indexed points and a failed document row."""
    vectors = FailingVectorStore()
    service = KnowledgeService(db_session, FakeEmbeddingService(), vectors, TEST_CONFIG)
    user_id = await _seed_user(db_session)

    with pytest.raises(HTTPException) as exc_info:
        await service.ingest_document(user_id, _create(content=_long_document(6, 6)))
    assert exc_info.value.status_code == status.HTTP_502_BAD_GATEWAY

    documents = await service.list_documents(user_id, limit=10)
    assert len(documents) == 1
    assert documents[0].status == KnowledgeDocument.STATUS_FAILED
    assert "RuntimeError" in (documents[0].error or "")
    assert documents[0].chunk_count == 0
    assert vectors._points == {}
    assert await service.list_chunks(user_id, documents[0].id) == []


@pytest.mark.asyncio
async def test_service_search_narrows_by_document(db_session: AsyncSession) -> None:
    vectors = FakeVectorStore()
    service = KnowledgeService(db_session, FakeEmbeddingService(), vectors, TEST_CONFIG)
    user_id = await _seed_user(db_session)
    garden = await service.ingest_document(user_id, _create(title="Garden", content="Tomatoes need sun and water."))
    await service.ingest_document(user_id, _create(title="Finance", content="Tomatoes cost more this quarter."))

    all_hits = await service.search(user_id, "tomatoes", limit=10)
    scoped = await service.search(user_id, "tomatoes", limit=10, document_id=garden.id)

    assert len(all_hits) > len(scoped) >= 1
    assert {hit.document_id for hit in scoped} == {garden.id}


@pytest.mark.asyncio
async def test_service_delete_removes_vectors_and_rows(db_session: AsyncSession) -> None:
    vectors = FakeVectorStore()
    service = KnowledgeService(db_session, FakeEmbeddingService(), vectors, TEST_CONFIG)
    user_id = await _seed_user(db_session)
    document = await service.ingest_document(user_id, _create(content=_long_document(4, 4)))

    await service.delete_document(user_id, document.id)

    assert vectors._points == {}
    assert await service.list_documents(user_id, limit=10) == []

