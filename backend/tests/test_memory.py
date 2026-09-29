"""Tests for the memory endpoints and the memory pipeline service."""

import pytest
from fastapi import HTTPException, status
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models.chat import Conversation, Message
from app.database.models.user import User
from app.memory.services.memory_service import MemoryService
from tests.conftest import FakeEmbeddingService, FakeVectorStore


async def _register_and_login(client: AsyncClient, email: str, username: str, password: str = "securepassword123") -> str:
    """Helper to create a user and return an auth token."""
    await client.post("/api/v1/auth/register", json={"email": email, "username": username, "password": password})
    response = await client.post("/api/v1/auth/login", json={"email": email, "password": password})
    return response.json()["access_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _create_conversation(client: AsyncClient, token: str, title: str = "Memory Test") -> int:
    response = await client.post("/api/v1/conversations", json={"title": title}, headers=_auth(token))
    return response.json()["id"]


async def _seed_content(session: AsyncSession, content: str) -> tuple[int, int, int]:
    """Create a user, conversation, and message row directly for service-level tests."""
    user = User(email="service@example.com", username="serviceuser", hashed_password="not-a-real-hash")
    session.add(user)
    await session.commit()
    await session.refresh(user)
    conversation = Conversation(user_id=user.id, title="Service Test")
    session.add(conversation)
    await session.commit()
    await session.refresh(conversation)
    message = Message(conversation_id=conversation.id, role=Message.ROLE_USER, content=content)
    session.add(message)
    await session.commit()
    await session.refresh(message)
    return user.id, conversation.id, message.id


@pytest.mark.asyncio
async def test_list_memories_requires_auth(client: AsyncClient) -> None:
    response = await client.get("/api/v1/memory")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
async def test_list_memories_empty(client: AsyncClient) -> None:
    token = await _register_and_login(client, "memory@example.com", "memoryuser")
    response = await client.get("/api/v1/memory", headers=_auth(token))
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == []


@pytest.mark.asyncio
async def test_chat_ingests_memories(client: AsyncClient) -> None:
    """Both messages created by chat run through the memory pipeline."""
    token = await _register_and_login(client, "memory@example.com", "memoryuser")
    conversation_id = await _create_conversation(client, token)
    chat_response = await client.post(
        f"/api/v1/conversations/{conversation_id}/chat",
        json={"content": "Hello there"},
        headers=_auth(token),
    )
    assert chat_response.status_code == status.HTTP_200_OK

    response = await client.get("/api/v1/memory", headers=_auth(token))
    assert response.status_code == status.HTTP_200_OK
    memories = response.json()
    assert len(memories) == 2
    contents = [memory["content"] for memory in memories]
    assert "Hello there" in contents
    first = memories[0]
    assert set(first) == {
        "id",
        "user_id",
        "conversation_id",
        "message_id",
        "content",
        "embedding_id",
        "importance_score",
        "created_at",
    }
    assert first["importance_score"] == 0.5
    assert first["embedding_id"]


@pytest.mark.asyncio
async def test_create_message_ingests_memory(client: AsyncClient) -> None:
    token = await _register_and_login(client, "memory@example.com", "memoryuser")
    conversation_id = await _create_conversation(client, token)
    post_response = await client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={"content": "Persist this into memory"},
        headers=_auth(token),
    )
    assert post_response.status_code == status.HTTP_201_CREATED

    response = await client.get("/api/v1/memory", headers=_auth(token))
    memories = response.json()
    assert len(memories) == 1
    assert memories[0]["content"] == "Persist this into memory"
    assert memories[0]["message_id"] == post_response.json()["id"]


@pytest.mark.asyncio
async def test_memories_scoped_per_user(client: AsyncClient) -> None:
    token_a = await _register_and_login(client, "alice@example.com", "alice")
    token_b = await _register_and_login(client, "bob@example.com", "bob")
    conversation_id = await _create_conversation(client, token_a)
    await client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={"content": "Alice private thought"},
        headers=_auth(token_a),
    )

    response = await client.get("/api/v1/memory", headers=_auth(token_b))
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == []


@pytest.mark.asyncio
async def test_search_memories(client: AsyncClient) -> None:
    token = await _register_and_login(client, "memory@example.com", "memoryuser")
    conversation_id = await _create_conversation(client, token)
    await client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={"content": "I love hiking in the mountains every weekend"},
        headers=_auth(token),
    )
    await client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={"content": "Quarterly financial reports are filed in April"},
        headers=_auth(token),
    )

    response = await client.get("/api/v1/memory/search", params={"q": "hiking mountains"}, headers=_auth(token))
    assert response.status_code == status.HTTP_200_OK
    results = response.json()
    assert len(results) == 2
    top = results[0]
    assert "hiking" in top["content"]
    assert "score" in top
    assert "embedding_id" in top
    assert results[0]["score"] >= results[1]["score"]


@pytest.mark.asyncio
async def test_search_scoped_to_user(client: AsyncClient) -> None:
    token_a = await _register_and_login(client, "alice@example.com", "alice")
    token_b = await _register_and_login(client, "bob@example.com", "bob")
    conversation_id = await _create_conversation(client, token_a)
    await client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={"content": "Alice only secret about penguins"},
        headers=_auth(token_a),
    )

    response = await client.get("/api/v1/memory/search", params={"q": "penguins"}, headers=_auth(token_b))
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == []


@pytest.mark.asyncio
async def test_search_requires_query(client: AsyncClient) -> None:
    token = await _register_and_login(client, "memory@example.com", "memoryuser")
    response = await client.get("/api/v1/memory/search", headers=_auth(token))
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_search_requires_auth(client: AsyncClient) -> None:
    response = await client.get("/api/v1/memory/search", params={"q": "anything"})
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
async def test_reindex_endpoint(client: AsyncClient) -> None:
    token = await _register_and_login(client, "memory@example.com", "memoryuser")
    conversation_id = await _create_conversation(client, token)
    await client.post(
        f"/api/v1/conversations/{conversation_id}/chat",
        json={"content": "Hello reindex"},
        headers=_auth(token),
    )

    response = await client.post("/api/v1/memory/reindex", headers=_auth(token))
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["created"] == 0
    assert data["reindexed"] == 2


@pytest.mark.asyncio
async def test_reindex_requires_auth(client: AsyncClient) -> None:
    response = await client.post("/api/v1/memory/reindex")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
async def test_create_memory_pipeline(db_session: AsyncSession) -> None:
    """create_memory embeds, stores the vector in Qdrant, then persists metadata."""
    vectors = FakeVectorStore()
    service = MemoryService(db_session, FakeEmbeddingService(), vectors)
    user_id, conversation_id, message_id = await _seed_content(db_session, "The pipeline remembers this")

    memory = await service.create_memory(user_id, conversation_id, message_id, "The pipeline remembers this")

    assert memory.id is not None
    assert memory.user_id == user_id
    assert memory.conversation_id == conversation_id
    assert memory.message_id == message_id
    assert memory.importance_score == 0.5
    assert vectors.collection_ready is True
    stored_vector, stored_payload = vectors._points[memory.embedding_id]
    assert len(stored_vector) == FakeEmbeddingService().dimension
    assert stored_payload["user_id"] == user_id
    assert stored_payload["content"] == "The pipeline remembers this"
    assert stored_payload["message_id"] == message_id


@pytest.mark.asyncio
async def test_delete_memory(db_session: AsyncSession) -> None:
    """delete_memory removes both the vector point and the metadata row."""
    vectors = FakeVectorStore()
    service = MemoryService(db_session, FakeEmbeddingService(), vectors)
    user_id, conversation_id, message_id = await _seed_content(db_session, "temporary memory")
    memory = await service.create_memory(user_id, conversation_id, message_id, "temporary memory")

    await service.delete_memory(user_id, memory.id)

    assert memory.embedding_id not in vectors._points
    assert await service.list_memories(user_id) == []


@pytest.mark.asyncio
async def test_delete_memory_not_owned_returns_404(db_session: AsyncSession) -> None:
    vectors = FakeVectorStore()
    service = MemoryService(db_session, FakeEmbeddingService(), vectors)
    user_id, conversation_id, message_id = await _seed_content(db_session, "owned memory")
    memory = await service.create_memory(user_id, conversation_id, message_id, "owned memory")

    with pytest.raises(HTTPException) as exc_info:
        await service.delete_memory(user_id + 1, memory.id)
    assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
    assert len(await service.list_memories(user_id)) == 1


@pytest.mark.asyncio
async def test_reindex_backfills_and_reembeds(db_session: AsyncSession) -> None:
    """reindex creates memories for messages that have none, then refreshes them."""
    vectors = FakeVectorStore()
    service = MemoryService(db_session, FakeEmbeddingService(), vectors)
    user_id, conversation_id, _ = await _seed_content(db_session, "first unmemorized message")
    second = Message(conversation_id=conversation_id, role=Message.ROLE_USER, content="second unmemorized message")
    db_session.add(second)
    await db_session.commit()
    await db_session.refresh(second)

    first_result = await service.reindex_memory(user_id)
    assert first_result.created == 2
    assert first_result.reindexed == 0
    assert len(await service.list_memories(user_id)) == 2

    second_result = await service.reindex_memory(user_id)
    assert second_result.created == 0
    assert second_result.reindexed == 2
    assert len(vectors._points) == 2


@pytest.mark.asyncio
async def test_try_create_memory_swallows_vector_store_failures(db_session: AsyncSession) -> None:
    """Ingestion failures are logged and never raised into the chat path."""

    class FailingVectorStore:
        async def ensure_collection(self, vector_size: int) -> None:
            return None

        async def upsert(self, point_id: str, vector: list[float], payload: dict[str, object]) -> None:
            raise RuntimeError("vector store unavailable")

        async def search(self, vector: list[float], limit: int, user_id: int) -> list[object]:
            raise RuntimeError("vector store unavailable")

        async def delete(self, point_id: str) -> None:
            raise RuntimeError("vector store unavailable")

    service = MemoryService(db_session, FakeEmbeddingService(), FailingVectorStore())
    user_id, conversation_id, message_id = await _seed_content(db_session, "message during outage")

    result = await service.try_create_memory(user_id, conversation_id, message_id, "message during outage")

    assert result is None
    assert await service.list_memories(user_id) == []
