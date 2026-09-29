"""Tests for conversation endpoints."""

import pytest
from httpx import AsyncClient


async def _create_user_and_login(client: AsyncClient) -> str:
    """Helper to create a user and return auth token."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@example.com",
            "username": "testuser",
            "password": "securepassword123",
        },
    )
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "test@example.com", "password": "securepassword123"},
    )
    return response.json()["access_token"]


@pytest.mark.asyncio
async def test_create_conversation(client: AsyncClient) -> None:
    """Create a conversation successfully."""
    token = await _create_user_and_login(client)
    response = await client.post(
        "/api/v1/conversations",
        json={"title": "Test Conversation"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "Test Conversation"
    assert "id" in data


@pytest.mark.asyncio
async def test_list_conversations(client: AsyncClient) -> None:
    """List conversations for authenticated user."""
    token = await _create_user_and_login(client)
    await client.post(
        "/api/v1/conversations",
        json={"title": "Test Conversation"},
        headers={"Authorization": f"Bearer {token}"},
    )
    response = await client.get(
        "/api/v1/conversations",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["title"] == "Test Conversation"


@pytest.mark.asyncio
async def test_get_conversation(client: AsyncClient) -> None:
    """Get a specific conversation by ID."""
    token = await _create_user_and_login(client)
    create_response = await client.post(
        "/api/v1/conversations",
        json={"title": "Test Conversation"},
        headers={"Authorization": f"Bearer {token}"},
    )
    conversation_id = create_response.json()["id"]
    response = await client.get(
        f"/api/v1/conversations/{conversation_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["id"] == conversation_id


@pytest.mark.asyncio
async def test_get_conversation_not_found(client: AsyncClient) -> None:
    """Get non-existent conversation returns 404."""
    token = await _create_user_and_login(client)
    response = await client.get(
        "/api/v1/conversations/999",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_chat_persists_messages(client: AsyncClient) -> None:
    """Chat endpoint saves both messages and returns the assistant reply."""
    token = await _create_user_and_login(client)
    create_response = await client.post(
        "/api/v1/conversations",
        json={"title": "Test Conversation"},
        headers={"Authorization": f"Bearer {token}"},
    )
    conversation_id = create_response.json()["id"]
    response = await client.post(
        f"/api/v1/conversations/{conversation_id}/chat",
        json={"content": "Hello"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["role"] == "assistant"
    assert data["content"] == "temporary placeholder"
    assert "id" in data

    messages_response = await client.get(
        f"/api/v1/conversations/{conversation_id}/messages",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert messages_response.status_code == 200
    messages = messages_response.json()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello"
    assert messages[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_chat_conversation_not_found(client: AsyncClient) -> None:
    """Chat in a non-existent conversation returns 404."""
    token = await _create_user_and_login(client)
    response = await client.post(
        "/api/v1/conversations/999/chat",
        json={"content": "Hello"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_create_and_list_messages(client: AsyncClient) -> None:
    """Persist a user message and list it back."""
    token = await _create_user_and_login(client)
    create_response = await client.post(
        "/api/v1/conversations",
        json={"title": "Test Conversation"},
        headers={"Authorization": f"Bearer {token}"},
    )
    conversation_id = create_response.json()["id"]

    post_response = await client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={"content": "Persisted message"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert post_response.status_code == 201
    message = post_response.json()
    assert message["role"] == "user"
    assert message["content"] == "Persisted message"
    assert "created_at" in message

    list_response = await client.get(
        f"/api/v1/conversations/{conversation_id}/messages",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert list_response.status_code == 200
    messages = list_response.json()
    assert len(messages) == 1
    assert messages[0]["id"] == message["id"]


@pytest.mark.asyncio
async def test_list_messages_conversation_not_found(client: AsyncClient) -> None:
    """Listing messages for a non-existent conversation returns 404."""
    token = await _create_user_and_login(client)
    response = await client.get(
        "/api/v1/conversations/999/messages",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_unauthorized_access(client: AsyncClient) -> None:
    """Accessing protected routes without token returns 401."""
    response = await client.get("/api/v1/conversations")
    assert response.status_code == 401
