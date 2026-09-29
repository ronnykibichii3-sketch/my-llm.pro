"""Tests for authentication endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_register_success(client: AsyncClient) -> None:
    """Register a new user successfully."""
    response = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@example.com",
            "username": "testuser",
            "password": "securepassword123",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "test@example.com"
    assert data["username"] == "testuser"
    assert "id" in data
    assert "hashed_password" not in data


@pytest.mark.asyncio
async def test_register_duplicate_email(client: AsyncClient) -> None:
    """Registering with duplicate email returns 409."""
    payload = {
        "email": "test@example.com",
        "username": "testuser",
        "password": "securepassword123",
    }
    await client.post("/api/v1/auth/register", json=payload)
    response = await client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_login_success(client: AsyncClient) -> None:
    """Login with valid credentials returns JWT."""
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
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_invalid_credentials(client: AsyncClient) -> None:
    """Login with invalid credentials returns 401."""
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "nonexistent@example.com", "password": "wrongpassword"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_me(client: AsyncClient) -> None:
    """GET /auth/me returns the authenticated user profile."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@example.com",
            "username": "testuser",
            "password": "securepassword123",
        },
    )
    login_response = await client.post(
        "/api/v1/auth/login",
        json={"email": "test@example.com", "password": "securepassword123"},
    )
    token = login_response.json()["access_token"]
    response = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "test@example.com"
    assert data["username"] == "testuser"
    assert "id" in data
    assert "hashed_password" not in data


@pytest.mark.asyncio
async def test_get_me_unauthorized(client: AsyncClient) -> None:
    """GET /auth/me without token returns 401."""
    response = await client.get("/api/v1/auth/me")
    assert response.status_code == 401
