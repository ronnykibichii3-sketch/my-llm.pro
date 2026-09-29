"""Tests for the user profile endpoints and the profile service."""

import pytest
from fastapi import status
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models.user import User
from app.profile.config import MAX_ITEMS_PER_LIST
from app.profile.schemas.profile import ProfileUpdateRequest
from app.profile.services.profile_service import ProfileService

PROFILE_FIELDS = {
    "id",
    "user_id",
    "display_name",
    "summary",
    "timezone",
    "preferred_language",
    "interests",
    "goals",
    "preferences",
    "created_at",
    "updated_at",
}


async def _register_and_login(
    client: AsyncClient, email: str, username: str, password: str = "securepassword123"
) -> str:
    """Helper to create a user and return an auth token."""
    await client.post("/api/v1/auth/register", json={"email": email, "username": username, "password": password})
    response = await client.post("/api/v1/auth/login", json={"email": email, "password": password})
    return response.json()["access_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _seed_user(session: AsyncSession, email: str = "profile-service@example.com") -> int:
    """Create a user row directly for service-level tests."""
    user = User(email=email, username="profileservice", hashed_password="not-a-real-hash")
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user.id


@pytest.mark.asyncio
async def test_get_profile_requires_auth(client: AsyncClient) -> None:
    response = await client.get("/api/v1/profile")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
async def test_get_profile_creates_empty_profile(client: AsyncClient) -> None:
    """The first read lazily provisions an empty profile row."""
    token = await _register_and_login(client, "profile@example.com", "profileuser")
    response = await client.get("/api/v1/profile", headers=_auth(token))
    assert response.status_code == status.HTTP_200_OK
    profile = response.json()
    assert set(profile) == PROFILE_FIELDS
    assert profile["display_name"] is None
    assert profile["interests"] == []
    assert profile["goals"] == []
    assert profile["preferences"] == {}

    again = await client.get("/api/v1/profile", headers=_auth(token))
    assert again.json()["id"] == profile["id"]


@pytest.mark.asyncio
async def test_update_profile_writes_scalars_and_lists(client: AsyncClient) -> None:
    token = await _register_and_login(client, "profile@example.com", "profileuser")
    response = await client.patch(
        "/api/v1/profile",
        json={
            "display_name": "  Ronny  ",
            "summary": "Builds personal AI tools.",
            "timezone": "Europe/Berlin",
            "preferred_language": "en",
            "interests": ["rust", "chess"],
            "goals": ["ship phase 4"],
            "preferences": {"tone": "concise"},
        },
        headers=_auth(token),
    )
    assert response.status_code == status.HTTP_200_OK
    profile = response.json()
    assert profile["display_name"] == "Ronny"
    assert profile["summary"] == "Builds personal AI tools."
@pytest.mark.asyncio
async def test_update_profile_is_partial(client: AsyncClient) -> None:
    """Absent fields keep their stored values."""
    token = await _register_and_login(client, "profile@example.com", "profileuser")
    await client.patch(
        "/api/v1/profile", json={"display_name": "Ronny", "timezone": "Europe/Berlin"}, headers=_auth(token)
    )
    response = await client.patch("/api/v1/profile", json={"display_name": "Ronny II"}, headers=_auth(token))
    profile = response.json()
    assert profile["display_name"] == "Ronny II"
    assert profile["timezone"] == "Europe/Berlin"


@pytest.mark.asyncio
async def test_update_profile_null_clears_scalar(client: AsyncClient) -> None:
    token = await _register_and_login(client, "profile@example.com", "profileuser")
    await client.patch("/api/v1/profile", json={"summary": "Initial summary"}, headers=_auth(token))
    response = await client.patch("/api/v1/profile", json={"summary": None}, headers=_auth(token))
    assert response.json()["summary"] is None


@pytest.mark.asyncio
async def test_update_profile_merges_lists_case_insensitively(client: AsyncClient) -> None:
    """Interests accumulate across patches and de-duplicate case-insensitively."""
    token = await _register_and_login(client, "profile@example.com", "profileuser")
    await client.patch("/api/v1/profile", json={"interests": ["Rust", "chess"]}, headers=_auth(token))
    response = await client.patch(
        "/api/v1/profile", json={"interests": ["rust", "sailing", "clean "]}, headers=_auth(token)
    )
    assert response.json()["interests"] == ["Rust", "chess", "sailing", "clean"]


@pytest.mark.asyncio
async def test_update_profile_merges_preferences(client: AsyncClient) -> None:
    token = await _register_and_login(client, "profile@example.com", "profileuser")
    await client.patch(
        "/api/v1/profile", json={"preferences": {"tone": "concise", "units": "metric"}}, headers=_auth(token)
    )
    response = await client.patch("/api/v1/profile", json={"preferences": {"tone": "detailed"}}, headers=_auth(token))
    assert response.json()["preferences"] == {"tone": "detailed", "units": "metric"}


@pytest.mark.asyncio
async def test_update_profile_rejects_empty_body(client: AsyncClient) -> None:
    token = await _register_and_login(client, "profile@example.com", "profileuser")
    response = await client.patch("/api/v1/profile", json={}, headers=_auth(token))
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_update_profile_rejects_oversized_value(client: AsyncClient) -> None:
    token = await _register_and_login(client, "profile@example.com", "profileuser")
    response = await client.patch("/api/v1/profile", json={"display_name": "x" * 121}, headers=_auth(token))
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_reset_profile_clears_every_field(client: AsyncClient) -> None:
    token = await _register_and_login(client, "profile@example.com", "profileuser")
    await client.patch(
        "/api/v1/profile",
        json={"display_name": "Ronny", "interests": ["rust"], "preferences": {"tone": "concise"}},
        headers=_auth(token),
    )
    response = await client.post("/api/v1/profile/reset", headers=_auth(token))
    assert response.status_code == status.HTTP_200_OK
    profile = response.json()
    assert profile["display_name"] is None
    assert profile["interests"] == []
    assert profile["preferences"] == {}


@pytest.mark.asyncio
async def test_reset_requires_auth(client: AsyncClient) -> None:
    response = await client.post("/api/v1/profile/reset")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
async def test_profiles_scoped_per_user(client: AsyncClient) -> None:
    token_a = await _register_and_login(client, "alice@example.com", "alice")
    token_b = await _register_and_login(client, "bob@example.com", "bob")
    await client.patch("/api/v1/profile", json={"display_name": "Alice"}, headers=_auth(token_a))

    response = await client.get("/api/v1/profile", headers=_auth(token_b))
    assert response.json()["display_name"] is None


@pytest.mark.asyncio
async def test_service_creates_profile_lazily(db_session: AsyncSession) -> None:
    user_id = await _seed_user(db_session)
    service = ProfileService(db_session)

    profile = await service.get_profile(user_id)
    assert profile.id is not None
    assert profile.user_id == user_id
    assert profile.interests == []


@pytest.mark.asyncio
async def test_service_caps_list_size(db_session: AsyncSession) -> None:
    """Stored lists never grow past MAX_ITEMS_PER_LIST, even when merging."""
    user_id = await _seed_user(db_session)
    profile = await ProfileService(db_session).get_profile(user_id)
    profile.interests = [f"interest-{index}" for index in range(MAX_ITEMS_PER_LIST)]
    await db_session.commit()

    capped = await ProfileService(db_session).update_profile(
        user_id, ProfileUpdateRequest(interests=["one-more", "and-another"])
    )

    assert len(capped.interests) == MAX_ITEMS_PER_LIST
    assert "one-more" not in capped.interests
    assert capped.interests[0] == "interest-0"
