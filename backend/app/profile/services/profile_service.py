"""User profile orchestration: lazy creation, merge updates, and reset.

The profile is the structured side of long-term memory. Unlike
``app.memory`` it is never searched by similarity: values are written exactly
and read exactly, which is what makes it safe to use as fact.
"""

from collections.abc import Iterable

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import logger
from app.database.models.profile import UserProfile
from app.profile.config import MAX_ITEMS_PER_LIST, MAX_PREFERENCE_KEYS
from app.profile.repositories.profile_repository import ProfileRepository
from app.profile.schemas.profile import ProfileUpdateRequest

SCALAR_FIELDS: tuple[str, ...] = ("display_name", "summary", "timezone", "preferred_language")
LIST_FIELDS: tuple[str, ...] = ("interests", "goals")


class ProfileService:
    """Owns the lifecycle of a single user's structured profile."""

    def __init__(self, session: AsyncSession) -> None:
        self._profiles = ProfileRepository(session)

    async def get_profile(self, user_id: int) -> UserProfile:
        """Return the user's profile, creating an empty one on first access."""
        profile = await self._profiles.get_for_user(user_id)
        if profile is None:
            profile = await self._profiles.create(user_id)
            logger.info("Created profile for user %s", user_id)
        return profile

    async def update_profile(self, user_id: int, payload: ProfileUpdateRequest) -> UserProfile:
        """Apply a partial update; absent fields are left untouched."""
        provided = payload.model_fields_set
        if not provided:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No profile fields provided")
        profile = await self.get_profile(user_id)
        self._apply_scalars(profile, payload, provided)
        self._apply_lists(profile, payload, provided)
        if "preferences" in provided:
            profile.preferences = _merge_preferences(profile.preferences, payload.preferences)
        await self._profiles.save(profile)
        logger.info("Updated profile for user %s fields=%s", user_id, sorted(provided))
        return profile

    async def reset_profile(self, user_id: int) -> UserProfile:
        """Clear every stored attribute without deleting the profile row."""
        profile = await self.get_profile(user_id)
        for name in SCALAR_FIELDS:
            setattr(profile, name, None)
        for name in LIST_FIELDS:
            setattr(profile, name, [])
        profile.preferences = {}
        await self._profiles.save(profile)
        logger.info("Reset profile for user %s", user_id)
        return profile

    @staticmethod
    def _apply_scalars(profile: UserProfile, payload: ProfileUpdateRequest, provided: set[str]) -> None:
        for name in SCALAR_FIELDS:
            if name in provided:
                setattr(profile, name, _normalise_text(getattr(payload, name)))

    @staticmethod
    def _apply_lists(profile: UserProfile, payload: ProfileUpdateRequest, provided: set[str]) -> None:
        for name in LIST_FIELDS:
            if name in provided:
                setattr(profile, name, _merge_unique(getattr(profile, name), getattr(payload, name)))


def _normalise_text(value: str | None) -> str | None:
    """Trim surrounding whitespace; treat an empty string as a cleared value."""
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed or None


def _merge_unique(existing: list[str] | None, incoming: list[str] | None) -> list[str]:
    """Append new items to the stored list, de-duplicated case-insensitively."""
    if incoming is None:
        return []
    merged: list[str] = []
    seen: set[str] = set()
    for item in _ordered_unique(existing or [], incoming):
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged[:MAX_ITEMS_PER_LIST]


def _merge_preferences(existing: dict[str, str] | None, incoming: dict[str, str] | None) -> dict[str, str]:
    """Overlay incoming preferences on the stored ones, capped in size."""
    if incoming is None:
        return {}
    merged = {**(existing or {}), **incoming}
    return dict(list(merged.items())[:MAX_PREFERENCE_KEYS])


def _ordered_unique(*sources: Iterable[str]) -> list[str]:
    """Flatten sources in order, trimming items and dropping blank ones."""
    return [stripped for source in sources for item in source if (stripped := item.strip())]
