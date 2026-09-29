"""Persistence for :class:`UserProfile`, one row per user."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models.profile import UserProfile


class ProfileRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_for_user(self, user_id: int) -> UserProfile | None:
        statement = select(UserProfile).where(UserProfile.user_id == user_id)
        return await self._session.scalar(statement)

    async def create(self, user_id: int) -> UserProfile:
        profile = UserProfile(
            user_id=user_id,
            interests=[],
            goals=[],
            preferences={},
        )
        self._session.add(profile)
        await self._session.commit()
        await self._session.refresh(profile)
        return profile

    async def save(self, profile: UserProfile) -> UserProfile:
        await self._session.commit()
        await self._session.refresh(profile)
        return profile
