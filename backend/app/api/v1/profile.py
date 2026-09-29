"""User profile API routes."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db_session
from app.dependencies.auth import CurrentUser
from app.profile.schemas.profile import ProfileRead, ProfileUpdateRequest
from app.profile.services.profile_service import ProfileService

router = APIRouter(tags=["profile"])


@router.get("/profile", response_model=ProfileRead)
async def get_profile(
    current_user: CurrentUser,
    session: AsyncSession = Depends(get_db_session),
) -> ProfileRead:
    """Return the authenticated user's profile, creating an empty one on first call."""
    profile = await ProfileService(session).get_profile(current_user.id)
    return ProfileRead.model_validate(profile)


@router.patch("/profile", response_model=ProfileRead)
async def update_profile(
    payload: ProfileUpdateRequest,
    current_user: CurrentUser,
    session: AsyncSession = Depends(get_db_session),
) -> ProfileRead:
    """Apply a partial profile update; absent fields are left untouched."""
    profile = await ProfileService(session).update_profile(current_user.id, payload)
    return ProfileRead.model_validate(profile)


@router.post("/profile/reset", response_model=ProfileRead)
async def reset_profile(
    current_user: CurrentUser,
    session: AsyncSession = Depends(get_db_session),
) -> ProfileRead:
    """Clear every stored profile attribute, keeping the profile row."""
    profile = await ProfileService(session).reset_profile(current_user.id)
    return ProfileRead.model_validate(profile)
