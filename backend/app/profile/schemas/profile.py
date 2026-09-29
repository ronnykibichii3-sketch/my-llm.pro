"""Request and response models for the user profile API."""

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from app.profile.config import (
    MAX_DISPLAY_NAME_LENGTH,
    MAX_ITEMS_PER_LIST,
    MAX_LANGUAGE_LENGTH,
    MAX_PREFERENCE_KEY_LENGTH,
    MAX_PREFERENCE_KEYS,
    MAX_PREFERENCE_VALUE_LENGTH,
    MAX_SUMMARY_LENGTH,
    MAX_TAG_LENGTH,
    MAX_TIMEZONE_LENGTH,
)

ProfileText = Annotated[str, Field(min_length=1, max_length=MAX_TAG_LENGTH)]
PreferenceKey = Annotated[str, Field(min_length=1, max_length=MAX_PREFERENCE_KEY_LENGTH)]
PreferenceValue = Annotated[str, Field(min_length=1, max_length=MAX_PREFERENCE_VALUE_LENGTH)]


class ProfileRead(BaseModel):
    id: int
    user_id: int
    display_name: str | None
    summary: str | None
    timezone: str | None
    preferred_language: str | None
    interests: list[str]
    goals: list[str]
    preferences: dict[str, str]
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)


class ProfileUpdateRequest(BaseModel):
    """Partial update. A field that is present but ``null`` clears it.

    ``interests`` and ``goals`` accumulate: incoming values are merged with the
    stored ones and de-duplicated, because a profile is built up over time.
    Use ``POST /api/v1/profile/reset`` to start over.
    """

    display_name: str | None = Field(default=None, max_length=MAX_DISPLAY_NAME_LENGTH)
    summary: str | None = Field(default=None, max_length=MAX_SUMMARY_LENGTH)
    timezone: str | None = Field(default=None, max_length=MAX_TIMEZONE_LENGTH)
    preferred_language: str | None = Field(default=None, max_length=MAX_LANGUAGE_LENGTH)
    interests: list[ProfileText] | None = Field(default=None, max_length=MAX_ITEMS_PER_LIST)
    goals: list[ProfileText] | None = Field(default=None, max_length=MAX_ITEMS_PER_LIST)
    preferences: dict[PreferenceKey, PreferenceValue] | None = Field(default=None, max_length=MAX_PREFERENCE_KEYS)
