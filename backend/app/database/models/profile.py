"""Structured long-term memory about a single user.

``app/memory`` stores the unstructured half of long-term memory as embeddings
in Qdrant. This model stores the structured half described in ARCHITECTURE.md
("structured memory stored in PostgreSQL"): durable attributes that should be
read and updated exactly, never approximated by a similarity search.
"""

from sqlalchemy import ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, TimestampMixin


class UserProfile(TimestampMixin, Base):
    """One structured profile row per user, created lazily on first access."""

    __tablename__ = "user_profiles"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True)
    display_name: Mapped[str | None] = mapped_column(String(120))
    summary: Mapped[str | None] = mapped_column(Text)
    timezone: Mapped[str | None] = mapped_column(String(64))
    preferred_language: Mapped[str | None] = mapped_column(String(32))
    interests: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    goals: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    preferences: Mapped[dict[str, str]] = mapped_column(JSON, default=dict, nullable=False)
