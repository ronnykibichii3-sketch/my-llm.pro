from datetime import datetime
from sqlalchemy import DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from app.database.base import Base

DEFAULT_IMPORTANCE_SCORE: float = 0.5


class Memory(Base):
    __tablename__ = "memories"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id", ondelete="CASCADE"), index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    importance_score: Mapped[float] = mapped_column(Float, nullable=False, default=DEFAULT_IMPORTANCE_SCORE, server_default="0.5")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())