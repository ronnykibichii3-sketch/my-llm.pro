"""Knowledge base models: an ingested document and its embedded chunks.

A document is the unit a user uploads; a chunk is the unit that gets embedded.
Chunk boundaries exist because the embedding model has a small useful context
window, so retrieval matches chunks and hydrates them back through PostgreSQL.
"""

from datetime import datetime
from typing import ClassVar

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.database.base import Base, TimestampMixin


class KnowledgeDocument(TimestampMixin, Base):
    """A single ingested source document owned by one user."""

    __tablename__ = "knowledge_documents"
    STATUS_PENDING: ClassVar[str] = "pending"
    STATUS_PROCESSING: ClassVar[str] = "processing"
    STATUS_READY: ClassVar[str] = "ready"
    STATUS_FAILED: ClassVar[str] = "failed"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    source: Mapped[str | None] = mapped_column(String(512))
    content_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=STATUS_PENDING, server_default=STATUS_PENDING)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    error: Mapped[str | None] = mapped_column(Text)


class KnowledgeChunk(Base):
    """One embedded span of a document; the row behind each Qdrant point."""

    __tablename__ = "knowledge_chunks"
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("knowledge_documents.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
