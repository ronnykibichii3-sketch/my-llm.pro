"""Request and response models for the knowledge base API."""

from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field

from app.knowledge.config import MAX_SOURCE_LENGTH, MAX_TITLE_LENGTH


class DocumentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=MAX_TITLE_LENGTH)
    content: str = Field(min_length=1)
    source: str | None = Field(default=None, max_length=MAX_SOURCE_LENGTH)
    content_type: str = Field(default="text/plain", min_length=1, max_length=64)


class ChunkRead(BaseModel):
    id: int
    document_id: int
    position: int
    content: str
    char_count: int
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class DocumentRead(BaseModel):
    id: int
    user_id: int
    title: str
    source: str | None
    content_type: str
    status: str
    char_count: int
    chunk_count: int
    error: str | None
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)


class DocumentDetailRead(DocumentRead):
    """A document plus its chunks.

    ``chunks`` defaults to an empty list so the ORM row validates on its own;
    the route fills it from the chunk rows in reading order.
    """

    chunks: list[ChunkRead] = Field(default_factory=list)


class KnowledgeSearchResult(BaseModel):
    score: float
    document_id: int
    document_title: str
    chunk_id: int
    position: int
    content: str
    char_count: int


@dataclass(frozen=True)
class DocumentDraft:
    """Internal hand-off between the request layer and the repository."""

    user_id: int
    title: str
    content: str
    source: str | None = None
    content_type: str = "text/plain"


class KnowledgePayload(TypedDict):
    """Structured metadata stored alongside each knowledge vector point."""

    user_id: int
    document_id: int
    position: int
    content: str
    kind: str


@dataclass(frozen=True)
class ChunkPlan:
    """Internal hand-off between the chunker, the embedder and the repository."""

    position: int
    content: str
    embedding_id: str
