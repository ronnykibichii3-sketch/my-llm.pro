from datetime import datetime
from typing import TypedDict

from pydantic import BaseModel, ConfigDict


class MemoryPayload(TypedDict):
    """Structured metadata stored alongside each vector point in Qdrant."""

    user_id: int
    conversation_id: int
    message_id: int
    content: str
    importance_score: float


class MemoryRead(BaseModel):
    id: int
    user_id: int
    conversation_id: int
    message_id: int
    content: str
    embedding_id: str
    importance_score: float
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class MemorySearchResult(MemoryRead):
    """A memory matched by vector search, with its relevance score."""

    score: float


class ReindexResult(BaseModel):
    created: int
    reindexed: int