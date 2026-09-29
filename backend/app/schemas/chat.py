from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ConversationCreate(BaseModel):
    title: str = Field(min_length=1, max_length=255)


class ConversationRead(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)


class MessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=20_000)


class MessageRead(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)
