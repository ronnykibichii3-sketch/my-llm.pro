from datetime import datetime
from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_]+$")
    password: str = Field(min_length=12, max_length=128)


class UserProfile(BaseModel):
    id: int
    email: EmailStr
    username: str
    model_config = ConfigDict(from_attributes=True)


class UserRead(BaseModel):
    id: int
    email: EmailStr
    username: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)
