"""Password hashing and JWT token helpers."""

from datetime import UTC, datetime, timedelta

import jwt
from passlib.context import CryptContext

from app.core.config import get_settings

JWT_ALGORITHM = "HS256"
TOKEN_TYPE = "access"
PASSWORD_CONTEXT = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return PASSWORD_CONTEXT.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    return PASSWORD_CONTEXT.verify(password, hashed_password)


def create_access_token(subject: str) -> str:
    settings = get_settings()
    expires_at = datetime.now(UTC) + timedelta(minutes=settings.access_token_expire_minutes)
    return jwt.encode({"sub": subject, "type": TOKEN_TYPE, "exp": expires_at}, settings.secret_key.get_secret_value(), algorithm=JWT_ALGORITHM)
