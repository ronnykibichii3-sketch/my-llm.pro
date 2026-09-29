from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.security import hash_password, verify_password
from app.database.models.user import User
from app.repositories.user_repository import UserRepository
from app.schemas.auth import LoginRequest
from app.schemas.user import UserCreate


class UserService:
    def __init__(self, session: AsyncSession) -> None:
        self._users = UserRepository(session)

    async def register(self, payload: UserCreate) -> User:
        email = str(payload.email).lower()
        if await self._users.get_by_email(email) or await self._users.get_by_username(payload.username):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email or username is already registered")
        return await self._users.create(email, payload.username, hash_password(payload.password))

    async def authenticate(self, payload: LoginRequest) -> User:
        user = await self._users.get_by_email(str(payload.email).lower())
        if user is None or not verify_password(payload.password, user.hashed_password) or not user.is_active:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials", headers={"WWW-Authenticate": "Bearer"})
        return user
