from typing import Annotated
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import get_settings
from app.core.security import JWT_ALGORITHM, TOKEN_TYPE
from app.database.models.user import User
from app.database.session import get_db_session
from app.repositories.user_repository import UserRepository

bearer_scheme = HTTPBearer(auto_error=False)
DbSession = Annotated[AsyncSession, Depends(get_db_session)]


async def get_current_user(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)], session: DbSession) -> User:
    unauthorized = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    if credentials is None:
        raise unauthorized
    try:
        payload = jwt.decode(credentials.credentials, get_settings().secret_key.get_secret_value(), algorithms=[JWT_ALGORITHM])
        subject = payload.get("sub")
        if not isinstance(subject, str) or payload.get("type") != TOKEN_TYPE:
            raise unauthorized
        user = await UserRepository(session).get_by_id(int(subject))
    except (jwt.PyJWTError, ValueError):
        raise unauthorized from None
    if user is None or not user.is_active:
        raise unauthorized
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
