from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.security import create_access_token
from app.database.session import get_db_session
from app.dependencies.auth import CurrentUser
from app.schemas.auth import LoginRequest, TokenResponse
from app.schemas.user import UserCreate, UserProfile, UserRead
from app.services.user_service import UserService

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def register(payload: UserCreate, session: AsyncSession = Depends(get_db_session)) -> UserRead:
    return UserRead.model_validate(await UserService(session).register(payload))


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, session: AsyncSession = Depends(get_db_session)) -> TokenResponse:
    user = await UserService(session).authenticate(payload)
    return TokenResponse(access_token=create_access_token(str(user.id)))


@router.get("/me", response_model=UserProfile)
async def me(current_user: CurrentUser) -> UserProfile:
    return UserProfile.model_validate(current_user)
