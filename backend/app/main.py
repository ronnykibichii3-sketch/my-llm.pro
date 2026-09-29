"""MyLLM.pro FastAPI application entrypoint."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import auth, chat, health, knowledge, memory, profile
from app.core.config import get_settings
from app.core.logging import configure_logging, logger
from app.exceptions.handlers import register_exception_handlers
from app.middleware.request_logging import RequestLoggingMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: configure logging on startup."""
    configure_logging()
    logger.info("Starting %s", get_settings().app_name)
    yield
    logger.info("Shutting down %s", get_settings().app_name)


def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)

    register_exception_handlers(app)

    app.include_router(health.router, prefix="/api/v1")
    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(chat.router, prefix="/api/v1")
    app.include_router(memory.router, prefix="/api/v1")
    app.include_router(profile.router, prefix="/api/v1")
    app.include_router(knowledge.router, prefix="/api/v1")

    return app


app = create_app()
