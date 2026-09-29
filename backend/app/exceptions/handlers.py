"""Global exception handlers for the application."""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core.logging import logger
from app.exceptions import AppException


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers on the application."""

    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
        logger.error("AppException: %s %s -> %s", request.method, request.url.path, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception: %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )
