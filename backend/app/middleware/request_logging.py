"""Request/response logging middleware."""

import time
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every incoming request and outgoing response."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start_time
        logger.info(
            "%s %s %s %.3fs",
            request.method,
            request.url.path,
            response.status_code,
            duration,
        )
        return response
