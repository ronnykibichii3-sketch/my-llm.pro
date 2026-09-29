"""Custom application exceptions."""


class AppException(Exception):
    """Base exception for application errors."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class NotFoundError(AppException):
    """Resource not found."""

    def __init__(self, resource: str, resource_id: int | str) -> None:
        super().__init__(f"{resource} with id {resource_id} not found", status_code=404)


class ConflictError(AppException):
    """Resource conflict."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=409)


class UnauthorizedError(AppException):
    """Authentication failed."""

    def __init__(self, message: str = "Unauthorized") -> None:
        super().__init__(message, status_code=401)
