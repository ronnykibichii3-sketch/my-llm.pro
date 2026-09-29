"""Turn a raw upload into validated, decodable text before chunking.

The API accepts raw request bodies instead of multipart form data because
``python-multipart`` is not a project dependency; this module is where the
content type, the size ceiling and the character encoding are policed.
"""

from fastapi import HTTPException, status

from app.knowledge.config import (
    MAX_TITLE_LENGTH,
    SUPPORTED_CONTENT_TYPES,
    UNTITLED_DOCUMENT,
)

CHARSET_PARAMETER = "charset="
BYTE_ORDER_MARK = "\ufeff"
NULL_BYTE = b"\x00"
DEFAULT_CHARSET = "utf-8"


def media_type_of(content_type: str | None) -> str:
    """Extract the bare media type, discarding parameters such as ``charset``."""
    return (content_type or "").split(";", 1)[0].strip().lower()


def ensure_supported(content_type: str | None) -> str:
    """Return the validated media type, or raise 415 for anything untextual."""
    media = media_type_of(content_type)
    if media not in SUPPORTED_CONTENT_TYPES and not media.startswith("text/"):
        supported = ", ".join(SUPPORTED_CONTENT_TYPES)
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type '{media or 'unknown'}'. Supported: {supported}",
        )
    return media


def decode_text(raw: bytes, content_type: str | None, max_bytes: int) -> str:
    """Decode an upload to text; 413 when too large, 415/400 when undecodable."""
    ensure_supported(content_type)
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Document exceeds the {max_bytes} byte upload limit",
        )
    if NULL_BYTE in raw:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Binary content is not supported; upload plain text",
        )
    charset = _charset_of(content_type)
    try:
        return raw.decode(charset).lstrip(BYTE_ORDER_MARK)
    except LookupError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported charset '{charset}'",
        ) from None
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Content is not valid '{charset}' text",
        ) from None


def title_from_text(text: str) -> str:
    """Derive a display title from the first non-blank line of a document."""
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped[:MAX_TITLE_LENGTH]
    return UNTITLED_DOCUMENT


def _charset_of(content_type: str | None) -> str:
    _, _, parameters = (content_type or "").partition(";")
    for parameter in parameters.split(";"):
        name, _, value = parameter.strip().partition("=")
        if name.strip().lower() == "charset" and value.strip():
            return value.strip().strip('"').strip("'")
    return DEFAULT_CHARSET
