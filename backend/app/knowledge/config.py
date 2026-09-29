"""Knowledge base module configuration and limits.

Chunk sizes, hard ceilings and the Qdrant payload keys all live here so the
ingest pipeline, the request validation and the stored payload cannot drift
apart (RULES.md Rule 7).
"""

from dataclasses import dataclass

from app.core.config import Settings

DEFAULT_CHUNK_MAX_CHARS = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_MAX_DOCUMENT_CHARS = 200_000
DEFAULT_MAX_UPLOAD_BYTES = 2_000_000

MAX_TITLE_LENGTH = 300
MAX_SOURCE_LENGTH = 512
MAX_QUERY_LENGTH = 2000
MAX_ERROR_LENGTH = 500
DEFAULT_SEARCH_LIMIT = 8
MAX_SEARCH_LIMIT = 50
DEFAULT_LIST_LIMIT = 50
MAX_LIST_LIMIT = 200
UNTITLED_DOCUMENT = "Untitled document"
DELETED_DOCUMENT_TITLE = "Deleted document"

PARAGRAPH_SEPARATOR = "\n\n"

# Documents are embedded and stored as plain text; binary formats such as PDF
# or DOCX need extra decoders that are not in requirements.txt and are rejected
# up front with 415 rather than ingested as mojibake.
SUPPORTED_CONTENT_TYPES: tuple[str, ...] = ("text/plain", "text/markdown", "text/csv", "application/json")

PAYLOAD_KIND_KEY = "kind"
PAYLOAD_DOCUMENT_ID_KEY = "document_id"
PAYLOAD_POSITION_KEY = "position"
PAYLOAD_CONTENT_KEY = "content"
PAYLOAD_KIND_VALUE = "knowledge_chunk"


@dataclass(frozen=True)
class KnowledgeConfig:
    """Tunables for a single ingest run, built once from application settings."""

    chunk_max_chars: int = DEFAULT_CHUNK_MAX_CHARS
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    max_document_chars: int = DEFAULT_MAX_DOCUMENT_CHARS
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES

    @classmethod
    def from_settings(cls, settings: Settings) -> "KnowledgeConfig":
        return cls(
            chunk_max_chars=settings.knowledge_chunk_max_chars,
            chunk_overlap=settings.knowledge_chunk_overlap,
            max_document_chars=settings.knowledge_max_document_chars,
            max_upload_bytes=settings.knowledge_max_upload_bytes,
        )
