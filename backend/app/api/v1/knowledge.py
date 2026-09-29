"""Knowledge base API routes."""

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import Response

from app.dependencies.auth import CurrentUser
from app.dependencies.knowledge import get_knowledge_service
from app.knowledge.config import (
    DEFAULT_LIST_LIMIT,
    DEFAULT_SEARCH_LIMIT,
    MAX_LIST_LIMIT,
    MAX_QUERY_LENGTH,
    MAX_SEARCH_LIMIT,
)
from app.knowledge.schemas.knowledge import (
    ChunkRead,
    DocumentCreate,
    DocumentDetailRead,
    DocumentRead,
    KnowledgeSearchResult,
)
from app.knowledge.services.knowledge_service import KnowledgeService

router = APIRouter(tags=["knowledge"])


@router.get("/knowledge/documents", response_model=list[DocumentRead])
async def list_documents(
    current_user: CurrentUser,
    limit: int = Query(default=DEFAULT_LIST_LIMIT, ge=1, le=MAX_LIST_LIMIT),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> list[DocumentRead]:
    """List the authenticated user's knowledge documents, newest first."""
    documents = await service.list_documents(current_user.id, limit)
    return [DocumentRead.model_validate(document) for document in documents]


@router.post("/knowledge/documents", response_model=DocumentRead, status_code=201)
async def create_document(
    payload: DocumentCreate,
    current_user: CurrentUser,
    service: KnowledgeService = Depends(get_knowledge_service),
) -> DocumentRead:
    """Ingest a document supplied as JSON: chunk, embed, index, store."""
    document = await service.ingest_document(current_user.id, payload)
    return DocumentRead.model_validate(document)


@router.post("/knowledge/documents/raw", response_model=DocumentRead, status_code=201)
async def create_document_from_raw_body(
    request: Request,
    current_user: CurrentUser,
    title: str | None = Query(default=None, min_length=1, max_length=300),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> DocumentRead:
    """Ingest a document supplied as a raw text body.

    Raw bodies are used instead of multipart uploads because the project does
    not depend on ``python-multipart``: ``curl --data-binary @notes.md`` works.
    """
    raw = await request.body()
    document = await service.ingest_raw(
        current_user.id,
        raw,
        request.headers.get("content-type"),
        title,
    )
    return DocumentRead.model_validate(document)


@router.get("/knowledge/documents/{document_id}", response_model=DocumentDetailRead)
async def get_document(
    document_id: int,
    current_user: CurrentUser,
    service: KnowledgeService = Depends(get_knowledge_service),
) -> DocumentDetailRead:
    """Fetch one document with its chunks in reading order."""
    document = await service.get_document(current_user.id, document_id)
    chunks = await service.list_chunks(current_user.id, document_id)
    return DocumentDetailRead.model_validate(document).model_copy(
        update={"chunks": [ChunkRead.model_validate(chunk) for chunk in chunks]}
    )


@router.delete("/knowledge/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: int,
    current_user: CurrentUser,
    service: KnowledgeService = Depends(get_knowledge_service),
) -> Response:
    """Delete a document, its chunks and its indexed vectors."""
    await service.delete_document(current_user.id, document_id)
    return Response(status_code=204)


@router.get("/knowledge/search", response_model=list[KnowledgeSearchResult])
async def search_knowledge(
    current_user: CurrentUser,
    q: str = Query(min_length=1, max_length=MAX_QUERY_LENGTH),
    limit: int = Query(default=DEFAULT_SEARCH_LIMIT, ge=1, le=MAX_SEARCH_LIMIT),
    document_id: int | None = Query(default=None, ge=1),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> list[KnowledgeSearchResult]:
    """Vector-search the user's knowledge; optionally narrowed to one document."""
    return await service.search(current_user.id, q, limit, document_id)

