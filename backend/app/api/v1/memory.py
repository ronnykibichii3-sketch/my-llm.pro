"""Memory API routes (foundation only: list, search, reindex)."""

from fastapi import APIRouter, Depends, Query
from app.dependencies.auth import CurrentUser
from app.dependencies.memory import get_memory_service
from app.memory.schemas.memory import MemoryRead, MemorySearchResult, ReindexResult
from app.memory.services.memory_service import MemoryService

router = APIRouter(tags=["memory"])

DEFAULT_LIST_LIMIT = 50
MAX_LIST_LIMIT = 200
DEFAULT_SEARCH_LIMIT = 10
MAX_SEARCH_LIMIT = 50
MAX_QUERY_LENGTH = 2000


@router.get("/memory", response_model=list[MemoryRead])
async def list_memories(
    current_user: CurrentUser,
    service: MemoryService = Depends(get_memory_service),
    limit: int = Query(DEFAULT_LIST_LIMIT, ge=1, le=MAX_LIST_LIMIT),
) -> list[MemoryRead]:
    """List the authenticated user's memories, newest first."""
    memories = await service.list_memories(current_user.id, limit)
    return [MemoryRead.model_validate(memory) for memory in memories]


@router.get("/memory/search", response_model=list[MemorySearchResult])
async def search_memory(
    current_user: CurrentUser,
    service: MemoryService = Depends(get_memory_service),
    q: str = Query(..., min_length=1, max_length=MAX_QUERY_LENGTH),
    limit: int = Query(DEFAULT_SEARCH_LIMIT, ge=1, le=MAX_SEARCH_LIMIT),
) -> list[MemorySearchResult]:
    """Semantic search across the authenticated user's memories."""
    return await service.search_memory(current_user.id, q, limit)


@router.post("/memory/reindex", response_model=ReindexResult)
async def reindex_memory(
    current_user: CurrentUser,
    service: MemoryService = Depends(get_memory_service),
) -> ReindexResult:
    """Backfill missing memories and re-embed existing ones for the user."""
    return await service.reindex_memory(current_user.id)