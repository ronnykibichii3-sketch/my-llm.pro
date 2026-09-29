"""Dependency providers for prompt-context assembly.

The context builder is the single composition point for everything the model is
told about a user: today the structured profile and knowledge retrieval, later
memory and personalisation. Wiring happens here so no other service has to know
how those sources are constructed.
"""

from functools import lru_cache

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.context.config import ContextConfig
from app.context.services.context_builder_service import ContextBuilderService
from app.core.config import get_settings
from app.database.session import get_db_session
from app.dependencies.knowledge import get_knowledge_service
from app.knowledge.services.knowledge_service import KnowledgeService
from app.profile.services.profile_service import ProfileService


@lru_cache
def get_context_config() -> ContextConfig:
    """Return the process-wide prompt-context budgets."""
    return ContextConfig.from_settings(get_settings())


def get_context_builder(
    session: AsyncSession = Depends(get_db_session),
    knowledge: KnowledgeService = Depends(get_knowledge_service),
    config: ContextConfig = Depends(get_context_config),
) -> ContextBuilderService:
    """Build a request-scoped context builder wired to the configured backends."""
    return ContextBuilderService(ProfileService(session), knowledge, config)
