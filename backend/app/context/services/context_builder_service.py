"""Assemble the prompt context handed to the chat service.

The pipeline (see app/context/README.md)::

    user message -> profile section + knowledge retrieval -> combined block -> chat service

This module is the only place that decides what the model is told about a user
before it answers. The chat service never queries profiles or knowledge itself;
memory, agents and training signals are meant to join here rather than be
threaded through the chat path.

Two rules shape the implementation:

* **Hard budgets.** The combined block is never longer than
  ``ContextConfig.max_chars``. Excerpts that do not fit are dropped from the
  bottom of the ranking upwards, and a dropped chunk is never cited.
* **Context is an enhancement, never a dependency.** A missing profile, an empty
  knowledge base or a vector store outage all produce a smaller context. Nothing
  here raises, so a chat turn cannot fail because retrieval returned nothing.
"""

from collections.abc import Sequence
from typing import Protocol

from app.context.config import (
    CHARS_PER_TOKEN_ESTIMATE,
    CONTEXT_PREAMBLE,
    EXCERPT_HEADER_TEMPLATE,
    EXCERPT_SEPARATOR,
    KNOWLEDGE_SECTION_HEADER,
    LABEL_SEPARATOR,
    LINE_SEPARATOR,
    LIST_SEPARATOR,
    PREFERENCE_SEPARATOR,
    PREFERENCE_VALUE_SEPARATOR,
    PROFILE_GOALS_LABEL,
    PROFILE_INTERESTS_LABEL,
    PROFILE_LANGUAGE_LABEL,
    PROFILE_PREFERENCES_LABEL,
    PROFILE_SECTION_HEADER,
    PROFILE_SUMMARY_LABEL,
    PROFILE_TIMEZONE_LABEL,
    SECTION_SEPARATOR,
    TRUNCATION_SUFFIX,
    ContextConfig,
)
from app.context.schemas.context import ContextCitation, ContextObject
from app.core.logging import logger
from app.database.models.profile import UserProfile
from app.knowledge.schemas.knowledge import KnowledgeSearchResult


class ProfileProvider(Protocol):
    """The slice of ``ProfileService`` the builder relies on."""

    async def get_profile(self, user_id: int) -> UserProfile: ...


class KnowledgeSearcher(Protocol):
    """The slice of ``KnowledgeService`` the builder relies on.

    Retrieval is scoped by ``user_id`` inside the knowledge service itself, so a
    caller cannot ask for another user's chunks by accident.
    """

    async def search(self, user_id: int, query: str, limit: int) -> list[KnowledgeSearchResult]: ...


class ContextBuilderService:
    """Turn a user's profile and knowledge into one bounded block of prompt text."""

    def __init__(
        self,
        profiles: ProfileProvider,
        knowledge: KnowledgeSearcher,
        config: ContextConfig | None = None,
    ) -> None:
        self._profiles = profiles
        self._knowledge = knowledge
        self._config = config if config is not None else ContextConfig()

    @property
    def config(self) -> ContextConfig:
        """The budgets this builder was configured with."""
        return self._config

    async def build_context(self, user_id: int, conversation_id: int, user_message: str) -> ContextObject:
        """Build the context for one turn; never raises because a source is empty or down."""
        profile_text = await self._profile_context(user_id)
        hits = await self._retrieve(user_id, conversation_id, user_message)
        knowledge_text, citations = _render_knowledge(hits, self._knowledge_budget(profile_text))
        combined = _combine(profile_text, knowledge_text, self._config.max_chars)
        context = ContextObject(
            profile_context=profile_text,
            knowledge_context=knowledge_text,
            combined_context=combined,
            citations=citations,
            token_estimate=estimate_tokens(combined),
        )
        logger.info(
            "Built context for user %s conversation %s profile_chars=%s knowledge_chars=%s "
            "combined_chars=%s citations=%s token_estimate=%s",
            user_id,
            conversation_id,
            len(profile_text),
            len(knowledge_text),
            len(combined),
            len(citations),
            context.token_estimate,
        )
        return context

    async def _profile_context(self, user_id: int) -> str:
        """Render the profile section, or an empty string when the profile is unusable."""
        try:
            profile = await self._profiles.get_profile(user_id)
        except Exception:
            logger.warning("Profile unavailable for user %s; continuing without profile context", user_id, exc_info=True)
            return ""
        return _render_profile(profile, self._config.profile_max_chars)

    async def _retrieve(self, user_id: int, conversation_id: int, user_message: str) -> list[KnowledgeSearchResult]:
        """Search the user's knowledge, or return nothing when there is nothing to search."""
        query = user_message.strip()
        if not query:
            logger.debug("Skipping knowledge retrieval for conversation %s: message is blank", conversation_id)
            return []
        try:
            hits = await self._knowledge.search(user_id, query, self._config.knowledge_top_k)
        except Exception:
            logger.warning(
                "Knowledge retrieval failed for user %s conversation %s; continuing without knowledge context",
                user_id,
                conversation_id,
                exc_info=True,
            )
            return []
        return list(hits)[: self._config.knowledge_top_k]

    def _knowledge_budget(self, profile_text: str) -> int:
        """Characters left for retrieval after the fixed preamble and the profile section."""
        used = len(_prefix(profile_text))
        remaining = self._config.max_chars - used
        return max(0, min(self._config.knowledge_max_chars, remaining))


def estimate_tokens(text: str) -> int:
    """Rough token count of a block of text; see ``CHARS_PER_TOKEN_ESTIMATE``."""
    if not text:
        return 0
    return (len(text) + CHARS_PER_TOKEN_ESTIMATE - 1) // CHARS_PER_TOKEN_ESTIMATE


def _prefix(profile_text: str) -> str:
    """The combined block up to the point where knowledge excerpts would start."""
    if profile_text:
        return CONTEXT_PREAMBLE + SECTION_SEPARATOR + profile_text + SECTION_SEPARATOR
    return CONTEXT_PREAMBLE + SECTION_SEPARATOR


def _combine(profile_text: str, knowledge_text: str, max_chars: int) -> str:
    """Join the sections under one preamble, keeping the result inside the budget."""
    if not profile_text and not knowledge_text:
        return ""
    if knowledge_text:
        combined = _prefix(profile_text) + knowledge_text
    elif profile_text:
        combined = CONTEXT_PREAMBLE + SECTION_SEPARATOR + profile_text
    else:
        combined = CONTEXT_PREAMBLE
    if len(combined) <= max_chars:
        return combined
    # Unreachable while the section budgets are smaller than the total; kept so a
    # misconfigured environment degrades into a truncated block instead of an
    # over-budget prompt.
    logger.error("Combined context of %s characters exceeds the %s character budget; hard-trimming", len(combined), max_chars)
    return _fit(combined, max_chars)


def _render_profile(profile: UserProfile | None, budget: int) -> str:
    """Render the populated profile attributes only; empty ones are skipped."""
    if profile is None:
        return ""
    lines: list[str] = []
    _add_line(lines, PROFILE_SUMMARY_LABEL, _clean(profile.summary))
    _add_line(lines, PROFILE_PREFERENCES_LABEL, _join_preferences(profile.preferences))
    _add_line(lines, PROFILE_INTERESTS_LABEL, _join_items(profile.interests))
    _add_line(lines, PROFILE_GOALS_LABEL, _join_items(profile.goals))
    _add_line(lines, PROFILE_TIMEZONE_LABEL, _clean(profile.timezone))
    _add_line(lines, PROFILE_LANGUAGE_LABEL, _clean(profile.preferred_language))
    if not lines:
        return ""
    return _fit(PROFILE_SECTION_HEADER + LINE_SEPARATOR + LINE_SEPARATOR.join(lines), budget)


def _render_knowledge(hits: Sequence[KnowledgeSearchResult], budget: int) -> tuple[str, list[ContextCitation]]:
    """Render a prefix of the ranking, stopping at the first excerpt that does not fit.

    Retrieval returns hits best-match first, so stopping early drops the weakest
    matches and keeps the excerpts that reach the prompt a strict prefix of the
    ranking. A dropped chunk is never cited: a citation means the text actually
    reached the prompt.
    """
    if not hits:
        return "", []
    if budget <= len(KNOWLEDGE_SECTION_HEADER):
        logger.debug("Knowledge budget of %s characters leaves no room for excerpts", budget)
        return "", []
    rendered: list[str] = []
    citations: list[ContextCitation] = []
    used = len(KNOWLEDGE_SECTION_HEADER) + len(LINE_SEPARATOR)
    for hit in hits:
        excerpt = _render_excerpt(len(rendered) + 1, hit)
        cost = len(excerpt) if not rendered else len(EXCERPT_SEPARATOR) + len(excerpt)
        if used + cost > budget:
            logger.debug(
                "Knowledge chunk %s does not fit the %s character budget; dropping it and everything ranked below it",
                hit.chunk_id,
                budget,
            )
            break
        rendered.append(excerpt)
        used += cost
        citations.append(
            ContextCitation(
                document_id=hit.document_id,
                chunk_id=hit.chunk_id,
                document_title=hit.document_title,
                position=hit.position,
                score=hit.score,
            )
        )
    if not rendered:
        return "", []
    return KNOWLEDGE_SECTION_HEADER + LINE_SEPARATOR + EXCERPT_SEPARATOR.join(rendered), citations


def _render_excerpt(index: int, hit: KnowledgeSearchResult) -> str:
    """One numbered excerpt: attribution line followed by the retrieved text."""
    header = EXCERPT_HEADER_TEMPLATE.format(index=index, title=hit.document_title, position=hit.position)
    return header + LINE_SEPARATOR + hit.content


def _add_line(lines: list[str], label: str, value: str) -> None:
    if value:
        lines.append(label + LABEL_SEPARATOR + value)


def _clean(value: str | None) -> str:
    return value.strip() if value else ""


def _join_items(items: Sequence[str] | None) -> str:
    if not items:
        return ""
    return LIST_SEPARATOR.join(item.strip() for item in items if item.strip())


def _join_preferences(preferences: dict[str, str] | None) -> str:
    if not preferences:
        return ""
    pairs = [f"{key.strip()}{PREFERENCE_VALUE_SEPARATOR}{_clean(value)}" for key, value in preferences.items() if key.strip()]
    return PREFERENCE_SEPARATOR.join(pairs)


def _fit(text: str, budget: int) -> str:
    """Trim ``text`` to ``budget``, preferring to drop a whole trailing line."""
    if len(text) <= budget:
        return text
    if budget <= len(TRUNCATION_SUFFIX):
        return ""
    trimmed = text[: budget - len(TRUNCATION_SUFFIX)]
    cut = trimmed.rfind(LINE_SEPARATOR)
    if cut > 0:
        trimmed = trimmed[:cut]
    return trimmed + TRUNCATION_SUFFIX

