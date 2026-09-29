"""Tests for prompt-context assembly (app/context).

Three levels are covered: the pure rendering and budget rules, the
graceful-degradation behaviour of the two context sources, and the chat endpoint
that consumes the assembled block.
"""

import pytest
from fastapi import FastAPI, status
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.context.config import (
    CONTEXT_PREAMBLE,
    EXCERPT_SEPARATOR,
    KNOWLEDGE_SECTION_HEADER,
    LINE_SEPARATOR,
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
from app.context.services.context_builder_service import ContextBuilderService, estimate_tokens
from app.core.config import Settings
from app.database.models.profile import UserProfile
from app.database.models.user import User
from app.database.session import get_db_session
from app.dependencies.knowledge import get_knowledge_config, get_knowledge_vector_store
from app.dependencies.memory import get_embedding_service, get_vector_store
from app.knowledge.config import KnowledgeConfig
from app.knowledge.schemas.knowledge import DocumentCreate, KnowledgeSearchResult
from app.knowledge.services.knowledge_service import KnowledgeService
from app.main import create_app
from app.memory.services.vector_store import VectorHit
from app.profile.schemas.profile import ProfileUpdateRequest
from app.profile.services.profile_service import ProfileService
from tests.conftest import FakeEmbeddingService, FakeVectorStore

KNOWLEDGE_TEST_CONFIG = KnowledgeConfig(chunk_max_chars=200, chunk_overlap=40, max_document_chars=5000, max_upload_bytes=10_000)


class StubProfiles:
    """Profile provider that replays one row, or fails the way a broken database would."""

    def __init__(self, profile: UserProfile | None = None, error: Exception | None = None) -> None:
        self._profile = profile
        self._error = error
        self.requested_user_ids: list[int] = []

    async def get_profile(self, user_id: int) -> UserProfile:
        self.requested_user_ids.append(user_id)
        if self._error is not None:
            raise self._error
        assert self._profile is not None, "stub profile not configured"
        return self._profile


class StubKnowledge:
    """Knowledge searcher that records how it was called and replays fixed hits."""

    def __init__(self, hits: list[KnowledgeSearchResult] | None = None, error: Exception | None = None) -> None:
        self._hits = list(hits or [])
        self._error = error
        self.calls: list[tuple[int, str, int]] = []

    async def search(self, user_id: int, query: str, limit: int) -> list[KnowledgeSearchResult]:
        self.calls.append((user_id, query, limit))
        if self._error is not None:
            raise self._error
        return self._hits[:limit]


class ScopedKnowledge:
    """Searcher whose answers depend on the user id, mirroring how retrieval is scoped."""

    def __init__(self, hits_by_user: dict[int, list[KnowledgeSearchResult]]) -> None:
        self._hits_by_user = hits_by_user

    async def search(self, user_id: int, query: str, limit: int) -> list[KnowledgeSearchResult]:
        return self._hits_by_user.get(user_id, [])[:limit]


def _profile(**values: object) -> UserProfile:
    """An in-memory profile row; attributes that are not passed stay null or empty."""
    defaults: dict[str, object] = {
        "user_id": 1,
        "summary": None,
        "timezone": None,
        "preferred_language": None,
        "interests": [],
        "goals": [],
        "preferences": {},
    }
    return UserProfile(**{**defaults, **values})  # type: ignore[arg-type]


def _hit(
    chunk_id: int,
    content: str,
    *,
    position: int | None = None,
    score: float = 0.9,
    document_id: int = 10,
    title: str = "Runbook",
) -> KnowledgeSearchResult:
    return KnowledgeSearchResult(
        score=score,
        document_id=document_id,
        document_title=title,
        chunk_id=chunk_id,
        position=position if position is not None else chunk_id,
        content=content,
        char_count=len(content),
    )


def _builder(
    profile: UserProfile | None = None,
    knowledge: list[KnowledgeSearchResult] | None = None,
    config: ContextConfig | None = None,
) -> ContextBuilderService:
    """Builder over stub sources: empty profile and no hits unless supplied."""
    return ContextBuilderService(
        StubProfiles(profile if profile is not None else _profile()),
        StubKnowledge(knowledge),
        config or ContextConfig(),
    )


@pytest.mark.asyncio
async def test_empty_sources_produce_an_empty_context() -> None:
    """A user with no profile and no matching documents simply gets no context."""
    context = await _builder().build_context(user_id=1, conversation_id=1, user_message="What should I read next?")

    assert context.profile_context == ""
    assert context.knowledge_context == ""
    assert context.combined_context == ""
    assert context.citations == []
    assert context.token_estimate == 0
    assert context.is_empty


@pytest.mark.asyncio
async def test_profile_section_lists_every_populated_attribute() -> None:
    profile = _profile(
        summary="  Prefers short answers  ",
        preferences={"tone": "concise", "format": "bullets"},
        interests=["rust", " chess "],
        goals=["Ship phase 5"],
        timezone="Europe/Berlin",
        preferred_language="en",
    )

    context = await _builder(profile=profile).build_context(user_id=1, conversation_id=1, user_message="hello")

    assert context.profile_context.startswith(PROFILE_SECTION_HEADER + LINE_SEPARATOR)
    assert f"{PROFILE_SUMMARY_LABEL}: Prefers short answers" in context.profile_context
    assert f"{PROFILE_PREFERENCES_LABEL}: tone=concise; format=bullets" in context.profile_context
    assert f"{PROFILE_INTERESTS_LABEL}: rust, chess" in context.profile_context
    assert f"{PROFILE_GOALS_LABEL}: Ship phase 5" in context.profile_context
    assert f"{PROFILE_TIMEZONE_LABEL}: Europe/Berlin" in context.profile_context
    assert f"{PROFILE_LANGUAGE_LABEL}: en" in context.profile_context
    assert context.knowledge_context == ""
    assert context.citations == []
    assert context.combined_context == CONTEXT_PREAMBLE + SECTION_SEPARATOR + context.profile_context
    assert KNOWLEDGE_SECTION_HEADER not in context.combined_context


@pytest.mark.asyncio
async def test_profile_section_skips_attributes_that_have_no_value() -> None:
    context = await _builder(profile=_profile(interests=["rowing"])).build_context(
        user_id=1, conversation_id=1, user_message="hello"
    )

    assert f"{PROFILE_INTERESTS_LABEL}: rowing" in context.profile_context
    unpopulated = (PROFILE_SUMMARY_LABEL, PROFILE_PREFERENCES_LABEL, PROFILE_GOALS_LABEL, PROFILE_TIMEZONE_LABEL, PROFILE_LANGUAGE_LABEL)
    assert all(label not in context.profile_context for label in unpopulated)


@pytest.mark.asyncio
async def test_knowledge_section_numbers_excerpts_and_cites_each_one() -> None:
    hits = [
        _hit(101, "Roll back by redeploying the previous tag.", position=0, score=0.91, document_id=7, title="Release runbook"),
        _hit(102, "Freeze the deploy queue first.", position=1, score=0.55, document_id=7, title="Release runbook"),
    ]

    context = await _builder(knowledge=hits).build_context(user_id=1, conversation_id=1, user_message="how do I roll back")

    lines = context.knowledge_context.split(LINE_SEPARATOR)
    assert lines[0] == KNOWLEDGE_SECTION_HEADER
    assert lines[1] == "[1] Release runbook (chunk 0)"
    assert lines[2] == "Roll back by redeploying the previous tag."
    assert lines[3] == ""  # excerpts are separated by a blank line
    assert lines[4] == "[2] Release runbook (chunk 1)"
    assert len(context.knowledge_context.split(EXCERPT_SEPARATOR)) == 2
    assert context.combined_context == CONTEXT_PREAMBLE + SECTION_SEPARATOR + context.knowledge_context
    assert [citation.chunk_id for citation in context.citations] == [101, 102]
    assert context.citations[0].document_id == 7
    assert context.citations[0].document_title == "Release runbook"
    assert context.citations[0].position == 0
    assert context.citations[1].score == pytest.approx(0.55)
    assert all(str(citation.chunk_id) not in context.knowledge_context for citation in context.citations)


@pytest.mark.asyncio
async def test_profile_is_rendered_before_knowledge_in_the_combined_block() -> None:
    profile = _profile(summary="Prefers short answers")
    hits = [_hit(1, "Roll back by redeploying the previous tag.")]

    context = await _builder(profile=profile, knowledge=hits).build_context(user_id=1, conversation_id=1, user_message="rollback?")

    assert context.combined_context.startswith(CONTEXT_PREAMBLE)
    assert context.combined_context.index(PROFILE_SECTION_HEADER) < context.combined_context.index(KNOWLEDGE_SECTION_HEADER)
    assert (
        context.combined_context
        == CONTEXT_PREAMBLE + SECTION_SEPARATOR + context.profile_context + SECTION_SEPARATOR + context.knowledge_context
    )


@pytest.mark.asyncio
async def test_profile_budget_trims_the_section_and_marks_the_trim() -> None:
    profile = _profile(summary="x" * 400, goals=["Ship phase 5"])

    context = await _builder(profile=profile, config=ContextConfig(profile_max_chars=120)).build_context(
        user_id=1, conversation_id=1, user_message="hello"
    )

    assert len(context.profile_context) <= 120
    assert context.profile_context.endswith(TRUNCATION_SUFFIX)
    assert "x" * 400 not in context.profile_context


@pytest.mark.asyncio
async def test_profile_trim_drops_whole_trailing_lines_rather_than_mid_sentence() -> None:
    profile = _profile(interests=["alpha", "beta", "gamma"], goals=["Ship phase 5"])
    roomy = await _builder(profile=profile).build_context(user_id=1, conversation_id=1, user_message="hello")
    kept = PROFILE_SECTION_HEADER + LINE_SEPARATOR + f"{PROFILE_INTERESTS_LABEL}: alpha, beta, gamma"
    assert roomy.profile_context == kept + LINE_SEPARATOR + f"{PROFILE_GOALS_LABEL}: Ship phase 5"

    trimmed = await _builder(
        profile=profile, config=ContextConfig(profile_max_chars=len(kept) + len(TRUNCATION_SUFFIX) + 1)
    ).build_context(user_id=1, conversation_id=1, user_message="hello")

    assert trimmed.profile_context == kept + TRUNCATION_SUFFIX
    assert PROFILE_GOALS_LABEL not in trimmed.profile_context
    assert len(trimmed.profile_context) <= len(kept) + len(TRUNCATION_SUFFIX) + 1


@pytest.mark.asyncio
async def test_knowledge_budget_drops_the_weakest_matches_first() -> None:
    hits = [
        _hit(1, "Roll back by redeploying the previous tag.", score=0.9),
        _hit(2, "Freeze the deploy queue before the rollback.", score=0.8),
        _hit(3, "The office kitchen closes at five.", score=0.1),
    ]
    roomy = ContextConfig(max_chars=10_000, profile_max_chars=10_000, knowledge_max_chars=10_000)
    message = "how do I roll back"

    all_three = await _builder(knowledge=hits, config=roomy).build_context(user_id=1, conversation_id=1, user_message=message)
    assert [citation.chunk_id for citation in all_three.citations] == [1, 2, 3]

    two_of_them = await _builder(knowledge=hits[:2], config=roomy).build_context(user_id=1, conversation_id=1, user_message=message)
    budget = len(two_of_them.knowledge_context)

    trimmed = await _builder(
        knowledge=hits,
        config=ContextConfig(max_chars=10_000, profile_max_chars=10_000, knowledge_max_chars=budget),
    ).build_context(user_id=1, conversation_id=1, user_message=message)

    assert [citation.chunk_id for citation in trimmed.citations] == [1, 2]
    assert "kitchen" not in trimmed.knowledge_context
    assert len(trimmed.knowledge_context) <= budget


@pytest.mark.asyncio
async def test_a_weak_chunk_never_takes_the_place_of_a_dropped_strong_one() -> None:
    """A short low-ranked chunk does not sneak in behind a strong chunk that did not fit."""
    strong = _hit(1, "Roll back by redeploying the previous tag and watch the error budget.", score=0.9)
    weak = _hit(2, "Too short to matter.", score=0.1)
    roomy = ContextConfig(max_chars=10_000, profile_max_chars=10_000, knowledge_max_chars=10_000)

    alone = await _builder(knowledge=[weak], config=roomy).build_context(user_id=1, conversation_id=1, user_message="rollback?")
    budget = len(alone.knowledge_context)
    assert [citation.chunk_id for citation in alone.citations] == [2]

    context = await _builder(
        knowledge=[strong, weak],
        config=ContextConfig(max_chars=10_000, profile_max_chars=10_000, knowledge_max_chars=budget),
    ).build_context(user_id=1, conversation_id=1, user_message="rollback?")

    assert context.knowledge_context == ""
    assert context.citations == []


@pytest.mark.asyncio
async def test_no_excerpt_is_rendered_or_cited_when_none_fit_the_budget() -> None:
    """A citation means the text reached the prompt, so a dropped chunk is never cited."""
    hits = [_hit(1, "Roll back by redeploying the previous tag.")]

    context = await _builder(
        knowledge=hits, config=ContextConfig(knowledge_max_chars=len(KNOWLEDGE_SECTION_HEADER))
    ).build_context(user_id=1, conversation_id=1, user_message="rollback?")

    assert context.knowledge_context == ""
    assert context.citations == []
    assert context.is_empty


@pytest.mark.asyncio
async def test_combined_context_never_exceeds_the_total_budget() -> None:
    """Section budgets bigger than the total are a misconfiguration, not an over-budget prompt."""
    profile = _profile(summary="Prefers short answers", interests=["rust"] * 20, goals=["Ship phase 5"])
    hits = [_hit(index, "Roll back by redeploying the previous tag. " * 3) for index in range(1, 6)]

    context = await _builder(
        profile=profile, knowledge=hits, config=ContextConfig(max_chars=200, profile_max_chars=1500, knowledge_max_chars=4500)
    ).build_context(user_id=1, conversation_id=1, user_message="rollback?")

    assert context.combined_context != ""
    assert len(context.combined_context) <= 200


@pytest.mark.asyncio
async def test_budget_left_by_the_profile_is_spent_on_knowledge() -> None:
    hits = [_hit(index, f"Roll back release {index} by redeploying the previous tag.") for index in range(1, 6)]
    config_without_profile = ContextConfig(max_chars=500, profile_max_chars=1500, knowledge_max_chars=4500)
    config_with_profile = ContextConfig(max_chars=500, profile_max_chars=300, knowledge_max_chars=4500)

    without_profile = await _builder(knowledge=hits, config=config_without_profile).build_context(
        user_id=1, conversation_id=1, user_message="rollback?"
    )
    with_profile = await _builder(
        profile=_profile(summary="s" * 200), knowledge=hits, config=config_with_profile
    ).build_context(user_id=1, conversation_id=1, user_message="rollback?")

    assert len(without_profile.combined_context) <= 500
    assert len(with_profile.combined_context) <= 500
    assert PROFILE_SUMMARY_LABEL in with_profile.profile_context
    assert len(with_profile.citations) < len(without_profile.citations)
    assert 0 < len(with_profile.knowledge_context) < len(without_profile.knowledge_context)


@pytest.mark.asyncio
async def test_knowledge_failure_degrades_to_profile_only_context() -> None:
    """A vector store outage must not turn into a failed chat turn."""
    profile = _profile(summary="Prefers short answers")
    builder = ContextBuilderService(
        StubProfiles(profile), StubKnowledge(error=RuntimeError("qdrant unreachable")), ContextConfig()
    )

    context = await builder.build_context(user_id=1, conversation_id=1, user_message="rollback?")

    assert context.knowledge_context == ""
    assert context.citations == []
    assert PROFILE_SUMMARY_LABEL in context.profile_context
    assert context.combined_context == CONTEXT_PREAMBLE + SECTION_SEPARATOR + context.profile_context


@pytest.mark.asyncio
async def test_profile_failure_degrades_to_knowledge_only_context() -> None:
    hits = [_hit(1, "Roll back by redeploying the previous tag.")]
    builder = ContextBuilderService(
        StubProfiles(error=RuntimeError("database unavailable")), StubKnowledge(hits), ContextConfig()
    )

    context = await builder.build_context(user_id=1, conversation_id=1, user_message="rollback?")

    assert context.profile_context == ""
    assert KNOWLEDGE_SECTION_HEADER in context.knowledge_context
    assert len(context.citations) == 1


@pytest.mark.asyncio
async def test_both_sources_failing_still_produces_a_usable_turn() -> None:
    builder = ContextBuilderService(
        StubProfiles(error=RuntimeError("database unavailable")),
        StubKnowledge(error=RuntimeError("qdrant unreachable")),
        ContextConfig(),
    )

    context = await builder.build_context(user_id=1, conversation_id=1, user_message="rollback?")

    assert context.is_empty
    assert context.combined_context == ""


@pytest.mark.asyncio
async def test_retrieval_is_scoped_capped_and_skipped_for_blank_messages() -> None:
    hits = [_hit(index, f"Roll back release {index}.") for index in range(1, 6)]
    knowledge = StubKnowledge(hits)
    builder = ContextBuilderService(StubProfiles(_profile()), knowledge, ContextConfig(knowledge_top_k=2))

    context = await builder.build_context(user_id=7, conversation_id=3, user_message="   how do I roll back?  \n")
    blank = await builder.build_context(user_id=7, conversation_id=3, user_message="  \n\t ")

    assert knowledge.calls == [(7, "how do I roll back?", 2)]
    assert [citation.chunk_id for citation in context.citations] == [1, 2]
    assert blank.knowledge_context == ""


@pytest.mark.asyncio
async def test_profile_lookup_uses_the_requesting_user_id() -> None:
    profiles = StubProfiles(_profile(summary="Prefers short answers"))
    builder = ContextBuilderService(profiles, StubKnowledge([]), ContextConfig())

    await builder.build_context(user_id=42, conversation_id=1, user_message="hello")

    assert profiles.requested_user_ids == [42]


@pytest.mark.asyncio
async def test_each_user_only_ever_sees_their_own_knowledge() -> None:
    searcher = ScopedKnowledge(
        {
            1: [_hit(1, "Alice rolls back with script A.")],
            2: [_hit(2, "Bob rolls back with script B.")],
        }
    )
    builder = ContextBuilderService(StubProfiles(_profile()), searcher, ContextConfig())

    alice = await builder.build_context(user_id=1, conversation_id=1, user_message="rollback?")
    bob = await builder.build_context(user_id=2, conversation_id=2, user_message="rollback?")
    stranger = await builder.build_context(user_id=3, conversation_id=3, user_message="rollback?")

    assert "script A" in alice.knowledge_context
    assert "script B" not in alice.knowledge_context
    assert "script B" in bob.knowledge_context
    assert "script A" not in bob.knowledge_context
    assert stranger.knowledge_context == ""
    assert stranger.citations == []


@pytest.mark.asyncio
async def test_token_estimate_is_a_character_estimate_of_the_combined_block() -> None:
    context = await _builder(
        profile=_profile(summary="Prefers short answers"), knowledge=[_hit(1, "Roll back safely.")]
    ).build_context(user_id=1, conversation_id=1, user_message="rollback?")

    assert context.token_estimate == estimate_tokens(context.combined_context) > 0


def test_estimate_tokens_rounds_up_and_counts_nothing_as_zero() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


@pytest.mark.parametrize(
    "budget",
    [
        {"max_chars": 0},
        {"max_chars": -1},
        {"profile_max_chars": 0},
        {"knowledge_max_chars": 0},
        {"knowledge_top_k": 0},
    ],
)
def test_context_config_rejects_non_positive_budgets(budget: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        ContextConfig(**budget)


def test_context_config_is_read_from_the_environment_settings() -> None:
    settings = Settings(context_max_chars=1234, profile_max_chars=111, knowledge_max_chars=222, knowledge_top_k=3)

    assert ContextConfig.from_settings(settings) == ContextConfig(
        max_chars=1234, profile_max_chars=111, knowledge_max_chars=222, knowledge_top_k=3
    )


class BrokenKnowledgeVectorStore(FakeVectorStore):
    """Knowledge vector store whose searches fail, the way an unreachable Qdrant would."""

    async def search(
        self,
        vector: list[float],
        limit: int,
        user_id: int,
        must_match: dict[str, object] | None = None,
    ) -> list[VectorHit]:
        raise RuntimeError("qdrant unreachable")


async def _register_and_login(
    client: AsyncClient, email: str, username: str, password: str = "securepassword123"
) -> str:
    await client.post("/api/v1/auth/register", json={"email": email, "username": username, "password": password})
    response = await client.post("/api/v1/auth/login", json={"email": email, "password": password})
    return response.json()["access_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _seed_user(session: AsyncSession, email: str, username: str) -> int:
    """Create a user row directly, for tests that drive services instead of endpoints."""
    user = User(email=email, username=username, hashed_password="not-a-real-hash")
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user.id


def _app_with_knowledge_store(session: AsyncSession, vectors: FakeVectorStore) -> FastAPI:
    """App wired to the in-memory fakes, with a caller-chosen knowledge vector store."""
    app = create_app()

    async def override_get_db() -> object:
        yield session

    embedding = FakeEmbeddingService()
    app.dependency_overrides[get_db_session] = override_get_db
    app.dependency_overrides[get_embedding_service] = lambda: embedding
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    app.dependency_overrides[get_knowledge_vector_store] = lambda: vectors
    app.dependency_overrides[get_knowledge_config] = lambda: KNOWLEDGE_TEST_CONFIG
    return app


_ROLLBACK_DOCUMENT = {
    "title": "Rollback runbook",
    "content": "Roll back the release by redeploying the previous tag. Freeze the deploy queue first.",
    "content_type": "text/plain",
}
_PAYROLL_DOCUMENT = {
    "title": "Payroll notes",
    "content": "Payroll runs on the last working day of the month after finance signs off.",
    "content_type": "text/plain",
}


@pytest.mark.asyncio
async def test_real_profile_and_retrieved_chunks_reach_the_context(db_session: AsyncSession) -> None:
    """The builder over the real profile and knowledge services, with no stubbed source."""
    alice = await _seed_user(db_session, "alice-context@example.com", "alicecontext")
    bob = await _seed_user(db_session, "bob-context@example.com", "bobcontext")
    knowledge = KnowledgeService(db_session, FakeEmbeddingService(), FakeVectorStore(), KNOWLEDGE_TEST_CONFIG)
    await knowledge.ingest_document(alice, DocumentCreate.model_validate(_ROLLBACK_DOCUMENT))
    await knowledge.ingest_document(bob, DocumentCreate.model_validate(_PAYROLL_DOCUMENT))
    await ProfileService(db_session).update_profile(alice, ProfileUpdateRequest(summary="Prefers short answers"))
    builder = ContextBuilderService(ProfileService(db_session), knowledge, ContextConfig())

    alice_context = await builder.build_context(user_id=alice, conversation_id=1, user_message="how do I roll back the release?")
    bob_context = await builder.build_context(user_id=bob, conversation_id=2, user_message="how do I roll back the release?")

    assert f"{PROFILE_SUMMARY_LABEL}: Prefers short answers" in alice_context.profile_context
    assert "Rollback runbook" in alice_context.knowledge_context
    assert alice_context.citations
    assert all(citation.document_title == "Rollback runbook" for citation in alice_context.citations)
    assert "Payroll" not in alice_context.combined_context
    assert bob_context.profile_context == ""
    assert "Rollback runbook" not in bob_context.combined_context


@pytest.mark.asyncio
async def test_chat_turn_with_profile_and_knowledge_still_returns_a_reply(client: AsyncClient) -> None:
    """The chat endpoint keeps working now that every turn assembles context first."""
    token = await _register_and_login(client, "context-chat@example.com", "contextchat")
    conversation = await client.post("/api/v1/conversations", json={"title": "Context"}, headers=_auth(token))
    conversation_id = conversation.json()["id"]
    await client.patch("/api/v1/profile", json={"summary": "Prefers short answers"}, headers=_auth(token))
    ingest = await client.post("/api/v1/knowledge/documents", json=_ROLLBACK_DOCUMENT, headers=_auth(token))
    assert ingest.status_code == status.HTTP_201_CREATED

    response = await client.post(
        f"/api/v1/conversations/{conversation_id}/chat",
        json={"content": "How do I roll back the release?"},
        headers=_auth(token),
    )

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["role"] == "assistant"
    assert body["content"]


@pytest.mark.asyncio
async def test_chat_turn_survives_a_failing_knowledge_search(db_session: AsyncSession) -> None:
    """A retrieval outage shortens the prompt; it never fails the turn."""
    app = _app_with_knowledge_store(db_session, BrokenKnowledgeVectorStore())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as anonymous:
        token = await _register_and_login(anonymous, "context-outage@example.com", "contextoutage")
        conversation = await anonymous.post("/api/v1/conversations", json={"title": "Outage"}, headers=_auth(token))
        conversation_id = conversation.json()["id"]
        await anonymous.post("/api/v1/knowledge/documents", json=_ROLLBACK_DOCUMENT, headers=_auth(token))

        response = await anonymous.post(
            f"/api/v1/conversations/{conversation_id}/chat",
            json={"content": "How do I roll back the release?"},
            headers=_auth(token),
        )

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["role"] == "assistant"




