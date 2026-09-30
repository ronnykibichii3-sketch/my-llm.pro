"""Tests for the inference provider layer (app/inference).

Four levels are covered: configuration resolution and validation, prompt
composition, the service's fallback and logging decisions with a stub provider, and
the Ollama client against a mocked HTTP transport. No test here opens a socket to a
model: the whole point of the provider boundary is that each side can be exercised
alone.
"""

import json
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, status
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.context.config import LINE_SEPARATOR, SECTION_SEPARATOR
from app.context.schemas.context import ContextCitation, ContextObject
from app.context.services.context_builder_service import estimate_tokens
from app.core.config import (
    ANTHROPIC_PROVIDER,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_PROVIDER,
    OPENAI_PROVIDER,
    Settings,
)
from app.database.session import get_db_session
from app.dependencies.inference import get_inference_service
from app.dependencies.knowledge import get_knowledge_config, get_knowledge_vector_store
from app.dependencies.memory import get_embedding_service, get_vector_store
from app.inference.config import (
    CONTEXT_SECTION_HEADER,
    DEFAULT_LLM_MAX_OUTPUT_TOKENS,
    DEFAULT_SYSTEM_PROMPT,
    FALLBACK_PROVIDER_NAME,
    FALLBACK_RESPONSE,
    IMPLEMENTATION_PENDING_MESSAGE,
    NO_CONTEXT_STATEMENT,
    OLLAMA_CHAT_PATH,
    SYSTEM_ROLE,
    USER_ROLE,
    InferenceConfig,
)
from app.inference.exceptions import (
    ProviderConfigurationError,
    ProviderResponseError,
    ProviderTransportError,
    InferenceUnavailableError,
)
from app.inference.providers.anthropic_provider import AnthropicProvider
from app.inference.providers.base import LLMProvider, truncate_detail
from app.inference.providers.factory import create_provider, create_provider_from_settings
from app.inference.providers.ollama_provider import (
    CONTENT_FIELD,
    EVAL_COUNT_FIELD,
    MESSAGE_FIELD,
    MESSAGES_FIELD,
    MODEL_FIELD,
    NUM_PREDICT_OPTION,
    OPTIONS_FIELD,
    OllamaProvider,
    PROMPT_EVAL_COUNT_FIELD,
    ROLE_FIELD,
    STREAM_FIELD,
    TEMPERATURE_OPTION,
)
from app.inference.providers.openai_provider import OpenAIProvider
from app.inference.services.inference_service import FALLBACK_EVENT, InferenceService
from app.inference.services.prompt_builder import PromptBuilder, compose_system_message
from app.knowledge.config import KnowledgeConfig
from app.main import create_app
from tests.conftest import FAKE_LLM_PROVIDER_NAME, FAKE_LLM_REPLY, FakeEmbeddingService, FakeLLMProvider, FakeVectorStore


OLLAMA_PROMPT_EVAL_COUNT = 31
OLLAMA_EVAL_COUNT = 7
OLLAMA_REPORTED_MODEL = "llama3.2:3b-instruct"
PROVIDER_ERROR_DETAIL = "model 'missing-model' not found"


# --- configuration -----------------------------------------------------------


def test_default_config_targets_ollama_without_a_credential() -> None:
    """The untouched configuration answers with the default local model, unsigned."""
    config = InferenceConfig()

    assert config.provider == OLLAMA_PROVIDER
    assert config.model == DEFAULT_OLLAMA_MODEL
    assert config.base_url == DEFAULT_OLLAMA_BASE_URL
    assert config.api_key is None
    assert config.max_output_tokens == DEFAULT_LLM_MAX_OUTPUT_TOKENS
    assert config.fallback_enabled


def test_from_settings_selects_the_selected_provider_settings() -> None:
    """Each provider reads its own model, endpoint and credential."""
    config = InferenceConfig.from_settings(
        Settings(llm_provider=OLLAMA_PROVIDER, ollama_model="qwen2.5:7b", ollama_base_url="http://nas:11434")
    )

    assert (config.provider, config.model, config.base_url, config.api_key) == (
        OLLAMA_PROVIDER,
        "qwen2.5:7b",
        "http://nas:11434",
        None,
    )


def test_from_settings_unwraps_the_openai_key_without_logging_it() -> None:
    """A configured key arrives as text for the provider, and an empty one counts as absent."""
    configured = InferenceConfig.from_settings(Settings(llm_provider=OPENAI_PROVIDER, openai_api_key=SecretStr("k-42")))
    blank = InferenceConfig.from_settings(Settings(llm_provider=OPENAI_PROVIDER, openai_api_key=SecretStr("   ")))

    assert configured.api_key == "k-42"
    assert configured.model and configured.base_url
    assert blank.api_key is None


def test_fallback_setting_reaches_the_config() -> None:
    """``LLM_FALLBACK_ENABLED=false`` is carried through so the service can raise instead."""
    config = InferenceConfig.from_settings(Settings(llm_fallback_enabled=False))

    assert config.fallback_enabled is False


@pytest.mark.parametrize("provider", ["gemini", "ollam", ""])
def test_unknown_provider_is_rejected_by_both_layers(provider: str) -> None:
    """An unsupported name fails at settings validation and at config construction."""
    with pytest.raises(ValidationError):
        Settings(llm_provider=provider)
    if provider:
        with pytest.raises(ValueError, match="LLM_PROVIDER must be one of"):
            InferenceConfig(provider=provider)


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": "   "},
        {"base_url": " "},
        {"request_timeout_seconds": 0},
        {"max_output_tokens": 0},
        {"temperature": 2.5},
    ],
)
def test_unusable_provider_settings_are_rejected(overrides: dict[str, Any]) -> None:
    """A provider is never constructed with a blank endpoint or a nonsensical sampler."""
    with pytest.raises(ValueError):
        InferenceConfig(**overrides)


def test_settings_provider_name_is_normalised() -> None:
    """``LLM_PROVIDER=Ollama`` in an environment file still selects ollama."""
    assert Settings(llm_provider="  Ollama ").llm_provider == OLLAMA_PROVIDER


# --- prompt composition ------------------------------------------------------


def _context(text: str, citations: int = 0) -> ContextObject:
    """A turn's assembled context, with as many citations as the test asks for."""
    return ContextObject(
        combined_context=text,
        token_estimate=estimate_tokens(text),
        citations=[
            ContextCitation(
                document_id=index, chunk_id=index, document_title=f"Doc {index}", position=index, score=0.9
            )
            for index in range(1, citations + 1)
        ],
    )


def test_prompt_builder_keeps_the_three_parts_separate() -> None:
    """The context is not folded into the instruction or into the message by the builder."""
    prompt = PromptBuilder().build_prompt(_context("User prefers evening sessions.", citations=2), "  Plan my day  ")

    assert prompt.user_prompt == "Plan my day"
    assert prompt.context == "User prefers evening sessions."
    assert prompt.context_size == len("User prefers evening sessions.")
    assert prompt.citation_count == 2


def test_prompt_builder_rejects_a_blank_message() -> None:
    """Whitespace is not a question worth a model call."""
    with pytest.raises(ValueError, match="user_prompt must not be empty"):
        PromptBuilder().build_prompt(_context(""), "   \n ")


def test_system_message_introduces_the_context_block() -> None:
    """The instruction comes first, then the label, then the context builder's block."""
    message = compose_system_message("Be brief.", "Known about the user:\n- prefers evenings")

    assert message == (
        "Be brief." + SECTION_SEPARATOR + CONTEXT_SECTION_HEADER + LINE_SEPARATOR + "Known about the user:\n- prefers evenings"
    )


def test_system_message_says_when_there_is_no_context() -> None:
    """An empty context is stated rather than left as a gap the model has to guess at."""
    assert compose_system_message("Be brief.", "   ").endswith(NO_CONTEXT_STATEMENT)


# --- the service around a provider -------------------------------------------


@pytest.mark.asyncio
async def test_service_hands_the_three_parts_to_the_provider() -> None:
    """The provider receives instruction, message and context as separate arguments."""
    provider = FakeLLMProvider()

    await InferenceService(provider=provider).answer(_context("Profile: likes trails."), "Where should I run?")

    assert len(provider.calls) == 1
    system_prompt, user_prompt, context = provider.calls[0]
    assert user_prompt == "Where should I run?"
    assert context == "Profile: likes trails."
    assert system_prompt == DEFAULT_SYSTEM_PROMPT
    assert CONTEXT_SECTION_HEADER not in system_prompt


@pytest.mark.asyncio
async def test_service_returns_the_provider_answer_and_its_measurements() -> None:
    """What the provider reports - model, provider, counts, latency - is what callers see."""
    response = await InferenceService(provider=FakeLLMProvider()).answer(_context(""), "Hello")

    assert response.content == FAKE_LLM_REPLY
    assert response.provider == "fake"
    assert response.model == "fake-model"
    assert (response.input_tokens_estimate, response.output_tokens_estimate) == (12, 4)
    assert response.latency_ms == 5


@pytest.mark.asyncio
async def test_unreachable_provider_is_answered_with_the_stored_apology() -> None:
    """A down provider still leaves the turn with a readable, persistable reply."""
    provider = FakeLLMProvider(error=ProviderTransportError("connection refused"))

    response = await InferenceService(provider=provider).answer(_context("Some context."), "Hello")

    assert response.content == FALLBACK_RESPONSE
    assert response.provider == FALLBACK_PROVIDER_NAME
    assert response.model == "fake-model"
    assert response.output_tokens_estimate == estimate_tokens(FALLBACK_RESPONSE)
    assert response.latency_ms == 0


@pytest.mark.asyncio
async def test_fallback_can_be_switched_off_to_surface_the_failure() -> None:
    """With ``LLM_FALLBACK_ENABLED=false`` the real reason reaches the caller."""
    service = InferenceService(provider=FakeLLMProvider(error=ProviderTransportError("connection refused")), fallback_enabled=False)

    with pytest.raises(InferenceUnavailableError, match="connection refused") as caught:
        await service.answer(_context(""), "Hello")

    assert caught.value.provider == "fake"


class MisconfiguredStub:
    """Provider whose failure is a deployment mistake rather than an outage."""

    @property
    def name(self) -> str:
        return OPENAI_PROVIDER

    @property
    def model(self) -> str:
        return "gpt-4o-mini"

    async def generate(self, system_prompt: str, user_prompt: str, context: str = "") -> Any:
        raise ProviderConfigurationError(OPENAI_PROVIDER, "not implemented yet")


@pytest.mark.asyncio
async def test_configuration_errors_are_never_fallen_back_over() -> None:
    """A broken deployment is reported, not hidden behind a polite apology."""
    service = InferenceService(provider=MisconfiguredStub())  # type: ignore[arg-type]

    with pytest.raises(ProviderConfigurationError):
        await service.answer(_context(""), "Hello")


def test_providers_satisfy_the_provider_contract() -> None:
    """The stub and the real client are interchangeable behind the protocol."""
    assert isinstance(FakeLLMProvider(), LLMProvider)
    assert isinstance(OllamaProvider(base_url=DEFAULT_OLLAMA_BASE_URL, model=DEFAULT_OLLAMA_MODEL), LLMProvider)


def test_error_detail_is_flattened_and_bounded() -> None:
    """A huge error page becomes one short loggable line."""
    detail = truncate_detail("line one\n" + "x" * 400 + "\nline two")

    assert "\n" not in detail
    assert detail.endswith("...")
    assert len(detail) <= 304


# --- the Ollama client, mocked at the HTTP boundary --------------------------


def _ollama(monkeypatch: pytest.MonkeyPatch, handler: httpx.MockTransport, **overrides: Any) -> OllamaProvider:
    """An Ollama client whose network calls are answered by ``handler``."""
    real_client = httpx.AsyncClient

    def client(**kwargs: Any) -> httpx.AsyncClient:
        return real_client(transport=handler, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    return OllamaProvider(
        base_url=DEFAULT_OLLAMA_BASE_URL,
        model=DEFAULT_OLLAMA_MODEL,
        request_timeout_seconds=float(overrides.pop("request_timeout_seconds", 5.0)),
        **overrides,
    )


def _handler(
    captured: list[httpx.Request],
    *,
    json_body: dict[str, Any] | None = None,
    status_code: int = status.HTTP_200_OK,
    text: str | None = None,
    failure: Exception | None = None,
) -> httpx.MockTransport:
    """A transport that records the request and replays a canned answer or fault."""

    def answer(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if failure is not None:
            raise failure
        if text is not None:
            return httpx.Response(status_code, text=text)
        return httpx.Response(status_code, json=json_body or {})

    return httpx.MockTransport(answer)


def _completion(
    content: str = "  Take the ridge trail at 6am.  ",
    *,
    prompt_eval_count: int | None = None,
    eval_count: int | None = None,
) -> dict[str, Any]:
    """A well-formed completion, with real token counts only when the test asks for them."""
    body: dict[str, Any] = {MODEL_FIELD: OLLAMA_REPORTED_MODEL, MESSAGE_FIELD: {CONTENT_FIELD: content}}
    if prompt_eval_count is not None:
        body[PROMPT_EVAL_COUNT_FIELD] = prompt_eval_count
    if eval_count is not None:
        body[EVAL_COUNT_FIELD] = eval_count
    return body


@pytest.mark.asyncio
async def test_ollama_sends_the_composed_turn_in_ollamas_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """One POST to /api/chat carrying system, then user, with streaming switched off."""
    captured: list[httpx.Request] = []
    provider = _ollama(monkeypatch, _handler(captured, json_body=_completion()))

    await provider.generate("Answer as the assistant.", "Where should I run?", "Profile: likes trails.")

    assert len(captured) == 1
    request = captured[0]
    assert str(request.url) == DEFAULT_OLLAMA_BASE_URL + OLLAMA_CHAT_PATH
    payload = json.loads(request.content)
    assert payload[MODEL_FIELD] == DEFAULT_OLLAMA_MODEL
    assert payload[STREAM_FIELD] is False
    assert payload[OPTIONS_FIELD][NUM_PREDICT_OPTION] == DEFAULT_LLM_MAX_OUTPUT_TOKENS
    assert [turn[ROLE_FIELD] for turn in payload[MESSAGES_FIELD]] == [SYSTEM_ROLE, USER_ROLE]
    assert payload[MESSAGES_FIELD][1][CONTENT_FIELD] == "Where should I run?"


@pytest.mark.asyncio
async def test_context_reaches_ollama_inside_the_system_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """The context block travels with the instruction, which is what steers the answer."""
    captured: list[httpx.Request] = []
    provider = _ollama(monkeypatch, _handler(captured, json_body=_completion()))

    await provider.generate("Answer as the assistant.", "Where should I run?", "Profile: likes trails.")

    system_message = json.loads(captured[0].content)[MESSAGES_FIELD][0][CONTENT_FIELD]
    assert CONTEXT_SECTION_HEADER + LINE_SEPARATOR + "Profile: likes trails." in system_message


@pytest.mark.asyncio
async def test_ollama_sends_the_sampling_configuration_as_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """The configured temperature and answer-length cap travel in the options object."""
    captured: list[httpx.Request] = []
    provider = _ollama(monkeypatch, _handler(captured, json_body=_completion()), temperature=0.2, max_output_tokens=64)

    await provider.generate("Answer as the assistant.", "Hello")

    options = json.loads(captured[0].content)[OPTIONS_FIELD]
    assert options[TEMPERATURE_OPTION] == 0.2
    assert options[NUM_PREDICT_OPTION] == 64


@pytest.mark.asyncio
async def test_ollama_answer_uses_the_models_own_token_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reported counts are stored instead of estimated ones, and the reply is trimmed."""
    provider = _ollama(monkeypatch, _handler([], json_body=_completion(prompt_eval_count=OLLAMA_PROMPT_EVAL_COUNT, eval_count=OLLAMA_EVAL_COUNT)))

    response = await provider.generate("Answer as the assistant.", "Hello")

    assert response.content == "Take the ridge trail at 6am."
    assert (response.input_tokens_estimate, response.output_tokens_estimate) == (OLLAMA_PROMPT_EVAL_COUNT, OLLAMA_EVAL_COUNT)
    assert response.model == OLLAMA_REPORTED_MODEL
    assert response.provider == OLLAMA_PROVIDER
    assert response.latency_ms >= 0


@pytest.mark.asyncio
async def test_ollama_without_reported_counts_falls_back_to_estimates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Some Ollama versions omit the counters; the turn still records measurable sizes."""
    provider = _ollama(monkeypatch, _handler([], json_body=_completion(content="Take the ridge trail at 6am.")))

    response = await provider.generate("Answer as the assistant.", "Hello", "Profile: likes trails.")

    assert response.input_tokens_estimate == estimate_tokens(
        compose_system_message("Answer as the assistant.", "Profile: likes trails.") + "Hello"
    )
    assert response.output_tokens_estimate == estimate_tokens("Take the ridge trail at 6am.")


@pytest.mark.asyncio
async def test_a_refused_connection_is_reported_as_a_transport_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ollama not running is a transport failure, not a missing answer."""
    provider = _ollama(monkeypatch, _handler([], failure=httpx.ConnectError("connection refused")))

    with pytest.raises(ProviderTransportError, match="connection refused"):
        await provider.generate("Answer as the assistant.", "Hello")


@pytest.mark.asyncio
async def test_a_slow_model_reports_the_seconds_it_was_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The timeout recorded in the log is the configured one, not a guess."""
    provider = _ollama(monkeypatch, _handler([], failure=httpx.ReadTimeout("read timed out")), request_timeout_seconds=3.0)

    with pytest.raises(ProviderTransportError, match="within 3.0 seconds"):
        await provider.generate("Answer as the assistant.", "Hello")


@pytest.mark.asyncio
async def test_an_http_error_keeps_ollamas_own_explanation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reason Ollama gives - a missing model, usually - survives into the error."""
    provider = _ollama(monkeypatch, _handler([], text=PROVIDER_ERROR_DETAIL, status_code=status.HTTP_404_NOT_FOUND))

    with pytest.raises(ProviderResponseError, match="not found"):
        await provider.generate("Answer as the assistant.", "Hello")


@pytest.mark.asyncio
async def test_a_malformed_body_is_a_response_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answering with anything but JSON is a response failure with a clear reason."""
    provider = _ollama(monkeypatch, _handler([], text="<html><body>proxy error</body></html>"))

    with pytest.raises(ProviderResponseError, match="invalid JSON"):
        await provider.generate("Answer as the assistant.", "Hello")


@pytest.mark.asyncio
async def test_an_empty_completion_is_refused_rather_than_stored(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank completion would be stored as an assistant reply the user never got."""
    provider = _ollama(monkeypatch, _handler([], json_body=_completion(content="   ")))

    with pytest.raises(ProviderResponseError, match="empty completion"):
        await provider.generate("Answer as the assistant.", "Hello")


@pytest.mark.asyncio
async def test_a_body_without_a_message_is_a_response_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the message field there is no answer to store, whatever else arrived."""
    provider = _ollama(monkeypatch, _handler([], json_body={MODEL_FIELD: OLLAMA_REPORTED_MODEL}))

    with pytest.raises(ProviderResponseError, match=MESSAGE_FIELD):
        await provider.generate("Answer as the assistant.", "Hello")


# --- which provider the configuration selects --------------------------------


def test_the_default_configuration_builds_an_ollama_client() -> None:
    """The shipped configuration selects the one provider that is implemented."""
    provider = create_provider(InferenceConfig())

    assert isinstance(provider, OllamaProvider)
    assert (provider.name, provider.model) == (OLLAMA_PROVIDER, DEFAULT_OLLAMA_MODEL)


def test_building_a_cloud_provider_without_a_key_is_refused() -> None:
    """Constructing OpenAI or Anthropic without a credential is caught at startup."""
    with pytest.raises(ProviderConfigurationError, match="API key"):
        create_provider(InferenceConfig(provider=OPENAI_PROVIDER))
    with pytest.raises(ProviderConfigurationError, match="API key"):
        create_provider(InferenceConfig(provider=ANTHROPIC_PROVIDER, api_key=None))


def test_cloud_providers_construct_once_a_key_is_configured() -> None:
    """With a key the clients exist and are selectable, even though they cannot answer yet."""
    openai = create_provider(InferenceConfig(provider=OPENAI_PROVIDER, api_key="sk-test"))
    anthropic = create_provider(InferenceConfig(provider=ANTHROPIC_PROVIDER, api_key="sk-test"))

    assert isinstance(openai, OpenAIProvider)
    assert isinstance(anthropic, AnthropicProvider)
    assert (openai.name, anthropic.name) == (OPENAI_PROVIDER, ANTHROPIC_PROVIDER)


@pytest.mark.asyncio
async def test_an_unimplemented_provider_declines_explaining_itself() -> None:
    """Asking OpenAI or Anthropic to answer says so, instead of pretending to work."""
    openai = create_provider(InferenceConfig(provider=OPENAI_PROVIDER, api_key="sk-test"))
    anthropic = create_provider(InferenceConfig(provider=ANTHROPIC_PROVIDER, api_key="sk-test"))

    with pytest.raises(ProviderConfigurationError, match=IMPLEMENTATION_PENDING_MESSAGE.format(provider=OPENAI_PROVIDER)):
        await openai.generate("Answer as the assistant.", "Hello")
    with pytest.raises(ProviderConfigurationError, match=IMPLEMENTATION_PENDING_MESSAGE.format(provider=ANTHROPIC_PROVIDER)):
        await anthropic.generate("Answer as the assistant.", "Hello")


def test_the_settings_driven_service_builds_the_provider_they_name() -> None:
    """The shipped settings resolve to the implemented local provider and its model."""
    service = InferenceService.from_settings(Settings())

    assert isinstance(service.provider, OllamaProvider)
    assert service.provider.model == DEFAULT_OLLAMA_MODEL


def test_environment_style_configuration_reaches_the_provider() -> None:
    """The values an operator puts in an environment file are the ones sent to Ollama."""
    provider = create_provider_from_settings(Settings(llm_provider=OLLAMA_PROVIDER, ollama_model="qwen2.5:7b"))

    assert isinstance(provider, OllamaProvider)
    assert provider.model == "qwen2.5:7b"


# --- the chat endpoint behind the new boundary -------------------------------


KNOWLEDGE_TEST_CONFIG = KnowledgeConfig(chunk_max_chars=200, chunk_overlap=40, max_document_chars=5000, max_upload_bytes=10_000)


def _app_with_inference(db_session: AsyncSession, service: InferenceService) -> FastAPI:
    """The real application with a real database, faked retrieval and one inference service."""
    app = create_app()
    app.dependency_overrides[get_db_session] = lambda: db_session
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    app.dependency_overrides[get_knowledge_vector_store] = lambda: FakeVectorStore()
    app.dependency_overrides[get_embedding_service] = lambda: FakeEmbeddingService()
    app.dependency_overrides[get_knowledge_config] = lambda: KNOWLEDGE_TEST_CONFIG
    app.dependency_overrides[get_inference_service] = lambda: service
    return app


async def _ask_the_assistant(
    app: FastAPI,
    email: str,
    username: str,
    content: str = "Where should I run this weekend?",
    password: str = "securepassword123",
) -> tuple[int, dict[str, Any]]:
    """One signed-in chat turn against ``app``, returning the status and body."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        registered = await client.post(
            "/api/v1/auth/register", json={"email": email, "username": username, "password": password}
        )
        assert registered.status_code == status.HTTP_201_CREATED
        login = await client.post("/api/v1/auth/login", json={"email": email, "password": password})
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
        conversation = await client.post("/api/v1/conversations", json={"title": "Weekend plan"}, headers=headers)
        conversation_id = conversation.json()["id"]

        response = await client.post(
            f"/api/v1/conversations/{conversation_id}/chat", json={"content": content}, headers=headers
        )
        return response.status_code, response.json()


@pytest.mark.asyncio
async def test_the_chat_endpoint_answers_with_the_providers_reply(db_session: AsyncSession) -> None:
    """The route's reply is now the provider's answer rather than a stored constant."""
    app = _app_with_inference(db_session, InferenceService(provider=FakeLLMProvider()))

    status_code, body = await _ask_the_assistant(app, "provider-chat@example.com", "providerchat")

    assert status_code == status.HTTP_200_OK
    assert body["role"] == "assistant"
    assert body["content"] == FAKE_LLM_REPLY


@pytest.mark.asyncio
async def test_the_chat_endpoint_still_answers_while_the_model_is_down(db_session: AsyncSession) -> None:
    """A down model is an apology in the transcript, not a failed request."""
    service = InferenceService(provider=FakeLLMProvider(error=ProviderTransportError("connection refused")))
    app = _app_with_inference(db_session, service)

    status_code, body = await _ask_the_assistant(app, "outage-chat@example.com", "outagechat")

    assert status_code == status.HTTP_200_OK
    assert body["content"] == FALLBACK_RESPONSE


@pytest.mark.asyncio
async def test_the_chat_endpoint_reports_a_misconfigured_provider(db_session: AsyncSession) -> None:
    """A deployment mistake reaches the client as a 502 stating the reason."""
    app = _app_with_inference(db_session, InferenceService(provider=MisconfiguredStub()))  # type: ignore[arg-type]

    status_code, body = await _ask_the_assistant(app, "broken-chat@example.com", "brokenchat")

    assert status_code == status.HTTP_502_BAD_GATEWAY
    assert "not implemented yet" in body["detail"]


@pytest.mark.asyncio
async def test_a_fallen_back_turn_is_recorded_in_the_log(db_session: AsyncSession, caplog: pytest.LogCaptureFixture) -> None:
    """The fallback decision is logged loudly - otherwise outages hide behind 200 responses."""
    app = _app_with_inference(
        db_session,
        InferenceService(provider=FakeLLMProvider(error=ProviderTransportError("connection refused"))),
    )

    with caplog.at_level("WARNING"):
        status_code, _ = await _ask_the_assistant(app, "logged-chat@example.com", "loggedchat")

    fallback_records = [record for record in caplog.records if record.getMessage() == FALLBACK_EVENT]

    assert status_code == status.HTTP_200_OK
    assert len(fallback_records) == 1
    assert fallback_records[0].extra_data["provider"] == FAKE_LLM_PROVIDER_NAME
    assert fallback_records[0].extra_data["reason"] == "ProviderTransportError"
    assert "connection refused" in fallback_records[0].extra_data["detail"]






