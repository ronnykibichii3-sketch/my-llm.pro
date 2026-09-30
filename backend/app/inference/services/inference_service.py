"""Ask the configured provider for an answer, and decide what happens when it fails.

Everything that is *policy* around a model call lives here, and nothing about
building a payload does (see app/inference/README.md):

* composition - hand the turn to ``PromptBuilder`` and nothing else;
* measurement - time the call, keep the provider's token counts, and log one
  structured record per turn so a slow or large turn is visible without a profiler;
* failure - a provider that cannot answer triggers the stored apology, because a
  user who just had a conversation persisted deserves an answer of some kind.
  ``LLM_FALLBACK_ENABLED=false`` inverts that for development, where a silent
  apology hides the real error; configuration errors are never swallowed.
"""

import logging
from time import perf_counter

from app.context.services.context_builder_service import estimate_tokens
from app.context.schemas.context import ContextObject
from app.core.config import Settings, get_settings
from app.inference.config import FALLBACK_PROVIDER_NAME, FALLBACK_RESPONSE, InferenceConfig
from app.inference.exceptions import InferenceUnavailableError, ProviderFailure
from app.inference.providers.base import LLMProvider, elapsed_ms
from app.inference.providers.factory import create_provider
from app.inference.schemas.inference import LLMResponse, Prompt
from app.inference.services.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)

# Structured log events; one per completed turn, whatever its outcome.
COMPLETED_EVENT = "inference.completed"
FALLBACK_EVENT = "inference.fallback"

# The fallback reply was never produced by a model, so these measurements describe
# the apology text rather than a completion.
FALLBACK_LATENCY_MS = 0


class InferenceService:
    """Compose, ask, measure."""

    def __init__(
        self,
        provider: LLMProvider,
        prompt_builder: PromptBuilder | None = None,
        fallback_enabled: bool = True,
    ) -> None:
        self._provider = provider
        self._prompt_builder = prompt_builder if prompt_builder is not None else PromptBuilder()
        self._fallback_enabled = fallback_enabled

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "InferenceService":
        """Build the service the environment describes: provider, prompt, fallback rule."""
        resolved = settings if settings is not None else get_settings()
        config = InferenceConfig.from_settings(resolved)
        return cls(
            provider=create_provider(config),
            prompt_builder=PromptBuilder(config),
            fallback_enabled=config.fallback_enabled,
        )

    @property
    def provider(self) -> LLMProvider:
        """The backend in use; its ``name`` and ``model`` appear in every log record."""
        return self._provider

    @property
    def prompt_builder(self) -> PromptBuilder:
        """The builder whose ``system_prompt`` this service sends; exposed for logging."""
        return self._prompt_builder

    async def generate(self, prompt: Prompt) -> LLMResponse:
        """Ask the provider for this turn's answer, falling back if it cannot give one.

        Provider failures are the interesting case: they are logged with the reason
        and turned into ``FALLBACK_RESPONSE`` unless the fallback is switched off, in
        which case ``InferenceUnavailableError`` reports the failure instead.
        """
        started_at = perf_counter()
        try:
            response = await self._provider.generate(prompt.system_prompt, prompt.user_prompt, prompt.context)
        except ProviderFailure as failure:
            if not self._fallback_enabled:
                raise InferenceUnavailableError(self._provider.name, str(failure)) from failure
            self._log_fallback(prompt, failure)
            return self._fallback_response(prompt)
        logger.info(
            COMPLETED_EVENT,
            extra={
                "extra_data": {
                    "provider": response.provider,
                    "model": response.model,
                    "latency_ms": response.latency_ms,
                    "input_tokens_estimate": response.input_tokens_estimate,
                    "output_tokens_estimate": response.output_tokens_estimate,
                    "response_size": len(response.content),
                    "context_size": prompt.context_size,
                    "context_token_estimate": prompt.context_token_estimate,
                    "citation_count": prompt.citation_count,
                    "total_ms": elapsed_ms(started_at),
                }
            },
        )
        return response

    async def answer(self, context: ContextObject, user_prompt: str) -> LLMResponse:
        """Build the prompt for a turn and answer it; the entry point ``ChatService`` uses."""
        return await self.generate(self._prompt_builder.build_prompt(context, user_prompt))

    def _fallback_response(self, prompt: Prompt) -> LLMResponse:
        """The apology the user sees when the provider could not answer.

        The token estimates describe what was attempted: the prompt really was
        assembled, and the apology really is the assistant turn that gets stored.
        """
        return LLMResponse(
            content=FALLBACK_RESPONSE,
            model=self._provider.model,
            provider=FALLBACK_PROVIDER_NAME,
            input_tokens_estimate=estimate_tokens(prompt.system_prompt + prompt.context + prompt.user_prompt),
            output_tokens_estimate=estimate_tokens(FALLBACK_RESPONSE),
            latency_ms=FALLBACK_LATENCY_MS,
        )

    def _log_fallback(self, prompt: Prompt, failure: ProviderFailure) -> None:
        """Record a failed turn at warning level - no traceback, the message is the point."""
        logger.warning(
            FALLBACK_EVENT,
            extra={
                "extra_data": {
                    "provider": self._provider.name,
                    "model": self._provider.model,
                    "reason": type(failure).__name__,
                    "detail": str(failure),
                    "context_size": prompt.context_size,
                    "user_prompt_size": len(prompt.user_prompt),
                }
            },
        )

