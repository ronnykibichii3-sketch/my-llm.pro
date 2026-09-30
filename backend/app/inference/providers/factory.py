"""Turn ``LLM_PROVIDER`` into a provider instance.

One switch, used by the dependency wiring and by tests, so adding a backend means
implementing its class and adding one branch here - nothing in the settings, the
service or ``ChatService`` learns the name of a provider (see
app/inference/README.md).
"""

from app.core.config import ANTHROPIC_PROVIDER, OLLAMA_PROVIDER, OPENAI_PROVIDER, SUPPORTED_LLM_PROVIDERS, Settings, get_settings
from app.inference.config import UNSUPPORTED_PROVIDER_MESSAGE, InferenceConfig
from app.inference.exceptions import ProviderConfigurationError
from app.inference.providers.anthropic_provider import AnthropicProvider
from app.inference.providers.base import LLMProvider
from app.inference.providers.ollama_provider import OllamaProvider
from app.inference.providers.openai_provider import OpenAIProvider


def create_provider(config: InferenceConfig) -> LLMProvider:
    """Instantiate the provider named by ``config.provider``.

    The configured-but-unimplemented providers validate their own settings in their
    constructor and raise ``ProviderConfigurationError``; they are constructed here
    like any other, so a missing API key is reported as a configuration error even
    though the client does not exist yet.
    """
    if config.provider == OLLAMA_PROVIDER:
        return OllamaProvider(
            base_url=config.base_url,
            model=config.model,
            request_timeout_seconds=config.request_timeout_seconds,
            max_output_tokens=config.max_output_tokens,
            temperature=config.temperature,
        )
    if config.provider == OPENAI_PROVIDER:
        return OpenAIProvider(
            base_url=config.base_url,
            model=config.model,
            api_key=config.api_key,
            request_timeout_seconds=config.request_timeout_seconds,
            max_output_tokens=config.max_output_tokens,
            temperature=config.temperature,
        )
    if config.provider == ANTHROPIC_PROVIDER:
        return AnthropicProvider(
            base_url=config.base_url,
            model=config.model,
            api_key=config.api_key,
            request_timeout_seconds=config.request_timeout_seconds,
            max_output_tokens=config.max_output_tokens,
            temperature=config.temperature,
        )
    # Unreachable through InferenceConfig, which rejects unknown providers; kept so a
    # provider built by hand still reports itself rather than returning None.
    raise ProviderConfigurationError(
        config.provider,
        UNSUPPORTED_PROVIDER_MESSAGE.format(providers=SUPPORTED_LLM_PROVIDERS, provider=config.provider),
    )


def create_provider_from_settings(settings: Settings | None = None) -> LLMProvider:
    """Resolve ``LLM_PROVIDER`` from settings and build the provider it names."""
    resolved = settings if settings is not None else get_settings()
    return create_provider(InferenceConfig.from_settings(resolved))
