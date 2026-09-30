"""Dependency providers for the inference layer.

Built once per process and shared: the service holds a provider that creates its
own HTTP client per call, so there is no connection pool to hand out and nothing
to close on shutdown - unlike the memory layer's vector store.

``LLM_PROVIDER`` is read here through ``InferenceConfig.from_settings``. Choosing
an interface-only provider (openai, anthropic) raises ``ProviderConfigurationError``
while this dependency resolves, which the exception handler renders as HTTP 502
with the reason, instead of failing at import time or answering with a stub.
"""

from functools import lru_cache

from app.inference.services.inference_service import InferenceService


@lru_cache
def get_inference_service() -> InferenceService:
    """Return the process-wide inference service for the configured provider."""
    return InferenceService.from_settings()
