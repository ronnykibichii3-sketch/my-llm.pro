"""Exceptions raised while asking a provider for a completion.

Two kinds, and the difference matters to the caller:

* ``ProviderFailure`` and its subclasses are *runtime* problems - the daemon is
  down, the request timed out, the reply came back empty. ``InferenceService``
  catches them and, unless the fallback is switched off, answers the turn with a
  stored apology instead of failing the user's request.
* ``ProviderConfigurationError`` and ``InferenceUnavailableError`` derive from
  ``AppException`` and are never swallowed: an unconfigured or unimplemented
  provider is a deployment bug, and hiding it behind a polite message would make
  it invisible. Both render as HTTP 502.
"""

from app.exceptions import AppException


class ProviderFailure(Exception):
    """The provider was reachable in principle but could not answer this turn."""


class ProviderTransportError(ProviderFailure):
    """The request never produced a response: refused connection, timeout, DNS."""


class ProviderResponseError(ProviderFailure):
    """A response arrived but was unusable: an HTTP error, bad JSON or no content."""


class ProviderConfigurationError(AppException):
    """The selected provider cannot be used as configured; never fallen back over."""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"LLM provider {provider!r}: {message}", status_code=502)
        self.provider = provider


class InferenceUnavailableError(AppException):
    """The provider failed and ``LLM_FALLBACK_ENABLED`` is off, so the turn has no answer."""

    def __init__(self, provider: str, reason: str) -> None:
        super().__init__(f"No answer from LLM provider {provider!r}: {reason}", status_code=502)
        self.provider = provider
