# Inference Provider Layer

One door to the model. `ChatService` asks for an answer and gets one - from Ollama
on the user's machine, or from the stored apology that follows a failure - without
ever learning which backend produced it, what its endpoint is, or that HTTP exists.

## Pipeline

```
ContextObject + user message     (assembled by app/context)
    |
PromptBuilder.build_prompt       (instruction / context / message, kept apart)
    |
InferenceService.answer          (timed, measured, decides whether to fall back)
    |
LLMProvider                      (ollama today; openai, anthropic configured only)
    |
LLMResponse                      (content, model, provider, token estimates, latency)
```

`ChatService._generate_reply` calls `answer(context, content)` and stores
`response.content` as the assistant message; everything below that call is swappable.

## Two rules

- **The three parts stay separate until the provider needs them.** `Prompt` carries
  the instruction, this turn's context and the user's message as its own fields, and
  `compose_system_message` - called by providers, never by a caller - is the only
  place they are joined. A backend that wants its context elsewhere can take it from
  `generate`'s third argument without any module outside this package deciding
  wording.
- **A model that cannot answer is a turn, not a crash.** Runtime failures
  (`ProviderTransportError`, `ProviderResponseError`) become `FALLBACK_RESPONSE`,
  which is written to the transcript as a normal assistant message and logged at
  warning level. Configuration mistakes are never fallen back over:
  `ProviderConfigurationError` and `InferenceUnavailableError` reach the client as
  HTTP 502 with the reason, because a deployment bug hidden behind a polite message
  stops being visible.

## Structure

- `app/inference/config.py` - every path, role, phrase and tunable, plus
  `InferenceConfig` (validated at construction, resolved from settings by
  `from_settings`, which reads the settings belonging to the *selected* provider)
- `app/inference/schemas/inference.py` - `Prompt` and `LLMResponse`
- `app/inference/services/prompt_builder.py` - `PromptBuilder` and `compose_system_message`
- `app/inference/services/inference_service.py` - the call, the measurements, the fallback
- `app/inference/providers/base.py` - the `LLMProvider` protocol plus `elapsed_ms`
  and `truncate_detail`
- `app/inference/providers/factory.py` - the one place `LLM_PROVIDER` is read
- `app/inference/exceptions.py` - which failures fall back and which do not
- `app/dependencies/inference.py` - `get_inference_service`, built once per process

## Response measurements

| Field                    | Source                                                            |
|--------------------------|-------------------------------------------------------------------|
| `content`                | The completion, trimmed of surrounding whitespace                  |
| `model`                  | What the provider says it served, else the configured model        |
| `provider`               | `LLM_PROVIDER`, or `fallback` for an apology                       |
| `input_tokens_estimate`  | Provider-reported counts when given, else a 4-chars-per-token rule |
| `output_tokens_estimate` | The same rule, applied to the answer alone                         |
| `latency_ms`             | The completion call only, excluding context assembly               |

Ollama reports `prompt_eval_count` and `eval_count` for most models and omits (or
zeros) them for others, so both paths are implemented and both are tested.

## What gets logged

Every turn logs one event with the numbers that make turns comparable:

- `inference.completed` (info) - provider, model, `latency_ms`, both token
  estimates, response size, `context_size`, `context_token_estimate`,
  `citation_count`, `total_ms`
- `inference.fallback` (warning) - provider, model, the failure class as `reason`, a
  flattened `detail`, `context_size`, `user_prompt_size`

Provider error bodies are untrusted and sometimes enormous, so `truncate_detail`
flattens one to a bounded line before it enters a message or a log. Credentials never
appear in either: Ollama has none, and the cloud providers report a missing key
without echoing it.

## Adding a provider

1. Add its name and defaults to `app/core/config.py`, extend
   `SUPPORTED_LLM_PROVIDERS`, and add the settings fields that provider needs.
2. Write `app/inference/providers/<name>_provider.py` as a class satisfying
   `LLMProvider`: it validates its own settings in `__init__` and raises
   `ProviderConfigurationError` naming whichever setting is missing.
3. Add one branch to `create_provider`. Neither `ChatService`, the settings, nor the
   service learns the provider's name.
4. Test it the way `test_inference.py` tests Ollama - against a mocked transport, so
   no test needs a model running.

`OpenAIProvider` and `AnthropicProvider` already do steps 1 to 3 and stop there:
their `generate` raises `IMPLEMENTATION_PENDING_MESSAGE`, so selecting one today
yields one clear 502 rather than a silent stub.

## Configuration

- `LLM_PROVIDER` (default `ollama`) - one of `ollama`, `openai`, `anthropic`
- `LLM_REQUEST_TIMEOUT_SECONDS` (default `120`) - the budget for one completion call
- `LLM_MAX_OUTPUT_TOKENS` (default `512`) - sent as Ollama's `num_predict`
- `LLM_TEMPERATURE` (default `0.7`, accepted range `0` to `2`)
- `LLM_FALLBACK_ENABLED` (default `true`) - off means a failure reaches the client
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL` - daemon address and model tag
- `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` - the key is a `SecretStr`, and
  a blank one counts as absent
- `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_MODEL` - same rules

## Tests

`backend/tests/test_inference.py` covers configuration validation, prompt
composition, the service's fallback and logging decisions against a stub provider,
the Ollama client against `httpx.MockTransport` (payload shape, reported counts,
refused connections, timeouts, HTTP errors, malformed bodies, empty completions),
provider selection from settings, and the chat endpoint behind the boundary. No test
opens a socket to a model or needs an API key.

