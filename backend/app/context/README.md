# Prompt Context Assembly

Backend-only foundation for the block of text the model is told about a user
before it answers. Nothing here calls a model: the chat service asks for a
context object and will pass it to the provider that replaces the placeholder
reply.

## Pipeline

```
User Message                 (the turn being answered)
    |
Profile Section              (UserProfile -> labelled lines, own budget)
    |
Knowledge Retrieval          (KnowledgeService.search, top_k, user-scoped)
    |
Combined Block               (preamble + sections, one total budget)
    |
ContextObject                (text + citations + token estimate -> ChatService)
```

`ChatService.chat` builds the context for every turn and hands it to
`_generate_reply`; the routes only wire the builder in. Memory, agents and
training signals are meant to join this pipeline rather than be threaded through
the chat path.

## Two rules

- **Hard budgets.** `max_chars` bounds the combined block, while
  `profile_max_chars` and `knowledge_max_chars` bound their own section. The
  builder spends whatever the profile section leaves over on knowledge, so an
  empty profile frees the full knowledge budget and a long profile loses
  retrieval rather than showing a truncated profile. Knowledge excerpts are
  rendered as a prefix of the ranking: the first excerpt that does not fit ends
  the section, so the weakest matches are the ones dropped, and a dropped chunk
  is never cited. A misconfiguration where the sections are larger than the
  total still yields a trimmed block rather than an over-budget prompt.
- **Context is an enhancement, never a dependency.** A missing profile, an empty
  knowledge base, a blank message or a vector store outage all produce a smaller
  context. Nothing in this module raises, so a turn cannot fail because retrieval
  returned nothing.

## Structure

- `app/context/config.py` — every budget, label and separator, plus `ContextConfig`
  (validated at construction, read from settings through `from_settings`)
- `app/context/schemas/context.py` — internal `ContextObject` and `ContextCitation`
- `app/context/services/context_builder_service.py` — the pipeline and its rendering
- `app/dependencies/context.py` — `get_context_builder`, the composition point that
  wires `ProfileService`, `KnowledgeService` and the config together

`ProfileProvider` and `KnowledgeSearcher` are narrow `Protocol`s, not the concrete
services: the builder only ever needs `get_profile` and `search`, which keeps it
testable without a database and stops it reaching further into those services.

## ContextObject

| Field              | Meaning                                                          |
|--------------------|------------------------------------------------------------------|
| `profile_context`  | The rendered profile section, or `""`                             |
| `knowledge_context`| The numbered excerpts, or `""`                                    |
| `combined_context` | The single string meant for the system prompt (`""` = no context) |
| `citations`        | One entry per excerpt that survived trimming                      |
| `token_estimate`   | Rough token count of the combined block (4 chars per token)       |

The combined block starts with a preamble that tells the model the text was
assembled automatically and may be stale. Every excerpt is headed by its rank,
source document and chunk position so a claim can be attributed, while the
machine-readable `document_id` / `chunk_id` stay in `citations` and out of the
prompt. The citations travel with the turn so a later step can persist or show
them; nothing stores or returns them to a client today.

## Configuration

- `CONTEXT_MAX_CHARS` (default `6000`) — the combined block
- `PROFILE_MAX_CHARS` (default `1500`) — the profile section
- `KNOWLEDGE_MAX_CHARS` (default `4500`) — the knowledge section
- `KNOWLEDGE_TOP_K` (default `5`) — excerpts retrieved per turn before trimming

## Tests

`backend/tests/test_context.py` covers the rendering and budget rules against
stubbed sources, the degradation paths (profile failure, retrieval failure, both
failing), user isolation, and the chat endpoint end to end — including a turn
against a knowledge store whose searches raise. No Qdrant server, model or API
key is needed for pytest.
