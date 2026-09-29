# User Profile Foundation

Backend-only structured memory: one durable profile row per user. No frontend
exposure and no chat augmentation yet.

## Why this module exists

`app/memory` stores the unstructured half of long-term memory as embeddings in
Qdrant, where a match is always an approximation. A profile is the structured
half described in ARCHITECTURE.md ("structured memory stored in PostgreSQL"):
values are written exactly and read exactly, which is what makes them safe to
use as fact — a display name, a timezone, a preferred language, a tone
preference.

## Pipeline

```
GET /profile                 (lazy: an empty profile row is created on first read)
PATCH /profile               (partial merge; absent fields are untouched)
    |
ProfileService               (trim, de-duplicate, merge, cap)
    |
ProfileRepository            (single UserProfile row per user)
    |
PostgreSQL                   (user_profiles, alembic migration 003)
```

There is no vector or embedding step: nothing in this module is searched by
similarity, so a profile stays correct even after many conversations.

## Structure

- `app/profile/config.py` — every length and count bound in one place
- `app/profile/services/profile_service.py` — lazy creation, merge updates, reset
- `app/profile/repositories/profile_repository.py` — persistence for one row
- `app/profile/schemas/profile.py` — Pydantic request/response models

## Merge rules

| Field                                      | Behaviour on PATCH                                        |
|--------------------------------------------|-----------------------------------------------------------|
| `display_name`, `summary`, `timezone`, `preferred_language` | Replaced; `null` (or blank) clears it          |
| `interests`, `goals`                       | Appended to the stored list, de-duplicated case-insensitively, capped |
| `preferences`                              | Incoming keys overlay stored keys, capped in size          |

A PATCH whose body sets no fields at all is rejected with `422`, so an accidental
empty request can never read as "clear everything". Use `POST /profile/reset` to
start over.

## Endpoints

| Method | Path                | Description                                             |
|--------|---------------------|---------------------------------------------------------|
| GET    | `/api/v1/profile`   | Return the profile, creating an empty one on first call  |
| PATCH  | `/api/v1/profile`   | Apply a partial update; absent fields are left untouched |
| POST   | `/api/v1/profile/reset` | Clear every stored attribute, keeping the row       |

All endpoints require a bearer token and can only touch the caller's own
profile.

## Configuration

No new environment variables: the module is configured by the limits in
`app/profile/config.py`, which match the column widths in the
`user_profiles` table.

## Tests

`backend/tests/test_profile.py` covers lazy creation, partial updates, merge and
cap behaviour, reset, and per-user scoping.
