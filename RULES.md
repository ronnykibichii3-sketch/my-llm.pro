# Development Rules

## Rule 1

No prototype code.

Production-grade only.

---

## Rule 2

No architecture rewrites.

Only additive improvements.

---

## Rule 3

Frontend must never contain business logic.

Business logic belongs to backend.

---

## Rule 4

No duplicate functionality.

Reuse existing services.

---

## Rule 5

Types everywhere.

Backend:
Pydantic

Frontend:
TypeScript

---

## Rule 6

Every module must contain:

README.md

tests/

config/

main implementation

---

## Rule 7

No magic strings.

Use constants and configs.

---

## Rule 8

Every API requires:

Request model

Response model

Validation

Logging

Error handling

---

## Rule 9

Every database change requires migration.

Alembic only.

---

## Rule 10

No direct model calls from frontend.

Frontend
→ API
→ Service
→ Model

Only.

``

---

## Rule 11

PostgreSQL runs on port 5433.

Never replace 5433 with 5432.

Do not modify PostgreSQL ports.

Every backend service, Alembic migration, test, Docker config and environment
file reads the database connection from DATABASE_URL only.

No hardcoded host, username, password or port values.
