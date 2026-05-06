# Admin API

The Admin API is the operator surface served on the admin listener
(default `:3001`). It owns CRUD for every configurable entity, the
playground proxy, the OpenAPI Scalar UI, and the Prometheus scrape
endpoint.

> **Network posture**: the admin listener is intended to be private —
> bind it to `127.0.0.1` or to a private subnet. The proxy listener
> is the public surface.

## 1. Authentication

Every `/admin/v1/*` endpoint requires an admin key, presented as
either:

- `Authorization: Bearer <admin-key>` (preferred), or
- `x-api-key: <admin-key>` (fallback for clients that can't set
  `Authorization`).

When both headers are present, `Authorization` wins. Admin keys are
configured in the bootstrap YAML under `admin.admin_keys` (a list).

The unauthenticated endpoints are:

- `/health` — liveness probe.
- `/metrics` — Prometheus scrape (intended to be private).
- `/admin/openapi.json`, `/admin/openapi-scalar` — the OpenAPI spec
  and the Scalar UI.

## 2. Error envelope

Admin errors use a deliberately simpler envelope than the proxy:

```json
{"error_msg": "name 'my-gpt4' already exists"}
```

| Status | When |
|---|---|
| 400 | Validation failed (missing required field, wrong type, schema rejection) |
| 401 | Missing or unknown admin key |
| 403 | Reserved for future per-key admin RBAC |
| 404 | Resource id not found |
| 409 | Duplicate name on POST or PUT-create |
| 500 | Internal error (etcd unavailable, etc.) |

## 3. Resource model

Every CRUD endpoint shares the same response shape:

```json
{
  "id": "uuid",
  "value": { …entity… },
  "revision": 42
}
```

Lists wrap items in `{"items": [...]}`. The `revision` field is the
etcd mod-revision at write time — useful for optimistic concurrency
in custom tooling.

## 4. Endpoints

### 4.1 Models — `/admin/v1/models`

The Model entity. See `crates/aisix-core/src/models/model.rs` for the
full schema.

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/admin/v1/models` | — | `{items: [{id, value: Model, revision}]}` |
| POST | `/admin/v1/models` | Model JSON | `{id, value, revision}` (id is server-assigned UUID v4) |
| GET | `/admin/v1/models/{id}` | — | `{id, value, revision}` |
| PUT | `/admin/v1/models/{id}` | Model JSON | `{id, value, revision}` (idempotent — first call 201, subsequent 200) |
| DELETE | `/admin/v1/models/{id}` | — | `{deleted: true}` |

POST + PUT both reject duplicate `name` with 409.

```bash
# Create a Model that wraps OpenAI's gpt-4o behind alias "my-gpt4"
curl -X POST http://localhost:3001/admin/v1/models \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-gpt4",
    "model": "openai/gpt-4o",
    "provider_config": {
      "api_key": "sk-real-openai-…",
      "api_base": "https://api.openai.com/v1"
    },
    "rate_limit": {"rpm": 100, "tpm": 100000},
    "cost": {"input_per_1k": 0.005, "output_per_1k": 0.015}
  }'
```

### 4.2 ApiKeys — `/admin/v1/apikeys`

Caller-facing keys. Empty `allowed_models` denies every model — it is
**not** a shortcut for "all". Use `["*"]` for wildcard.

| Method | Path |
|---|---|
| GET | `/admin/v1/apikeys` |
| POST | `/admin/v1/apikeys` |
| GET | `/admin/v1/apikeys/{id}` |
| PUT | `/admin/v1/apikeys/{id}` |
| DELETE | `/admin/v1/apikeys/{id}` |
| POST | `/admin/v1/apikeys/{id}/rotate` — returns a new `key` value, invalidates the old |

```bash
curl -X POST http://localhost:3001/admin/v1/apikeys \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "key": "sk-aisix-app-prod",
    "allowed_models": ["my-gpt4"],
    "rate_limit": {"rpm": 60, "concurrency": 10},
    "max_budget_usd": 500.0
  }'
```

### 4.3 Credentials — `/admin/v1/credentials`

Centralised credential list (spec §3.8). Use this when multiple
Models share the same upstream provider key — the Model can then
reference the credential by id rather than embedding the secret.

| Method | Path |
|---|---|
| GET / POST | `/admin/v1/credentials` |
| GET / PUT / DELETE | `/admin/v1/credentials/{id}` |

### 4.4 Budgets — `/admin/v1/budgets`

Per-ApiKey monthly USD spend cap (spec §3.4). The proxy reads the
budget at request start (pre-check) and adds the cost at end of
request.

| Method | Path |
|---|---|
| GET / POST | `/admin/v1/budgets` |
| GET / PUT / DELETE | `/admin/v1/budgets/{id}` |

```bash
curl -X POST http://localhost:3001/admin/v1/budgets \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "api_key_id": "uuid-of-apikey",
    "monthly_usd_cap": 1000.0,
    "usd_per_1k_tokens": 0.01
  }'
```

### 4.5 Teams — `/admin/v1/teams`

Group of ApiKeys with shared rate limits and credential overrides
(spec §3.8).

| Method | Path |
|---|---|
| GET / POST | `/admin/v1/teams` |
| GET / PUT / DELETE | `/admin/v1/teams/{id}` |

### 4.6 Health — `GET /admin/v1/health`

Per-Model health from the in-process `HealthTracker`:

```json
{
  "status": "ok",
  "models": [
    {"id": "uuid", "name": "my-gpt4", "health": 0},
    {"id": "uuid", "name": "my-claude", "health": 1}
  ]
}
```

`health` is `0` (Healthy), `1` (Degraded — 4–7 consecutive upstream
failures), or `2` (Down — 8+).

### 4.7 Spend — `GET /admin/v1/spend`

Current-month accumulated USD spend per ApiKey from the in-process
`BudgetTracker`. Returns the period (`YYYY-MM`) plus per-key entries.

```json
{
  "period": "2026-04",
  "entries": [
    {"api_key_id": "uuid", "spend_usd": 12.34}
  ]
}
```

### 4.8 Playground — `POST /playground/chat/completions`

Proxies a chat completion through the proxy router **in-process** —
no extra network hop, but the request is fully audited as if it had
arrived on the proxy listener.

This endpoint expects a **proxy** API key (an `ApiKey` from the
snapshot), not an admin key. The admin key only protects the rest
of `/admin/v1/*`.

### 4.9 OpenAPI

- `GET /admin/openapi.json` — machine-readable OpenAPI 3 document
  generated by `utoipa` from the same handler signatures.
- `GET /admin/openapi-scalar` — Scalar HTML UI for browsing /
  trying the API.

These are unauthenticated by design: knowing the API surface should
not require credentials.

## 5. Working with the snapshot

Admin writes are NOT immediately visible to the proxy. The flow is:

1. Admin handler validates the body, generates a UUID (POST) or
   resolves an id (PUT/DELETE).
2. Handler writes to etcd transactionally.
3. The watch supervisor observes the change (~50 ms typical).
4. The supervisor publishes a new `AisixSnapshot` via `ArcSwap`.
5. Subsequent proxy requests see the change.

End-to-end the propagation is bounded at 500 ms in practice. The
E2E harness sleeps 500 ms between Admin write and Proxy assertion;
production tooling should do the same when chaining writes.

## 6. Versioning

Admin endpoints live under `/admin/v1/`. The same compatibility
rules as the proxy API apply: optional fields are added without
breaking existing clients; removed fields stay accepted-but-ignored
for at least one minor version. A v2 surface, when needed, will
mount in parallel.

## 7. See also

- [`architecture.md`](./architecture.md) — bootstrap, snapshot, watch
  supervisor.
- [`api-proxy.md`](./api-proxy.md) — caller-facing surface.
- The auto-generated OpenAPI lives at `/admin/openapi.json` (or
  browsable at `/admin/openapi-scalar`).
