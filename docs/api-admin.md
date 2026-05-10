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

Every single-entity CRUD endpoint shares the same response shape:

```json
{
  "id": "uuid",
  "value": { …entity… },
  "revision": 42
}
```

Lists return a **bare JSON array** of those entries — there is no
`{"items": [...]}` envelope. The `revision` field on each entry is
the etcd mod-revision at write time, useful for optimistic
concurrency in custom tooling.

## 4. Endpoints

### 4.1 Models — `/admin/v1/models`

The Model entity. See `crates/aisix-core/src/models/model.rs` for the
full schema.

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/admin/v1/models` | — | `[{id, value: Model, revision}, …]` (bare array) |
| POST | `/admin/v1/models` | Model JSON | `{id, value, revision}` (id is server-assigned UUID v4) |
| GET | `/admin/v1/models/{id}` | — | `{id, value, revision}` |
| PUT | `/admin/v1/models/{id}` | Model JSON | `{id, value, revision}` (idempotent — first call 201, subsequent 200) |
| DELETE | `/admin/v1/models/{id}` | — | `{deleted: true}` |

POST + PUT both reject duplicate `display_name` with 409.

The Model schema (see `crates/aisix-core/src/models/model.rs`) is
`deny_unknown_fields` and accepts two mutually-exclusive shapes:

- **Direct** — set `provider` + `model_name` + `provider_key_id`
  (the secret lives on a separate ProviderKey resource — see §4.3).
- **Routing** — omit those three; provide a `routing` block listing
  weighted upstream targets.

```bash
# Direct-mode Model: alias "my-gpt4" → openai/gpt-4o, with the
# upstream credential resolved through ProviderKey id <pk-id>.
curl -X POST http://localhost:3001/admin/v1/models \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "my-gpt4",
    "provider": "openai",
    "model_name": "gpt-4o",
    "provider_key_id": "<pk-id-from-/admin/v1/provider_keys>",
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
| POST | `/admin/v1/apikeys/{id}/rotate` — returns `{entry, plaintext}`; the new plaintext is shown **once** here and never again |

The ApiKey schema (see `crates/aisix-core/src/models/apikey.rs`) is
`deny_unknown_fields` and stores **only the SHA-256 hex digest** of
the caller's plaintext (`key_hash`), not the plaintext itself. The
operator (or `/rotate`) generates the plaintext, hashes it, and
ships only the hash to the admin API.

```bash
PLAINTEXT="sk-aisix-app-prod"
KEY_HASH=$(printf "%s" "$PLAINTEXT" | shasum -a 256 | awk '{print $1}')

curl -X POST http://localhost:3001/admin/v1/apikeys \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "key_hash": "'"$KEY_HASH"'",
    "allowed_models": ["my-gpt4"],
    "rate_limit": {"rpm": 60, "concurrency": 10},
    "max_budget_usd": 500.0
  }'
```

> `max_budget_usd` is enforced only in SaaS / managed mode. See
> §4.4 for the standalone caveat and the SaaS propagation model.

The `/rotate` endpoint replaces the stored hash and returns the new
plaintext directly, so the operator does not need to compute the
hash themselves on rotation:

```json
{
  "entry": {"id": "uuid", "value": {"key_hash": "<new-hash>", …}, "revision": 43},
  "plaintext": "sk-aisix-…"
}
```

### 4.3 ProviderKeys — `/admin/v1/provider_keys`

Centralised list of upstream provider credentials (OpenAI, Anthropic,
Gemini, DeepSeek). Use this when multiple Models share the same
upstream provider key — the Model can then reference the
ProviderKey by id rather than embedding the secret. Naming aligns
with the AISIX-Cloud control plane's `ProviderKey` table.

| Method | Path |
|---|---|
| GET / POST | `/admin/v1/provider_keys` |
| GET / PUT / DELETE | `/admin/v1/provider_keys/{id}` |

```bash
curl -X POST http://localhost:3001/admin/v1/provider_keys \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "openai-prod",
    "secret": "sk-prod-xxxx",
    "api_base": "https://api.openai.com/v1"
  }'
```

### 4.4 Budgets

Per-ApiKey USD spend caps are expressed via `max_budget_usd` on the
ApiKey resource — there is no separate `/admin/v1/budgets`
collection. **Enforcement is a SaaS-tier feature**: the data-plane
proxy never reads `max_budget_usd` from the etcd snapshot. The
field is populated by the SaaS control plane for parity with its
own `api_keys` table and is informational on the DP side.

#### SaaS / Managed mode

Two flows run in parallel between the DP and cp-api:

- **Pre-check (pull)** — every request triggers
  `BudgetClient::check` (`crates/aisix-proxy/src/budget.rs`), which
  calls cp-api over mTLS at `GET /dp/budget_check?api_key_id=<uuid>`
  with a 5 s LRU cache per api_key. `allow=false` becomes a 429
  `BudgetExceeded` before upstream dispatch.
- **Usage push** — each `/v1/chat/completions` and `/v1/messages`
  request enqueues a `UsageEvent`; a worker
  (`crates/aisix-server/src/telemetry.rs`) batches up to 100
  events or flushes every 5 s and POSTs them to `/dp/telemetry`.
  cp-api consumes this stream to update the authoritative spend,
  which feeds subsequent budget-check calls. **`/v1/responses`,
  `/v1/embeddings`, `/v1/audio*`, `/v1/images*`, and `/v1/rerank`
  do not emit usage events today** (see the comment in
  `crates/aisix-proxy/src/chat.rs`), so spend on those endpoints
  is not yet reflected in cp-api's ledger — tracked as #226.

Worst-case propagation between an over-cap completion and a
follow-up 429 is therefore ≈ 10 s (5 s telemetry flush + 5 s
budget-check cache). When cp-api is unreachable, the last good
decision sticks for up to `AISIX_DP_BUDGET_STALE_MAX_SECONDS`
(default 600 s); past that the proxy applies the `fail_mode`
(`open` / `closed` / `sticky`) the most recent successful response
carried.

Because spend lives on cp-api, multiple DP instances behind the
same cp-api stay consistent without DP-side coordination.

#### Standalone mode

Budget enforcement is **not implemented**. The admin API still
accepts `max_budget_usd` on POST/PUT (the field passes through
schema validation and persists to etcd) so the wire shape stays
compatible with managed mode, but no part of the proxy reads it.
Operators who need per-key spend caps must run in managed mode.

Team-level budgets are SaaS-tier (cp-api owns cross-key
aggregation) and have no standalone counterpart.

### 4.5 Health — `GET /admin/v1/health`

Per-Model health from the in-process `HealthTracker`, plus a
`config` block with the watch-supervisor's snapshot freshness:

```json
{
  "status": "ok",
  "models": [
    {"id": "uuid", "name": "my-gpt4", "health": 0},
    {"id": "uuid", "name": "my-claude", "health": 1}
  ],
  "config": {
    "snapshot_revision": 1234567,
    "snapshot_age_seconds": 5
  }
}
```

`health` is `0` (Healthy), `1` (Degraded — 4–7 consecutive upstream
failures), or `2` (Down — 8+). The `config` block surfaces the etcd
watch supervisor's freshness — a wedged watch can otherwise let the
gateway serve a frozen snapshot for hours while still reporting
every Model healthy. The block is omitted when the supervisor isn't
wired (rare; e.g. a config-file-only test rig).

### 4.6 Playground — `POST /playground/chat/completions`

Proxies a chat completion through the proxy router **in-process** —
no extra network hop, but the request is fully audited as if it had
arrived on the proxy listener.

This endpoint expects a **proxy** API key (an `ApiKey` from the
snapshot), not an admin key. The admin key only protects the rest
of `/admin/v1/*`.

### 4.7 OpenAPI

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
