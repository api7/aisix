---
title: Build A Virtual Model With Failover
description: Create a routing model in AISIX AI Gateway that fails over from a primary upstream to a secondary upstream, and verify the failover by deliberately breaking the primary.
sidebar_position: 81
---

This tutorial builds a routing model that fails over from a primary upstream-backed model to a secondary upstream-backed model, then verifies the failover by deliberately breaking the primary. The caller keeps using one stable alias — the failover decision is invisible on the client side.

You end with one caller-facing alias `chat-prod` that clients call exactly the same way as a direct model.

## Prerequisites

- A running gateway from the [Self-Hosted Quickstart](../quickstart/self-hosted.md)
- A caller API key from the [First Model, First Key, First Request](../quickstart/first-model-first-key-first-request.md) quickstart, configured with `"allowed_models": ["chat-prod"]` (or `["*"]` for evaluation)
- Two OpenAI-compatible upstream endpoints — for evaluation you can use the **same** account for both and break the primary in Step 6 by pointing it at an unreachable host

## Architecture

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant Proxy as AISIX proxy (:3000)
    participant Primary as gpt-4o-primary
    participant Secondary as gpt-4o-secondary

    Client->>Proxy: POST /v1/chat/completions (model: chat-prod)
    Note over Proxy: resolve chat-prod → routing block
    Proxy->>Primary: dispatch (attempt 1)
    Primary--xProxy: 502 retryable error
    Note over Proxy: retries exhausted on primary; fall over
    Proxy->>Secondary: dispatch
    Secondary-->>Proxy: 200 OK
    Proxy-->>Client: 200 OK (caller blind to failover)
```

## Step 1: Create Two Provider Keys

Create one provider key per upstream. Capture the returned `id` from each response.

```bash title="Create the primary provider key"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/provider_keys \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "openai-primary",
    "secret": "YOUR_PRIMARY_PROVIDER_KEY",
    "api_base": "https://api.openai.com/v1"
  }'
```

Capture as `PRIMARY_PK_ID`.

```bash title="Create the secondary provider key"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/provider_keys \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "openai-secondary",
    "secret": "YOUR_SECONDARY_PROVIDER_KEY",
    "api_base": "https://api.openai.com/v1"
  }'
```

Capture as `SECONDARY_PK_ID`.

## Step 2: Create The Two Direct Target Models

These are the real upstream-backed models the routing alias will choose between. Capture each returned `id` if you want to delete them in cleanup.

```bash title="Create gpt-4o-primary"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/models \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "gpt-4o-primary",
    "provider": "openai",
    "model_name": "gpt-4o",
    "provider_key_id": "PRIMARY_PK_ID"
  }'
```

```bash title="Create gpt-4o-secondary"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/models \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "gpt-4o-secondary",
    "provider": "openai",
    "model_name": "gpt-4o-mini",
    "provider_key_id": "SECONDARY_PK_ID"
  }'
```

## Step 3: Create The Routing Model

The routing model holds a `routing` block instead of `provider`/`model_name`/`provider_key_id`. With `strategy: "failover"` it always starts at the first target and only falls forward on retryable failures.

```bash title="Create the failover virtual model chat-prod"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/models \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "chat-prod",
    "routing": {
      "strategy": "failover",
      "targets": [
        {"model": "gpt-4o-primary"},
        {"model": "gpt-4o-secondary"}
      ],
      "retries": 1,
      "max_fallbacks": 1,
      "retry_on_429": true
    }
  }'
```

Field meanings (full reference in [Routing And Failover](../configuration/routing-and-failover.md)):

- `retries: 1` — one extra attempt on the current target before failing over
- `max_fallbacks: 1` — attempt at most one later target after the initial target
- `retry_on_429: true` — let upstream `429` participate in retry and failover (off by default; without this, `429` from primary would propagate to the client)

## Step 4: Allow The Routing Alias On The Caller Key

If your caller key has `"allowed_models": ["*"]`, it already covers `chat-prod`. Otherwise update it:

```bash title="Grant chat-prod on the caller key"
curl -sS -X PUT http://127.0.0.1:3001/admin/v1/apikeys/YOUR_CALLER_KEY_ID \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "key_hash": "YOUR_CALLER_KEY_HASH",
    "allowed_models": ["chat-prod"]
  }'
```

You do **not** need to add the direct target aliases (`gpt-4o-primary`, `gpt-4o-secondary`) to `allowed_models` — the routing model resolves them server-side.

## Step 5: Verify The Happy Path

```bash title="Call the routing alias"
curl -sS -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer sk-demo-caller" \
  -H "Content-Type: application/json" \
  -d '{"model": "chat-prod", "messages": [{"role":"user","content":"Say hello."}]}'
```

You should receive a normal OpenAI-shaped chat-completions response. The client cannot tell that `chat-prod` is a routing alias — that is the point.

## Step 6: Force Failover And Verify

Make the primary target unreachable so the proxy is forced to walk forward to the secondary. Update its provider key to point at an invalid host:

```bash title="Break the primary upstream"
curl -sS -X PUT http://127.0.0.1:3001/admin/v1/provider_keys/PRIMARY_PK_ID \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "openai-primary",
    "secret": "YOUR_PRIMARY_PROVIDER_KEY",
    "api_base": "https://api.openai.invalid/v1"
  }'
```

Wait briefly for the snapshot to propagate:

```bash title="Wait for propagation"
sleep 1
```

On slow CI runners or a cold etcd, the broken-primary update may not be visible after one second. See [Wait for configuration propagation](../quickstart/first-model-first-key-first-request.md#step-4-wait-for-configuration-propagation) for the polling alternative.

Re-issue the same client call from Step 5. Expected: still `200 OK`, but the response is now served by `gpt-4o-secondary`. The client sees one successful response and never sees the retry.

To confirm the failover landed on the secondary target, check the per-model runtime state:

```bash title="Inspect per-model runtime state"
curl -sS http://127.0.0.1:3001/admin/v1/models/status \
  -H "Authorization: Bearer YOUR_ADMIN_KEY"
```

`gpt-4o-primary` should be in `cooldown` (set by the request-path retryable failure) while `gpt-4o-secondary` reports `healthy`. The routing alias `chat-prod` reports `not_applicable` — routing models themselves are not runtime-filtered.

## What Just Happened

1. The proxy resolved `chat-prod` to its routing block (`strategy: "failover"`).
2. It tried `gpt-4o-primary` first; the transport-level failure to `api.openai.invalid` was retryable.
3. `retries: 1` gave one extra attempt on the primary, which also failed.
4. `max_fallbacks: 1` allowed walking forward to `gpt-4o-secondary`, which served the response.
5. The proxy marked `gpt-4o-primary` `cooldown` in its runtime tracker so future requests skip it for the cooldown window.

This is the same path exercised by the `routing-strategies-e2e` test in `tests/e2e/src/cases/routing-strategies-e2e.test.ts` — that test asserts `primary.receivedRequests.length === 2` (initial + one retry) and `secondary.receivedRequests.length === 1` (the fallback that served the response). If the wire shape ever changes, that test fails first.

## Cleanup

Restore the primary upstream and remove the resources you created. Delete in reverse dependency order.

```bash title="Restore the primary upstream"
curl -sS -X PUT http://127.0.0.1:3001/admin/v1/provider_keys/PRIMARY_PK_ID \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "openai-primary",
    "secret": "YOUR_PRIMARY_PROVIDER_KEY",
    "api_base": "https://api.openai.com/v1"
  }'
```

```bash title="Remove the tutorial resources"
curl -sS -X DELETE http://127.0.0.1:3001/admin/v1/models/CHAT_PROD_ID         -H "Authorization: Bearer YOUR_ADMIN_KEY"
curl -sS -X DELETE http://127.0.0.1:3001/admin/v1/models/PRIMARY_MODEL_ID     -H "Authorization: Bearer YOUR_ADMIN_KEY"
curl -sS -X DELETE http://127.0.0.1:3001/admin/v1/models/SECONDARY_MODEL_ID   -H "Authorization: Bearer YOUR_ADMIN_KEY"
curl -sS -X DELETE http://127.0.0.1:3001/admin/v1/provider_keys/PRIMARY_PK_ID    -H "Authorization: Bearer YOUR_ADMIN_KEY"
curl -sS -X DELETE http://127.0.0.1:3001/admin/v1/provider_keys/SECONDARY_PK_ID  -H "Authorization: Bearer YOUR_ADMIN_KEY"
```

## Variations And Next Steps

- **Add a third target** — set `max_fallbacks: 2` to allow the proxy to walk through all three on a bad day.
- **Switch to `round_robin`** — for load-balancing peers rather than primary/secondary. See [`round_robin` semantics](../configuration/routing-and-failover.md#round_robin).
- **Switch to `weighted`** — for skewed primary share. Set `weight` on each target.
- **Add `background_model_check`** to each direct model — the gateway probes the upstream periodically and marks the target `unhealthy` independent of in-flight request failures, so the routing filter excludes it before the first user request fails. See [Models § `background_model_check`](../configuration/models.md#direct-models).
- **Wire up an observability exporter** to see failover events in your telemetry pipeline. See [Observability Exporters](../configuration/observability-exporters.md) for the resource shape.

## Related Pages

- [Models](../configuration/models.md) — direct vs routing model shape and the full field reference
- [Routing And Failover](../configuration/routing-and-failover.md) — strategy semantics, retry semantics, runtime filtering
- [`GET /admin/v1/models/status`](../configuration/models.md#operational-notes) — per-model runtime state
- [Troubleshooting](../operations/troubleshooting.md) — failure modes if the verification step doesn't behave as documented
