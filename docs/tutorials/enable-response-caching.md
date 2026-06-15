---
title: Enable Response Caching
description: Enable prompt-response caching in AISIX AI Gateway and verify cache hit and miss behavior using the x-aisix-cache header.
sidebar_position: 83
---

This tutorial turns on the chat-completion cache so identical caller requests are served from the cache instead of forwarding to the upstream, and verifies the result with the `x-aisix-cache` response header.

You end with one enabled `CachePolicy` and a reproducible header-level proof that the cache works.

## Prerequisites

- A running gateway from the [Self-Hosted Quickstart](../quickstart/self-hosted.md)
- A direct model and caller API key from the [First Model, First Key, First Request](../quickstart/first-model-first-key-first-request.md) quickstart — this tutorial reuses `gpt-4o-prod` and `sk-demo-caller` as canonical names
- The caller key must include the model in `allowed_models` (or be a wildcard `["*"]`)

## How It Works

The proxy keys each request on a fingerprint built from the resolved model alias and the normalized request body — `messages`, `temperature`, `top_p`, `max_tokens`, and any other top-level request fields (tool-calling fields, future OpenAI-shape fields, anything the proxy doesn't pre-process).

**The caller key is intentionally excluded from the fingerprint.** Two callers asking the same prompt to the same model alias hit the same cache entry, so the second caller sees the first caller's cached response.

Treat prompts as crossing caller boundaries: don't cache content that shouldn't be visible to other callers in your deployment.

When an enabled `CachePolicy` matches the request, the proxy:

1. computes the fingerprint
2. looks up the cache
3. on **miss**, dispatches to the upstream, writes the response into the cache, returns the response with `x-aisix-cache: miss`
4. on **hit**, returns the cached response without calling the upstream, with `x-aisix-cache: hit`

If no enabled policy matches, the cache gate stays closed and the response has no `x-aisix-cache` header.

## Step 1: Create The Cache Policy

```bash title="Create the cache policy"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/cache_policies \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "default-chat-cache",
    "enabled": true,
    "applies_to": "all",
    "ttl_seconds": 3600
  }'
```

Field meanings (full reference in [Caching](../configuration/caching.md)):

- `enabled: true` — the cache gate consults this policy on every request
- `applies_to: "all"` — matches every request. For targeted rollouts use `model:<display_name>` or `api_key:<api_key_id>`.
- `ttl_seconds: 3600` — cache entry lifetime hint. Defaults to `3600` if omitted.
- `backend: "memory"` is the default and uses the in-process cache. `redis` uses the shared Redis cache and requires `cache.redis` in the gateway's bootstrap config — without it, requests matching the policy get no caching at all (no silent fallback to memory).

Wait for the snapshot to propagate:

```bash title="Wait for propagation"
sleep 1
```

If the Step 2 call returns no `x-aisix-cache` header on a slow runner, the policy has not propagated yet. See [Wait for configuration propagation](../quickstart/first-model-first-key-first-request.md#step-4-wait-for-configuration-propagation) for the polling alternative.

## Step 2: Send The First Request — Cache Miss

The proxy emits the `x-aisix-cache` header on every response that participates in the cache path. `curl -i` prints headers so we can see it directly.

```bash title="First call — should report cache miss"
curl -sSi -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer sk-demo-caller" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-prod",
    "messages": [{"role":"user","content":"cached prompt"}]
  }'
```

Look for this line in the response headers:

```text
x-aisix-cache: miss
```

`miss` means the gateway dispatched to the upstream and wrote the response into the cache.

## Step 3: Send The Same Request Again — Cache Hit

Repeat the request with the same body and model alias:

```bash title="Second call — should report cache hit"
curl -sSi -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer sk-demo-caller" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-prod",
    "messages": [{"role":"user","content":"cached prompt"}]
  }'
```

Look for:

```text
x-aisix-cache: hit
```

The response body is the cached copy of the first response — the upstream was not called.

## Step 4: Send A Different Request — Cache Miss Again

Change the prompt to confirm the fingerprint is not "always hit":

```bash title="Different prompt — should report cache miss"
curl -sSi -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer sk-demo-caller" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-prod",
    "messages": [{"role":"user","content":"a different prompt"}]
  }'
```

`x-aisix-cache: miss` proves the cache key reflects the request, not a constant.

## What Just Happened

The proxy hashed the resolved model alias and the request body for each request. The first call missed the cache and went to the upstream; the second call had the same fingerprint and hit the cache; the third call had a different prompt and therefore a different fingerprint, so it missed again. The upstream saw only two calls instead of three.

This contract is exercised by `tests/e2e/src/cases/cache-policy-e2e.test.ts` (identical-request hit) and `tests/e2e/src/cases/cache-scenarios-e2e.test.ts` (different-prompt miss). Both tests assert the `x-aisix-cache` header values and the upstream `receivedRequests.length` count — the cache header is a published contract that cp-api and the dashboard's `/logs` view depend on.

## Cleanup

```bash title="Delete the cache policy"
curl -sS -X DELETE http://127.0.0.1:3001/admin/v1/cache_policies/YOUR_CACHE_POLICY_ID \
  -H "Authorization: Bearer YOUR_ADMIN_KEY"
```

Use the `id` returned in Step 1. Deleting the policy is enough to disable caching for that scope; the cached entries in memory are dropped when the gateway restarts.

## Variations And Next Steps

- **Target one model only** — set `applies_to: "model:gpt-4o-prod"` to scope the cache to a specific alias.
- **Target one caller** — set `applies_to: "api_key:<api_key_id>"` using the `id` returned from `POST /admin/v1/apikeys`.
- **Tune TTL** — for short-lived agent traces, drop `ttl_seconds` to something like `60`; for FAQ-style traffic, raise it.
- **Stage a policy** — write it with `enabled: false` first, sanity-check the resource shape, then flip it to `true` without delete + recreate.
- **Wire up an observability exporter** to see cache hit-rate as a metric, not just a header. See [Observability Exporters](../configuration/observability-exporters.md) for the resource shape.

## Related Pages

- [Caching](../configuration/caching.md) — full field reference and scope matcher details
- [Headers And Error Codes](../reference/headers-and-error-codes.md) — `x-aisix-cache` and other published proxy headers
- [Metrics And Logs](../operations/metrics-and-logs.md) — how cache hit rate shows up in metrics
