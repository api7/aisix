---
title: Enable Response Caching
description: Enable prompt-response caching in AISIX AI Gateway and verify cache hit and miss behavior using the x-aisix-cache header.
sidebar_position: 83
---

This tutorial shows you how to enable response caching for chat-completion requests and verify cache behavior with the `x-aisix-cache` response header.

You will:

1. Create a cache policy.
2. Send a request that misses the cache.
3. Send the same request again and verify a cache hit.
4. Delete the cache policy.

## Prerequisites

- A running gateway from the [Quickstart](../quickstart)
- A direct model and caller API key from [Understand admin resources](../quickstart/first-model-first-key-first-request.md) — this tutorial reuses `gpt-4o-prod` and `sk-demo-caller` as canonical names
- The caller key must include the model in `allowed_models` (or be a wildcard `["*"]`)
- `jq`, used to capture the cache policy ID

## Set variables

```shell
export AISIX_ADMIN_KEY="admin-local-only-change-me"
export AISIX_API_KEY="sk-demo-caller"
export AISIX_MODEL="gpt-4o-prod"
```

## Create a cache policy

```shell
CACHE_POLICY_ID=$(curl -sS -X POST http://127.0.0.1:3001/admin/v1/cache_policies \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "default-chat-cache",
    "enabled": true,
    "applies_to": "all",
    "ttl_seconds": 3600
  }' | jq -r .id)
```

This policy applies to all chat-completion requests. See [Caching](../configuration/caching.md) for scoped policies.

## Verify a cache miss

The proxy emits the `x-aisix-cache` header on every response that participates in the cache path. Because admin writes propagate asynchronously, poll until the first cache-participating request returns `miss`:

```shell
for i in $(seq 1 20); do
  RESPONSE=$(curl -sSi -X POST http://127.0.0.1:3000/v1/chat/completions \
    -H "Authorization: Bearer ${AISIX_API_KEY}" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "'"${AISIX_MODEL}"'",
      "messages": [{"role":"user","content":"cached prompt"}]
    }')

  echo "${RESPONSE}"

  if echo "${RESPONSE}" | grep -qi '^x-aisix-cache: miss'; then
    break
  fi
  sleep 0.5
done
```

Look for this line in the response headers:

```text
x-aisix-cache: miss
```

`miss` means the gateway dispatched to the upstream and wrote the response into the cache.

## Verify a cache hit

Repeat the request with the same body and model alias:

```shell
curl -sSi -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer ${AISIX_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"${AISIX_MODEL}"'",
    "messages": [{"role":"user","content":"cached prompt"}]
  }'
```

Look for:

```text
x-aisix-cache: hit
```

The response body is the cached copy of the first response — the upstream was not called.

## Verify a different request

Change the prompt to confirm the fingerprint is not "always hit":

```shell
curl -sSi -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer ${AISIX_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"${AISIX_MODEL}"'",
    "messages": [{"role":"user","content":"a different prompt"}]
  }'
```

`x-aisix-cache: miss` proves the cache key reflects the request, not a constant.

## Delete the cache policy

```shell
curl -sS -X DELETE "http://127.0.0.1:3001/admin/v1/cache_policies/${CACHE_POLICY_ID}" \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}"
```

Deleting the policy disables caching for that scope. In-memory cache entries are dropped when the gateway restarts.

## Next steps

- [Caching](../configuration/caching.md) — full field reference and scope matcher details
- [Headers and error codes](../reference/headers-and-error-codes.md) — `x-aisix-cache` and other published proxy headers
- [Metrics and logs](../operations/metrics-and-logs.md) — how cache hit rate shows up in metrics
