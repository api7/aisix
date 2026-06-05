---
title: Enable Response Caching
description: Enable prompt-response caching in AISIX AI Gateway and verify cache hit and miss behavior with the x-aisix-cache header.
sidebar_position: 83
toc_max_heading_level: 2
---

Enable response caching for chat-completion requests and verify cache behavior with the `x-aisix-cache` response header. You create a cache policy, check a miss, repeat the request to confirm a hit, and remove the policy at the end.

## Prerequisites

Before you start, run the gateway from the [Quickstart](../quickstart) and create a direct model plus caller API key with [Understand Admin Resources](../quickstart/first-model-first-key-first-request.md). The commands use `gpt-4o-prod` and `sk-demo-caller`. The caller key must include the model in `allowed_models`, or use the wildcard value `["*"]`. Install `jq` to capture the cache policy ID from the admin API response.

## Configure Caching

In this section, you will create the cache policy used by the test requests.

### Set Variables

Export the values used by the cache policy commands:

```shell
export AISIX_ADMIN_KEY="admin-local-only-change-me"
export AISIX_API_KEY="sk-demo-caller"
export AISIX_MODEL="gpt-4o-prod"
```

### Create a Cache Policy

Create a cache policy that applies to chat-completion requests:

```shell
CACHE_POLICY_ID=$(curl -sS -X POST "http://127.0.0.1:3001/admin/v1/cache_policies" \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "default-chat-cache",
    "enabled": true,
    "applies_to": "all",
    "ttl_seconds": 3600
  }' | jq -r .id)
```

The policy applies to all chat-completion requests. For scoped cache policies, see [Caching](../configuration/caching.md).

## Verify Cache Behavior

After the cache policy is active, send repeated requests to compare cache miss and cache hit behavior.

### Verify a Cache Miss

The proxy emits the `x-aisix-cache` header on every response that participates in the cache path. Because admin writes propagate asynchronously, poll until the first cache-participating request returns `miss`:

```shell
for i in $(seq 1 20); do
  RESPONSE=$(curl -sSi -X POST "http://127.0.0.1:3000/v1/chat/completions" \
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

`miss` means the gateway reached the upstream provider and wrote the response into the cache.

### Verify a Cache Hit

Repeat the request with the same body and model alias:

```shell
curl -sSi -X POST "http://127.0.0.1:3000/v1/chat/completions" \
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

The response body is the cached copy of the first response. AISIX serves it without calling the upstream again.

### Verify a Different Request

Change the prompt to confirm the cache key is tied to the request:

```shell
curl -sSi -X POST "http://127.0.0.1:3000/v1/chat/completions" \
  -H "Authorization: Bearer ${AISIX_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"${AISIX_MODEL}"'",
    "messages": [{"role":"user","content":"a different prompt"}]
  }'
```

`x-aisix-cache: miss` shows that the cache key reflects the request, not a constant.

## Delete the Cache Policy

Delete the cache policy when you finish the tutorial:

```shell
curl -sS -X DELETE "http://127.0.0.1:3001/admin/v1/cache_policies/${CACHE_POLICY_ID}" \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}"
```

Deleting the policy disables caching for that scope. In-memory cache entries are dropped when the gateway restarts.

## Related Reading

For field details and scope matcher behavior, see [Caching](../configuration/caching.md). For `x-aisix-cache`, other proxy headers, and cache metrics, see [Headers and error codes](../reference/headers-and-error-codes.md) and [Metrics and logs](../operations/metrics-and-logs.md).
