---
title: Caching
description: Configure cache policies, TTL scope matching, and cache-backend behavior in AISIX AI Gateway.
toc_max_heading_level: 2
sidebar_position: 39
---

Caching requires a process backend and at least one matching cache policy. The process backend makes cache storage available to the data plane. `CachePolicy` resources decide which requests may use that storage.

Runtime caching is exact-match response caching for non-streaming chat-completions requests. Streaming responses are not cached.

## Cache Layers

| Layer | Configured by | What it controls |
| --- | --- | --- |
| Process cache backend | Bootstrap configuration | Whether the data plane uses in-memory cache or Redis. |
| Cache policy | Dynamic `CachePolicy` resource | Which non-streaming chat-completions requests may use the configured backend. |

Both layers must line up before a response can be cached. A Redis value on a policy does not move that policy to Redis; the process backend is selected at startup.

## Configure Caching

Enable the process backend first, then create a cache policy that matches the requests you want to cache.

### Configure the Process Backend

The server selects one cache backend at startup.

Memory cache is the default in-process backend. It is useful for a single data plane instance or local testing.

Redis can be configured through bootstrap config when multiple data-plane instances need to share cached responses. The Redis bootstrap path uses a single-node connection.

```yaml title="config.yaml"
cache:
  backend: redis
  redis:
    url: redis://127.0.0.1:6379/
```

See [Bootstrap configuration](bootstrap-config.md) for process configuration.

### Create a Cache Policy

A cache policy allows matching requests to use the configured cache backend.

```shell
curl -sS -X POST "http://127.0.0.1:3001/admin/v1/cache_policies" \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "default-chat-cache",
    "enabled": true,
    "ttl_seconds": 3600,
    "applies_to": "model:gpt-4o-prod"
  }'
```

This policy does not choose the process backend. It only says that matching requests may use whichever cache backend the process was started with.

### Policy Scope

`applies_to` controls which requests match the policy.

Use `all` to match every non-streaming chat-completions request:

```json
{
  "applies_to": "all"
}
```

Use `model:<display_name>` to match the caller-visible model alias:

```json
{
  "applies_to": "model:gpt-4o-prod"
}
```

Use `api_key:<api_key_id>` to match the authenticated API-key resource id:

```json
{
  "applies_to": "api_key:550e8400-e29b-41d4-a716-446655440000"
}
```

The runtime matcher compares against the request's model alias and the authenticated API-key id. For routing models, the cache key uses the virtual model alias the caller requested, not the direct target that served the miss.

Avoid unsupported matcher prefixes. The data plane treats unknown forms as `all`, so a typo can make a policy broader than intended.

## Behavior Details

For each non-streaming chat-completions request, the proxy finds the first enabled cache policy whose `applies_to` matcher accepts the request.

If a policy matches, the proxy checks the cache. A miss is written back with the policy's `ttl_seconds`. If no enabled policy matches, the cache path stays closed for that request.

When the request participates in caching, the proxy can emit:

```text
x-aisix-cache: miss
x-aisix-cache: hit
```

If no policy matches, the response is neither a cache hit nor a cache miss.

### Backend Field on a Policy

The `CachePolicy` schema includes `backend` with `memory` and `redis` values.

Treat this as a persisted hint, not as a per-policy backend selector. Runtime traffic uses the backend selected by bootstrap config for the whole process. Changing `backend` on an individual policy does not move that policy to a different cache backend.

### Choose Safe Defaults

Start with a narrow policy, such as `model:<alias>` or `api_key:<id>`.

Use `all` only when every non-streaming chat-completions request in the environment needs to participate in caching.

Use Redis at bootstrap time when several data-plane instances need to share cached responses.

Disable a policy with `enabled: false` when you want to stage or temporarily turn off caching without deleting the policy.

## Troubleshooting

Use these checks when cache policy configuration does not show up in responses.

### Responses Never Show the Cache Header

Check the three caching requirements: the process started with a cache backend, an enabled cache policy matches the request, and the request is a non-streaming chat-completions request.

### A Policy Matches Too Broadly

Check `applies_to`. Unknown matcher prefixes fall back to `all`, so stick to `all`, `model:<display_name>`, or `api_key:<api_key_id>`.

### Redis Is Configured on the Policy but Traffic Still Uses Memory

Set Redis in bootstrap config. `CachePolicy.backend` is not a runtime selector for individual policies.

## Related Reading

[Bootstrap configuration](bootstrap-config.md) covers process-level cache backend settings, and [Configuration overview](overview.md) explains the split between bootstrap settings and dynamic resources. To manage standalone admin writes, see [Admin API](admin-api.md). For another request-control policy layer, see [Rate limits](rate-limits.md).
