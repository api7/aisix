---
title: Caching
description: Configure cache policies, TTL scope matching, and current cache-backend behavior in AISIX AI Gateway.
sidebar_position: 39
---

Caching is controlled by dynamic `CachePolicy` resources plus the bootstrap cache backend selection.

## Current Fields

- `name`
- `enabled`
- `backend`
- `ttl_seconds`
- `applies_to`

Example:

```bash title="Create a cache policy"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/cache_policies \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "default-chat-cache",
    "backend": "memory",
    "ttl_seconds": 3600,
    "applies_to": "all"
  }'
```

## Scope Matching

`applies_to` currently supports:

- `all`
- `model:<display_name>`
- `api_key:<api_key_id>`

Current matching is done against:

- the caller-visible model alias in the request
- the authenticated API key resource `id`

Unknown `applies_to` prefixes currently fall back to `all` on the data-plane side, so operators should rely on the documented forms only.

## Runtime Behavior

Current cache gating behavior is:

- the proxy selects the first enabled policy whose `applies_to` matcher accepts the request
- the selected policy's `ttl_seconds` is used for the cache write
- if no policy matches, the cache gate stays closed for that request

On chat responses, the proxy can emit `x-aisix-cache` with:

- `hit`
- `miss`

If no enabled policy matches the request, the response should not be treated as a cache hit or miss path.

## Backend Boundary

Current schema supports:

- `memory`
- `redis`

Current runtime boundary:

- `memory` is the reliable default path
- bootstrap config can wire a Redis backend at process start
- the dynamic `CachePolicy.backend` field should still be treated conservatively because broader Redis support boundaries are still being expanded

## Related Pages

- [Bootstrap Configuration](bootstrap-config.md)
- [Admin API](admin-api.md)
- [Roadmap](../roadmap.md)
