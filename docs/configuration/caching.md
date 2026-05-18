---
title: Caching
description: Configure cache policies, TTL scope matching, and current cache-backend behavior in AISIX AI Gateway.
sidebar_position: 39
---

Caching is controlled by dynamic `CachePolicy` resources plus the bootstrap cache backend selection.

Use this page to answer two separate questions:

- is a cache backend available in the process
- which requests are allowed to use it

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

This example only defines the policy. The process also needs a compatible bootstrap cache backend.

## Scope Matching

`applies_to` currently supports:

- `all`
- `model:<display_name>`
- `api_key:<api_key_id>`

Current matching is done against:

- the caller-visible model alias in the request
- the authenticated API key resource `id`

Unknown `applies_to` prefixes currently fall back to `all` on the data-plane side, so operators should rely on the documented forms only.

That means undocumented matcher prefixes are unsafe from an operator predictability standpoint.

## Runtime Behavior

Current cache gating behavior is:

- the proxy selects the first enabled policy whose `applies_to` matcher accepts the request
- the selected policy's `ttl_seconds` is used for the cache write
- if no policy matches, the cache gate stays closed for that request

On chat responses, the proxy can emit `x-aisix-cache` with:

- `hit`
- `miss`

Those headers are the easiest caller-visible sign that the request participated in the cache path.

If no enabled policy matches the request, the response should not be treated as a cache hit or miss path.

## Backend Boundary

Current schema supports:

- `memory`
- `redis`

Current runtime boundary:

- `memory` is the reliable default path
- bootstrap config can wire a Redis backend at process start
- the dynamic `CachePolicy.backend` field should still be treated conservatively because broader Redis support boundaries are still being expanded

Note: the per-policy `backend` field is currently parsed and stored on the `CachePolicy` row but is not consulted by the runtime proxy. The proxy uses the cache backend selected via bootstrap-config (`cache.backend`) regardless of what each policy specifies. The field is preserved for forward compatibility; do not depend on it to override the runtime backend.

## Operator Guidance

- start with `memory` plus a narrowly scoped policy
- use `all` only when you truly want broad cache participation
- prefer `model:<alias>` or `api_key:<id>` when you need targeted rollout

## Troubleshooting

### Responses never show `x-aisix-cache`

Check both sides:

- a bootstrap cache backend must be available
- an enabled cache policy must match the request

### A policy matches too broadly

Revisit `applies_to` and avoid undocumented matcher forms.

## Related Pages

- [Bootstrap Configuration](bootstrap-config.md)
- [Admin API](admin-api.md)
- [Roadmap](../roadmap.md)
