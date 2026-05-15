---
title: Errors And Retries
description: Understand the shared proxy error envelope, endpoint-specific status boundaries, and retry behavior on AISIX AI Gateway.
sidebar_position: 30
---

AISIX AI Gateway uses a shared proxy error envelope across its client-facing proxy endpoints.

Use this page to understand what a caller should do after a failed request, not just what status code was returned.

## Error Envelope

The proxy returns an OpenAI-compatible error body:

```json
{
  "error": {
    "message": "...",
    "type": "invalid_api_key",
    "param": null,
    "code": null
  }
}
```

In practice, `param` and `code` are omitted when they are not set, so client code should not assume those fields are always present.

The admin surface (`/admin/v1/*`) uses a **different**, simpler envelope: `{"error_msg": "..."}`. Treat them as distinct contracts — the proxy envelope is the OpenAI-compatible one above; the admin envelope is operator-facing.

## Gateway Status Codes And Error Types

The proxy emits the following gateway-generated failures. The `error.type` strings are stable and match upstream OpenAI conventions where possible:

| Status | `error.type`             | Trigger |
|--------|--------------------------|---------|
| `400`  | `invalid_request_error`  | Malformed payload or endpoint-specific invalid usage |
| `401`  | `invalid_api_key`        | Missing, malformed, or unknown caller API key |
| `403`  | `permission_denied`      | Valid key, but the resolved model is not in `allowed_models` |
| `404`  | `model_not_found`        | The requested model alias does not resolve in the current snapshot |
| `413`  | `invalid_request_error`  | Request body exceeded `proxy.request_body_limit_bytes` |
| `422`  | `content_filter`         | Guardrail blocked the request or response content |
| `429`  | `rate_limit_exceeded`    | Rate-limit rejection (per-key or per-model) |
| `429`  | `billing_error` (with `code: "budget_exceeded"`) | Budget rejection from the ApiKey's `max_budget_usd` |
| `503`  | `provider_unavailable`   | No provider bridge is registered for the resolved provider |
| `503`  | `all_candidates_unavailable` | Every routing candidate was excluded by runtime status (cooldown or background-unhealthy) and the routing model is configured with `on_all_filtered: fail`. The response carries `Retry-After: 30`. See [Routing And Failover § All-Targets-Filtered Policy](../configuration/routing-and-failover.md#all-targets-filtered-policy) |

Bridge-level upstream failures inherit their `status` and `error.type` from the upstream provider response (see "Upstream Error Mapping" below).

The `billing_error` row is the one case where `error.code` is set on the wire. The envelope looks like:

```json
{
  "error": {
    "message": "budget exceeded for ApiKey \"<id>\"",
    "type": "billing_error",
    "code": "budget_exceeded"
  }
}
```

`503 provider_unavailable` is emitted on the direct-dispatch path when no bridge is registered for the resolved provider. On a routing model the same condition is absorbed into the retry/failover loop and surfaces through the per-target runtime state on `GET /admin/v1/models/status` rather than as a top-level `503` to the caller.

`503 all_candidates_unavailable` is the routing-model fail-fast response when every candidate has been removed by the runtime filter. This is distinct from `provider_unavailable`: the bridge is registered and the model is well-configured, but every target is currently in cooldown or has been marked unhealthy by `background_model_check`.

## Upstream Error Mapping

When the upstream returns `4xx`, that client-visible error class is preserved through the proxy mapping.

When the upstream returns `5xx`, the proxy collapses that class to `502`.

That design keeps transient upstream failures out of the caller-visible `5xx` taxonomy and presents them as gateway-bad-upstream behavior.

## Retry-After

For rate-limit-style rejections, the proxy may return a `Retry-After` header.

Use that header as the first retry signal when present.

If your client has both automatic retry logic and server-provided delay handling, prefer the server hint.

## Endpoint-Specific Notes

- `/v1/embeddings`, `/v1/completions`, and `/v1/images/generations` can return `501` with error type `not_implemented` when the resolved provider does not support that endpoint
- `/v1/responses` returns `400` when the resolved model is not an OpenAI provider
- `/passthrough/:provider/*rest` follows its own raw upstream status behavior after proxy auth and provider resolution

## Caller Strategy

As a practical rule:

- treat `400`, `401`, `403`, and `404` as configuration or request bugs
- treat `429` as backoff-and-retry territory
- treat `502` as an upstream/transient class worth cautious retry
- treat `501` as a capability mismatch that needs a different provider or endpoint choice

## Retry Guidance

Safe retry behavior depends on the failure class:

- retry `429` using backoff and `Retry-After` when present
- retry transient transport or `502` errors carefully with idempotency in mind
- do not retry `400`, `401`, `403`, or `404` without changing the request or configuration

## Troubleshooting

### The same request sometimes returns `429`

Inspect caller-key rate limits or budget checks first.

### The same request returns `502` only for one upstream-backed model

That usually points to upstream-side instability or provider-path issues rather than caller auth.

## Related Pages

- [OpenAI-Compatible API](openai-compatible-api.md)
- [Provider Passthrough](passthrough.md)
- [Headers And Error Codes](../reference/headers-and-error-codes.md)
