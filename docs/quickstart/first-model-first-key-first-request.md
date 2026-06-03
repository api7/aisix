---
title: Understand Admin Resources
description: Learn how provider keys, model aliases, and caller API keys work together in AISIX AI Gateway.
sidebar_position: 11
---

This guide explains the resources you created in the [Quickstart](../quickstart) and shows how to prove that the proxy is enforcing them.

Use it as the second page in the getting-started path. The quickstart proves that one request can pass through the gateway; this page explains how to inspect the resource chain and diagnose the first layer that rejects a request.

By the end of the guide, you will understand:

- which resource stores the upstream provider credential
- which resource exposes the caller-facing model alias
- which resource authenticates callers and limits model access
- how to verify propagation, authentication, and authorization from the proxy surface

:::note Standalone only
This guide uses the standalone `admin/v1` API on `127.0.0.1:3001`. A [Cloud managed data plane](aisix-cloud-managed-dp.md) only exposes proxy APIs locally and does **not** bind the standalone admin listener. In managed deployments, create provider keys, models, and caller API keys through the AISIX Cloud control plane.
:::

## Resource chain

The minimum request path uses three dynamic resources:

```text
caller bearer token
  -> ApiKey.key_hash
  -> ApiKey.allowed_models
  -> Model.display_name
  -> Model.provider_key_id
  -> ProviderKey.secret
  -> upstream provider
```

`ProviderKey` stores the upstream credential and provider connection details.

`Model` exposes the model alias callers send to AISIX. In the quickstart, callers send `gpt-4o-prod`, while AISIX forwards the upstream model name `gpt-4o-mini`.

`ApiKey` authenticates caller traffic. The gateway stores `key_hash`, the SHA-256 hash of the plaintext caller key, and uses `allowed_models` to decide which model aliases the caller can use.

## Prerequisites

Complete the [Quickstart](../quickstart) first, or start from a gateway that already has:

- a provider key
- a direct model alias
- a caller API key

This guide uses the quickstart values:

```shell
export AISIX_ADMIN_KEY="admin-local-only-change-me"
export CALLER_KEY="sk-demo-caller"
export MODEL_ALIAS="gpt-4o-prod"
```

If you just completed the quickstart, keep the captured `PROVIDER_KEY_ID`, `MODEL_ID`, and `APIKEY_ID` variables in the same shell. The cleanup commands at the end of this guide use them.

## Inspect the resources

List the provider keys, models, and caller API keys that the admin API stores:

```shell
curl -sS http://127.0.0.1:3001/admin/v1/provider_keys \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}"
```

```shell
curl -sS http://127.0.0.1:3001/admin/v1/models \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}"
```

```shell
curl -sS http://127.0.0.1:3001/admin/v1/apikeys \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}"
```

The provider-key response includes `secret` in plaintext. Treat provider-key command output as sensitive, including output from later `GET /admin/v1/provider_keys/:id` calls.

Check that the resources line up with the request path:

- the provider key has `display_name: "openai-upstream"` and the OpenAI adapter settings
- the model has `display_name: "gpt-4o-prod"` and references the provider key by using `provider_key_id`
- the API key has `allowed_models: ["gpt-4o-prod"]`

If one of those links is missing, fix the admin resource before debugging the proxy surface.

## Verify propagation to the proxy

Admin writes do not become visible to the proxy instantly. AISIX publishes dynamic resources through the watch-driven snapshot path, so propagation is fast but asynchronous.

Poll the proxy until the model alias is visible to the caller key:

```shell
MODEL_VISIBLE=false
for i in $(seq 1 20); do
  MODELS_RESPONSE=$(curl -sS http://127.0.0.1:3000/v1/models \
    -H "Authorization: Bearer ${CALLER_KEY}")

  if echo "${MODELS_RESPONSE}" \
    | jq -e --arg model "${MODEL_ALIAS}" \
      '.data[]? | select(.id == $model)' >/dev/null; then
    MODEL_VISIBLE=true
    echo "model alias is visible"
    break
  fi
  sleep 0.5
done

if [ "${MODEL_VISIBLE}" != "true" ]; then
  echo "model alias is not visible yet; check the admin resources and proxy logs" >&2
fi
```

Then inspect the visible models:

```shell
curl -sS http://127.0.0.1:3000/v1/models \
  -H "Authorization: Bearer ${CALLER_KEY}"
```

Expected response shape:

```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-4o-prod",
      "object": "model",
      "created": 1715000000,
      "owned_by": "openai"
    }
  ]
}
```

`created` is a gateway-side unix timestamp, so the exact value differs between runs.

## Verify the proxy contract

If the final quickstart request already succeeded, you can skip to [Verify auth and allowlist enforcement](#verify-auth-and-allowlist-enforcement). Otherwise, send one normal request with the caller key and the allowed model alias:

```shell
curl -sS -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer ${CALLER_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-prod",
    "messages": [
      {"role": "user", "content": "Say hello from AISIX."}
    ]
  }'
```

With a valid upstream provider key, the response follows the OpenAI chat-completions shape.

The important boundary is that the application never sends the upstream provider key. The caller sends only the gateway-issued caller key. AISIX resolves the model alias and uses the provider key on the upstream side.

## Verify auth and allowlist enforcement

Two negative-path checks prove the proxy is enforcing the admin resources, not only forwarding traffic.

### Missing bearer returns `401`

Send the same request without `Authorization`:

```shell
curl -sS -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-prod","messages":[{"role":"user","content":"hi"}]}'
```

Expected status: `401`.

The proxy response body uses the OpenAI-compatible error envelope:

```json
{
  "error": {
    "message": "missing or malformed Authorization header",
    "type": "invalid_api_key"
  }
}
```

### Unauthorized model returns `403` or `404`

Ask for a model alias the caller key cannot use:

```shell
curl -sS -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer ${CALLER_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"model":"some-model-not-in-allowed-models","messages":[{"role":"user","content":"hi"}]}'
```

Expected result:

- `403` with `"type": "permission_denied"` if the alias exists but is not listed in `allowed_models`
- `404` with `"type": "model_not_found"` if the alias does not exist in the current proxy snapshot

These checks exercise the same authentication and authorization path that gates production traffic.

## Troubleshoot the resource chain

Use the first failing status code to locate the failing part of the chain:

- `401 invalid_api_key` means the caller key is missing, malformed, or unknown to the proxy snapshot.
- `403 permission_denied` means the key exists, but the resolved model alias is not in `allowed_models`.
- `404 model_not_found` means the model alias does not resolve in the current proxy snapshot.
- `503 provider_unavailable` means no provider bridge is registered for the resolved provider, or every routing candidate is unavailable.

Admin API errors use a different envelope:

```json
{
  "error_msg": "..."
}
```

See [Headers and error codes](../reference/headers-and-error-codes.md) for the proxy and admin error boundaries.

## Clean up when done

:::warning Keep the quickstart resources if you are still learning
Do not clean up if you want to continue to the SDK quickstarts. [OpenAI SDK quickstart](openai-sdk.md) reuses the same caller key and model alias, and [Anthropic SDK quickstart](anthropic-sdk.md) can reuse the same caller key after you add an Anthropic-backed alias.
:::

Skip this section if you want to continue with the same local gateway, caller key, and model alias.

Delete the quickstart resources in reverse dependency order:

```shell
curl -sS -X DELETE http://127.0.0.1:3001/admin/v1/apikeys/${APIKEY_ID} \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}"
```

```shell
curl -sS -X DELETE http://127.0.0.1:3001/admin/v1/models/${MODEL_ID} \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}"
```

```shell
curl -sS -X DELETE http://127.0.0.1:3001/admin/v1/provider_keys/${PROVIDER_KEY_ID} \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}"
```

Then remove the local gateway stack:

```shell
docker compose down -v
```

## Next steps

- [What is AISIX AI Gateway](../overview/what-is-aisix-ai-gateway.md) — learn where the gateway fits and when to use it.
- [Core concepts](../overview/core-concepts.md) — learn the broader resource model.
- [Client APIs overview](../integration/overview.md) — choose the caller-facing API surface for your application.
- [OpenAI SDK quickstart](openai-sdk.md) — call the gateway from an OpenAI SDK client.
- [Anthropic SDK quickstart](anthropic-sdk.md) — call the gateway from an Anthropic-style client.
- [Models](../configuration/models.md) — configure direct and routing model aliases.
- [API keys](../configuration/api-keys.md) — configure caller access and model allowlists.
