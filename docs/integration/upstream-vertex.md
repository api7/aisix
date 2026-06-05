---
title: Google Vertex AI Upstream
description: Route AISIX AI Gateway to Google Vertex AI with a service-account credential and Vertex model aliases.
sidebar_position: 32
toc_max_heading_level: 2
keywords:
  - AISIX AI Gateway
  - Google Vertex AI
  - Gemini
  - service account
  - AI gateway
---

AISIX AI Gateway can route OpenAI-compatible chat requests to [Google Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs). Callers can reach Vertex-hosted Gemini and partner models through the gateway.

Register GCP credentials, map a caller-facing alias to a Vertex model ID, and send requests through the gateway's proxy API. The provider key uses `adapter: vertex`, and AISIX authenticates to Vertex with a GCP OAuth2 bearer token.

## Use Cases

This setup is for models that run on Google Vertex AI and need gateway authentication, model allowlists, rate limiting, and usage accounting. The Vertex adapter can mint and cache GCP access tokens from a service-account key, so callers do not manage token refresh.

For models you host yourself, use [Bring Your Own Endpoint](../configuration/byo-endpoint.md) instead.

## Vertex Request Routing

AISIX chooses the Vertex route from the configured `model_name`.

Gemini models such as `gemini-*` use the Google publisher route, `publishers/google/models/<model>:generateContent`, or `:streamGenerateContent?alt=sse` for streaming.

Anthropic models such as `claude-*` use the Anthropic publisher raw-predict route with an Anthropic Messages body and `anthropic_version: "vertex-2023-10-16"`.

OpenAI-compatible Model Garden partner models, including `meta/*`, `llama*`, `deepseek*`, `qwen*`, `openai/gpt-oss*`, `minimaxai/*`, `moonshotai/*`, and `zai-org/*`, use the Vertex `endpoints/openapi/chat/completions` route with the model ID in the body.

Mistral models such as `mistral-*` and `codestral-*` use the Mistral publisher raw-predict route with an OpenAI-compatible body. AI21 models such as `jamba-*` use the AI21 publisher raw-predict route with an OpenAI-compatible body.

For every supported publisher, AISIX reads the GCP credential from the provider key's `secret`, obtains an OAuth2 access token, and sends the Vertex request with `Authorization: Bearer <token>`.

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant Gateway as AISIX AI Gateway
    participant Google as Google OAuth2
    participant Vertex as Vertex AI

    Client->>Gateway: POST /v1/chat/completions (model = alias)
    Gateway->>Gateway: Authenticate caller and resolve model alias
    alt service_account_json
        Gateway->>Google: signed JWT -> access token (cached)
        Google-->>Gateway: access_token
    else pre-minted access_token
        Note over Gateway: Use the supplied token
    end
    Gateway->>Vertex: POST publisher-specific Vertex route (Bearer token)
    Vertex-->>Gateway: Vertex response
    Gateway-->>Client: OpenAI-compatible response with caller-facing alias
```

## Credential Modes

The provider key's `secret` is a JSON object carrying `project`, `region`, and exactly one credential mode. Use `service_account_json` for the full GCP service-account JSON key. Use `access_token` only when you already manage and refresh a pre-minted GCP OAuth2 bearer token.

With `service_account_json`, AISIX signs a JWT with the service account's RSA private key and exchanges it for an OAuth2 access token at the service account's `token_uri`. The gateway caches the token for reuse and refreshes it about 60 seconds before its reported expiry.

With `access_token`, the gateway uses the provided bearer token directly. GCP token TTL is roughly one hour, so this mode is best suited to short-lived test environments or an existing token-minting pipeline.

AISIX rejects a provider key that sets both credential modes or neither mode.

## Prerequisites

Before you start, run the gateway with the admin API on `:3001` and the proxy API on `:3000`, prepare your admin key from the bootstrap config, and enable the Vertex AI API in the target GCP project. You also need a region, such as `us-central1`, and a service-account key with the Vertex AI user role.

Prepare the GCP project ID, Vertex region, service-account JSON key or pre-minted access token, Vertex publisher model ID, caller-facing alias, and optional proxy or private endpoint host.

## Configure the Vertex Upstream

Create a Vertex provider key, model alias, and caller API key. The provider key stores the GCP project, region, and credential mode; the model selects the Vertex publisher model ID.

### Create a Vertex Provider Key

The `secret` is a JSON string. The example below uses the `service_account_json` mode. Embed the service-account JSON as a nested object inside the secret.

:::warning Production Credentials
The standalone gateway stores `secret` as plaintext under the etcd `prefix` from [`config.yaml`](../configuration/bootstrap-config.md). For production, protect etcd with encryption at rest and restricted network access, or use AISIX Cloud's managed [Provider Key Rotation](../cloud/provider-key-rotation.md).
:::

Create the Vertex provider key with the GCP credential settings:

```shell
curl -sS -X POST "http://127.0.0.1:3001/admin/v1/provider_keys" \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "vertex-prod",
    "provider": "google-vertex",
    "adapter": "vertex",
    "secret": "{\"project\":\"my-gcp-project\",\"region\":\"us-central1\",\"service_account_json\":{\"type\":\"service_account\",\"private_key\":\"-----BEGIN PRIVATE KEY-----\\nYOUR_SERVICE_ACCOUNT_PRIVATE_KEY\\n-----END PRIVATE KEY-----\\n\",\"client_email\":\"vertex-sa@my-gcp-project.iam.gserviceaccount.com\",\"token_uri\":\"https://oauth2.googleapis.com/token\"}}"
  }'
```

The `secret` must include the GCP project id and Vertex region. The region drives the `<region>-aiplatform.googleapis.com` host unless you override `api_base`.

For credentials, include exactly one mode: `service_account_json` for OAuth token minting, or `access_token` for a pre-minted token you refresh yourself.

Set `adapter` to `vertex`. Use a provider label that identifies the upstream; the example uses `google-vertex`.

If you route Vertex traffic through a corporate proxy, set `api_base` on the provider key to the proxy host. AISIX appends the publisher-specific `/v1/projects/...` path.

Save the returned `id` for the model resource.

### Create a Model

`model_name` is the Vertex publisher model ID. The caller-facing alias is `display_name`.

```shell
curl -sS -X POST "http://127.0.0.1:3001/admin/v1/models" \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "gemini-prod",
    "provider": "google-vertex",
    "model_name": "gemini-1.5-pro",
    "provider_key_id": "YOUR_PROVIDER_KEY_ID"
  }'
```

Other supported examples include `claude-sonnet-4-5` for Anthropic on Vertex, `meta/llama-3.3-70b-instruct-maas` for OpenAI-compatible partner models, `mistral-large-2411` for Mistral, and `jamba-1.5-large` for AI21.

### Create a Caller API Key

Hash the plaintext caller key before storing it in AISIX:

```shell
if command -v sha256sum >/dev/null 2>&1; then
  printf '%s' 'sk-demo-caller' | sha256sum | cut -d' ' -f1
else
  printf '%s' 'sk-demo-caller' | shasum -a 256 | awk '{print $1}'
fi
```

Create the API key resource with access to the Vertex-backed model alias:

```shell
curl -sS -X POST "http://127.0.0.1:3001/admin/v1/apikeys" \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "key_hash": "YOUR_CALLER_KEY_HASH",
    "allowed_models": ["gemini-prod"]
  }'
```

## Send a Test Request

Admin API writes propagate to the proxy asynchronously. If the alias is not visible immediately, check configuration propagation and retry after the proxy has loaded the model alias. The example below uses Gemini. Gemini requires at least one user or assistant turn; a system-only request is rejected before the provider request.

```shell
curl -sS -X POST "http://127.0.0.1:3000/v1/chat/completions" \
  -H "Authorization: Bearer sk-demo-caller" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-prod",
    "messages": [
      {"role": "user", "content": "Say hello from Vertex."}
    ]
  }'
```

The gateway returns an OpenAI-compatible response with the caller-facing alias:

```json
{
  "object": "chat.completion",
  "model": "gemini-prod",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Hello from Vertex!"},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8}
}
```

## Verify the Vertex Upstream

After the test request succeeds, confirm the caller-facing alias and Vertex request.

```shell
curl -sS -X POST "http://127.0.0.1:3000/v1/chat/completions" \
  -H "Authorization: Bearer sk-demo-caller" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-prod","messages":[{"role":"user","content":"ping"}]}' \
  | grep -o '"model":"[^"]*"'
```

The output should be `"model":"gemini-prod"`, your caller-facing alias, not `gemini-1.5-pro`.

AISIX sends `Authorization: Bearer <token>` and calls the route family selected from `model_name`: Gemini uses the Google publisher `:generateContent` route, Anthropic on Vertex uses the Anthropic `:rawPredict` route, OpenAI-compatible partner models use `endpoints/openapi/chat/completions`, and Mistral or AI21 models use their publisher `:rawPredict` routes.

Check Vertex logs, metrics, quota usage, or provider-side request records for the test request. If AISIX returns a token-minting or upstream authentication error, check the service-account key, region, Vertex API enablement, IAM role, and model access.

## Limitations

Publisher selection is prefix-based. If `model_name` does not match a supported prefix, the gateway rejects the request before the provider request with an unsupported-publisher configuration error.

The end-to-end example uses Gemini because it is the most common Vertex path. For partner models, validate the exact `model_name`, quota, and regional availability in your Vertex project before exposing the alias to callers.

Provider-key request and response overrides can apply on Vertex routes, but they are most directly useful on OpenAI-compatible routes. Gemini's native `contents` format does not match every OpenAI-style override target.

## Related Reading

[Choose a Provider Upstream](provider-upstreams.md) compares setup paths, and [Adapter Protocol Families](../reference/adapters.md) shows where Vertex fits among adapter families. Configure credentials and `api_base` behavior with [Provider Keys](../configuration/provider-keys.md). For other upstream families, see [AWS Bedrock Upstream](upstream-bedrock.md) and [Azure OpenAI Upstream](upstream-azure-openai.md).
