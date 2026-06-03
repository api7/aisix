---
title: Google Vertex AI Upstream
description: Route AISIX AI Gateway to Google Vertex AI with a service-account credential, including in-process OAuth token minting, publisher dispatch, and an end-to-end chat example.
sidebar_position: 32
keywords:
  - AISIX AI Gateway
  - Google Vertex AI
  - Gemini
  - service account
  - AI gateway
---

AISIX AI Gateway can route requests to [Google Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs) so callers reach Vertex-hosted Gemini and partner models through the gateway's OpenAI-compatible proxy. This page shows how to register a GCP credential, how the gateway mints OAuth2 tokens, how publisher dispatch works, and how to verify a request reached Vertex with Bearer auth.

Vertex uses the `vertex` adapter family. The gateway resolves the Vertex publisher from the model id, authenticates with a GCP OAuth2 Bearer token, dispatches to the publisher-specific Vertex route, and renders the response back to the caller as an OpenAI chat-completions envelope.

## When to use this

- Use this when your models run on Google Vertex AI and you want them behind the gateway's auth, allowlist, rate limiting, and usage accounting.
- Use this when you want the gateway to mint and cache GCP access tokens from a service-account key, rather than managing token refresh yourself.
- For models you host yourself, see [Bring your own endpoint](../configuration/byo-endpoint.md) instead.

## How it works

The gateway resolves the Vertex publisher from the model id and chooses the
matching Vertex route.

Gemini models such as `gemini-*` use the Google publisher route,
`publishers/google/models/<model>:generateContent`, or
`:streamGenerateContent?alt=sse` for streaming.

Anthropic models such as `claude-*` use the Anthropic publisher raw-predict
route and carry an Anthropic Messages body with
`anthropic_version: "vertex-2023-10-16"`.

OpenAI-compatible MaaS models, including `meta/*`, `llama*`, `deepseek*`,
`qwen*`, `openai/gpt-oss*`, `minimaxai/*`, `moonshotai/*`, and `zai-org/*`, use
the Vertex `endpoints/openapi/chat/completions` route with the model id in the
body.

Mistral models such as `mistral-*` and `codestral-*` use the Mistral publisher
raw-predict route. AI21 models such as `jamba-*` use the AI21 publisher
raw-predict route. In both cases, AISIX sends an OpenAI-compatible body and keeps
the model id in both the URL and body.

For every supported publisher, the gateway:

1. Reads the GCP credential from the provider key's `secret`.
2. Obtains an OAuth2 access token — either the pre-minted `access_token` you supplied, or one minted in-process from a `service_account_json` (see [Token minting](#token-minting)).
3. POSTs the publisher-specific Vertex request with `Authorization: Bearer <token>`.

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant Proxy as AISIX proxy (:3000)
    participant Bridge as Vertex bridge
    participant Google as Google OAuth2
    participant Vertex as Vertex AI (Gemini)

    Client->>Proxy: POST /v1/chat/completions (model = your-alias)
    Note over Proxy: resolve alias → Model + ProviderKey (adapter=vertex)
    Proxy->>Bridge: dispatch
    alt service_account_json
        Bridge->>Google: signed JWT → access token (cached)
        Google-->>Bridge: access_token
    else pre-minted access_token
        Note over Bridge: use token verbatim
    end
    Bridge->>Vertex: POST publisher-specific Vertex route (Bearer token)
    Vertex-->>Bridge: Gemini response
    Bridge-->>Proxy: normalized chat response
    Note over Proxy: restore response.model = your-alias
    Proxy-->>Client: OpenAI-shaped JSON
```

## Token minting

The provider key's `secret` is a JSON object carrying `project`, `region`, and **exactly one** of two credential modes:

- **`service_account_json`** (recommended) — the full GCP service-account JSON key, as emitted by `gcloud iam service-accounts keys create`. The gateway signs a JWT with the service account's RSA private key (RS256), exchanges it for an OAuth2 access token at the service account's `token_uri` (the standard [JWT-bearer assertion grant](https://developers.google.com/identity/protocols/oauth2/service-account)), and caches the token in-process. The cached token is refreshed about 60 seconds before its reported expiry, so an in-flight request never lands on an expired token.
- **`access_token`** — a pre-minted GCP OAuth2 bearer token you manage and refresh yourself (GCP token TTL is roughly one hour). Useful for short-lived test rigs or when you already operate a token-mint pipeline.

Setting both, or neither, fails at registration time with a clear error.

## Prerequisites

- A running gateway (admin on `:3001`, proxy on `:3000`). See the [Quickstart](../quickstart).
- Your admin key from the bootstrap config.
- A GCP project with the Vertex AI API enabled, a region (for example `us-central1`), and a service-account key with the Vertex AI user role.

## Values to collect

Before creating AISIX resources, collect these upstream values:

| Value | Where it is used |
| --- | --- |
| GCP project id | `secret.project` on the provider key |
| Vertex region | `secret.region`; also determines the standard Vertex AI host |
| Service-account JSON key or pre-minted access token | `secret.service_account_json` or `secret.access_token` |
| Vertex publisher model id | `model_name` on the model resource |
| Caller-facing alias | `display_name` on the model resource and `allowed_models` on the caller API key |
| Optional proxy or private endpoint host | `api_base` on the provider key |

## Create a Vertex provider key

The `secret` is a JSON string. The example below uses the `service_account_json` mode. Embed the service-account JSON as a nested object inside the secret.

:::warning Production credentials
The standalone gateway stores `secret` as plaintext under the etcd `prefix` from [`config.yaml`](../configuration/bootstrap-config.md). For production, front etcd with encryption-at-rest, restrict etcd network access to the gateway, or use AISIX Cloud's managed [Provider Key Rotation](../cloud/provider-key-rotation.md), where the secret stays in the control plane and only the projected reference reaches the data plane.
:::

```shell
curl -sS -X POST http://127.0.0.1:3001/admin/v1/provider_keys \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "vertex-prod",
    "provider": "google-vertex",
    "adapter": "vertex",
    "secret": "{\"project\":\"my-gcp-project\",\"region\":\"us-central1\",\"service_account_json\":{\"type\":\"service_account\",\"private_key\":\"-----BEGIN PRIVATE KEY-----\\nYOUR_SERVICE_ACCOUNT_PRIVATE_KEY\\n-----END PRIVATE KEY-----\\n\",\"client_email\":\"vertex-sa@my-gcp-project.iam.gserviceaccount.com\",\"token_uri\":\"https://oauth2.googleapis.com/token\"}}"
  }'
```

The `secret` must include the GCP project id and Vertex region. The region
drives the `<region>-aiplatform.googleapis.com` host unless you override
`api_base`.

For credentials, include exactly one mode:

- `service_account_json`, the full GCP service-account JSON key. AISIX mints and
  refreshes OAuth tokens in-process.
- `access_token`, a pre-minted GCP OAuth2 token that you refresh yourself.

`adapter` must be `vertex`. `provider` is a free-form vendor label (`google-vertex` matches the AISIX Cloud catalog id).

If you operate behind a corporate proxy, set `api_base` on the provider key to your proxy host; it overrides the regional `<region>-aiplatform.googleapis.com` host. The bridge appends the publisher-specific `/v1/projects/...` path itself.

Capture the returned `id` for the next step.

## Create a model

`model_name` is the Vertex publisher model id. The customer-facing alias is `display_name`.

```shell
curl -sS -X POST http://127.0.0.1:3001/admin/v1/models \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "gemini-prod",
    "provider": "google-vertex",
    "model_name": "gemini-1.5-pro",
    "provider_key_id": "YOUR_PROVIDER_KEY_ID"
  }'
```

Other supported examples include `claude-sonnet-4-5` for Anthropic on Vertex,
`meta/llama-3.3-70b-instruct-maas` for OpenAI-compatible MaaS, `mistral-large-2411`
for Mistral, and `jamba-1.5-large` for AI21.

## Create a caller API key

```shell
if command -v sha256sum >/dev/null 2>&1; then
  printf '%s' 'sk-demo-caller' | sha256sum | cut -d' ' -f1
else
  printf '%s' 'sk-demo-caller' | shasum -a 256 | awk '{print $1}'
fi
```

```shell
curl -sS -X POST http://127.0.0.1:3001/admin/v1/apikeys \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "key_hash": "YOUR_CALLER_KEY_HASH",
    "allowed_models": ["gemini-prod"]
  }'
```

## Send a Request

Admin writes propagate to the proxy asynchronously. Before sending traffic, poll `/v1/models` until the alias appears for the caller key. The example below uses Gemini. Gemini requires at least one user or assistant turn; a system-only request is rejected before dispatch.

```shell
curl -sS -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer sk-demo-caller" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-prod",
    "messages": [
      {"role": "user", "content": "Say hello from Vertex."}
    ]
  }'
```

Expected response (OpenAI-shaped, alias restored):

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

## Verify

Confirm the two observable facts a `200` does not, by itself, prove.

### `response.model` is the alias, not the Gemini id

```shell
curl -sS -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer sk-demo-caller" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-prod","messages":[{"role":"user","content":"ping"}]}' \
  | grep -o '"model":"[^"]*"'
```

Expected: `"model":"gemini-prod"` — your alias, not `gemini-1.5-pro`. This is the gateway-wide alias-restore contract.

### The outbound request uses the expected Vertex route

The gateway sends `Authorization: Bearer <token>` and calls the route family selected from `model_name`:

- Gemini: `/v1/projects/<project>/locations/<region>/publishers/google/models/<model>:generateContent`
- Anthropic on Vertex: `/v1/projects/<project>/locations/<region>/publishers/anthropic/models/<model>:rawPredict`
- Llama and other OpenAI-compatible MaaS models: `/v1/projects/<project>/locations/<region>/endpoints/openapi/chat/completions`
- Mistral and AI21: `/v1/projects/<project>/locations/<region>/publishers/<publisher>/models/<model>:rawPredict`

Confirm the auth and token-mint path indirectly:

```shell
curl -sS -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer sk-demo-caller" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-prod","messages":[{"role":"user","content":"ping"}]}'
```

With a valid service-account key, expect `200`. With an invalid private key or revoked service account, expect a configuration or upstream error — confirming the gateway minted (or attempted to mint) a token and dispatched to Vertex. Upstream Vertex error envelopes (which can contain your GCP project id) are redacted to a canned, status-keyed message before reaching the caller.

## Limitations

- Publisher selection is prefix-based. If `model_name` does not match a supported prefix, the gateway rejects the request before dispatch with a publisher-unknown configuration error.
- This page uses Gemini for the end-to-end example because it is the most common Vertex path. For partner models, validate the exact `model_name`, quota, and regional availability in your Vertex project before exposing the alias to callers.
- Provider-key request and response overrides can apply on Vertex routes, but they are most directly useful on the OpenAI-compatible rails. Gemini's native `contents` shape does not match every OpenAI-style override target.

## Next steps

- [Choose a provider upstream](provider-upstreams.md) — compare upstream setup paths.
- [Adapter protocol families](../reference/adapters.md) — where Vertex fits among the five adapters.
- [Provider keys](../configuration/provider-keys.md) — the credential resource and `api_base` behavior.
- [AWS Bedrock upstream](upstream-bedrock.md) and [Azure OpenAI upstream](upstream-azure-openai.md) — the other specialized-family guides.
