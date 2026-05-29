---
title: Google Vertex AI upstream
description: Route AISIX AI Gateway to Google Vertex AI Gemini with a service-account credential, including in-process OAuth token minting and an end-to-end chat example.
sidebar_position: 32
keywords:
  - AISIX AI Gateway
  - Google Vertex AI
  - Gemini
  - service account
  - AI gateway
---

AISIX AI Gateway can route requests to [Google Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs) so callers reach Gemini models through the gateway's OpenAI-compatible proxy. This page shows how to register a GCP credential, how the gateway mints an OAuth2 token for each call, and how to verify a request reached Vertex with Bearer auth.

Vertex uses the `vertex` adapter family. The gateway builds Gemini's native `:generateContent` request, authenticates with a GCP OAuth2 Bearer token, and renders the response back to the caller as an OpenAI chat-completions envelope.

## When to use this

- Use this when your Gemini models run on Google Vertex AI and you want them behind the gateway's auth, allowlist, rate limiting, and usage accounting.
- Use this when you want the gateway to mint and cache GCP access tokens from a service-account key, rather than managing token refresh yourself.
- For models you host yourself, see [Bring your own endpoint](../configuration/byo-endpoint.md) instead.

## How it works

The gateway resolves the Vertex publisher from the model id. **Gemini** (`gemini-*`) is the currently wired publisher; see [Limitations](#limitations) for the rest.

For a Gemini model the gateway:

1. Reads the GCP credential from the provider key's `secret`.
2. Obtains an OAuth2 access token — either the pre-minted `access_token` you supplied, or one minted in-process from a `service_account_json` (see [Token minting](#token-minting)).
3. POSTs the Gemini `generateContent` body to `{base}/v1/projects/<project>/locations/<region>/publishers/google/models/<model>:generateContent` with `Authorization: Bearer <token>`. Streaming uses `:streamGenerateContent?alt=sse`.

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
    Bridge->>Vertex: POST ...:generateContent (Bearer token)
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

- A running self-hosted gateway (admin on `:3001`, proxy on `:3000`). See the [Self-Hosted Quickstart](../quickstart/self-hosted.md).
- Your admin key from the bootstrap config.
- A GCP project with the Vertex AI API enabled, a region (for example `us-central1`), and a service-account key with the Vertex AI user role.

## Configuration

### Step 1: Create the Vertex provider key

The `secret` is a JSON string. The example below uses the `service_account_json` mode. Embed the service-account JSON as a nested object inside the secret.

:::warning Production credentials
The standalone gateway stores `secret` as plaintext under the etcd `prefix` from [`config.yaml`](../configuration/bootstrap-config.md). For production, front etcd with encryption-at-rest, restrict etcd network access to the gateway, or use AISIX Cloud's managed [Provider Key Rotation](../cloud/provider-key-rotation.md), where the secret stays in the control plane and only the projected reference reaches the data plane.
:::

```bash title="Create a Vertex provider key (service-account mode)"
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

The `secret` fields are:

| Field | Required | Description |
|---|---|---|
| `project` | Yes | GCP project id (named or numeric), e.g. `my-gcp-project`. |
| `region` | Yes | Vertex AI region, e.g. `us-central1`, `europe-west4`. Drives the `<region>-aiplatform.googleapis.com` host. |
| `service_account_json` | One of the two | The full GCP service-account JSON key. The gateway mints tokens in-process. Mutually exclusive with `access_token`. |
| `access_token` | One of the two | A pre-minted GCP OAuth2 token you refresh yourself. Mutually exclusive with `service_account_json`. |

`adapter` must be `vertex`. `provider` is a free-form vendor label (`google-vertex` matches the AISIX Cloud catalog id).

If you operate behind a corporate proxy, set `api_base` on the provider key to your proxy host; it overrides the regional `<region>-aiplatform.googleapis.com` host. The bridge appends the `/v1/projects/.../publishers/google/models/<model>:generateContent` path itself.

Capture the returned `id` for the next step.

### Step 2: Create the model

`model_name` is the Gemini publisher model id. The customer-facing alias is `display_name`.

```bash title="Create a Gemini model"
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

### Step 3: Create a caller API key

```bash title="Hash a plaintext caller key"
printf 'sk-demo-caller' | sha256sum | cut -d' ' -f1
```

```bash title="Create a caller API key"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/apikeys \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "key_hash": "YOUR_CALLER_KEY_HASH",
    "allowed_models": ["gemini-prod"]
  }'
```

### Step 4: Send a request

Allow about a second for the configuration to propagate, then call the gateway. Gemini requires at least one user or assistant turn — a system-only request is rejected before dispatch.

```bash title="Send a chat completion to Vertex"
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

```json title="200 OK"
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

## Verification

Confirm the two observable facts a `200` does not, by itself, prove.

### `response.model` is the alias, not the Gemini id

```bash title="Confirm alias restore"
curl -sS -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer sk-demo-caller" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-prod","messages":[{"role":"user","content":"ping"}]}' \
  | grep -o '"model":"[^"]*"'
```

Expected: `"model":"gemini-prod"` — your alias, not `gemini-1.5-pro`. This is the gateway-wide alias-restore contract.

### The outbound request hits `:generateContent` with a Bearer token

The gateway's e2e coverage asserts the outbound shape against a mock Vertex: the request carries `Authorization: Bearer <token>` and hits `/v1/projects/<project>/locations/<region>/publishers/google/models/<model>:generateContent`. Confirm the auth/mint path indirectly:

```bash title="Negative check — bad credentials surface as upstream auth failure"
curl -sS -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer sk-demo-caller" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-prod","messages":[{"role":"user","content":"ping"}]}'
```

With a valid service-account key, expect `200`. With an invalid private key or revoked service account, expect a configuration or upstream error — confirming the gateway minted (or attempted to mint) a token and dispatched to Vertex. Upstream Vertex error envelopes (which can contain your GCP project id) are redacted to a canned, status-keyed message before reaching the caller.

## Limitations

The Vertex adapter currently supports **Gemini** models. The following Vertex publishers are recognized by the model-id resolver but are **not yet implemented** for dispatch:

- **Anthropic on Vertex** (`claude-*`, the `rawPredict` wire shape)
- **Llama on Vertex** (`meta/llama-*` / `llama*`)
- Mistral and AI21 on Vertex

Requests for these publishers return a clear "not yet implemented" error. Do not register them as live models. See the [Roadmap](../roadmap.md) for direction.

## Related pages

- [Adapter protocol families](../reference/adapters.md) — where Vertex fits among the five adapters.
- [Provider keys](../configuration/provider-keys.md) — the credential resource and `api_base` behavior.
- [AWS Bedrock upstream](upstream-bedrock.md) and [Azure OpenAI upstream](upstream-azure-openai.md) — the other specialized-family guides.
- [Roadmap](../roadmap.md) — planned Vertex publisher support.
