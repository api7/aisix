---
title: Guardrails
description: Configure keyword, AWS Bedrock, Azure AI Content Safety, and Aliyun content-moderation guardrails in AISIX AI Gateway. Understand hook points, fail-open behavior, latency modes, streaming output handling, and how blocked requests return HTTP 422 content_filter.
sidebar_position: 38
---

Guardrails are content-policy resources that AISIX AI Gateway enforces on the request and response lifecycle. A guardrail can block a prompt before it reaches the upstream model (the input hook), block a model response before it reaches the caller (the output hook), or both. When a guardrail blocks a request, the gateway returns HTTP `422` with `error.type: content_filter` and the blocked content never crosses the gateway.

This page covers every guardrail kind the gateway supports today — `keyword`, AWS Bedrock, Azure AI Content Safety (Prompt Shield and Text Moderation), and Aliyun content moderation — and the runtime behavior that applies to all of them.

## How guardrails work

The data plane (DP) resolves a guardrail chain for each request and runs every guardrail attached to it. Each guardrail returns one of three verdicts:

- **Allow** — the content passed the policy.
- **Block** — the content violated the policy. The gateway returns `422 content_filter`. An input block means the prompt never reaches the upstream; an output block means the model response never reaches the caller.
- **Bypass** — a remote-API guardrail could not reach its provider and `fail_open` is `true`, so the request proceeds. The bypass is recorded on the usage event (`guardrail_bypassed_reason`) for audit.

Guardrails run on every guarded proxy surface on the **input** hook — `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages`, `/v1/embeddings`, `/v1/images/generations`, `/v1/audio/speech`, and `/v1/rerank` — so a prompt is scanned before dispatch regardless of which API the caller uses. The **output** hook runs on the surfaces that return gateway-scannable text — `/v1/chat/completions`, `/v1/responses` (non-streaming and streaming), and `/v1/messages`.

## Guardrail kinds

| `kind` | What it checks | Where it runs | Input | Output |
|---|---|---|---|---|
| `keyword` | Case-insensitive literal and regex blocklist | In-process on the DP | Yes | Yes |
| `bedrock` | AWS Bedrock managed guardrail (content filters, denied topics, word filters, sensitive-information/PII) via `ApplyGuardrail` | AWS API call (SigV4) | Yes | Yes |
| `azure_content_safety` | Azure AI Content Safety **Prompt Shield** — jailbreak and indirect prompt-injection detection | Azure API call | Yes | Yes |
| `azure_content_safety_text_moderation` | Azure AI Content Safety **Text Moderation** — Hate / Sexual / SelfHarm / Violence severity plus custom blocklists | Azure API call | Yes | Yes (windowed streaming) |
| `aliyun_text_moderation` | Aliyun content-safety guardrail (`TextModerationPlus`) — risk-level moderation | Aliyun API call | Yes | Yes (windowed streaming) |

`keyword` runs entirely inside the gateway process and needs no external credentials. The other four kinds call an external moderation service; their behavior on a provider error is governed by `fail_open`.

## Common fields

Every guardrail shares this top-level shape, then adds kind-specific fields.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | string | — | Operator-facing name. Surfaces in metric labels and block reasons. |
| `enabled` | boolean | `true` | When `false`, the chain skips the guardrail entirely. Use it to stage a rule before turning it on. |
| `hook_point` | string | `both` | `input`, `output`, or `both`. Narrows a rule to one side of the lifecycle. |
| `fail_open` | boolean | `true` | For remote-API kinds, the verdict when the provider is unreachable: `true` allows the request through (bypass, recorded); `false` blocks it. No-op for `keyword`. |
| `kind` | string | — | The discriminator: `keyword`, `bedrock`, `azure_content_safety`, `azure_content_safety_text_moderation`, or `aliyun_text_moderation`. |

:::note Self-hosted vs Cloud
In self-hosted mode you create guardrails through the admin API (`/admin/v1/guardrails`), and a guardrail applies to **every request** in the gateway. In AISIX Cloud you create guardrails from the dashboard and scope them to specific environments, models, API keys, or teams — see [Scoping guardrails in AISIX Cloud](#scoping-guardrails-in-aisix-cloud).
:::

## Fail-open and fail-closed

For the four remote-API kinds, `fail_open` decides what happens when the moderation provider is unreachable, throttled, or times out:

- `fail_open: true` (default) — the gateway **bypasses** the guardrail and lets the request through. The bypass is recorded on the usage event so a compliance audit can see what slipped past. Choose this when availability matters more than strict moderation.
- `fail_open: false` — the gateway **fails closed** and blocks the request with `422`. Choose this when a moderation outage must never release unscanned content.

`keyword` runs in-process, so it has no provider to fail against and ignores `fail_open`.

The Azure Text Moderation and Aliyun kinds additionally expose `output_fail_open` (default `false`) so an outage on the **output** hook fails closed even when the input hook is configured to fail open.

## Streaming output handling

When a guardrail runs on the output hook of a streamed response, the gateway must decide how much to release before the scan completes. By default it holds the whole response back, scans once, then releases or blocks — secure by default, so an output-blocking guardrail can never leak content onto the wire before its check runs.

- `keyword`, `bedrock`, and Azure Prompt Shield use **whole-response hold-back**: the gateway buffers the response (up to `262144` bytes), scans it, and either releases it or returns `422`. If the buffer cap is exceeded it fails closed.
- Azure Text Moderation and Aliyun support **windowed** incremental release: they release a sliding window of content only after it scans clean, carrying `window_overlap_size` characters between windows so a span split across a boundary is still caught.

## Keyword guardrails

`kind: "keyword"` is an in-process literal and regex blocklist. It is the simplest guardrail and needs no external service.

| Field | Type | Description |
|---|---|---|
| `patterns` | array | List of `{ "kind": "literal" \| "regex", "value": "..." }`. Literals match case-insensitively; regexes are compiled with the Rust `regex` engine. An invalid regex is rejected at load time. |

```bash title="Create a keyword guardrail"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/guardrails \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "block-secrets",
    "hook_point": "input",
    "kind": "keyword",
    "patterns": [
      { "kind": "literal", "value": "AKIA" },
      { "kind": "regex", "value": "\\bssn:\\s*\\d{3}-\\d{2}-\\d{4}" }
    ]
  }'
```

For a full walkthrough including verification, see the [keyword guardrails tutorial](../tutorials/add-keyword-guardrails.md).

## AWS Bedrock guardrails

`kind: "bedrock"` calls the AWS Bedrock [`ApplyGuardrail`](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html) API with a SigV4-signed request. The policy itself — content filters, denied topics, word filters, sensitive-information (PII) filters — lives in the AWS Bedrock guardrail you reference; the gateway forwards the request's text and enforces AWS's verdict. A `GUARDRAIL_INTERVENED` response becomes a `422 content_filter` block; `NONE` passes.

| Field | Type | Description |
|---|---|---|
| `guardrail_id` | string | The AWS-console-issued guardrail identifier. |
| `guardrail_version` | string | A published version number (`1`, `2`, …) or `DRAFT`. |
| `region` | string | AWS region of the Bedrock guardrail, e.g. `us-east-1`. |
| `aws_credentials` | object | `{ "kind": "static", "access_key_id": "...", "secret_access_key": "..." }`. The IAM principal needs `bedrock:ApplyGuardrail` on the guardrail. |
| `latency_mode` | object | `{ "kind": "serial" }` waits for the call unconditionally, or `{ "kind": "timed", "timeout_ms": N }` aborts after `N` ms (100–5000) and applies `fail_open`. |

```json title="AWS Bedrock guardrail"
{
  "name": "bedrock-review",
  "kind": "bedrock",
  "hook_point": "both",
  "fail_open": true,
  "guardrail_id": "abcdefgh1234",
  "guardrail_version": "DRAFT",
  "region": "us-east-1",
  "aws_credentials": {
    "kind": "static",
    "access_key_id": "YOUR_ACCESS_KEY_ID",
    "secret_access_key": "YOUR_SECRET_ACCESS_KEY"
  },
  "latency_mode": { "kind": "serial" }
}
```

- Use `latency_mode: timed` with a `fail_open` policy to bound the latency a slow `ApplyGuardrail` call can add to a request.
- The input hook concatenates the request's text messages into a single `ApplyGuardrail` call; the output hook scans the model's response text.

:::caution Static credentials only
Bedrock guardrails authenticate with static AWS access keys today. Role-based credentials (`sts:AssumeRole`) are on the [roadmap](../roadmap.md), not yet available.
:::

## Azure AI Content Safety — Prompt Shield

`kind: "azure_content_safety"` calls Azure AI Content Safety Prompt Shield (`/contentsafety/text:shieldPrompt`) to detect jailbreak and indirect prompt-injection attacks. A detected attack becomes a `422 content_filter` block.

| Field | Type | Default | Description |
|---|---|---|---|
| `endpoint` | string | — | Azure Cognitive Services resource endpoint, e.g. `https://my-resource.cognitiveservices.azure.com`. |
| `api_key` | string | — | Subscription key (`Ocp-Apim-Subscription-Key`). |
| `timeout_ms` | integer | `5000` | HTTP call timeout. When it elapses, `fail_open` governs the verdict. `0` fires immediately; use `u32::MAX` for effectively unlimited. |

```json title="Azure Prompt Shield guardrail"
{
  "name": "prompt-shield",
  "kind": "azure_content_safety",
  "hook_point": "input",
  "fail_open": false,
  "endpoint": "https://my-resource.cognitiveservices.azure.com",
  "api_key": "YOUR_AZURE_CONTENT_SAFETY_KEY",
  "timeout_ms": 3000
}
```

## Azure AI Content Safety — Text Moderation

`kind: "azure_content_safety_text_moderation"` calls Azure AI Content Safety `text:analyze` for category-severity moderation (Hate, Sexual, SelfHarm, Violence) plus custom blocklists. It runs on input and output, including streaming output.

| Field | Type | Default | Description |
|---|---|---|---|
| `endpoint` | string | — | Azure Cognitive Services resource endpoint. |
| `api_key` | string | — | Subscription key. |
| `timeout_ms` | integer | `5000` | HTTP call timeout. |
| `output_type` | string | `FourSeverityLevels` | `FourSeverityLevels` (0,2,4,6) or `EightSeverityLevels` (0–7). |
| `categories` | array | `["Hate","Sexual","SelfHarm","Violence"]` | Categories to analyze. |
| `severity_threshold` | integer | `2` | A category at or above this severity blocks. |
| `severity_threshold_by_category` | object | `{}` | Per-category overrides that take precedence over the general threshold. |
| `blocklist_names` | array | `[]` | Azure self-managed blocklist names to match against. |
| `halt_on_blocklist_hit` | boolean | `false` | Forwarded to Azure's `haltOnBlocklistHit`. |
| `text_source` | string | `concatenate_user_content` | Input-hook text selection: `concatenate_user_content` or `concatenate_all_content` (includes system messages). Ignored on the output hook. |
| `stream_processing_mode` | string | `window` | `window` (sliding-window incremental release) or `buffer_full` (whole-response hold-back). |
| `window_size` | integer | `10000` | Sliding-window size in characters. Azure caps this at 10000. |
| `window_overlap_size` | integer | `256` | Characters carried between windows. |
| `max_buffer_bytes` | integer | `262144` | Max bytes buffered in `buffer_full` mode. |
| `on_buffer_exceeded` | string | `fail_closed` | `fail_closed` or `fail_open` when the buffer cap is hit. |
| `output_fail_open` | boolean | `false` | Fail-open policy for the output hook. |

```json title="Azure Text Moderation guardrail"
{
  "name": "text-moderation",
  "kind": "azure_content_safety_text_moderation",
  "hook_point": "both",
  "fail_open": false,
  "endpoint": "https://my-resource.cognitiveservices.azure.com",
  "api_key": "YOUR_AZURE_CONTENT_SAFETY_KEY",
  "categories": ["Hate", "Violence"],
  "severity_threshold": 4
}
```

cp-api omits unset fields, so a minimal row inherits the defaults above.

## Aliyun text moderation

`kind: "aliyun_text_moderation"` calls Aliyun's `TextModerationPlus` action on `green-cip.<region>.aliyuncs.com`. The input hook uses the `llm_query_moderation` service code and the output hook `llm_response_moderation`. Aliyun grades each call with a `RiskLevel` (`none` < `low` < `medium` < `high`); the gateway blocks when the returned level reaches `risk_level_threshold`.

| Field | Type | Default | Description |
|---|---|---|---|
| `region` | string | — | Aliyun region, e.g. `cn-shanghai`. The gateway builds `https://green-cip.<region>.aliyuncs.com`. |
| `endpoint` | string | unset | Explicit endpoint override. When set, it wins over `region`. |
| `access_key_id` | string | — | Aliyun AccessKey ID. |
| `access_key_secret` | string | — | Aliyun AccessKey secret. |
| `risk_level_threshold` | string | `high` | `low`, `medium`, or `high`. A returned level at or above this blocks. |
| `timeout_ms` | integer | `5000` | HTTP call timeout. |
| `output_fail_open` | boolean | `false` | Fail-open policy for the output hook. |
| `stream_processing_mode` | string | `window` | `window` or `buffer_full`. |
| `window_size` | integer | `2000` | Sliding-window size — Aliyun's `llm_response_moderation` per-call cap. |
| `window_overlap_size` | integer | `128` | Characters carried between windows. |
| `max_buffer_bytes` | integer | `262144` | Max bytes buffered in `buffer_full` mode. |
| `on_buffer_exceeded` | string | `fail_closed` | `fail_closed` or `fail_open` when the buffer cap is hit. |

```json title="Aliyun text-moderation guardrail"
{
  "name": "aliyun-review",
  "kind": "aliyun_text_moderation",
  "hook_point": "both",
  "fail_open": false,
  "region": "cn-shanghai",
  "access_key_id": "YOUR_ACCESS_KEY_ID",
  "access_key_secret": "YOUR_ACCESS_KEY_SECRET",
  "risk_level_threshold": "high"
}
```

## Credential handling

The provider secrets a guardrail carries — `aws_credentials.secret_access_key`, `api_key`, `access_key_secret` — are sensitive. The gateway never logs them.

- **AISIX Cloud**: cp-api envelope-encrypts the secret at rest (the same trust boundary as provider keys) and decrypts it only at projection time, so the data plane holds the plaintext in memory and never needs a master key.
- **Self-hosted**: the admin API persists the guardrail to your etcd. Secure your etcd store and restrict access to the admin endpoint accordingly.

## Scoping guardrails in AISIX Cloud

In AISIX Cloud you attach a guardrail to one or more scopes so it governs only the traffic that needs it:

- `env` — every request in the environment
- `model` — requests routed to a specific model
- `api_key` — requests from a specific caller API key
- `team` — requests attributed to a team

Each attachment carries a priority; when the same guardrail resolves through multiple matching scopes, the highest-priority attachment wins and duplicates are dropped. Scoped attachments are an AISIX Cloud capability. In self-hosted mode, a guardrail with no attachment applies to every request in the gateway.

## Verification

The runnable check below uses a `keyword` guardrail because it needs no external provider. Create the guardrail, then send a request whose prompt contains the blocked token.

```bash title="Send a blocked prompt"
curl -sS -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_CALLER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "YOUR_MODEL",
    "messages": [{ "role": "user", "content": "my key is AKIAEXAMPLE" }]
  }'
```

Expected result: the request is blocked before it reaches the upstream.

```text title="Expected output"
422
```

The response body carries the OpenAI-shaped error envelope with the content-filter type:

```json title="Expected 422 body"
{ "error": { "type": "content_filter", "message": "request blocked by content policy (guardrail 'block-secrets')" } }
```

A benign prompt to the same model returns `200`. Propagation from the admin API to the running data plane is not instant; if the guardrail does not block immediately after creation, retry for a few seconds.

## Limitations

- `enforcement_mode: "monitor"` is **not yet implemented**. The field is stored and shown in the dashboard, but the data plane always blocks when a guardrail fires — do not rely on `monitor` for pass-through behavior.
- The `mandatory` and `direction` fields are stored and forwarded to the dashboard but **not yet consulted** by the data plane; `fail_open` and `hook_point` govern behavior today.
- Bedrock guardrails support static AWS credentials only; role-based (`sts:AssumeRole`) credentials are on the [roadmap](../roadmap.md).
- Guardrails scan text. Image and audio payloads themselves are not content-moderated, though the text parts of such requests are.

## Troubleshooting

### The guardrail saves but nothing is blocked

Confirm the guardrail is `enabled`, that its `hook_point` matches the side you are testing (input vs output), and that you are sending traffic through a guarded surface. In self-hosted mode, allow a few seconds for the new resource to propagate to the running data plane.

### A blocked request returns `422`

That is the expected outcome for a guardrail denial. The client-facing body uses `error.type: content_filter` and the message names the guardrail that fired (`guardrail '<name>'`); the specific reason (which pattern or policy matched) is logged on the data plane, not returned to the caller.

### A remote-API guardrail lets everything through

The provider is likely unreachable and the guardrail is configured with `fail_open: true`, so requests bypass it. Check the data-plane logs for a guardrail-call-failed warning and the usage event's `guardrail_bypassed_reason`. Set `fail_open: false` to fail closed instead.

## Related pages

- [Add keyword guardrails (tutorial)](../tutorials/add-keyword-guardrails.md)
- [Admin API](admin-api.md)
- [Headers and error codes](../reference/headers-and-error-codes.md)
- [Roadmap](../roadmap.md)
