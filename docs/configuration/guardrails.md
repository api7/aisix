---
title: Guardrails
description: Configure content checks for chat requests and responses in AISIX AI Gateway.
sidebar_position: 38
toc_max_heading_level: 2
---

Guardrails apply content policy at the gateway. They can block prompts before
they reach an upstream provider, block responses before they reach a caller, or
record a bypass when a remote guardrail is unavailable and the policy is
configured to fail open.

Guardrails run on `POST /v1/chat/completions` and `POST /v1/messages`.

## Choose a Guardrail Kind

| Kind | Use When | Operational Dependency |
| --- | --- | --- |
| `keyword` | You need deterministic local blocking for literals or regex patterns. | Runs inside the data plane and does not depend on an external provider. |
| `bedrock` | You already operate AWS Bedrock guardrails and want AISIX to call Bedrock `ApplyGuardrail` during proxy handling. | Requires Bedrock credentials, network reachability, and the default `bedrock` build feature. |
| `azure_content_safety` | You want Azure AI Content Safety Prompt Shield checks for jailbreak or indirect prompt-injection detection. | Requires Azure Content Safety credentials, network reachability, and the default `azure-content-safety` build feature. |
| `aliyun_text_moderation` | You use Aliyun content-safety moderation for prompt or response review. | Requires Aliyun credentials, network reachability, and the Aliyun guardrail runtime feature. |

Remote guardrails depend on external services. Treat credentials, network
reachability, timeouts, and `fail_open` as part of the policy.

## Prerequisites

Before creating a guardrail, decide which traffic should be inspected, whether
remote-guardrail outages should block or fail open, and whether the policy
should apply environment-wide or through an attachment.

In standalone mode, `/admin/v1/guardrails` creates guardrail definitions. It
does not create `GuardrailAttachment` rows, so attachment-scoped
rollout requires AISIX Cloud projection or direct config-store management.

## Configure Guardrails

Choose the hook point, then create the guardrail definition for the policy
backend you want to use.

### Hook Point

`hook_point` controls where the guardrail runs:

- `input` checks the caller request before it reaches the provider.
- `output` checks the upstream response before it is returned to the caller.
- `both` checks both sides.

Input blocking prevents the prompt from reaching the provider. Output blocking
prevents the provider response from reaching the caller.

### Create a Keyword Guardrail

Keyword guardrails support literal and regex patterns:

```shell
curl -sS -X POST http://127.0.0.1:3001/admin/v1/guardrails \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "block-secrets",
    "hook_point": "input",
    "kind": "keyword",
    "patterns": [
      {"kind": "literal", "value": "AKIA"},
      {"kind": "regex", "value": "\\bssn:\\s*\\d{3}-\\d{2}-\\d{4}"}
    ]
  }'
```

If a keyword guardrail blocks content, the proxy returns `422`. Invalid regex
patterns are rejected before the rule is applied, so a typo does not silently
disable the policy.

An empty keyword pattern list is valid but inert. It behaves like a guardrail
that allows every request.

### Create a Bedrock Guardrail

`kind: "bedrock"` calls AWS Bedrock `ApplyGuardrail`:

```json
{
  "name": "bedrock-review",
  "kind": "bedrock",
  "hook_point": "input",
  "fail_open": true,
  "guardrail_id": "gr-123456789abc",
  "guardrail_version": "DRAFT",
  "region": "us-east-1",
  "aws_credentials": {
    "kind": "static",
    "access_key_id": "YOUR_ACCESS_KEY_ID",
    "secret_access_key": "YOUR_SECRET_ACCESS_KEY"
  },
  "latency_mode": {
    "kind": "timed",
    "timeout_ms": 3000
  }
}
```

When Bedrock returns an intervention, AISIX blocks with `422`. When Bedrock is
unavailable, throttled, or times out, `fail_open` decides whether the request
continues as a bypass or is blocked.

`bedrock_endpoint_url` in bootstrap configuration overrides the Bedrock endpoint
for every Bedrock guardrail in the deployment. Use it for private or test
Bedrock endpoints; it is not configured per guardrail row.

### Create an Azure Prompt Shield Guardrail

`kind: "azure_content_safety"` calls Azure AI Content Safety Prompt Shield:

```json
{
  "name": "prompt-shield",
  "kind": "azure_content_safety",
  "hook_point": "input",
  "fail_open": true,
  "endpoint": "https://YOUR_RESOURCE.cognitiveservices.azure.com",
  "api_key": "YOUR_AZURE_CONTENT_SAFETY_KEY",
  "timeout_ms": 5000
}
```

The data plane calls `/contentsafety/text:shieldPrompt?api-version=2024-09-01`
on the configured endpoint and authenticates with
`Ocp-Apim-Subscription-Key`.

`timeout_ms` defaults to `5000`. A timeout, throttling response, 5xx response,
or configuration error follows `fail_open`.

### Create an Aliyun Text Moderation Guardrail

`kind: "aliyun_text_moderation"` calls Aliyun's content-safety guardrail
(`TextModerationPlus`) on the `green-cip.<region>.aliyuncs.com` endpoint:

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

The input hook uses `llm_query_moderation`; the output hook uses
`llm_response_moderation`. AISIX blocks when Aliyun returns a `RiskLevel` at or
above `risk_level_threshold`. The default threshold is `high`.

| Field | Behavior |
| --- | --- |
| `region` | Builds the default endpoint. Set `endpoint` to override it for a private proxy or custom endpoint. |
| `output_fail_open` | Defaults to `false`, so an Aliyun outage cannot release unscanned model output. The request-level `fail_open` field governs the input hook. |
| Streaming controls | `stream_processing_mode`, `window_size`, `window_overlap_size`, `max_buffer_bytes`, and `on_buffer_exceeded` control streaming-output moderation. |

## Runtime Behavior

Review how AISIX scopes guardrails, handles managed-resource fields, and
returns guardrail decisions to callers.

### Scope Guardrails

AISIX resolves guardrails from two resource types:

- `Guardrail`, the policy definition.
- `GuardrailAttachment`, the binding between a guardrail and a scope.

Attachments can bind a guardrail to the whole environment, a model entry, an API
key entry, or a team bucket. When the same guardrail matches through more than
one attachment, the highest `priority` wins. If priority is tied, the more
specific scope wins.

The standalone `/admin/v1/guardrails` API manages guardrail definitions.
Attachment rows come from managed projection or a direct config-store workflow.

A guardrail definition with no attachment rows applies environment-wide at
priority `0`. As soon as any attachment row exists for that guardrail,
attachment semantics take over. If all of those attachments are disabled, the
guardrail does not run.

### Fields Accepted Without Enforcement

Some fields are accepted with the resource but do not currently change how
AISIX evaluates guardrails.

| Field | Accepted On | Runtime Behavior |
| --- | --- | --- |
| `enforcement_mode: "monitor"` | Guardrail definition | Records the requested posture. AISIX still blocks when the guardrail fires. |
| `mandatory` | Guardrail definition | Accepted but does not currently change enforcement. `fail_open` controls the remote-guardrail error path. |
| `direction` | Guardrail attachment | Accepted on projected resources. Configure `hook_point` on the guardrail definition to control input and output checking. |

For accepted fields, use [Resource schemas](../reference/resource-schemas.md)
and the [Admin API reference](/ai-gateway/reference/admin-api).

### Response Behavior

Guardrail denials return `422`.

For remote guardrails with `fail_open: true`, an upstream failure can produce a
bypass instead of a denial. The proxy continues the request and records the first
bypass reason in usage telemetry.

## Troubleshooting

### The Resource Saves but Nothing Is Blocked

Confirm you are testing `POST /v1/chat/completions` or `POST /v1/messages`.
Those are the proxy routes that run the guardrail chain.

Then check that the guardrail is `enabled`, the `hook_point` covers the side you
are testing, the request or response contains inspectable content, and the rule
has a matching attachment or is using the environment-wide default behavior.

For remote guardrails, also check credentials, endpoint reachability, timeout
settings, and whether your data-plane build includes the relevant feature.

### A Blocked Request Returns `422`

Guardrail denials return `422`.

### A Remote Guardrail Lets Traffic Through During an Outage

Check `fail_open`. When it is `true`, remote guardrail failures bypass blocking
and appear in telemetry as a guardrail bypass reason.

### Monitor Mode Still Blocks Traffic

The stored `monitor` posture does not make the data plane allow a matched
guardrail violation.

## Related Reading

For standalone guardrail CRUD, see [Admin API](admin-api.md). For the
dynamic-resource model, see [Configuration overview](overview.md). For
caller-visible denial responses, see
[Headers and error codes](../reference/headers-and-error-codes.md).
