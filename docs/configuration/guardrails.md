---
title: Guardrails
description: Configure content checks for chat requests and responses in AISIX AI Gateway.
sidebar_position: 38
---

Guardrails apply content policy at the gateway boundary. They can block prompts
before they reach an upstream provider, block responses before they reach a
caller, or record a bypass when a remote guardrail is unavailable and the policy
is configured to fail open.

Current guardrails run on `POST /v1/chat/completions` and `POST /v1/messages`.
Other proxy endpoints do not run the same guardrail chain today.

## Choose a guardrail kind

| Kind | Use when | Operational dependency |
| --- | --- | --- |
| `keyword` | You need deterministic local blocking for literals or regex patterns. | Runs inside the data plane and does not depend on an external provider. |
| `bedrock` | You already operate AWS Bedrock guardrails and want AISIX to call Bedrock `ApplyGuardrail` at the proxy boundary. | Requires Bedrock credentials, network reachability, and the default `bedrock` build feature. |
| `azure_content_safety` | You want Azure AI Content Safety Prompt Shield checks for jailbreak or indirect prompt-injection detection. | Requires Azure Content Safety credentials, network reachability, and the default `azure-content-safety` build feature. |

Remote guardrails are operational dependencies. Treat credentials, network
reachability, timeouts, and `fail_open` as part of the policy.

## Prerequisites

Decide three things before creating a guardrail:

- which traffic should be inspected: request input, provider output, or both
- what should happen when a remote guardrail is unavailable: block or fail open
- whether the guardrail should apply environment-wide or through an attachment

In standalone mode, `/admin/v1/guardrails` creates guardrail definitions. It
does not currently create `GuardrailAttachment` rows, so attachment-scoped
rollout requires a managed projection path or direct config-store workflow.

## Hook point

`hook_point` controls where the guardrail runs:

- `input` checks the caller request before upstream dispatch.
- `output` checks the upstream response before it is returned to the caller.
- `both` checks both sides.

Input blocking prevents the prompt from reaching the provider. Output blocking
prevents the provider response from reaching the caller.

## Create a keyword guardrail

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

## Create a Bedrock guardrail

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

## Create an Azure Prompt Shield guardrail

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

## Scope guardrails

The runtime resolves guardrails from two resource types:

- `Guardrail`, the policy definition.
- `GuardrailAttachment`, the binding between a guardrail and a scope.

Attachments can bind a guardrail to the whole environment, a model entry, an API
key entry, or a team bucket. When the same guardrail matches through more than
one attachment, the highest `priority` wins. If priority is tied, the more
specific scope wins.

Current standalone admin boundary:

- `/admin/v1/guardrails` manages guardrail definitions.
- There is no standalone admin route for `guardrail_attachments` yet.
- Managed control-plane projection or direct config-store writes can provide
  attachment rows.

For compatibility with older control planes, a guardrail definition with no
attachment rows applies environment-wide at priority `0`. As soon as any
attachment row exists for that guardrail, attachment semantics take over. If all
of those attachments are disabled, the guardrail does not run.

## Stored fields that are not full runtime controls

The generated schema includes fields that are stored for control-plane
compatibility but are not full runtime switches today.

`enforcement_mode: "monitor"` is not monitor-only in the current data plane. If
a guardrail fires, it blocks traffic even when the row says `monitor`.

`mandatory` is stored and forwarded, but the data plane does not consult it.
`fail_open` controls the current remote-guardrail error path.

`direction` on attachments is stored for dashboard/control-plane compatibility.
Use `hook_point` on the guardrail definition to control input and output
checking.

For the exact accepted shape, use the generated
[Resource schemas](../reference/resource-schemas.md) and the live
[Admin API reference](/ai-gateway/reference/admin-api).

## Response behavior

Guardrail denials return `422`.

For remote guardrails with `fail_open: true`, an upstream failure can produce a
bypass instead of a denial. The proxy continues the request and records the first
bypass reason in usage telemetry.

## Troubleshooting

### The resource saves but nothing is blocked

Confirm you are testing `POST /v1/chat/completions` or `POST /v1/messages`.
Other proxy endpoints do not run the same guardrail chain today.

Then check that the guardrail is `enabled`, the `hook_point` covers the side you
are testing, the request or response contains inspectable content, and the rule
has a matching attachment or is relying on the environment-wide compatibility
fallback.

For remote guardrails, also check credentials, endpoint reachability, timeout
settings, and whether your data-plane build includes the relevant feature.

### A blocked request returns `422`

That is expected for current guardrail denials.

### A remote guardrail lets traffic through during an outage

Check `fail_open`. When it is `true`, remote guardrail failures bypass blocking
and appear in telemetry as a guardrail bypass reason.

### `enforcement_mode: "monitor"` still blocks traffic

That is expected today. Monitor-only enforcement is not implemented in the data
plane.

## Next steps

- [Admin API](admin-api.md) explains standalone guardrail CRUD.
- [Configuration overview](overview.md) explains dynamic resources and managed
  projection.
- [Headers and error codes](../reference/headers-and-error-codes.md) describes
  caller-visible denial responses.
