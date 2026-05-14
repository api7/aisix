---
title: Guardrails
description: Configure keyword and Bedrock-shaped guardrail resources and understand their current runtime behavior in AISIX AI Gateway.
sidebar_position: 38
---

Guardrails are content-policy resources attached to the gateway's chat path.

Current guardrails run on `POST /v1/chat/completions` through the live guardrail chain.

Use this page to understand where guardrails execute today, not just what the schema can store.

## Current Fields

- `name`
- `enabled`
- `hook_point`
- `fail_open`
- `kind`

`hook_point` currently supports:

- `input`
- `output`
- `both`

These settings control where in the chat request/response lifecycle the current guardrail is asked to act.

## Keyword Guardrails

`kind: "keyword"` is the current generally usable guardrail type.

Example:

```bash title="Create a keyword guardrail"
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

Current runtime behavior:

- keyword guardrails run in-process on the data plane
- blocked requests return `422`
- input blocking prevents the prompt from reaching the upstream
- output blocking prevents the upstream response from reaching the caller

That makes keyword guardrails the currently reliable operator tool for in-process content blocking.

## Bedrock-Shaped Guardrails

`kind: "bedrock"` is part of the current resource schema.

Example shape:

```json title="Bedrock-shaped guardrail"
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
    "kind": "serial"
  }
}
```

Current runtime boundary:

- the gateway accepts and stores this shape
- the live chain does not document it as generally available runtime enforcement yet

This is the key difference between schema support and dependable runtime support.

Keep Bedrock runtime support in the roadmap and limited-capability framing, not as fully available behavior.

## Operator Guidance

- use `keyword` for production behavior you need to rely on today
- treat `bedrock` rows as an advanced or staged capability until your own deployment proves the runtime path you want

## Troubleshooting

### The resource saves but nothing is blocked

First confirm you are testing the `POST /v1/chat/completions` path and not assuming every proxy endpoint runs the guardrail chain.

### A blocked request returns `422`

That is expected for current guardrail denials.

## Related Pages

- [Admin API](admin-api.md)
- [Headers And Error Codes](../reference/headers-and-error-codes.md)
- [Roadmap](../roadmap.md)
