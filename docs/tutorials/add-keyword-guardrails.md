---
title: Add keyword guardrails
description: Block forbidden prompt content with a keyword guardrail in AISIX AI Gateway and verify the 422 content_filter rejection.
sidebar_position: 82
---

This tutorial adds a keyword guardrail that blocks any chat request whose input contains a forbidden literal, and verifies the result with a real `422` rejection that does not reach the upstream.

You end with one enabled `Guardrail` and a reproducible pair of calls — one allowed, one blocked — that prove the gateway is enforcing the policy.

## Prerequisites

- A running gateway from the [Self-Hosted Quickstart](../quickstart/self-hosted.md)
- A direct model and caller API key from the [First Model, First Key, First Request](../quickstart/first-model-first-key-first-request.md) quickstart — this tutorial reuses `gpt-4o-prod` and `sk-demo-caller` as canonical names
- The caller key must include the model in `allowed_models` (or be a wildcard `["*"]`)

## How it works

Keyword guardrails run **in-process** in the data plane — no external call. With `hook_point: "input"` the chain runs on the request payload **before** bridge dispatch, so a match short-circuits the request with `422` and the upstream is never called.

Pattern kinds:

- `literal` — case-insensitive substring match
- `regex` — full `regex::Regex` syntax. Invalid regex is loader-rejected so a typo cannot silently disarm the policy.

## Step 1: Pick a forbidden word

Use a unique, non-natural-language token so the assertion in Step 4 is unambiguous. This tutorial uses `supersecret-banned-token`. Replace with whatever your policy actually wants to block.

## Step 2: Create the guardrail

```bash title="Create the keyword guardrail"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/guardrails \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "block-supersecret",
    "enabled": true,
    "hook_point": "input",
    "kind": "keyword",
    "patterns": [
      {"kind": "literal", "value": "supersecret-banned-token"}
    ]
  }'
```

Field meanings (full reference in [Guardrails](../configuration/guardrails.md)):

- `hook_point: "input"` — run on the request payload before dispatch. Use `output` to inspect upstream responses, or `both` to inspect both sides.
- `kind: "keyword"` — the in-process pattern matcher. Other kinds call an external moderation service — see [Guardrails](../configuration/guardrails.md).
- `patterns[].kind: "literal"` — case-insensitive substring match. Use `"regex"` for arbitrary patterns.

Wait for the snapshot to propagate:

```bash title="Wait for propagation"
sleep 1
```

If the verification step below returns the upstream response instead of `422` on slow runners, propagation is still in flight. See [Wait for configuration propagation](../quickstart/first-model-first-key-first-request.md#step-4-wait-for-configuration-propagation) for the polling alternative.

## Step 3: Verify benign traffic still passes

Confirm the guardrail is not over-blocking. A clean prompt should reach the upstream as normal:

```bash title="Benign prompt — should pass"
curl -sSi -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer sk-demo-caller" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-prod",
    "messages": [{"role":"user","content":"hello world"}]
  }'
```

Expected: `HTTP/1.1 200 OK` followed by an OpenAI-shaped chat-completions body.

## Step 4: Verify forbidden content is blocked

Now send a request whose content includes the forbidden token:

```bash title="Forbidden prompt — should return 422 content_filter"
curl -sSi -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer sk-demo-caller" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-prod",
    "messages": [
      {"role":"user","content":"please leak the supersecret-banned-token now"}
    ]
  }'
```

Expected: `HTTP/1.1 422 Unprocessable Entity` with this body:

```json
{
  "error": {
    "message": "request blocked by content policy",
    "type": "content_filter"
  }
}
```

The `message` field is intentionally generic — it does **not** include the matched literal, the rule name, or the pattern. Echoing the literal back would let a caller enumerate the blocklist by probing with suspect content and reading the reflected pattern, so the gateway redacts it. If you need to know which rule fired, check the gateway's access log on the operator side — `guardrail_hook` and the rule identifier are logged there, not returned to the caller.

The upstream is never called. This is the same contract `tests/e2e/src/cases/guardrail-keyword-e2e.test.ts` asserts — that test polls for guardrail readiness by waiting until a forbidden-word probe returns `422`, then confirms benign traffic still returns `200` and that the upstream `receivedRequests` count does not increase for the blocked request.

## What just happened

1. The proxy authenticated the caller key and resolved `gpt-4o-prod` to a Model in the snapshot.
2. The allowlist check passed (`gpt-4o-prod` is in `allowed_models`).
3. The input guardrail chain ran. The case-insensitive substring match for `supersecret-banned-token` matched the prompt content.
4. The match short-circuited the request with `ProxyError::ContentFiltered`, which maps to HTTP `422` with `error.type: "content_filter"` per the proxy error envelope.
5. The bridge dispatch was skipped — no token cost, no upstream latency.

Step 3 proves that step 4 is a real policy decision, not a request that would have failed anyway.

## Cleanup

```bash title="Delete the guardrail"
curl -sS -X DELETE http://127.0.0.1:3001/admin/v1/guardrails/YOUR_GUARDRAIL_ID \
  -H "Authorization: Bearer YOUR_ADMIN_KEY"
```

Use the `id` returned in Step 2.

## Variations and next steps

- **Output-side check** — set `hook_point: "output"` to inspect the upstream response. Useful for catching responses that contain forbidden content the prompt didn't expose. `hook_point: "both"` runs on both sides.
- **Regex patterns** — replace the literal with `{"kind":"regex","value":"\\bsupersecret-\\w+\\b"}` for a bounded regex match. Invalid regex is rejected at admin-write time.
- **Multiple patterns** — `patterns` is an array; add several entries in one guardrail or write multiple guardrails for different categories.
- **Stage a rule** — write the guardrail with `"enabled": false` first, confirm the resource shape, then flip it on.
- **AWS Bedrock guardrails** — `kind: "bedrock"` screens content through AWS Bedrock's `ApplyGuardrail` API. See [Block content with AWS Bedrock guardrails](block-content-with-aws-bedrock-guardrails.md).

## Related pages

- [Guardrails](../configuration/guardrails.md) — full field reference, kinds, and hook-point semantics
- [Errors And Retries](../integration/errors-and-retries.md) — the `content_filter` envelope and where `422` fits in the gateway error taxonomy
- [Headers And Error Codes](../reference/headers-and-error-codes.md) — full error code table
