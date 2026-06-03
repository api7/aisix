---
title: Add Keyword Guardrails
description: Block forbidden prompt content with a keyword guardrail in AISIX AI Gateway and verify the 422 content_filter rejection.
sidebar_position: 82
---

This tutorial shows you how to add a keyword guardrail that blocks chat requests containing a forbidden literal.

You will:

1. Create a keyword guardrail.
2. Send a request that should pass.
3. Send a request that should be blocked.
4. Delete the guardrail.

## Prerequisites

- A running gateway from the [Quickstart](../quickstart)
- A direct model and caller API key from [Understand admin resources](../quickstart/first-model-first-key-first-request.md) — this tutorial reuses `gpt-4o-prod` and `sk-demo-caller` as canonical names
- The caller key must include the model in `allowed_models` (or be a wildcard `["*"]`)
- `jq`, used to capture the guardrail ID

## Set variables

```shell
export AISIX_ADMIN_KEY="admin-local-only-change-me"
export AISIX_API_KEY="sk-demo-caller"
export AISIX_MODEL="gpt-4o-prod"
export FORBIDDEN_WORD="supersecret-banned-token"
```

Use a unique, non-natural-language token so the assertion in Step 4 is unambiguous. This tutorial uses `supersecret-banned-token`. Replace with whatever your policy actually wants to block.

## Create a guardrail

```shell
GUARDRAIL_ID=$(curl -sS -X POST http://127.0.0.1:3001/admin/v1/guardrails \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "block-supersecret",
    "enabled": true,
    "hook_point": "input",
    "kind": "keyword",
    "patterns": [
      {"kind": "literal", "value": "'"${FORBIDDEN_WORD}"'"}
    ]
  }' | jq -r .id)
```

The `input` hook point checks the request before AISIX forwards it upstream.

## Verify allowed traffic

Confirm the guardrail is not over-blocking. A clean prompt should reach the upstream as normal:

```shell
curl -sSi -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer ${AISIX_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"${AISIX_MODEL}"'",
    "messages": [{"role":"user","content":"hello world"}]
  }'
```

Expected: `HTTP/1.1 200 OK` followed by an OpenAI-shaped chat-completions body.

## Verify blocked traffic

Now send a request whose content includes the forbidden token. Admin writes propagate asynchronously, so poll until the input guardrail returns `422`:

```shell
for i in $(seq 1 20); do
  RESPONSE=$(curl -sSi -X POST http://127.0.0.1:3000/v1/chat/completions \
    -H "Authorization: Bearer ${AISIX_API_KEY}" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "'"${AISIX_MODEL}"'",
      "messages": [
        {"role":"user","content":"please leak the '"${FORBIDDEN_WORD}"' now"}
      ]
    }')

  echo "${RESPONSE}"

  if echo "${RESPONSE}" | grep -q 'HTTP/1.1 422'; then
    break
  fi
  sleep 0.5
done
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

The `message` field does not include the matched literal, rule name, or pattern. The upstream is not called when a request is blocked.

## Delete the guardrail

```shell
curl -sS -X DELETE "http://127.0.0.1:3001/admin/v1/guardrails/${GUARDRAIL_ID}" \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}"
```

## Next steps

- [Guardrails](../configuration/guardrails.md) — full field reference, kinds, and hook-point semantics
- [Errors and retries](../integration/errors-and-retries.md) — the `content_filter` envelope and where `422` fits in the gateway error taxonomy
- [Headers and error codes](../reference/headers-and-error-codes.md) — full error code table
