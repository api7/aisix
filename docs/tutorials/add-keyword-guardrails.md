---
title: Add Keyword Guardrails
description: Block forbidden prompt content with a keyword guardrail in AISIX AI Gateway and verify the 422 content_filter rejection.
sidebar_position: 82
toc_max_heading_level: 2
---

Add a keyword guardrail that blocks chat requests containing a forbidden literal. You create the guardrail, verify both allowed and blocked traffic, and remove the guardrail at the end.

## Prerequisites

Before you start, run the gateway from the [Quickstart](../quickstart) and create a direct model plus caller API key with [Understand Admin Resources](../quickstart/first-model-first-key-first-request.md). The commands use `gpt-4o-prod` and `sk-demo-caller`. The caller key must include the model in `allowed_models`, or use the wildcard value `["*"]`. Install `jq` to capture the guardrail ID from the admin API response.

## Configure the Guardrail

In this section, you will create the guardrail resource and attach it to the request path.

### Set Variables

Export the values used by the guardrail commands:

```shell
export AISIX_ADMIN_KEY="admin-local-only-change-me"
export AISIX_API_KEY="sk-demo-caller"
export AISIX_MODEL="gpt-4o-prod"
export FORBIDDEN_WORD="supersecret-banned-token"
```

Use a unique, non-natural-language token so the blocked-traffic check is unambiguous. The commands use `supersecret-banned-token`; replace it with a token that matches your policy.

### Create a Guardrail

Create an input guardrail that blocks the configured literal token:

```shell
GUARDRAIL_ID=$(curl -sS -X POST "http://127.0.0.1:3001/admin/v1/guardrails" \
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

## Verify Guardrail Behavior

After the guardrail is attached, send allowed and blocked requests to confirm the policy behavior.

### Verify Allowed Traffic

Confirm that the guardrail allows unrelated prompts. A clean prompt should reach the upstream as normal:

```shell
curl -sSi -X POST "http://127.0.0.1:3000/v1/chat/completions" \
  -H "Authorization: Bearer ${AISIX_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"${AISIX_MODEL}"'",
    "messages": [{"role":"user","content":"hello world"}]
  }'
```

A successful response starts with `HTTP/1.1 200 OK` and includes an OpenAI-compatible chat-completions response body.

### Verify Blocked Traffic

Now send a request whose content includes the forbidden token. Admin writes propagate asynchronously, so poll until the input guardrail returns `422`:

```shell
for i in $(seq 1 20); do
  RESPONSE=$(curl -sSi -X POST "http://127.0.0.1:3000/v1/chat/completions" \
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

A blocked response starts with `HTTP/1.1 422 Unprocessable Entity` and includes this body:

```json
{
  "error": {
    "message": "request blocked by content policy",
    "type": "content_filter"
  }
}
```

The `message` field does not include the matched literal, rule name, or pattern. The upstream is not called when a request is blocked.

## Delete the Guardrail

Delete the guardrail when you finish the tutorial:

```shell
curl -sS -X DELETE "http://127.0.0.1:3001/admin/v1/guardrails/${GUARDRAIL_ID}" \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}"
```

## Related Reading

For field details, guardrail kinds, and hook-point semantics, see [Guardrails](../configuration/guardrails.md). For the `content_filter` error envelope and status code behavior, see [Errors and retries](../integration/errors-and-retries.md) and [Headers and error codes](../reference/headers-and-error-codes.md).
