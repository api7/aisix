---
title: Responses API
description: Learn how AISIX AI Gateway handles the OpenAI Responses API across OpenAI and non-OpenAI providers.
sidebar_position: 25
---

AISIX AI Gateway exposes `POST /v1/responses` as a proxy for the OpenAI Responses API.

Use this endpoint when your application (or a tool such as the OpenAI Codex CLI) speaks the Responses API surface rather than chat completions. It works regardless of which provider backs the resolved model.

## Provider Support

`/v1/responses` works for **any** configured provider:

- **OpenAI models** — the request body is forwarded verbatim to the upstream's own `/v1/responses` endpoint (a thin proxy), so every Responses feature the upstream supports passes through unchanged.
- **Non-OpenAI models** (Anthropic, Gemini, DeepSeek, …) — the gateway **bridges** the request: it translates the Responses payload into its internal chat format, dispatches through the same provider adapter `/v1/chat/completions` uses, and re-encodes the reply back into the Responses shape (non-streaming JSON and streaming SSE alike). This is what lets a Responses-only client such as Codex point at an Anthropic model.

The bridge is the same machinery `/v1/chat/completions` and `/v1/messages` use for cross-provider translation, so behavior (tool calling, failover within a Model Group, usage accounting) stays consistent across surfaces.

## Gateway Behavior

For every request the gateway authenticates and authorizes the caller key, resolves the model alias, runs the configured input guardrails, then:

**OpenAI provider (verbatim passthrough)**

1. rewrites `model` to the upstream provider model id
2. forwards the body to the upstream `/v1/responses` endpoint
3. returns JSON or streaming SSE depending on the request

**Non-OpenAI provider (cross-provider bridge)**

1. translates the Responses request (`instructions`, `input`, `tools`, `tool_choice`, `temperature`, `top_p`, `max_output_tokens`, `stream`) into the internal chat format
2. dispatches through the provider adapter (e.g. Anthropic Messages)
3. re-encodes the response into the Responses shape — a `message` output item for assistant text and a `function_call` output item per tool call, plus the streaming event sequence (`response.created` → `response.output_item.added` → `response.output_text.delta` / `response.function_call_arguments.delta` → `response.completed`)

Multi-turn agent loops work across providers: `function_call` and `function_call_output` items in `input` are translated into the upstream's tool-use / tool-result turns.

### What the bridge does not carry

Responses fields that have no cross-provider equivalent are dropped rather than forwarded (forwarding an OpenAI-only field would make a provider like Anthropic reject the request):

- `reasoning` (effort/summary) — extended-thinking is not mapped to a backend today
- `store`, `previous_response_id` — the gateway is stateless; replay the full `input` each turn
- hosted tools (`web_search`, `file_search`, `code_interpreter`, …) — only `type: "function"` tools translate
- `text`/`metadata`/`service_tier` and other OpenAI-only knobs

These limitations apply only to the non-OpenAI bridge path; OpenAI models forward the body verbatim.

## Usage Accounting

Both non-streaming and streaming requests emit a usage event carrying the upstream-reported token counts (`input_tokens`, `output_tokens`, plus the `reasoning_tokens` and `cached_tokens` sub-counts when present), so Responses-API traffic shows up in the logs and counts toward budget the same way chat completions do.

For a streamed request the counts are read from the terminal `response.completed` event (and from `response.incomplete` / `response.failed` on truncation or cancellation), so the usage event is emitted at end of stream. This matters for clients that always stream — for example the OpenAI Codex CLI — whose successful calls would otherwise be invisible to accounting.

## Example

```bash title="Call the Responses API"
curl -sS -X POST http://127.0.0.1:3000/v1/responses \
  -H "Authorization: Bearer YOUR_CALLER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-prod",
    "input": "Say hello from AISIX."
  }'
```

### Point Codex at a non-OpenAI model

```bash title="Codex (or any Responses client) against an Anthropic-backed alias"
curl -sS -X POST http://127.0.0.1:3000/v1/responses \
  -H "Authorization: Bearer YOUR_CALLER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-prod",
    "input": "Say hello from AISIX.",
    "stream": true
  }'
```

The gateway bridges this to the Anthropic Messages API and streams back the Responses SSE event sequence.

## When To Use Responses Instead Of Chat Completions

- use `/v1/responses` when your application or tool is standardized on that OpenAI API surface (for example the Codex CLI)
- use `/v1/chat/completions` when you want the broadest feature coverage; the Responses bridge carries the common path (text, tool calls, streaming) but drops OpenAI-only knobs listed above

## Troubleshooting

### Tool calls aren't replayed correctly across turns

Send the full conversation in `input` each turn (the gateway is stateless): include the assistant's prior `function_call` items and the matching `function_call_output` items. The gateway translates them into the backend's tool-use / tool-result turns.

### `reasoning` has no effect on a non-OpenAI model

Extended-thinking config isn't bridged today (see [What the bridge does not carry](#what-the-bridge-does-not-carry)). The request still succeeds; the field is ignored.

## Related Pages

- [Streaming](streaming.md)
- [OpenAI-Compatible API](openai-compatible-api.md)
- [Errors And Retries](errors-and-retries.md)
