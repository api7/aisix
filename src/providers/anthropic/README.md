# Anthropic Provider

Bridges OpenAI-compatible Chat Completion API requests to Anthropic's [Messages API](https://docs.anthropic.com/en/api/messages).

## Request Conversion

`ChatCompletionRequest` → `AnthropicMessagesRequest` via `impl From`.

### Field Mapping

| OpenAI Field | Anthropic Field | Notes |
|---|---|---|
| `model` | `model` | Passed through (provider prefix already stripped) |
| `messages` (role=system) | `system` | Hoisted to top-level `system` parameter as array of `TextBlockParam` (`[{"type": "text", "text": "..."}]`). Each system message becomes a separate block. |
| `messages` (role≠system) | `messages` | Mapped 1:1 as `{"role": "...", "content": "..."}` |
| `max_tokens` | `max_tokens` | Required by Anthropic; defaults to `4096` if not provided |
| `temperature` | `temperature` | Anthropic range is 0.0–1.0; values >1.0 will be capped by the API |
| `top_p` | `top_p` | Direct mapping |
| `stop` | `stop_sequences` | Array of strings |
| `stream` | `stream` | Boolean |
| `user` | `metadata.user_id` | Maps to `{"user_id": "..."}` |

### Unsupported Fields (Ignored)

These `ChatCompletionRequest` fields have no semantically equivalent Anthropic parameter and are silently dropped:

| OpenAI Field | Reason |
|---|---|
| `frequency_penalty` | No Anthropic equivalent |
| `presence_penalty` | No Anthropic equivalent |
| `logprobs` | No Anthropic equivalent |
| `top_logprobs` | No Anthropic equivalent |
| `n` | Anthropic always returns exactly 1 completion |
| `response_format` | No direct equivalent (Anthropic uses tool-based structured output) |

### Anthropic-Only Parameters (Not Mapped)

These Anthropic parameters have no `ChatCompletionRequest` equivalent:

- `top_k` — No OpenAI field
- `tools` / `tool_choice` — Not yet in `ChatCompletionRequest` (upstream TODO)

## Response Conversion

`AnthropicMessagesResponse` → `ChatCompletionResponse` via `impl From`.

### Field Mapping

| Anthropic Field | OpenAI Field | Notes |
|---|---|---|
| `id` | `id` | Passed through (e.g., `msg_...`) |
| `model` | `model` | Passed through |
| `content` | `choices[0].message.content` | All `text` blocks concatenated into a single string |
| `role` | `choices[0].message.role` | Always `"assistant"` |
| `stop_reason` | `choices[0].finish_reason` | See stop reason mapping below |
| `usage.input_tokens` | `usage.prompt_tokens` | Direct mapping |
| `usage.output_tokens` | `usage.completion_tokens` | Direct mapping |
| — | `usage.total_tokens` | Computed: `input_tokens + output_tokens` |
| — | `object` | Always `"chat.completion"` |
| — | `created` | Set to current Unix timestamp |

### Stop Reason Mapping

| Anthropic `stop_reason` | OpenAI `finish_reason` |
|---|---|
| `end_turn` | `stop` |
| `max_tokens` | `length` |
| `stop_sequence` | `stop` |
| Other values | Passed through unchanged |

## Streaming

Anthropic uses typed SSE events (`event: <type>\ndata: <json>`) unlike OpenAI's `data:`-only format with `[DONE]` sentinel. The `type` field is inside the JSON payload, so only `data:` lines are parsed.

### Event → Chunk Mapping

| Anthropic Event | OpenAI Chunk | Notes |
|---|---|---|
| `message_start` | Initial chunk | Sets `delta.role = "assistant"`, captures `id` and `model` for subsequent chunks |
| `content_block_delta` (text_delta) | Content chunk | `delta.content` = text fragment |
| `message_delta` | Final chunk | `finish_reason` from stop reason mapping, `usage` with output token count |
| `content_block_start` | — | Skipped |
| `content_block_stop` | — | Skipped |
| `message_stop` | — | Skipped |
| `ping` | — | Skipped |
| `error` | Error chunk | Formatted as `[Anthropic error: {type} - {message}]` with `finish_reason = "stop"` |

## Authentication

Requests to Anthropic use:
- `x-api-key` header (from `provider_config.api_key`)
- `anthropic-version: 2023-06-01` header

## Configuration

```json
{
  "name": "@my-anthropic/sonnet",
  "model": "anthropic/claude-sonnet-4-6",
  "provider_config": {
    "api_key": "<your_key>",
    "api_base": "https://api.anthropic.com/v1"
  }
}
```

`api_base` is optional; defaults to `https://api.anthropic.com/v1`.

**Note**: Anthropic requires the `anthropic-version: 2023-06-01` header for Messages API requests. This is handled automatically by the provider.
