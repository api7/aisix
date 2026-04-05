# LLM Types Deep Dive

This document describes the Layer 1 type system for the LLM subsystem. It covers the wire models for the supported chat APIs plus the shared metadata and error types used by later bridge and provider code.

## Namespace boundaries

The OpenAI namespace owns both Chat Completions and Responses. They are related APIs from the same vendor, but they are not peers of Anthropic types.

## OpenAI Chat as the hub format

The types in `gateway::types::openai` are both the OpenAI Chat Completions wire models and the internal hub format used by the N:M bridge architecture.

Three choices matter here:

- `ChatCompletionRequest.extra` uses `#[serde(flatten)]` so provider-specific fields can survive round-trips without polluting the hub model.
- `MessageContent` is untagged because OpenAI accepts either a plain string or a structured content array.
- `StopCondition` is untagged because the API accepts either a single stop string or a list.

That keeps the hub representation close to the public OpenAI schema while still being permissive enough for bridge code.

## OpenAI Responses under the OpenAI namespace

The Responses API types now live in `gateway::types::openai::responses`.

That module models the parts that are specific to Responses rather than Chat Completions:

- polymorphic input (`ResponsesInput`)
- built-in OpenAI tools (`ResponsesTool`)
- richer output items (`ResponsesOutputItem`)
- fine-grained SSE event types (`ResponsesApiStreamEvent`)

Keeping these types under the OpenAI namespace avoids presenting `responses` as a top-level peer alongside provider-agnostic or vendor-level modules.

## Anthropic message models

`gateway::types::anthropic` stays separate because it describes a different vendor protocol.

Its main differences from the hub format are:

- system prompts are top-level, not embedded in the message list
- content blocks are internally tagged by `type`
- streaming uses event-specific records instead of a single chunk envelope
- prompt caching metadata is part of the native schema

## Shared bridge metadata

`gateway::types::common::BridgeContext` carries information that cannot be represented cleanly in the hub request alone.

It currently has three buckets:

- `anthropic_messages_extras` for Anthropic Messages-specific request data
- `openai_responses_extras` for OpenAI Responses-specific state such as `previous_response_id`
- `passthrough` for arbitrary provider-specific values

This lets future `to_hub()` implementations return a normalized hub request without losing format-specific data that must be restored later.

## Unified usage accounting

`gateway::types::common::Usage` is intentionally sparse: every field is optional.

That matches real provider behavior. Some providers report only prompt tokens, some only final totals, and some stream usage late. `Usage::merge()` therefore follows two rules:

- overwrite only fields that are present in the incoming value
- derive `total_tokens` only when it was not explicitly provided and both prompt and completion counts exist

The tests cover overwrite behavior, derived totals, and preservation of explicit totals.

## GatewayError

`gateway::error::GatewayError` is the common error surface for the LLM subsystem.

It separates four concerns:

- client-side request problems (`Validation`, `Bridge`)
- data conversion problems (`Transform`)
- upstream/provider failures (`Provider`, `Http`)
- stream lifecycle failures (`Stream`)

Two helper methods make this usable from higher layers:

- `is_retryable()` centralizes retry policy
- `status_code()` maps failures to proxy-facing HTTP status codes

This keeps the later LLM runtime and proxy code from duplicating provider error classification logic.
