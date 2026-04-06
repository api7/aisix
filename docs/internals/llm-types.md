# LLM Type and Trait System

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

## Trait stack

The type layer and the trait layer are tightly coupled, so they are documented together here.

Two independent axes shape the design:

- API format semantics such as OpenAI Chat, Anthropic Messages, and OpenAI Responses
- provider-specific behavior such as endpoint shape, auth headers, request transforms, and native bypasses

`ChatFormat` models the first axis. `ProviderMeta`, `ChatTransform`, and `ProviderCapabilities` model the second.

### `ChatFormat`

`ChatFormat` defines a complete external chat protocol:

- request type
- non-streaming response type
- streaming chunk type
- bridge rules to and from the hub format

The hub format remains OpenAI Chat. Every format therefore explains how to:

- convert its request into a hub request plus `BridgeContext`
- convert a hub response back into its own response type
- convert a hub stream into its own stream events

The trait also includes a native escape hatch for providers that can serve the source format directly.

### Explicit stream state

Two stream-state associated types are explicit:

- `BridgeState` for hub-to-format conversion
- `NativeStreamState` for provider-native streaming conversion

That keeps stream state typed and local to the format implementation instead of hiding it behind erased containers.

Hub stream state also keeps partially assembled tool calls keyed by `(choice_index, tool_call_index)`, because tool call indices are scoped to a streamed choice rather than globally unique across the whole chunk stream.

### Provider layering

The provider side is split into three layers.

`ProviderMeta` contains stable metadata such as provider name, default base URL, endpoint path, stream reader kind, and auth header construction.

`ChatTransform` contains hub-to-provider request and response mapping. Its default behavior is intentionally OpenAI-compatible: serialize the request, apply `CompatQuirks`, deserialize the response, and treat SSE `data:` lines as OpenAI-style chunks.

`ProviderCapabilities` is capability discovery. It returns typed trait objects such as `as_native_anthropic_messages()` and `as_native_openai_responses()` instead of booleans, so a provider cannot claim support for a feature without also exposing the methods behind that feature.

### Native support traits

`NativeAnthropicMessagesSupport` and `NativeOpenAIResponsesSupport` are optional extensions layered on top of `ChatTransform`.

`NativeHandler` is the small type-erased enum that carries those typed trait objects across format dispatch boundaries.

### `CompatQuirks`

`CompatQuirks` is the declarative escape hatch for OpenAI-compatible providers that are almost, but not exactly, compatible.

The current implementation supports:

- removing unsupported parameters
- renaming request parameters
- forcing `stream_options.include_usage` when a provider requires usage in streaming mode
- recording provider-specific stream termination markers and tool-argument behavior

This keeps provider-specific compatibility patches out of custom transform implementations unless the provider genuinely needs bespoke logic.
