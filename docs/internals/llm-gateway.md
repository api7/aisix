# LLM Gateway

This document describes the current Layer 3 gateway entry point.

The current implementation is intentionally narrow:

- it handles both complete and streaming chat requests
- it routes either through native format support or through the OpenAI Chat hub
- it exposes a typed `ChatResponse<F>` envelope that can return either `Complete` or `Stream`

## Files

The current slice spans three areas:

- `gateway::gateway` in `src/gateway/gateway.rs`
- `gateway::formats` in `src/gateway/formats/`
- `gateway::types::response` in `src/gateway/types/response.rs`

`gateway::mod.rs` re-exports `Gateway`, `gateway::formats::mod.rs` re-exports the current chat format entry points, and `gateway::types::mod.rs` re-exports `ChatResponse`.

## Core Flow

`Gateway::chat<F>()` is the typed entry point.

Its control flow is:

1. ask the format whether the request is streaming
2. ask the format whether the provider supports a native path
3. if native support exists, call the native path
4. otherwise use the hub request/response path for complete calls or the hub streaming path for stream calls

That keeps the current implementation limited to one closed loop: typed request in, typed response out, with `Usage` attached either directly or through a oneshot receiver.

## Hub Path

The hub path is used when the format does not have native support for the chosen provider.

The sequence is:

1. `F::to_hub()` converts the request into `ChatCompletionRequest`
2. `Gateway::call_chat_hub()` runs provider-side request transformation and the HTTP POST
3. the provider definition converts the JSON response back into hub `ChatCompletionResponse`
4. `extract_chat_usage_from_response()` maps OpenAI-style usage into the shared `Usage` type
5. `F::from_hub()` converts the hub response back into `F::Response`

This keeps provider-specific JSON shape handling in the provider layer and format-specific response shape handling in the format layer. The gateway itself only orchestrates the sequence.

For streaming hub calls, the sequence is:

1. `F::to_hub()` converts the request into `ChatCompletionRequest`
2. `Gateway::call_chat_hub_stream()` runs provider-side request transformation and the HTTP POST
3. `select_chat_stream_reader()` chooses the raw response reader based on `StreamReaderKind`
4. `HubChunkStream` parses the provider stream into hub `ChatCompletionChunk` values
5. `BridgedStream<F>` converts those hub chunks into `F::StreamChunk` and sends final `Usage` through a oneshot channel

Today the gateway only wires `StreamReaderKind::Sse`. Other reader kinds return a validation error until their readers are implemented.

## Native Path

The native path is used when `F::native_support()` returns a `NativeHandler` for the chosen provider.

The sequence is:

1. `F::call_native()` chooses the endpoint path and request body
2. `Gateway::call_chat_native()` executes the HTTP POST against the provider instance base URL
3. for complete calls, `F::parse_native_response()` parses the JSON response into `F::Response`, then `F::response_usage()` can extract a `Usage` snapshot from that typed response
4. for stream calls, `NativeStream<F>` converts provider-native chunks into `F::StreamChunk` and sends final `Usage` through a oneshot channel

Native complete calls no longer hard-code `Usage::default()`. Formats can now report native complete-call usage through `ChatFormat::response_usage()`, while formats that keep the default hook still return an empty `Usage` value.

## `ChatResponse<F>`

`ChatResponse<F>` uses a single public shape for both complete and stream responses.

That is deliberate for three reasons:

- the public typed entry point should not need a return-type rewrite once streaming is enabled
- usage has different delivery timing for complete vs stream responses
- the gateway can box either bridged hub streams or native streams behind one alias

The stream field uses a type-erased alias:

- `ChatResponseStream<F> = Pin<Box<dyn Stream<Item = Result<F::StreamChunk>> + Send>>`

That avoids hard-coding either `BridgedStream<F>` or `NativeStream<F>` into the response type. The gateway can box either stream adapter without changing the outer `ChatResponse<F>` shape.

## Helper Naming

The internal HTTP helpers are named `call_chat_hub()` and `call_chat_native()` on purpose.

The gateway layer will later grow non-chat entry points such as embeddings, TTS, STT, and image generation. Keeping the current helpers explicitly chat-scoped prevents ambiguity once those additional call paths exist.

## Current Limits

This module does not attempt to finish the full Layer 3 design.

- `SessionStore` is not wired yet
- `chat_completion()` and `messages()` are implemented as convenience helpers today
- `responses()` remains deferred until its corresponding format lands
- only `StreamReaderKind::Sse` is wired today; `AwsEventStream` and `JsonArrayStream` are still deferred
- native complete-call usage reporting depends on each format implementing `ChatFormat::response_usage()`; formats that do not override it still return empty usage

## Why This Slice Exists

This is the first point where the new provider layer, format layer, and response envelope meet under one runtime entry point.

Without this slice, the current gateway would still be missing:

- a typed orchestration entry point
- a place to choose native vs hub routing
- a shared `ChatResponse<F>` shape
- a common provider error mapping path

That is why the implementation first established typed complete-call orchestration and then extended the same entry point to stream transport integration.
