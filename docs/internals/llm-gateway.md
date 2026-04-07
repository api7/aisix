# LLM Gateway

This document describes the current Layer 3 gateway entry point.

The current implementation is intentionally narrow:

- it only handles non-streaming chat requests
- it routes either through native format support or through the OpenAI Chat hub
- it exposes a typed `ChatResponse<F>` envelope even though the current implementation only returns the `Complete` variant

## Files

The current slice is implemented in two files:

- `gateway::gateway` in `src/gateway/gateway.rs`
- `gateway::types::response` in `src/gateway/types/response.rs`

`gateway::mod.rs` re-exports `Gateway`, and `gateway::types::mod.rs` re-exports `ChatResponse`.

## Core Flow

`Gateway::chat<F>()` is the typed entry point.

Its control flow is:

1. reject `stream=true` up front
2. ask the format whether the provider supports a native path
3. if native support exists, call the native non-streaming path
4. otherwise bridge through the hub request/response path

That keeps the current implementation limited to one closed loop: typed request in, typed response out, with `Usage` attached.

## Hub Path

The hub path is used when the format does not have native support for the chosen provider.

The sequence is:

1. `F::to_hub()` converts the request into `ChatCompletionRequest`
2. `Gateway::call_chat_hub()` runs provider-side request transformation and the HTTP POST
3. the provider definition converts the JSON response back into hub `ChatCompletionResponse`
4. `extract_chat_usage_from_response()` maps OpenAI-style usage into the shared `Usage` type
5. `F::from_hub()` converts the hub response back into `F::Response`

This keeps provider-specific JSON shape handling in the provider layer and format-specific response shape handling in the format layer. The gateway itself only orchestrates the sequence.

## Native Path

The native path is used when `F::native_support()` returns a `NativeHandler` for the chosen provider.

The native path is still non-streaming only.

The sequence is:

1. `F::call_native()` chooses the endpoint path and request body
2. `Gateway::call_chat_native()` executes the HTTP POST against the provider instance base URL
3. `F::parse_native_response()` parses the JSON response into `F::Response`

The gateway currently returns `Usage::default()` for native non-streaming calls because there is not yet a generic format hook for extracting usage out of arbitrary native response types.

## `ChatResponse<F>`

`ChatResponse<F>` is introduced now even though the current code path only emits `Complete`.

That is deliberate for two reasons:

- the public typed entry point should not need a return-type rewrite once streaming is enabled
- usage has different delivery timing for complete vs stream responses

The stream field uses a type-erased alias:

- `ChatResponseStream<F> = Pin<Box<dyn Stream<Item = Result<F::StreamChunk>> + Send>>`

That avoids hard-coding either `BridgedStream<F>` or `NativeStream<F>` into the response type. The later streaming work can box either stream adapter without changing the outer `ChatResponse<F>` shape.

## Helper Naming

The internal HTTP helpers are named `call_chat_hub()` and `call_chat_native()` on purpose.

The gateway layer will later grow non-chat entry points such as embeddings, TTS, STT, and image generation. Keeping the current helpers explicitly chat-scoped prevents ambiguity once those additional call paths exist.

## Current Limits

This module does not attempt to finish the full Layer 3 design.

- streaming requests are rejected explicitly and deferred to the later streaming gateway work
- `SessionStore` is not wired yet
- only `chat_completion()` is implemented as a convenience helper today
- `messages()` and `responses()` remain deferred until their corresponding formats land
- native non-streaming usage extraction is still format-specific future work

## Why This Slice Exists

This is the first point where the new provider layer, format layer, and response envelope meet under one runtime entry point.

Without this slice, later stream work would still be missing:

- a typed orchestration entry point
- a place to choose native vs hub routing
- a shared `ChatResponse<F>` shape
- a common provider error mapping path

That is why the implementation stops at non-streaming correctness first and leaves stream transport integration to later gateway work.
