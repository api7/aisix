# LLM Stream Core

This document describes the current Layer 3 stream pipeline.

The current implementation introduces four building blocks:

- `sse_reader` in `gateway::streams::reader`
- `HubChunkStream` in `gateway::streams::hub`
- `BridgedStream` in `gateway::streams::bridged`
- `NativeStream` in `gateway::streams::native`

## Scope

This slice now covers both hub-facing and native stream adapters.

- `sse_reader` turns a byte stream into complete SSE lines.
- `HubChunkStream` turns provider stream lines into hub `ChatCompletionChunk` values.
- `BridgedStream` turns hub chunks into a concrete `ChatFormat` stream.
- `NativeStream` bypasses hub chunks and lets a `ChatFormat` decode native provider stream lines directly.

Gateway request execution still sits in a later step, but the reusable stream adapters are now in place.

## `sse_reader`

`sse_reader` keeps the contract simple: it emits raw SSE lines as strings.

Three details matter:

- it preserves the original line content instead of stripping `data:` prefixes
- it appends a synthetic trailing newline so the last partial line is flushed on EOF
- it drops empty separator lines so downstream transforms only see meaningful records

That behavior matches the current provider transforms, which already parse SSE framing themselves.

## `HubChunkStream`

`HubChunkStream` is the first stream adapter that works on top of provider transforms.

Its polling behavior is deliberately ordered:

1. drain the internal buffer first
2. poll the raw line stream only when the buffer is empty
3. call `ProviderCapabilities::transform_stream_chunk()` on each raw line
4. return the first produced hub chunk immediately and queue the rest

That fixes the earlier class of bug where a provider transform could return multiple chunks for one raw input line and only the first chunk would be observed.

## `BridgedStream`

`BridgedStream` sits one layer above `HubChunkStream`.

Its behavior mirrors the hub adapter:

1. drain any already buffered format-specific chunks
2. poll `HubChunkStream` only when that buffer is empty
3. call `ChatFormat::from_hub_stream()` for each hub chunk
4. return the first bridged chunk immediately and queue the rest

When the hub stream ends, `BridgedStream` also calls `ChatFormat::stream_end_events()` so formats can emit explicit terminators such as final SSE events.

## `NativeStream`

`NativeStream` is the direct counterpart for native-format paths.

Instead of going through hub `ChatCompletionChunk` values, it passes each raw provider stream line to `ChatFormat::transform_native_stream_chunk()`. Buffering rules are the same: if one input line expands into multiple output items, the adapter returns the first one immediately and preserves the rest for later polls.

## Usage Accumulation

`HubChunkStream` still centralizes hub-token tracking.

Whenever a transformed hub chunk carries `usage`, the stream copies `prompt_tokens` and `completion_tokens` into `ChatStreamState`. This keeps token accounting outside individual provider transforms while still making the latest usage totals available to later pipeline stages.

`BridgedStream` reports those latest hub totals through a oneshot channel on both normal completion and premature drop. It only fills fields that were actually observed in the hub stream, and it derives `total_tokens` when both sides are known.

`NativeStream` exposes the same completion and drop hook through `ChatFormat::native_usage()`. Formats that do not override that hook still send an empty `Usage` value, but native-capable formats can now report their own accumulated usage snapshot without coupling the generic stream adapter to any one state shape.

## Stream State

`ChatStreamState` now carries both aggregation data and provider stream metadata.

It currently tracks:

- buffered tool call assembly state
- latest input and output token counts
- streamed response metadata such as `id`, `model`, and `created`

Those metadata fields are required because some providers only emit response identity once at stream start, while later events still need to be converted into well-formed hub chunks.

## Current Limits

This implementation is intentionally narrow.

- only the SSE reader kind is implemented in this slice
- `JsonArrayStream` and `AwsEventStream` readers are still future work
- the legacy providers under `src/providers/` still keep their own SSE splitting logic
- no production native format has started overriding `ChatFormat::native_usage()` yet

That keeps the stream-layer work focused on buffering correctness, polling order, and handoff between provider, hub, and format-specific stream representations.
