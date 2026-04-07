# LLM Stream Core

This document describes the first part of the Layer 3 stream pipeline.

The current implementation introduces two building blocks:

- `sse_reader` in `gateway::streams::reader`
- `HubChunkStream` in `gateway::streams::hub`

## Scope

This slice only covers the hub-facing stream foundation.

- `sse_reader` turns a byte stream into complete SSE lines.
- `HubChunkStream` turns provider stream lines into hub `ChatCompletionChunk` values.

`BridgedStream` and `NativeStream` are intentionally deferred to later steps.

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

## Usage Accumulation

`HubChunkStream` also centralizes usage tracking.

Whenever a transformed hub chunk carries `usage`, the stream copies `prompt_tokens` and `completion_tokens` into `ChatStreamState`. This keeps token accounting outside individual provider transforms while still making the latest usage totals available to later pipeline stages.

## Stream State

`ChatStreamState` now carries both aggregation data and provider stream metadata.

It currently tracks:

- buffered tool call assembly state
- latest input and output token counts
- streamed response metadata such as `id`, `model`, and `created`

Those metadata fields are required because some providers only emit response identity once at stream start, while later events still need to be converted into well-formed hub chunks.

## Current Limits

This implementation is intentionally narrow.

- only the SSE reader is implemented in this slice
- `JsonArrayStream` and `AwsEventStream` readers are still future work
- no format bridging happens here yet; this stream only produces hub chunks

That keeps the first stream-layer step focused on correctness of buffering, polling order, and usage propagation.
