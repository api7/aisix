# Roadmap

This roadmap tracks planned capabilities and areas that are not yet ready to document as generally available product behavior.

Use this roadmap to understand direction. Use the [official documentation](https://docs.api7.ai/ai-gateway/) to understand what is available today.

## Principles

- Official documentation describes current, verified behavior.
- This roadmap collects planned or incomplete capabilities.
- Presence in this roadmap is not a delivery commitment.

## Now

### Provider Compatibility Expansion

Current status:
- The gateway already exposes multiple client-facing endpoints across OpenAI-compatible and Anthropic-style paths.
- Support depth still varies by endpoint and provider combination.

Planned outcome:
- Broader parity across providers and endpoint families.

Applies to:
- `AISIX AI Gateway`

## Next

### Redis-Backed Cache Policy Completion

Current status:
- The data plane enforces `CachePolicy.backend` per matched policy: `memory` uses the in-process cache; `redis` uses the shared Redis cache when the bootstrap config provides `cache.redis`, otherwise caching is disabled for matching requests (no silent memory fallback).
- Redis cluster/sentinel modes and broader support boundaries are still being expanded.

Planned outcome:
- Clear, fully supported Redis-backed cache policy behavior across all Redis deployment modes.

Applies to:
- `AISIX AI Gateway`

### Cloud Playground Parity With Data-Plane Path

Current status:
- AISIX Cloud playground is a preview path that sends requests from the control plane directly to the upstream provider.
- It does not pass through the managed data plane, so it does not exercise data-plane routing, cache, guardrails, or rate limiting.

Planned outcome:
- A more production-representative playground experience.

Applies to:
- `AISIX Cloud`

### First-Party Data-Warehouse Sinks

Current status:
- The `object_store` observability exporter writes batched NDJSON telemetry to Amazon S3, Google Cloud Storage, or Azure Blob today. Loading that telemetry into a warehouse such as Snowflake or Databricks is a customer-run step.
- A first-party sink that manages the pipe, the schema, and exactly-once streaming directly to the warehouse is not yet available.

Planned outcome:
- First-party Snowflake and Databricks sinks where the gateway owns the end-to-end delivery to the warehouse table, including managed schema and exactly-once semantics.

Applies to:
- `AISIX Cloud`

## Later

### Advanced Governance And Multi-Team Controls

Current status:
- Core environment and resource management are the active focus.

Planned outcome:
- Richer organization and governance controls.

Applies to:
- `AISIX Cloud`

### Expanded Advanced Cache Backends

Current status:
- The current docs and runtime center on prompt-response caching with the currently implemented backends.

Planned outcome:
- Additional backend strategies where they are backed by real runtime support.

Applies to:
- `AISIX AI Gateway`

## Not Current Product Behavior

These areas are not current product behavior unless implementation status changes:

- planned-only MCP or agent-gateway features
- planned-only control-plane governance features not yet backed by code
- provider or endpoint support that is not yet reflected in the current implementation

## Related Pages

- [AISIX AI Gateway documentation](https://docs.api7.ai/ai-gateway/)
- [AISIX AI Gateway quickstart](https://docs.api7.ai/ai-gateway/quickstart/)
- [AISIX Cloud](https://api7.ai/ai-gateway)
