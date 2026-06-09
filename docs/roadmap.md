---
title: Roadmap
description: Planned and not-yet-fully-available capabilities for AISIX AI Gateway and AISIX Cloud.
sidebar_position: 2
---

This page tracks planned capabilities and areas that are not yet ready to document as generally available product behavior.

Use this page to understand direction. Use the rest of the documentation set to understand what is available today.

## Principles

- Main documentation pages describe current, verified behavior.
- This roadmap collects planned or incomplete capabilities.
- Presence on this page is not a delivery commitment.

## Now

### Gateway Documentation Rebuild

Current status:
- The old flat docs set is being replaced with a new customer-facing structure organized around overview, quickstart, integration, configuration, Cloud, operations, reference, and tutorials.

Planned outcome:
- A full official docs set for platform engineers and AI agent developers.

Applies to:
- `AISIX AI Gateway`
- `AISIX Cloud`

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
- Cache policy schema supports `memory` and `redis` as backend hints.
- Current runtime behavior is still centered on `memory`, and Redis should not yet be treated as fully available customer behavior.

Planned outcome:
- Clear, fully supported Redis-backed cache policy behavior.

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
- The `object_store` observability exporter writes batched NDJSON telemetry to Amazon S3, Google Cloud Storage, or Azure Blob today. Loading that telemetry into a warehouse such as Snowflake or Databricks is a customer-run step — for example, Snowflake's Snowpipe auto-ingesting from the bucket, as covered in [Load AI gateway logs into Snowflake](tutorials/load-gateway-logs-into-snowflake.md).
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

## Out Of Scope For Current Docs

These areas should not be described as current product behavior unless implementation status changes:

- planned-only MCP or agent-gateway features
- planned-only control-plane governance features not yet backed by code
- provider or endpoint support that is not yet reflected in the current implementation

## Related Pages

- [Feature Matrix](overview/feature-matrix.md)
- [What Is AISIX AI Gateway](overview/what-is-aisix-ai-gateway.md)
- [Deployment Modes](overview/deployment-modes.md)
