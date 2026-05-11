---
title: Provider Compatibility
description: Reference for current provider coverage and compatibility boundaries in AISIX AI Gateway.
sidebar_position: 64
---

## Current Provider Enum

The current provider set is:

- `openai`
- `anthropic`
- `gemini`
- `deepseek`

## Compatibility Boundary

Provider support is not identical across every endpoint and behavior surface.

Current reference point:

- the gateway exposes a mixed OpenAI-compatible and Anthropic-style surface
- support depth varies by provider and endpoint family

Use the feature matrix and integration docs as the current contract, and treat broader provider parity as ongoing work.

## Related Pages

- [Feature Matrix](../overview/feature-matrix.md)
- [OpenAI-Compatible API](../integration/openai-compatible-api.md)
- [Roadmap](../roadmap.md)
