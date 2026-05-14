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

This means provider compatibility is not a single yes/no question.

The real questions are:

- which endpoint family are you using
- which provider backs the resolved model
- whether the path is provider-native or translated

## Practical Reading Guide

- start with integration docs for endpoint-family behavior
- use the feature matrix for current breadth versus limited support
- use roadmap items for parity you expect but do not yet see documented

Use the feature matrix and integration docs as the current contract, and treat broader provider parity as ongoing work.

## Related Pages

- [Feature Matrix](../overview/feature-matrix.md)
- [OpenAI-Compatible API](../integration/openai-compatible-api.md)
- [Roadmap](../roadmap.md)
