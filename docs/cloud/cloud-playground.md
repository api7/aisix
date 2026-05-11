---
title: Cloud Playground
description: Understand the current AISIX Cloud playground behavior and its current limitations relative to the managed data plane.
sidebar_position: 74
---

The current AISIX Cloud playground is a preview path.

## Current Behavior

The control plane sends the playground request directly to the upstream provider.

That means the current playground path does **not** exercise the managed data plane's:

- routing
- cache
- guardrails
- rate limits

Use it as a preview and configuration-checking surface, not as a perfect production-path simulation.

## Related Pages

- [AISIX Cloud Overview](overview.md)
- [Roadmap](../roadmap.md)
