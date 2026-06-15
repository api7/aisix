---
title: Cloud Playground
description: Understand the current AISIX Cloud playground behavior and its current limitations relative to the managed data plane.
sidebar_position: 74
---

The current AISIX Cloud playground is a preview path.

Use it as a fast configuration or model-selection preview, not as a production-path simulator.

## Current Behavior

The control plane sends the playground request directly to the upstream provider.

That means the current playground path does **not** exercise the managed data plane's:

- routing
- cache
- guardrails
- rate limits

It is best understood as a control-plane convenience feature.

Use it as a preview and configuration-checking surface, not as a perfect production-path simulation.

## When To Use It

- quick sanity checks
- early model configuration validation
- exploratory prompt testing from the Cloud UI path

## When Not To Use It As Proof

- validating managed data-plane routing behavior
- validating cache behavior
- validating guardrail behavior
- validating live budget or rate-limit behavior

## Troubleshooting

### The Cloud playground succeeds but real managed traffic behaves differently

That is expected with the current preview-only boundary.

## Related Pages

- [AISIX Cloud Overview](overview.md)
- [Roadmap](../roadmap.md)
