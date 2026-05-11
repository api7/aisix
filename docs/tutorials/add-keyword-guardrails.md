---
title: Add Keyword Guardrails
description: Add a keyword guardrail that blocks forbidden prompt or response content in AISIX AI Gateway.
sidebar_position: 82
---

This tutorial shows the current generally usable guardrail path: `kind: "keyword"`.

## Goal

Block a known forbidden pattern before it reaches the upstream.

## Flow

1. create a keyword guardrail with `hook_point: "input"`
2. wait for propagation
3. verify that benign traffic still succeeds
4. verify that matching traffic returns `422` with `error.type = "content_filter"`

## Related Pages

- [Guardrails](../configuration/guardrails.md)
- [Headers And Error Codes](../reference/headers-and-error-codes.md)
- [Testing And Verification](../operations/testing-and-verification.md)
