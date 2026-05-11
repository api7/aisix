---
title: Enable Response Caching
description: Enable prompt-response caching in AISIX AI Gateway and verify cache hit and miss behavior.
sidebar_position: 83
---

This tutorial shows how to turn on the current chat caching path.

## Goal

Serve the second identical chat request from cache and confirm the behavior with `x-aisix-cache`.

## Flow

1. create a direct model and caller API key
2. create a cache policy with `backend: "memory"`
3. send the first request and confirm `x-aisix-cache: miss`
4. send the same request again and confirm `x-aisix-cache: hit`

## Related Pages

- [Caching](../configuration/caching.md)
- [Metrics And Logs](../operations/metrics-and-logs.md)
- [Testing And Verification](../operations/testing-and-verification.md)
