---
title: Use An OpenAI Client With An Anthropic Upstream
description: Route an OpenAI-style client through AISIX AI Gateway to an Anthropic upstream model.
sidebar_position: 80
---

This tutorial shows the core gateway pattern: keep the client contract stable while changing the upstream provider.

## Goal

Use an OpenAI-compatible client path against a model backed by an Anthropic upstream.

## Flow

1. create a provider key for the Anthropic upstream
2. create a model alias backed by `provider: "anthropic"`
3. create a caller API key that allows that alias
4. send the request through the gateway

## Why This Works

The gateway resolves the caller-visible model alias, then dispatches through the provider bridge registered for that model's provider.

## Related Pages

- [Models](../configuration/models.md)
- [Provider Keys](../configuration/provider-keys.md)
- [OpenAI-Compatible API](../integration/openai-compatible-api.md)
