---
title: Rotate A Provider Key Without Client Impact
description: Rotate upstream credentials by switching models to a new provider key while keeping caller-facing API keys stable.
sidebar_position: 85
---

This tutorial shows the safest current credential-rotation pattern.

## Goal

Rotate an upstream provider credential without reissuing caller API keys.

## Flow

1. create a new provider key with the rotated upstream credential
2. update the model to reference the new provider key
3. wait for propagation
4. verify that client traffic continues through the same caller API key

## Related Pages

- [Provider Keys](../configuration/provider-keys.md)
- [Provider Key Rotation](../cloud/provider-key-rotation.md)
- [Configuration Propagation](../configuration/configuration-propagation.md)
