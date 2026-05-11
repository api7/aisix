---
title: Provider Key Rotation
description: Rotate upstream provider credentials in AISIX Cloud without forcing caller API key changes.
sidebar_position: 76
---

The current provider-key rotation pattern in AISIX Cloud is:

1. create a new provider key with the rotated upstream credential
2. update the model to reference the new provider key
3. let the change propagate to the managed data plane
4. continue serving callers without reissuing caller API keys

This keeps caller-facing credentials stable while upstream credentials change.

## Related Pages

- [Provider Keys](../configuration/provider-keys.md)
- [Resource Projection](resource-projection.md)
- [Configuration Propagation](../configuration/configuration-propagation.md)
