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

## Why This Matters

This is one of the clearest examples of separating caller identity from upstream provider identity.

Callers continue using the same AISIX API key and model alias while the control plane changes which upstream credential backs that model.

## Operational Sequence

The important operator checkpoints are:

1. create the replacement provider key correctly
2. update the model reference
3. wait for projection to the managed data plane
4. verify that live traffic still succeeds

## Troubleshooting

### Rotation is complete in Cloud, but live traffic still uses the old behavior

Check projection timing before assuming the model update itself failed.

## Related Pages

- [Provider Keys](../configuration/provider-keys.md)
- [Resource Projection](resource-projection.md)
- [Configuration Propagation](../configuration/configuration-propagation.md)
