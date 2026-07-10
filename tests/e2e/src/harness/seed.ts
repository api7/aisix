import { randomUUID } from "node:crypto";

import { EtcdClient } from "./etcd.js";

/**
 * Seeds resources by writing canonical resource documents straight to
 * etcd — the same front door the control plane uses in managed mode,
 * where the Admin API is not in the write path. The
 * interface mirrors `AdminClient`'s create methods (same body shapes,
 * same `{id, value}` return with a generated id), so call sites migrate
 * mechanically: `admin.createModel({...})` → `seed.createModel({...})`.
 *
 * The document written is exactly the caller-supplied body — the
 * canonical resource shape from `schemas/resources/`. The loader fills
 * serde defaults on load, so a sparse document loads identically to an
 * Admin-API-written one; that equivalence is pinned by
 * `cases/seed-vs-admin-characterization-e2e.test.ts`.
 *
 * Unlike the Admin API there is no synchronous validation: a malformed
 * document is silently skipped by the loader and the test then times
 * out in `waitConfigPropagation`. Keep seed bodies aligned with the
 * schemas, and probe propagation with a positive condition.
 */
export class SeedClient {
  constructor(
    private readonly etcd: EtcdClient,
    private readonly prefix: string,
  ) {}

  async createModel(
    model: Record<string, unknown>,
  ): Promise<{ id: string; value: Record<string, unknown> }> {
    return this.put("models", model);
  }

  async createApiKey(
    key: Record<string, unknown>,
  ): Promise<{ id: string; value: Record<string, unknown> }> {
    return this.put("api_keys", key);
  }

  async createProviderKey(
    pk: Record<string, unknown>,
  ): Promise<{ id: string; value: Record<string, unknown> }> {
    // Same defaulting as AdminClient.createProviderKey: cp-api always
    // writes `provider` + `adapter`, so the seeded document carries the
    // OpenAI-compatible pair unless a test overrides them.
    return this.put("provider_keys", { provider: "openai", adapter: "openai", ...pk });
  }

  async createObservabilityExporter(
    exporter: Record<string, unknown>,
  ): Promise<{ id: string; value: Record<string, unknown> }> {
    return this.put("observability_exporters", exporter);
  }

  private async put(
    kind: string,
    value: Record<string, unknown>,
  ): Promise<{ id: string; value: Record<string, unknown> }> {
    // The Admin API generates a UUID server-side; here the harness is
    // the writer, so it generates one — the id lives in the key
    // (`<prefix>/<kind>/<id>`), not in the document.
    const id = randomUUID();
    await this.etcd.put(`${this.prefix}/${kind}/${id}`, JSON.stringify(value));
    return { id, value };
  }
}
