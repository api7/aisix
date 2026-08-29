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
 * serde defaults on load, so a sparse document loads with the same
 * defaults the schema documents.
 *
 * There is no synchronous validation on this path: a malformed
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

  /**
   * Seeds a guardrail and, unless `attach` is false, the env-scoped
   * attachment that puts it in force.
   *
   * A guardrail's scope comes only from its attachments — an unattached
   * one governs nothing (AISIX-Cloud#1450 retired the fallback that used
   * to apply a zero-attachment guardrail to the whole environment, because
   * keying on the ABSENCE of rows made removing a guardrail's last
   * attachment WIDEN it). Attaching by default mirrors the console, which
   * always writes an attachment alongside the guardrail; a test that
   * manages its own scope passes `{ attach: false }` and writes the
   * attachment it wants.
   */
  async createGuardrail(
    guardrail: Record<string, unknown>,
    opts: { attach?: boolean } = {},
  ): Promise<{ id: string; value: Record<string, unknown> }> {
    const created = await this.put("guardrails", guardrail);
    if (opts.attach !== false) {
      await this.attachGuardrailToEnv(created.id);
    }
    return created;
  }

  /** Env-scope attachment: the guardrail applies to every request. */
  async attachGuardrailToEnv(
    guardrailID: string,
    priority = 100,
  ): Promise<{ id: string; value: Record<string, unknown> }> {
    return this.put("guardrail_attachments", {
      guardrail_id: guardrailID,
      scope_type: "env",
      priority,
    });
  }

  async createCachePolicy(
    policy: Record<string, unknown>,
  ): Promise<{ id: string; value: Record<string, unknown> }> {
    return this.put("cache_policies", policy);
  }

  async createRateLimitPolicy(
    policy: Record<string, unknown>,
  ): Promise<{ id: string; value: Record<string, unknown> }> {
    return this.put("rate_limit_policies", policy);
  }

  async createOidcProvider(
    provider: Record<string, unknown>,
  ): Promise<{ id: string; value: Record<string, unknown> }> {
    return this.put("oidc_providers", provider);
  }

  async createClaimMapping(
    mapping: Record<string, unknown>,
  ): Promise<{ id: string; value: Record<string, unknown> }> {
    return this.put("claim_mappings", mapping);
  }

  async createPassthroughRoute(
    route: Record<string, unknown>,
  ): Promise<{ id: string; value: Record<string, unknown> }> {
    return this.put("passthrough_routes", route);
  }

  /**
   * Overwrite the document at `<prefix>/<kind>/<id>` — the seed-side
   * equivalent of an Admin API PUT. Propagation is asynchronous; probe
   * it with the case's `waitConfigPropagation` condition as with
   * creates.
   */
  async update(
    kind: string,
    id: string,
    value: Record<string, unknown>,
  ): Promise<void> {
    await this.etcd.put(`${this.prefix}/${kind}/${id}`, JSON.stringify(value));
  }

  /** Remove `<prefix>/<kind>/<id>` so the loader drops the resource. */
  async delete(kind: string, id: string): Promise<void> {
    await this.etcd.delete(`${this.prefix}/${kind}/${id}`);
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
