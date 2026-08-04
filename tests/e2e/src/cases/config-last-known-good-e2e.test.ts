import { createHash, randomUUID } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E for RED last-known-good retention (issue #871, PR 2). When an etcd
// document is updated to bytes this data-plane version cannot represent
// (RED — schema violation), the previously accepted value must keep
// serving real traffic for as long as the etcd key exists:
//
// - immediately after the rejected watch update (already true pre-PR);
// - across a full restart, which replays the snapshot cache AND runs a
//   fresh load_all + resync against the still-rejected etcd bytes — the
//   two paths that used to silently drop the row (the "cliff");
// - with the staleness observable every cycle: `rejected[]` on
//   `GET /status/config` carries `serving_stale_since` (stable across
//   the restart) and a recomputed `serving_stale_age_seconds`, and the
//   metrics listener exposes a per-kind stale-served gauge;
// - until the etcd key is DELETED, at which point the pinned value dies
//   with it — retention must never outlive the key.

const CALLER_PLAINTEXT = "sk-last-known-good-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

interface StatusConfig {
  state: string;
  applied?: { resource_counts: Record<string, number> };
  rejected: Array<{
    resource_kind: string;
    resource_id: string;
    serving_stale_since?: string;
    serving_stale_age_seconds?: number;
  }>;
  partially_compatible: Array<{ resource_kind: string; field: string; count: number }>;
}

async function getStatusConfig(app: SpawnedApp): Promise<StatusConfig> {
  const res = await fetch(`${app.metricsUrl}/status/config`);
  expect(res.status).toBe(200);
  return (await res.json()) as StatusConfig;
}

async function scrape(app: SpawnedApp): Promise<string> {
  const res = await fetch(`${app.metricsUrl}/metrics`);
  expect(res.status).toBe(200);
  return res.text();
}

describe("config last-known-good: rejected updates keep serving across resync and restart", () => {
  let app: SpawnedApp | undefined;
  let stoppedApp: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcd: EtcdClient | undefined;
  let etcdReachable = false;
  let cacheDir: string | undefined;

  const etcdPrefix = `/aisix-e2e-lkg-${randomUUID()}`;
  let modelId: string;
  let modelKey: string;
  let staleSinceBeforeRestart: string | undefined;

  beforeAll(async () => {
    etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    cacheDir = await mkdtemp(join(tmpdir(), "aisix-lkg-cache-"));
    app = await spawnApp({
      etcdPrefix,
      snapshotCachePath: join(cacheDir, "config_cache.json"),
    });

    const seed = new SeedClient(etcd, etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "lkg-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    const model = await seed.createModel({
      display_name: "lkg-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    modelId = model.id;
    modelKey = `${etcdPrefix}/models/${modelId}`;
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["lkg-model"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    // The pre-restart app was stop()ped without cleanup; exit() is
    // idempotent on the dead process and reclaims its tmp dir.
    await stoppedApp?.exit();
    await upstream?.close();
    if (cacheDir) await rm(cacheDir, { recursive: true, force: true });
  });

  test("a rejected update leaves the old value serving real traffic, with staleness reported", async (ctx) => {
    if (!etcdReachable || !app || !etcd) {
      ctx.skip();
      return;
    }

    await waitConfigPropagation(async () => {
      const cfg = await getStatusConfig(app!);
      return (cfg.applied?.resource_counts.models ?? 0) >= 1;
    });

    // Baseline: the model serves.
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    const before = await proxy.chat({
      model: "lkg-model",
      messages: [{ role: "user", content: "baseline" }],
    });
    expect(before.status, JSON.stringify(before.body)).toBe(200);

    // A newer control plane (or a bug) replaces the document with bytes
    // this DP rejects: empty display_name violates the schema (RED).
    await etcd.put(
      modelKey,
      JSON.stringify({
        display_name: "",
        provider: "openai",
        model_name: "gpt-4o-mini",
      }),
    );

    let cfg: StatusConfig | undefined;
    await waitConfigPropagation(async () => {
      cfg = await getStatusConfig(app!);
      return cfg.rejected.some((r) => r.resource_id === modelId);
    });

    // The old value keeps serving REAL traffic — the user journey the
    // cliff used to break.
    const during = await proxy.chat({
      model: "lkg-model",
      messages: [{ role: "user", content: "still serving?" }],
    });
    expect(during.status, JSON.stringify(during.body)).toBe(200);

    // The rejection is reported with the staleness attached.
    expect(cfg!.state).toBe("degraded");
    const rejection = cfg!.rejected.find((r) => r.resource_id === modelId)!;
    expect(rejection.resource_kind).toBe("models");
    expect(rejection.serving_stale_since).toBeTypeOf("string");
    expect(rejection.serving_stale_age_seconds).toBeTypeOf("number");
    staleSinceBeforeRestart = rejection.serving_stale_since;
    // The served row keeps counting.
    expect(cfg!.applied?.resource_counts.models).toBe(1);

    // And the per-kind gauge on the metrics listener.
    const text = await scrape(app);
    expect(text).toMatch(/aisix_config_stale_served_resources\{kind="models"\} 1/);
  });

  test("the last known good survives a restart (cache replay + live resync against rejected bytes)", async (ctx) => {
    if (!etcdReachable || !app || !etcd) {
      ctx.skip();
      return;
    }

    // Full process restart on the same etcd prefix + snapshot cache.
    stoppedApp = app;
    await app.stop();
    app = await spawnApp({
      etcdPrefix,
      snapshotCachePath: join(cacheDir!, "config_cache.json"),
    });

    // Prove the LIVE etcd read completed (not just the cache replay):
    // a sentinel written after the restart can only appear via the new
    // process's load_all/watch. By then the boot resync has re-read the
    // rejected bytes for the model key — the exact path that used to
    // drop the row.
    await etcd.put(
      `${etcdPrefix}/api_keys/${randomUUID()}`,
      JSON.stringify({
        key_hash: createHash("sha256").update(`sentinel-${randomUUID()}`).digest("hex"),
        allowed_models: [],
      }),
    );
    await waitConfigPropagation(async () => {
      const cfg = await getStatusConfig(app!);
      return (cfg.applied?.resource_counts.api_keys ?? 0) >= 2;
    });

    // The model still serves real traffic from its pinned value.
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    const after = await proxy.chat({
      model: "lkg-model",
      messages: [{ role: "user", content: "post-restart" }],
    });
    expect(after.status, JSON.stringify(after.body)).toBe(200);

    // The rejection + staleness survive, and the stale-since instant is
    // CONTINUOUS across the restart (persisted in the snapshot cache),
    // so the age keeps growing instead of resetting.
    const cfg = await getStatusConfig(app);
    const rejection = cfg.rejected.find((r) => r.resource_id === modelId)!;
    expect(rejection).toBeDefined();
    expect(rejection.serving_stale_since).toBe(staleSinceBeforeRestart);
    expect(rejection.serving_stale_age_seconds).toBeTypeOf("number");
    expect(cfg.applied?.resource_counts.models).toBe(1);

    const text = await scrape(app);
    expect(text).toMatch(/aisix_config_stale_served_resources\{kind="models"\} 1/);
  });

  test("deleting the etcd key kills the pinned value — no zombie config", async (ctx) => {
    if (!etcdReachable || !app || !etcd) {
      ctx.skip();
      return;
    }

    await etcd.delete(modelKey);
    await waitConfigPropagation(async () => {
      const cfg = await getStatusConfig(app!);
      return (cfg.applied?.resource_counts.models ?? 0) === 0;
    });

    // The resource is gone for real traffic...
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    const gone = await proxy.chat({
      model: "lkg-model",
      messages: [{ role: "user", content: "should be gone" }],
    });
    expect(gone.status, JSON.stringify(gone.body)).toBe(404);

    // ...and every stale/rejected signal clears with it.
    const cfg = await getStatusConfig(app);
    expect(cfg.rejected).toHaveLength(0);
    const text = await scrape(app);
    expect(text).toMatch(/aisix_config_stale_served_resources\{kind="models"\} 0/);
  });
});
