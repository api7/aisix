import { randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startMockControlPlane,
  waitConfigPropagation,
  type MockControlPlane,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: a managed gateway that starts while the shared `<base>/global/`
// catalog is still empty watches it all the same, so a price the control
// plane projects later reaches the gateway without a restart, and the
// catalog's `applied_revisions` value moves past the write.

interface Beat {
  applied_revisions?: Record<string, number>;
}

describe("shared catalog empty at startup", () => {
  let cp: MockControlPlane | undefined;
  let app: SpawnedApp | undefined;
  let etcd: EtcdClient | undefined;
  let envPrefix = "";
  let globalPrefix = "";
  let etcdReachable = false;

  const latest = (): Beat | undefined => {
    const last = cp!.heartbeatBodies.at(-1);
    return last === undefined ? undefined : (JSON.parse(last) as Beat);
  };

  const globalPricingRows = async (): Promise<number> => {
    const res = await fetch(`${app!.metricsUrl}/status/config`);
    if (!res.ok) return -1;
    const body = (await res.json()) as { applied?: { resource_counts?: Record<string, number> } };
    return body.applied?.resource_counts?.global_pricing ?? 0;
  };

  beforeAll(async () => {
    etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    cp = await startMockControlPlane();
    const prefix = `/aisix-e2e-empty-catalog-${randomUUID()}`;
    envPrefix = `${prefix}/${cp.envId}/`;
    globalPrefix = `${prefix}/global/`;
    app = await spawnApp({
      etcdPrefix: prefix,
      awaitProxyListener: false,
      extra: { managed: { ...cp.managed, heartbeat_interval_secs: 2 } },
    });
    await waitConfigPropagation(
      async () => Object.keys(latest()?.applied_revisions ?? {}).length === 2,
      30_000,
    );
  });

  afterAll(async () => {
    await app?.exit();
    await cp?.close();
  });

  test("a price written after startup is applied through the catalog's watch", async (ctx) => {
    if (!etcdReachable || !app || !cp) return ctx.skip();
    expect(Object.keys(latest()!.applied_revisions!).sort()).toEqual([envPrefix, globalPrefix].sort());
    expect(await globalPricingRows()).toBe(0);
    const before = latest()!.applied_revisions![globalPrefix]!;

    const global = new SeedClient(etcd!, globalPrefix.slice(0, -1));
    const { id } = await global.createPricing({
      key: `vendor/${randomUUID()}`,
      input_per_1k: 1,
      output_per_1k: 1,
    });
    const revision = await etcd!.modRevision(`${globalPrefix}pricing/${id}`);
    expect(revision).toBeGreaterThan(before);

    await waitConfigPropagation(async () => (await globalPricingRows()) === 1, 30_000);
    await waitConfigPropagation(
      async () => (latest()?.applied_revisions?.[globalPrefix] ?? 0) >= revision!,
      30_000,
    );
  }, 90_000);
});
