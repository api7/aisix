import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startMockControlPlane,
  waitConfigPropagation,
  type MockControlPlane,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: a managed gateway heartbeats `applied_revisions`, one value per
// prefix it watches. Every configuration event under a prefix at or below
// its value has been processed (applied or rejected), so the control plane
// can compare a write's revision with the value of the prefix it wrote to
// and tell a node that is behind from one that is not.

interface Beat {
  applied_revision: number;
  applied_revisions?: Record<string, number>;
}

const hash = (plaintext: string) => createHash("sha256").update(plaintext).digest("hex");
// Connections to etcd in the order the gateway opens them at boot: the
// environment prefix's, then the shared catalog's.
const CATALOG_CONNECTION = 2;

describe("heartbeat applied_revisions", () => {
  let cp: MockControlPlane | undefined;
  let app: SpawnedApp | undefined;
  let etcd: EtcdClient | undefined;
  let envPrefix = "";
  let globalPrefix = "";
  let etcdReachable = false;
  let catalogRevision = 0;

  const latest = (): Beat | undefined => {
    const last = cp!.heartbeatBodies.at(-1);
    return last === undefined ? undefined : (JSON.parse(last) as Beat);
  };
  // A heartbeat sent after `since` heartbeats had arrived.
  const nextBeat = async (since: number): Promise<Beat> => {
    await waitConfigPropagation(async () => cp!.heartbeatBodies.length > since, 15_000);
    return latest()!;
  };
  const reaches = (prefix: string, revision: number) => async () =>
    (latest()?.applied_revisions?.[prefix] ?? 0) >= revision;
  const putCatalog = async (): Promise<number> => {
    const key = `${globalPrefix}pricing/${randomUUID()}`;
    await etcd!.put(key, JSON.stringify({ key: `vendor/${randomUUID()}`, input_per_1k: 1, output_per_1k: 1 }));
    return revisionOf(key);
  };
  const revisionOf = async (key: string): Promise<number> => {
    const revision = await etcd!.modRevision(key);
    if (!Number.isSafeInteger(revision) || revision! <= 0) throw new Error(`no revision for ${key}: ${revision}`);
    return revision!;
  };

  beforeAll(async () => {
    etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    cp = await startMockControlPlane();
    const prefix = `/aisix-e2e-applied-rev-${randomUUID()}`;
    envPrefix = `${prefix}/${cp.envId}/`;
    globalPrefix = `${prefix}/global/`;
    // Hold the catalog's connection so a catalog write lands after the
    // environment has been read and before the catalog is.
    cp.holdEtcdConnection(CATALOG_CONNECTION);
    app = await spawnApp({
      etcdPrefix: prefix,
      awaitProxyListener: false,
      extra: { managed: { ...cp.managed, heartbeat_interval_secs: 5 } },
    });
    await waitConfigPropagation(async () => cp!.etcdConnectionHeld(), 30_000);
    catalogRevision = await putCatalog();
    cp.releaseEtcdConnection();
    // The first live read has completed once both prefixes are reported.
    await waitConfigPropagation(async () => Object.keys(latest()?.applied_revisions ?? {}).length === 2, 30_000);
  });

  afterAll(async () => {
    await app?.exit();
    await cp?.close();
  });

  test("a catalog write landing during the initial reads is covered by the catalog's value", async (ctx) => {
    if (!etcdReachable || !app || !cp) return ctx.skip();
    await waitConfigPropagation(reaches(globalPrefix, catalogRevision), 30_000);
    expect(Object.keys(latest()!.applied_revisions!).sort()).toEqual([envPrefix, globalPrefix].sort());
  }, 60_000);

  test("a rejected write still advances its prefix's value", async (ctx) => {
    if (!etcdReachable || !app || !cp) return ctx.skip();
    const key = `${envPrefix}models/rejected-${randomUUID()}`;
    await etcd!.put(key, "{not json");
    const revision = await revisionOf(key);
    await waitConfigPropagation(reaches(envPrefix, revision), 30_000);
  }, 60_000);

  test("one prefix's events do not move a prefix whose stream is behind", async (ctx) => {
    if (!etcdReachable || !app || !cp) return ctx.skip();
    // The catalog's stream stops delivering; its write stays undelivered
    // while the environment's later write arrives on the other stream.
    cp.stallEtcdConnection(CATALOG_CONNECTION);
    const undelivered = await putCatalog();
    const seed = new SeedClient(etcd!, envPrefix.slice(0, -1));
    const caller = `sk-ar-${randomUUID()}`;
    await seed.createApiKey({ key_hash: hash(caller), allowed_models: ["*"] });
    await waitConfigPropagation(
      async () => (await new ProxyClient(app!.proxyUrl, caller).listModels()).status === 200,
      15_000,
    );
    const beat = await nextBeat(cp.heartbeatBodies.length);
    expect(beat.applied_revisions![envPrefix]).toBeGreaterThan(undelivered);
    expect(beat.applied_revisions![globalPrefix]).toBeLessThan(undelivered);

    // Delivered, it is covered.
    cp.resumeEtcdConnection(CATALOG_CONNECTION);
    await waitConfigPropagation(reaches(globalPrefix, undelivered), 30_000);
  }, 90_000);
});
