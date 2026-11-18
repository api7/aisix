import { randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  spawnApp,
  startMockControlPlane,
  waitConfigPropagation,
  type MockControlPlane,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: the `applied_revision` a managed gateway heartbeats is exact. Every
// configuration event at or below it, in every prefix the gateway watches,
// has been processed — so the control plane can compare it against the
// revision of its own writes and tell a node that is behind from one that
// is not. The heartbeat says so with `applied_revision_complete: true`.

interface Beat {
  applied_revision: number;
  applied_revision_complete?: boolean;
}

describe("heartbeat applied_revision", () => {
  let cp: MockControlPlane | undefined;
  let app: SpawnedApp | undefined;
  let etcd: EtcdClient | undefined;
  let prefix = "";
  let envPrefix = "";
  let etcdReachable = false;
  let catalogRevision = 0;

  const latestBeat = (): Beat | undefined => {
    const last = cp!.heartbeatBodies.at(-1);
    return last === undefined ? undefined : (JSON.parse(last) as Beat);
  };
  const reportsAtLeast = (revision: number) => async () =>
    (latestBeat()?.applied_revision ?? 0) >= revision;

  beforeAll(async () => {
    etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    cp = await startMockControlPlane();
    prefix = `/aisix-e2e-applied-rev-${randomUUID()}`;
    envPrefix = `${prefix}/${cp.envId}`;
    // The second connection to etcd is the shared catalog's; hold it so a
    // catalog write lands after the environment has been read and before
    // the catalog is.
    cp.holdEtcdConnection(2);
    app = await spawnApp({
      etcdPrefix: prefix,
      awaitProxyListener: false,
      extra: { managed: { ...cp.managed, heartbeat_interval_secs: 5 } },
    });
    await waitConfigPropagation(async () => cp!.etcdConnectionHeld(), 30_000);
    const catalogKey = `${prefix}/global/pricing/${randomUUID()}`;
    await etcd.put(
      catalogKey,
      JSON.stringify({ key: "vendor/applied-rev", input_per_1k: 1, output_per_1k: 1 }),
    );
    catalogRevision = (await etcd.modRevision(catalogKey))!;
    cp.releaseEtcdConnection();
  });

  afterAll(async () => {
    await app?.exit();
    await cp?.close();
  });

  test("a catalog write landing during the initial reads is covered by the reported revision", async (ctx) => {
    if (!etcdReachable || !app || !cp) return ctx.skip();
    await waitConfigPropagation(reportsAtLeast(catalogRevision), 30_000);
    expect(latestBeat()!.applied_revision_complete).toBe(true);
  }, 60_000);

  test("a rejected write still advances the reported revision", async (ctx) => {
    if (!etcdReachable || !app || !cp) return ctx.skip();
    const key = `${envPrefix}/models/rejected-${randomUUID()}`;
    await etcd!.put(key, "{not json");
    const revision = (await etcd!.modRevision(key))!;
    await waitConfigPropagation(reportsAtLeast(revision), 30_000);
    expect(latestBeat()!.applied_revision_complete).toBe(true);
  }, 60_000);
});
