import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  etcdEndpoint,
  spawnApp,
  startMockControlPlane,
  waitConfigPropagation,
  waitForLogLine,
  type MockControlPlane,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: a managed gateway whose path to the control plane's etcd endpoint
// goes bad and comes back must pick configuration up again on its own.
//
// The fault is the one a lossy path leaves behind: after a run of reset
// connections, one reconnect lands on a peer that accepts TCP and never
// answers, so its TLS handshake never finishes. Once the path is healthy
// again, a change written to etcd has to reach the gateway in bounded time,
// without a restart — with `etcd.request_timeout_ms` unset (the default) and
// with it set.

const hash = (plaintext: string) => createHash("sha256").update(plaintext).digest("hex");
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const authenticates = async (app: SpawnedApp, key: string): Promise<boolean> => {
  try {
    return (await new ProxyClient(app.proxyUrl, key).listModels()).status === 200;
  } catch (e) {
    // The proxy listener binds once the first configuration applies.
    const err = e as { code?: string; cause?: { code?: string } };
    if ((err.code ?? err.cause?.code) === "ECONNREFUSED") return false;
    throw e;
  }
};

for (const requestTimeoutMs of [undefined, 3000]) {
  const label = requestTimeoutMs === undefined ? "unset" : `${requestTimeoutMs} ms`;

  describe(`etcd reconnect into a silent peer (request_timeout_ms ${label})`, () => {
    let cp: MockControlPlane | undefined;
    let app: SpawnedApp | undefined;
    let seed: SeedClient | undefined;
    let etcdReachable = false;
    const tag = randomUUID().slice(0, 8);

    beforeAll(async () => {
      const etcd = new EtcdClient();
      etcdReachable = await etcd.ping();
      if (!etcdReachable) return;
      cp = await startMockControlPlane();
      const prefix = `/aisix-e2e-reconnect-${randomUUID()}`;
      app = await spawnApp({
        etcdPrefix: prefix,
        awaitProxyListener: false,
        logLevel: "info",
        extra: {
          managed: cp.managed,
          etcd: {
            endpoints: [etcdEndpoint()],
            prefix,
            dial_timeout_ms: 1000,
            ...(requestTimeoutMs === undefined ? {} : { request_timeout_ms: requestTimeoutMs }),
          },
        },
      });
      seed = new SeedClient(etcd, `${prefix}/${cp.envId}`);
      await seed.createApiKey({ key_hash: hash(`sk-rb-${tag}-first`), allowed_models: ["*"] });
      await waitConfigPropagation(() => authenticates(app!, `sk-rb-${tag}-first`));
    });

    afterAll(async () => {
      await app?.exit();
      await cp?.close();
    });

    test("a change written after the path recovers is applied", async (ctx) => {
      if (!etcdReachable || !app || !cp) return ctx.skip();

      // Connections reset: the watch fails and reconnect attempts start.
      cp.setEtcdPath("reset");
      await sleep(3000);
      // The next attempt lands on a peer that never answers.
      cp.setEtcdPath("blackhole");
      const deadline = Date.now() + 30_000;
      while (cp.etcdBlackholed() === 0) {
        if (Date.now() > deadline) throw new Error("no reconnect attempt reached the blackhole");
        await sleep(100);
      }
      await sleep(1000);
      cp.setEtcdPath("pass");

      await seed!.createApiKey({ key_hash: hash(`sk-rb-${tag}-second`), allowed_models: ["*"] });
      await waitConfigPropagation(() => authenticates(app!, `sk-rb-${tag}-second`), 60_000);
      expect(await authenticates(app, `sk-rb-${tag}-second`)).toBe(true);

      await waitForLogLine(
        app,
        (line) => line.includes("etcd watch re-established after a failure"),
        "the INFO line announcing the recovered watch",
      );
    }, 120_000);
  });
}
