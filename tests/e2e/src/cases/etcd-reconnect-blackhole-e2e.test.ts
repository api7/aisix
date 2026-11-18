import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  completeLines,
  etcdEndpoint,
  scrapeMetrics,
  sumMetric,
  waitForLogLine,
  spawnApp,
  startMockControlPlane,
  waitConfigPropagation,
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
const RECOVERED = "etcd watch established after a failed cycle";

/**
 * The number of complete log lines once everything the gateway wrote
 * before this call is in: a request's access-log line is behind all of it
 * in the gateway's FIFO log queue.
 */
const logBarrier = async (app: SpawnedApp, key: string): Promise<number> => {
  const { requestId } = await new ProxyClient(app.proxyUrl, key).listModels();
  await waitForLogLine(app, (line) => line.includes(requestId), `the access-log line of ${requestId}`);
  return completeLines(app).length;
};

const authenticates = async (app: SpawnedApp, key: string): Promise<boolean> => {
  let status: number;
  try {
    status = (await new ProxyClient(app.proxyUrl, key).listModels()).status;
  } catch (e) {
    // The proxy listener binds once the first configuration applies.
    const err = e as { code?: string; cause?: { code?: string } };
    if ((err.code ?? err.cause?.code) === "ECONNREFUSED") return false;
    throw e;
  }
  // 401 is a key that has not propagated yet; anything else is a failure.
  if (status === 401) return false;
  if (status !== 200) throw new Error(`GET /v1/models answered ${status}`);
  return true;
};

for (const requestTimeoutMs of [undefined, 3000]) {
  const label = requestTimeoutMs === undefined ? "unset" : `${requestTimeoutMs} ms`;

  describe(`etcd reconnect into a silent peer (request_timeout_ms ${label})`, () => {
    let cp: MockControlPlane | undefined;
    let app: SpawnedApp | undefined;
    let seed: SeedClient | undefined;
    let envPrefix = "";
    let etcdReachable = false;
    const tag = randomUUID().slice(0, 8);
    const key = (n: string) => `sk-rb-${tag}-${n}`;
    const addKey = (n: string) => seed!.createApiKey({ key_hash: hash(key(n)), allowed_models: ["*"] });

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
      envPrefix = `${prefix}/${cp.envId}`;
      seed = new SeedClient(etcd, envPrefix);
      await addKey("first");
      await waitConfigPropagation(() => authenticates(app!, key("first")));
      // Written after the first configuration is serving, so it can only
      // arrive over the watch: the watch is established before the fault.
      await addKey("warm");
      await waitConfigPropagation(() => authenticates(app!, key("warm")));
    });

    afterAll(async () => {
      await app?.exit();
      await cp?.close();
    });

    test("a change written after the path recovers is applied", async (ctx) => {
      if (!etcdReachable || !app || !cp) return ctx.skip();
      const checkpoint = await logBarrier(app, key("warm"));

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

      await addKey("second");
      await waitConfigPropagation(() => authenticates(app!, key("second")), 90_000);

      // The recovered environment watch, from this outage.
      await waitConfigPropagation(
        async () =>
          completeLines(app!)
            .slice(checkpoint)
            .some((line) => line.includes(RECOVERED) && line.includes(envPrefix)),
        10_000,
      );

      // And it delivers: a change written now arrives as a watch apply, with
      // no full (re)load in between.
      const applies = async (trigger: string) =>
        sumMetric(await scrapeMetrics(app!.metricsUrl), "aisix_config_apply_duration_seconds_count", { trigger });
      const [fullBefore, watchBefore] = [await applies("full"), await applies("watch")];
      await addKey("third");
      await waitConfigPropagation(() => authenticates(app!, key("third")));
      await waitConfigPropagation(async () => (await applies("watch")) > watchBefore, 10_000);
      expect(await applies("full")).toBe(fullBefore);
    }, 180_000);
  });
}
