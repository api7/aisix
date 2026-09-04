import { createHash, randomUUID } from "node:crypto";
import { connect } from "node:net";
import { afterAll, afterEach, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startEtcdRelay,
  startOpenAiUpstream,
  type AppOverrides,
  type EtcdRelay,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// The proxy listener binds only after the gateway has applied a
// configuration. Before, it bound while the first configuration read was
// still in flight, so a platform that reads "the TCP port accepts" as
// "this instance is ready" routed client traffic to a gateway that knew
// no API keys and answered every request 401 invalid_api_key.
//
// Both specs drive the gateway through a TCP relay standing in for etcd,
// which is what makes the first read's timing controllable: held (the
// read never completes), refused (the read fails and the supervisor
// retries), or forwarding.

const MODEL = "startup-gate-model";
const CALLER_PLAINTEXT = "sk-startup-gate-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

function proxyPort(app: SpawnedApp): number {
  return Number(new URL(app.proxyUrl).port);
}

/** True when a TCP connection to `port` on loopback is accepted. */
function tcpAccepts(port: number, timeoutMs = 1_000): Promise<boolean> {
  return new Promise((resolve) => {
    const sock = connect(port, "127.0.0.1");
    const settle = (accepted: boolean) => {
      sock.destroy();
      resolve(accepted);
    };
    sock.once("connect", () => settle(true));
    sock.once("error", () => settle(false));
    sock.setTimeout(timeoutMs, () => settle(false));
  });
}

/** Poll until the port accepts, or the budget runs out. */
async function waitForTcp(port: number, timeoutMs: number): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await tcpAccepts(port)) return true;
    await sleep(100);
  }
  return false;
}

/** The port must stay closed for the whole window, not merely at one instant. */
async function staysClosed(port: number, windowMs: number): Promise<void> {
  const deadline = Date.now() + windowMs;
  while (Date.now() < deadline) {
    expect(await tcpAccepts(port)).toBe(false);
    await sleep(100);
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

describe("the proxy listener waits for the first configuration", () => {
  let etcd: EtcdClient | undefined;
  let etcdReachable = false;
  let upstream: OpenAiUpstream | undefined;
  const apps: SpawnedApp[] = [];
  const relays: EtcdRelay[] = [];

  beforeAll(async () => {
    etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    upstream = await startOpenAiUpstream();
  });

  afterEach(async () => {
    await Promise.all(apps.splice(0).map((a) => a.exit()));
    await Promise.all(relays.splice(0).map((r) => r.stop()));
  });

  afterAll(async () => {
    await upstream?.close();
  });

  /**
   * Everything a caller needs to get a 200 out of the gateway, written
   * straight to the real etcd. The relay decides when the gateway is
   * allowed to read it.
   */
  async function seedFixtures(prefix: string): Promise<void> {
    const seed = new SeedClient(etcd!, prefix);
    const pk = await seed.createProviderKey({
      display_name: "startup-gate-pk",
      secret: "sk-mock",
      api_base: `${upstream!.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: [MODEL] });
  }

  async function spawnBehindRelay(
    relay: EtcdRelay,
    prefix: string,
    overrides: Partial<AppOverrides> = {},
  ): Promise<SpawnedApp> {
    const app = await spawnApp({
      // Spread first: the three keys below are what make these specs mean
      // anything, and a caller must not be able to replace the relay
      // endpoint or re-arm the readiness gate without a type error.
      ...overrides,
      etcdPrefix: prefix,
      // The subject of both specs is that this listener is NOT up yet.
      awaitProxyListener: false,
      // No dial/request timeouts. `etcd.request_timeout_ms` would abort
      // the held read, and writing one here would suggest a timeout is
      // driving the retries when the supervisor's backoff is. Unset —
      // the shipped default — leaves the read unbounded, which is what
      // makes "held" mean "still in flight" for these two specs.
      extra: { etcd: { endpoints: [relay.endpoint], prefix } },
    });
    apps.push(app);
    return app;
  }

  /** Poll the captured output until `needle` shows up. */
  async function waitForOutput(
    app: SpawnedApp,
    needle: string,
    timeoutMs: number,
  ): Promise<boolean> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      if (app.output().includes(needle)) return true;
      await sleep(100);
    }
    return false;
  }

  test("refuses TCP while the first read is in flight, then serves that snapshot", async (ctx) => {
    if (!etcdReachable) {
      ctx.skip();
      return;
    }
    const prefix = `/aisix-e2e-gate-hold-${randomUUID()}`;
    const relay = await startEtcdRelay();
    relays.push(relay);
    // Accepted but not forwarded: the range read never completes, and
    // nothing has failed — the configuration is simply slow to arrive.
    await relay.hold();
    await seedFixtures(prefix);

    const app = await spawnBehindRelay(relay, prefix);
    const port = proxyPort(app);

    await staysClosed(port, 2_000);
    // The readiness surface that IS up while the proxy listener is not.
    const ready = await fetch(`${app.metricsUrl}/status/ready`);
    expect(ready.status).toBe(503);

    await relay.release();
    expect(await waitForTcp(port, 70_000)).toBe(true);

    // Serving the snapshot the gate waited for: the caller key and the
    // model were seeded before the gateway ever read anything.
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    const res = await proxy.chat({
      model: MODEL,
      messages: [{ role: "user", content: "hello" }],
    });
    expect(res.status).toBe(200);
  });

  test("a first read that fails and recovers binds the listener automatically, exactly once", async (ctx) => {
    if (!etcdReachable) {
      ctx.skip();
      return;
    }
    const prefix = `/aisix-e2e-gate-fail-${randomUUID()}`;
    const relay = await startEtcdRelay();
    relays.push(relay);
    // Nothing is listening: every dial is refused, which is what the
    // watch supervisor retries on with its own backoff.
    await relay.refuse();
    await seedFixtures(prefix);

    const app = await spawnBehindRelay(relay, prefix, {
      // "aisix listening" is emitted once per worker on the
      // thread-per-core path, so the count below only means "bound once"
      // with the single-listener mode pinned. The gate itself sits
      // outside `serve_http` and covers both modes; the sibling spec
      // exercises whichever mode the suite is running.
      threadPerCore: false,
      // The bind line is INFO.
      logLevel: "info",
    });
    const port = proxyPort(app);

    // Several supervisor retries go by without a listener appearing.
    await staysClosed(port, 3_000);
    expect(
      await waitForOutput(
        app,
        "waiting for the first configuration before binding the proxy listener",
        5_000,
      ),
    ).toBe(true);

    await relay.release();
    expect(await waitForTcp(port, 70_000)).toBe(true);

    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    const res = await proxy.chat({
      model: MODEL,
      messages: [{ role: "user", content: "hello" }],
    });
    expect(res.status).toBe(200);

    const bindLines = app
      .output()
      .split("\n")
      .filter((line) => line.includes("aisix listening") && line.includes('label="proxy"'));
    expect(bindLines).toHaveLength(1);
  });
});
