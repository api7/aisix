import { createHash, randomUUID } from "node:crypto";
import { connect, createServer, type Server, type Socket } from "node:net";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  etcdEndpoint,
  SeedClient,
  ProxyClient,
  spawnApp,
  startOpenAiUpstream,
  awaitWindowHeadroom,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";
import { metricDelta, scrapeMetrics } from "../harness/metrics.js";

// E2E: cluster-level rate limiting (api7/AISIX-Cloud#798).
//
// Two DP replicas behind one shared etcd (same config → same ApiKey
// entry id → same rate-limit bucket) and one shared Redis. With an
// ApiKey capped at RPM=1, the first request to replica A succeeds and a
// second request to replica B — a DIFFERENT process — is already
// rate-limited (429 + Retry-After). This is the exact repro from the
// issue (curl :3000 then :3001).
//
// The contrast suite below runs the same shape with the default
// `memory` backend and shows BOTH replicas serve the request: per-
// process counters multiply the limit by the replica count, which is
// the bug #798 fixes.

const CALLER_PLAINTEXT = "sk-rl-cluster-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const ETCD_ENDPOINT = etcdEndpoint();
const REDIS_URL = process.env.AISIX_E2E_REDIS ?? "redis://127.0.0.1:6379";

/** RESP-level PING so the suite skips honestly when no redis is reachable
 *  (CI provisions redis:7-alpine on :6379). */
async function redisPing(url: string): Promise<boolean> {
  const m = /^redis:\/\/(?:[^@/]*@)?([^:/]+)(?::(\d+))?/.exec(url);
  if (!m) return false;
  const host = m[1];
  const port = m[2] ? Number(m[2]) : 6379;
  return new Promise((resolve) => {
    const sock = connect({ host, port }, () => sock.write("PING\r\n"));
    const done = (ok: boolean) => {
      sock.destroy();
      resolve(ok);
    };
    sock.once("data", (buf) => done(buf.toString().startsWith("+PONG")));
    sock.once("error", () => done(false));
    sock.setTimeout(1000, () => done(false));
  });
}

/** A shared etcd block so two replicas read ONE config namespace — the
 *  ApiKey then has a single entry id across both, which is the rate-limit
 *  bucket key. (`spawnApp` otherwise gives each app a unique prefix.) */
function sharedEtcd(prefix: string) {
  return {
    endpoints: [ETCD_ENDPOINT],
    prefix,
  };
}

function chatRequest(proxyUrl: string, model: string): Promise<Response> {
  return fetch(`${proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model,
      messages: [{ role: "user", content: "hello" }],
    }),
  });
}

/** Seed one model + an RPM=1 ApiKey into the SHARED config namespace —
 *  both replicas pick it up over the same etcd watch. */
async function seed(etcdRoot: string, upstreamBase: string, model: string) {
  const seed = new SeedClient(new EtcdClient(), etcdRoot);
  const pk = await seed.createProviderKey({
    display_name: `${model}-pk`,
    secret: "sk-mock",
    api_base: `${upstreamBase}/v1`,
  });
  await seed.createModel({
    display_name: model,
    provider: "openai",
    model_name: "gpt-4o-mini",
    provider_key_id: pk.id,
  });
  await seed.createApiKey({
    key_hash: CALLER_KEY_HASH,
    allowed_models: [model],
    rate_limit: { rpm: 1 },
  });
}

/** Wait until `model` is visible on `proxyUrl` without spending the RPM=1
 *  budget (listModels does not consume a request slot). */
async function waitModelLive(proxyUrl: string, model: string) {
  const probe = new ProxyClient(proxyUrl, CALLER_PLAINTEXT);
  await waitConfigPropagation(async () => {
    const res = await probe.listModels();
    if (res.status !== 200) return false;
    const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
    return data.some((m) => m.id === model);
  });
}

describe("rate limit is shared across replicas with backend=redis (#798)", () => {
  let appA: SpawnedApp | undefined;
  let appB: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let infraReady = false;
  const prefix = `/aisix-e2e-rl-${randomUUID()}`;
  const model = "rl-cluster";

  beforeAll(async () => {
    infraReady = (await new EtcdClient().ping()) && (await redisPing(REDIS_URL));
    if (!infraReady) return;

    upstream = await startOpenAiUpstream();
    const extra = {
      etcd: sharedEtcd(prefix),
      ratelimit: { backend: "redis", redis: { url: REDIS_URL } },
    };
    appA = await spawnApp({ extra });
    appB = await spawnApp({ extra });
    await seed(prefix, upstream.baseUrl, model);
    await waitModelLive(appA.proxyUrl, model);
    await waitModelLive(appB.proxyUrl, model);
  });

  afterAll(async () => {
    await appA?.exit();
    await appB?.exit();
    await upstream?.close();
    // The harness cleans the unique prefixes it generated, not our shared
    // override — drop it ourselves. Skip when infra was unavailable (the
    // suite skipped) so teardown doesn't fail on an unreachable etcd.
    if (infraReady) await new EtcdClient().deletePrefix(prefix);
  });

  test("first call on A succeeds, second call on B is 429", async (ctx) => {
    if (!infraReady || !appA || !appB) {
      ctx.skip();
      return;
    }

    // The limiter buckets on fixed wall-clock minutes, so a burst that
    // straddles a boundary gets a fresh allowance and the 429 assertion
    // below flaps. Keep the whole burst inside one window.
    await awaitWindowHeadroom();
    const first = await chatRequest(appA.proxyUrl, model);
    expect(first.status).toBe(200);
    await first.body?.cancel();

    // Different process, shared Redis counter → already over the cap.
    const second = await chatRequest(appB.proxyUrl, model);
    expect(second.status).toBe(429);
    // Retry-After is the load-bearing SDK back-off contract.
    expect(second.headers.get("retry-after")).toBeTruthy();
    await second.body?.cancel();
  });
});

describe("rate limit is NOT shared with backend=memory (per-replica, the #798 bug)", () => {
  let appA: SpawnedApp | undefined;
  let appB: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReady = false;
  const prefix = `/aisix-e2e-rl-mem-${randomUUID()}`;
  const model = "rl-cluster-mem";

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReady = await etcd.ping();
    if (!etcdReady) return;

    upstream = await startOpenAiUpstream();
    // Shared etcd (same ApiKey entry id) but default memory backend — the
    // counters live per-process, so the cap does NOT span replicas.
    const extra = { etcd: sharedEtcd(prefix) };
    appA = await spawnApp({ extra });
    appB = await spawnApp({ extra });
    await seed(prefix, upstream.baseUrl, model);
    await waitModelLive(appA.proxyUrl, model);
    await waitModelLive(appB.proxyUrl, model);
  });

  afterAll(async () => {
    await appA?.exit();
    await appB?.exit();
    await upstream?.close();
    if (etcdReady) await new EtcdClient().deletePrefix(prefix);
  });

  test("first call on A and first call on B both succeed", async (ctx) => {
    if (!etcdReady || !appA || !appB) {
      ctx.skip();
      return;
    }

    const first = await chatRequest(appA.proxyUrl, model);
    expect(first.status).toBe(200);
    await first.body?.cancel();

    // Default memory backend: B has its own counter → still allowed. With
    // N replicas the effective limit is N×, which is what #798 reports.
    const second = await chatRequest(appB.proxyUrl, model);
    expect(second.status).toBe(200);
    await second.body?.cancel();
  });
});


/**
 * A TCP relay in front of Redis that the test can break, in either of the
 * two shapes a real outage takes.
 *
 * `cut()` answers each client request with a Redis error instead of
 * forwarding it: Redis is reachable and refusing, so the gateway learns
 * of the failure immediately.
 *
 * `blackhole()` keeps the socket open and never forwards or answers
 * anything, which is what a stopped container, a downed host or a
 * partitioned network looks like from the client end. Nothing arrives and
 * nothing is refused, so without a command budget the gateway waits on TCP
 * retransmission — for minutes.
 */
async function startRedisCutoff(upstreamUrl: string): Promise<{
  url: string;
  cut(): void;
  blackhole(): void;
  close(): Promise<void>;
}> {
  const m = /^redis:\/\/(?:[^@/]*@)?([^:/]+)(?::(\d+))?/.exec(upstreamUrl);
  if (!m) throw new Error(`unparseable redis url: ${upstreamUrl}`);
  const host = m[1];
  const port = m[2] ? Number(m[2]) : 6379;
  let cut = false;
  let hole = false;
  const live = new Set<Socket>();
  const server: Server = createServer((client) => {
    live.add(client);
    const server = connect({ host, port });
    live.add(server);
    client.on("data", (buf) => {
      // Swallowed under blackhole: no forward, no reply, no close.
      if (hole) return;
      if (cut) client.write("-ERR simulated redis outage\r\n");
      else server.write(buf);
    });
    server.on("data", (buf) => {
      if (hole) return;
      client.write(buf);
    });
    const bin = () => {
      client.destroy();
      server.destroy();
      live.delete(client);
      live.delete(server);
    };
    client.on("error", bin);
    server.on("error", bin);
    client.on("close", bin);
    server.on("close", bin);
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  const addr = server.address();
  if (typeof addr === "string" || addr === null) throw new Error("no relay port");
  return {
    url: `redis://127.0.0.1:${addr.port}`,
    cut: () => {
      cut = true;
    },
    blackhole: () => {
      hole = true;
    },
    close: () =>
      new Promise<void>((r) => {
        for (const s of live) s.destroy();
        server.close(() => r());
      }),
  };
}

// #1060: `aisix_redis_failures_total` had no caller at all, so an operator
// querying it got an empty result whether the shared backend was healthy or
// failing constantly. The limiter fails OPEN — it degrades to per-replica
// in-memory counting and keeps answering 200 — so nothing else about the
// request changes and this counter is the only signal the degradation ever
// produces. Asserted through a real gateway's `GET /metrics`, because an
// emit that is only exercised by a unit test is exactly the shape that
// shipped uncalled three times before.
describe("a Redis outage on the shared rate-limit backend is scrapeable (#1060)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let relay: Awaited<ReturnType<typeof startRedisCutoff>> | undefined;
  let infraReady = false;
  const prefix = `/aisix-e2e-rl-redisfail-${randomUUID()}`;
  const model = "rl-redis-fail";

  beforeAll(async () => {
    infraReady = (await new EtcdClient().ping()) && (await redisPing(REDIS_URL));
    if (!infraReady) return;

    upstream = await startOpenAiUpstream();
    relay = await startRedisCutoff(REDIS_URL);
    app = await spawnApp({
      extra: {
        etcd: sharedEtcd(prefix),
        ratelimit: { backend: "redis", redis: { url: relay.url } },
      },
    });
    await seed(prefix, upstream.baseUrl, model);
    await waitModelLive(app.proxyUrl, model);
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await relay?.close();
    if (infraReady) await new EtcdClient().deletePrefix(prefix);
  });

  test("the failure counter rises while the gateway keeps serving", async (ctx) => {
    if (!infraReady || !app || !relay) {
      ctx.skip();
      return;
    }

    // Healthy: the request is served and nothing is counted.
    await awaitWindowHeadroom();
    const healthy = await chatRequest(app.proxyUrl, model);
    expect(healthy.status).toBe(200);
    await healthy.body?.cancel();
    const before = await scrapeMetrics(app.metricsUrl);

    relay.cut();

    // Still served — that is the fail-open contract, and exactly why the
    // outage is invisible without the counter. (The seeded key is RPM=1, so
    // this second call would have been a 429 had the shared counter still
    // been readable; per-replica fallback starts from an empty window.)
    const degraded = await chatRequest(app.proxyUrl, model);
    expect(degraded.status).toBe(200);
    await degraded.body?.cancel();

    const after = await scrapeMetrics(app.metricsUrl);
    expect(
      metricDelta(before, after, "aisix_redis_failures_total", (labels) =>
        labels.operation.startsWith("ratelimit_"),
      ),
    ).toBeGreaterThan(0);
  });
});


/** Seed one model + an ApiKey whose limit is high enough that every
 *  request in the outage suite is admitted — what is under test is how
 *  long the limiter takes to answer, not whether it refuses. */
async function seedGenerousLimit(
  etcdRoot: string,
  upstreamBase: string,
  model: string,
) {
  const seed = new SeedClient(new EtcdClient(), etcdRoot);
  const pk = await seed.createProviderKey({
    display_name: `${model}-pk`,
    secret: "sk-mock",
    api_base: `${upstreamBase}/v1`,
  });
  await seed.createModel({
    display_name: model,
    provider: "openai",
    model_name: "gpt-4o-mini",
    provider_key_id: pk.id,
  });
  await seed.createApiKey({
    key_hash: CALLER_KEY_HASH,
    allowed_models: [model],
    rate_limit: { rpm: 1000 },
  });
}

// A Redis that stops answering without closing the socket must degrade the
// request, not hold it.
//
// The limiter has always failed open on a Redis *error* — the describe
// above pins that — but with no bound on a command the error never
// arrived: `docker stop` of the Redis container left every rate-limited
// request unanswered past three minutes (curl exit 28), while requests
// matching no policy answered in milliseconds. Redis unreachable is the
// case the fail-open path exists for, and it was the one case it did not
// cover.
//
// Two properties, because each is worthless without the other: the request
// that meets the silence must come back inside the command budget, and the
// requests behind it must not each pay that budget again for as long as
// the outage lasts.
describe("an unreachable Redis degrades the limiter instead of hanging the request", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let relay: Awaited<ReturnType<typeof startRedisCutoff>> | undefined;
  let infraReady = false;
  const prefix = `/aisix-e2e-rl-redisblackhole-${randomUUID()}`;
  const model = "rl-redis-blackhole";
  // Short enough that the assertions below fit a normal test timeout, and
  // still long enough that no healthy round trip on this host trips it.
  // It is also deliberately below the 5s default: the first-request bound
  // is what shows the per-block field reached the connection, and a bound
  // above the default would pass either way.
  const TIMEOUT_SECS = 2;

  beforeAll(async () => {
    infraReady = (await new EtcdClient().ping()) && (await redisPing(REDIS_URL));
    if (!infraReady) return;

    upstream = await startOpenAiUpstream();
    relay = await startRedisCutoff(REDIS_URL);
    app = await spawnApp({
      extra: {
        etcd: sharedEtcd(prefix),
        ratelimit: {
          backend: "redis",
          redis: { url: relay.url, timeout_secs: TIMEOUT_SECS },
        },
      },
    });
    await seedGenerousLimit(prefix, upstream.baseUrl, model);
    await waitModelLive(app.proxyUrl, model);
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await relay?.close();
    if (infraReady) await new EtcdClient().deletePrefix(prefix);
  });

  test("the request completes on the local fallback, and the next one costs nothing", async (ctx) => {
    if (!infraReady || !app || !relay) {
      ctx.skip();
      return;
    }

    const healthy = await chatRequest(app.proxyUrl, model);
    expect(healthy.status).toBe(200);
    await healthy.body?.cancel();

    relay.blackhole();

    const firstStarted = Date.now();
    const first = await chatRequest(app.proxyUrl, model);
    const firstMs = Date.now() - firstStarted;
    expect(first.status).toBe(200);
    await first.body?.cancel();
    // Before the budget existed this never returned at all. The bound is
    // the configured budget plus room for the mock upstream and process
    // scheduling, and stays under the 5s default so a gateway that ignored
    // `timeout_secs` fails here.
    expect(firstMs).toBeGreaterThanOrEqual(TIMEOUT_SECS * 1000);
    expect(firstMs).toBeLessThan(TIMEOUT_SECS * 1000 + 2000);

    // And the one behind it short-circuits: without the cool-off every
    // request for the length of the outage carries the budget as added
    // latency.
    const secondStarted = Date.now();
    const second = await chatRequest(app.proxyUrl, model);
    const secondMs = Date.now() - secondStarted;
    expect(second.status).toBe(200);
    await second.body?.cancel();
    expect(secondMs).toBeLessThan(1000);
  });
});
