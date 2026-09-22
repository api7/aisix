import { createHash, randomUUID } from "node:crypto";
import { createServer as createHttpServer, type Server as HttpServer } from "node:http";
import { connect, createServer, type Server, type Socket } from "node:net";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  etcdEndpoint,
  ProxyClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";
import { pickFreePort } from "../harness/ports.js";

// E2E: what a Redis outage costs ONE request when the cache is on the
// shared backend.
//
// The cache subsystem holds two connections to the same `cache.redis` —
// exact-KV and vector search — and a single chat request against a
// policy with a `semantic` block touches both twice: exact lookup,
// semantic lookup, exact write, semantic write. Each failure was bounded
// (that is #1147) but each was bounded SEPARATELY, because the cool-off
// belonged to the connection rather than to the subsystem, so the request
// paid the command budget once per operation. Release QA measured 20.0s
// at the default 5s budget against 5.0s for an exact-only policy.
//
// The outage shape is a black hole — the socket stays open and nothing is
// ever answered, which is what a paused container or a dropped-packet
// policy looks like from the client end. A stopped container can answer
// with a refusal instead, and a refusal is the cheap failure: the client
// learns immediately and no budget is spent, so it would not exercise
// this at all.

const ETCD_ENDPOINT = etcdEndpoint();
const REDIS_URL = process.env.AISIX_E2E_REDIS ?? "redis://127.0.0.1:6379";

const CALLER_PLAINTEXT = "sk-cache-outage-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

// Below the 5s default on purpose: a bound derived from it cannot pass on
// a gateway that ignored the configured budget.
const TIMEOUT_SECS = 3;
// One budget plus room for the mock upstream, the embedding call and
// process scheduling. It has to stay under TWO budgets, which is what the
// defect costs at this setting (measured 4.0s).
const ONE_BUDGET_MS = TIMEOUT_SECS * 1000 + 1_500;

/** Does the server speak FT.* (vector search)? `null` = unreachable.
 *  Without it the semantic half of a policy never wires up and the case
 *  under test cannot occur, so the suite skips honestly. */
async function redisVectorSupport(url: string): Promise<boolean | null> {
  const m = /^redis:\/\/(?:[^@/]*@)?([^:/]+)(?::(\d+))?/.exec(url);
  if (!m) return null;
  const host = m[1];
  const port = m[2] ? Number(m[2]) : 6379;
  return new Promise((resolve) => {
    const sock = connect({ host, port }, () => sock.write("FT._LIST\r\n"));
    const done = (v: boolean | null) => {
      sock.destroy();
      resolve(v);
    };
    sock.once("data", (buf) => {
      const head = buf.toString();
      if (head.startsWith("*")) return done(true);
      if (/^-ERR unknown command/i.test(head)) return done(false);
      done(null);
    });
    sock.once("error", () => done(null));
    sock.setTimeout(1000, () => done(null));
  });
}

/** A TCP relay in front of Redis that the test can black-hole: after
 *  `blackhole()` the sockets stay open and nothing is forwarded or
 *  answered, ever. */
async function startRedisBlackhole(
  upstreamUrl: string,
  opts: { blackholed?: boolean } = {},
): Promise<{
  url: string;
  blackhole(): void;
  heal(): void;
  close(): Promise<void>;
}> {
  const m = /^redis:\/\/(?:[^@/]*@)?([^:/]+)(?::(\d+))?/.exec(upstreamUrl);
  if (!m) throw new Error(`unparseable redis url: ${upstreamUrl}`);
  const host = m[1];
  const port = m[2] ? Number(m[2]) : 6379;
  let hole = opts.blackholed ?? false;
  const live = new Set<Socket>();
  const server: Server = createServer((client) => {
    live.add(client);
    const upstream = connect({ host, port });
    live.add(upstream);
    client.on("data", (buf) => {
      if (hole) return;
      upstream.write(buf);
    });
    upstream.on("data", (buf) => {
      if (hole) return;
      client.write(buf);
    });
    const bin = () => {
      client.destroy();
      upstream.destroy();
      live.delete(client);
      live.delete(upstream);
    };
    for (const s of [client, upstream]) {
      s.on("error", bin);
      s.on("close", bin);
    }
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  const addr = server.address();
  if (typeof addr === "string" || addr === null) throw new Error("no relay port");
  return {
    url: `redis://127.0.0.1:${addr.port}`,
    blackhole: () => {
      hole = true;
    },
    // Forwarding resumes for connections opened from now on; the sockets
    // swallowed mid-handshake are dropped, which is what a Redis coming
    // back up looks like from the client end.
    heal: () => {
      hole = false;
      for (const s of live) s.destroy();
      live.clear();
    },
    close: () =>
      new Promise<void>((resolve, reject) => {
        for (const s of live) s.destroy();
        server.close((err) => (err ? reject(err) : resolve()));
      }),
  };
}

/** Deterministic 4-d embedding so the semantic layer has a real vector
 *  to store and search on. */
function keywordVector(text: string): number[] {
  return text.toLowerCase().includes("alpha") ? [1, 0, 0, 0] : [0, 0, 0, 1];
}

async function startEmbeddingMock(): Promise<{
  baseUrl: string;
  close(): Promise<void>;
}> {
  const server: HttpServer = createHttpServer((req, res) => {
    res.on("error", () => {});
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      const body = JSON.parse(raw || "{}") as { input?: string | string[] };
      const inputs = Array.isArray(body.input) ? body.input : [body.input ?? ""];
      res.statusCode = 200;
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          object: "list",
          model: "embed-mock",
          data: inputs.map((text, index) => ({
            object: "embedding",
            index,
            embedding: keywordVector(text),
          })),
          usage: { prompt_tokens: inputs.length, total_tokens: inputs.length },
        }),
      );
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((r) => server.listen(port, "127.0.0.1", r));
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    close: () =>
      new Promise<void>((resolve, reject) =>
        server.close((err) => (err ? reject(err) : resolve())),
      ),
  };
}

const SEMANTIC_MODEL = "cache-outage-semantic";
const EXACT_MODEL = "cache-outage-exact";
/// Served by a `backend: memory` policy — the other half of a mixed
/// deployment, which is the shape that can re-arm the degradation latch
/// on a backend that never fails.
const MEMORY_MODEL = "cache-outage-memory";
// Longer than the OLD 5s cool-off and shorter than the 30s one, so the
// window length alone decides whether the cache write pays a second
// budget. Non-streaming completions — the only responses this cache
// stores — routinely run this long.
const UPSTREAM_DELAY_MS = 7_000;

/** Two chat models, each with its own redis cache policy — one carrying a
 *  `semantic` block, one exact-only — plus the embedding model the
 *  semantic layer needs. The caller key is seeded LAST, per
 *  tests/e2e/AGENTS.md, so gating on it implies the whole set. */
async function seed(etcdRoot: string, embedBase: string, upstreamBase: string) {
  const seed = new SeedClient(new EtcdClient(), etcdRoot);
  const embedPk = await seed.createProviderKey({
    display_name: "cache-outage-embed-pk",
    secret: "sk-mock",
    api_base: `${embedBase}/v1`,
  });
  await seed.createModel({
    display_name: "cache-outage-embed",
    provider: "openai",
    model_name: "embed-mock",
    provider_key_id: embedPk.id,
    embedding: { dimensions: 4, normalize: true },
  });
  const chatPk = await seed.createProviderKey({
    display_name: "cache-outage-chat-pk",
    secret: "sk-mock",
    api_base: `${upstreamBase}/v1`,
  });
  for (const model of [SEMANTIC_MODEL, EXACT_MODEL, MEMORY_MODEL]) {
    await seed.createModel({
      display_name: model,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: chatPk.id,
    });
  }
  await seed.createCachePolicy({
    name: "cache-outage-semantic-policy",
    backend: "redis",
    applies_to: `model:${SEMANTIC_MODEL}`,
    ttl_seconds: 600,
    semantic: { embedding_model: "cache-outage-embed", threshold: 0.85 },
  });
  await seed.createCachePolicy({
    name: "cache-outage-exact-policy",
    backend: "redis",
    applies_to: `model:${EXACT_MODEL}`,
    ttl_seconds: 600,
  });
  // A memory-backed policy on its own model, so a spec can drive traffic
  // that does NOT touch redis while redis is down. Its cache essentially
  // cannot fail, so if its successes counted as "redis recovered" the
  // outage would be re-reported on every alternation.
  await seed.createCachePolicy({
    name: "cache-outage-memory-policy",
    backend: "memory",
    applies_to: `model:${MEMORY_MODEL}`,
    ttl_seconds: 600,
  });
  await seed.createApiKey({
    key_hash: CALLER_KEY_HASH,
    allowed_models: ["*"],
  });
}

/** Returns the wall-clock cost of one completed chat request. */
async function timeChat(
  proxyUrl: string,
  model: string,
  prompt: string,
): Promise<{ status: number; ms: number }> {
  const started = Date.now();
  const res = await fetch(`${proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
    },
    body: JSON.stringify({ model, messages: [{ role: "user", content: prompt }] }),
  });
  await res.text();
  return { status: res.status, ms: Date.now() - started };
}

interface Fixture {
  app: SpawnedApp;
  upstream: OpenAiUpstream;
  embed: Awaited<ReturnType<typeof startEmbeddingMock>>;
  relay: Awaited<ReturnType<typeof startRedisBlackhole>>;
  prefix: string;
}

/** One gateway with its own relay, so each case starts on a cool-off that
 *  nothing has opened. Waiting one out instead would mean sleeping 30s. */
async function bringUp(tag: string, responseDelayMs?: number): Promise<Fixture> {
  const prefix = `/aisix-e2e-cache-outage-${tag}-${randomUUID()}`;
  const upstream = await startOpenAiUpstream(
    responseDelayMs ? { responseDelayMs } : {},
  );
  const embed = await startEmbeddingMock();
  const relay = await startRedisBlackhole(REDIS_URL);
  const app = await spawnApp({
    extra: {
      etcd: { endpoints: [ETCD_ENDPOINT], prefix },
      cache: {
        backend: "redis",
        redis: { url: relay.url, timeout_secs: TIMEOUT_SECS },
      },
    },
  });
  await seed(prefix, embed.baseUrl, upstream.baseUrl);
  const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
  await waitConfigPropagation(
    async () => (await probe.listModels()).status === 200,
  );
  return { app, upstream, embed, relay, prefix };
}

async function tearDown(f: Fixture | undefined) {
  if (!f) return;
  await f.app.exit();
  await f.upstream.close();
  await f.embed.close();
  await f.relay.close();
  await new EtcdClient().deletePrefix(f.prefix);
}

async function vectorRedisReady(): Promise<boolean> {
  return (await new EtcdClient().ping()) && (await redisVectorSupport(REDIS_URL)) === true;
}

describe("a Redis outage costs one request one budget for the whole cache", () => {
  let f: Fixture | undefined;
  let ready = false;

  beforeAll(async () => {
    ready = await vectorRedisReady();
    if (ready) f = await bringUp("fast");
  });
  afterAll(async () => {
    await tearDown(f);
  });

  test("a semantic policy pays one budget, not one per cache connection", async (ctx) => {
    if (!ready || !f) {
      ctx.skip();
      return;
    }

    // Healthy first, so both cache connections are established and the
    // semantic index exists before anything is broken.
    const warm = await timeChat(f.app.proxyUrl, SEMANTIC_MODEL, "alpha warm");
    expect(warm.status).toBe(200);

    f.relay.blackhole();

    const degraded = await timeChat(f.app.proxyUrl, SEMANTIC_MODEL, "alpha cold");
    expect(degraded.status).toBe(200);
    expect(degraded.ms).toBeGreaterThanOrEqual(TIMEOUT_SECS * 1000);
    expect(degraded.ms).toBeLessThan(ONE_BUDGET_MS);

    // Behind it the cool-off is open, so the next request costs nothing.
    const behind = await timeChat(f.app.proxyUrl, SEMANTIC_MODEL, "alpha behind");
    expect(behind.status).toBe(200);
    expect(behind.ms).toBeLessThan(1_000);
  }, 60_000);
});

// The cache read and the cache write of one request straddle the upstream
// call, so "one budget per request" holds only while the cool-off outlasts
// that call. At the 5s window this case paid a second budget on the write
// — the reason the window is 30s. The case above cannot see it: its
// upstream answers instantly, so its write lands inside any window.
describe("an upstream slower than the old cool-off still costs one budget", () => {
  let f: Fixture | undefined;
  let ready = false;

  beforeAll(async () => {
    ready = await vectorRedisReady();
    if (ready) f = await bringUp("slow", UPSTREAM_DELAY_MS);
  }, 60_000);
  afterAll(async () => {
    await tearDown(f);
  });

  test("the cache write is still covered by the cool-off its read opened", async (ctx) => {
    if (!ready || !f) {
      ctx.skip();
      return;
    }

    const warm = await timeChat(f.app.proxyUrl, SEMANTIC_MODEL, "alpha warm");
    expect(warm.status).toBe(200);

    f.relay.blackhole();

    const degraded = await timeChat(f.app.proxyUrl, SEMANTIC_MODEL, "alpha cold");
    expect(degraded.status).toBe(200);
    // Both cache layers are unreachable, so neither can serve this — the
    // request must have gone upstream, and the timing below is only
    // meaningful if it did. Asserted on the upstream's own record rather
    // than inferred from the clock.
    expect(f.upstream.receivedRequests.length).toBe(2);
    // The exact lookup spends one budget, then the upstream runs; the
    // writes that follow must short-circuit.
    expect(degraded.ms).toBeGreaterThanOrEqual(
      UPSTREAM_DELAY_MS + TIMEOUT_SECS * 1000,
    );
    // The bound sits between "one budget" (delay + budget) and "two"
    // (delay + 2 x budget), with at least a second of room on each side
    // so neither verdict rides on scheduling noise.
    expect(degraded.ms).toBeLessThan(
      UPSTREAM_DELAY_MS + TIMEOUT_SECS * 1000 + 2_000,
    );
  }, 60_000);
});

// The cool-off ends on its own, and what used to happen then was that the
// next request in was elected to re-test Redis — paying the full budget to
// discover what the request before it already knew, once per window for as
// long as the outage lasted. A background probe does that now, so the
// request after the window costs nothing while Redis is still down.
//
// This one waits out the real 30s window rather than a shortened one: the
// constant is the thing under test.
describe("the request after the cool-off does not re-test Redis", () => {
  let f: Fixture | undefined;
  let ready = false;

  beforeAll(async () => {
    ready = await vectorRedisReady();
    if (ready) f = await bringUp("window");
  });
  afterAll(async () => {
    await tearDown(f);
  });

  test("a request arriving after the window expires is not delayed", async (ctx) => {
    if (!ready || !f) {
      ctx.skip();
      return;
    }

    const warm = await timeChat(f.app.proxyUrl, EXACT_MODEL, "window warm");
    expect(warm.status).toBe(200);

    f.relay.blackhole();

    const first = await timeChat(f.app.proxyUrl, EXACT_MODEL, "window cold");
    expect(first.status).toBe(200);
    expect(first.ms).toBeGreaterThanOrEqual(TIMEOUT_SECS * 1000);

    // Past the 30s cool-off, with Redis still black-holed.
    await new Promise((r) => setTimeout(r, 33_000));

    const after = await timeChat(f.app.proxyUrl, EXACT_MODEL, "window after");
    expect(after.status).toBe(200);
    expect(after.ms).toBeLessThan(1_000);
  }, 120_000);
});

describe("an exact-only policy still pays one budget", () => {
  let f: Fixture | undefined;
  let ready = false;

  beforeAll(async () => {
    ready = await vectorRedisReady();
    if (ready) f = await bringUp("exact");
  });
  afterAll(async () => {
    await tearDown(f);
  });

  test("one connection, one budget", async (ctx) => {
    if (!ready || !f) {
      ctx.skip();
      return;
    }

    const warm = await timeChat(f.app.proxyUrl, EXACT_MODEL, "exact warm");
    expect(warm.status).toBe(200);

    f.relay.blackhole();

    const degraded = await timeChat(f.app.proxyUrl, EXACT_MODEL, "exact cold");
    expect(degraded.status).toBe(200);
    expect(degraded.ms).toBeGreaterThanOrEqual(TIMEOUT_SECS * 1000);
    expect(degraded.ms).toBeLessThan(ONE_BUDGET_MS);
  }, 60_000);
});


// A cache Redis that is already unreachable when the gateway STARTS used
// to end the boot. Neither half of that was right.
//
// The connect ran on the driver's own retry schedule rather than on
// `timeout_secs`, so the process sat before any listener bind for about
// eight minutes with no log line, and only then exited. And exiting at
// all was the odd one out: every cache operation already fails open to a
// miss, and a RUNNING gateway rides out an unbounded cache-Redis outage
// that way — only boot was fatal.
//
// The deployment that makes it concrete is the one `config.example.yaml`
// recommends: `ratelimit.redis` pointing at the same Redis as
// `cache.redis`. With both blocks on one unreachable server the limiter
// degraded correctly and the cache then killed the process anyway, so the
// gateway served nothing at all — which is the shape this whole release
// is fixing.
describe("a cache Redis unreachable at startup degrades the cache, not the boot", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let embed: Awaited<ReturnType<typeof startEmbeddingMock>> | undefined;
  let relay: Awaited<ReturnType<typeof startRedisBlackhole>> | undefined;
  let ready = false;
  const prefix = `/aisix-e2e-cache-boot-${randomUUID()}`;

  beforeAll(async () => {
    ready = await vectorRedisReady();
    if (!ready) return;

    upstream = await startOpenAiUpstream();
    embed = await startEmbeddingMock();
    // Silent from the first SYN: the gateway's own boot connect is what
    // meets it. A refused connection is a different, fast path.
    relay = await startRedisBlackhole(REDIS_URL, { blackholed: true });
    app = await spawnApp({
      // `info`, so the background attach announces itself in `output()`.
      logLevel: "info",
      // Both blocks meet the same black hole, and the limiter's connect
      // and the cache's run one after the other — two budgets before a
      // listener binds, against a 10s harness readiness gate. Waiting
      // here ourselves keeps the case measuring its own subject instead
      // of racing that gate; the assertion that it served at all is the
      // `/livez` poll below.
      awaitListeners: false,
      extra: {
        etcd: { endpoints: [ETCD_ENDPOINT], prefix },
        cache: {
          backend: "redis",
          redis: { url: relay.url, timeout_secs: TIMEOUT_SECS },
        },
        // The shared-Redis deployment, pointed at the same dead relay.
        ratelimit: {
          backend: "redis",
          redis: { url: relay.url, timeout_secs: TIMEOUT_SECS },
        },
      },
    });
    // `awaitListeners: false` means nothing has waited yet, and the
    // config probe below would meet a refused connection rather than a
    // not-ready one. The bound is generous on purpose: the subject is
    // that the boot no longer takes MINUTES, and the two budgets it does
    // take are the gateway's business, not this case's.
    const deadline = Date.now() + 30_000;
    let live = false;
    while (!live && Date.now() < deadline) {
      live = await fetch(`${app.proxyUrl}/livez`)
        .then((r) => r.ok)
        .catch(() => false);
      if (!live) await new Promise((r) => setTimeout(r, 200));
    }
    if (!live) throw new Error("the gateway never bound its proxy listener");

    await seed(prefix, embed.baseUrl, upstream.baseUrl);
    const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(
      async () => (await probe.listModels()).status === 200,
    );
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await embed?.close();
    await relay?.close();
    if (ready) await new EtcdClient().deletePrefix(prefix);
  });


  test("it serves with every backend=redis policy a miss, then starts caching", async (ctx) => {
    if (!ready || !app || !relay || !upstream) {
      ctx.skip();
      return;
    }

    // 1. It serves at all. On the pre-change binary the process exits
    //    during `beforeAll` instead, so `waitConfigPropagation` never
    //    sees a 200 and the case fails there.
    expect(app.output()).toContain("aisix listening");
    const first = await timeChat(app.proxyUrl, EXACT_MODEL, "boot degraded one");
    expect(first.status).toBe(200);

    // 2. One WARN names the backend and WHICH Redis, with no credentials.
    const warn = app
      .output()
      .split("\n")
      .find((l) => l.includes("cache backend unreachable at startup"));
    expect(warn).toBeDefined();
    expect(warn).toContain(new URL(relay.url).host);
    expect(warn).not.toContain("redis://");

    // 3. Every backend=redis policy is a miss while degraded, so the same
    //    prompt reaches the upstream twice. This is what says "serving
    //    uncached" rather than "serving from some other cache".
    const before = upstream.receivedRequests.length;
    await timeChat(app.proxyUrl, EXACT_MODEL, "boot degraded two");
    await timeChat(app.proxyUrl, EXACT_MODEL, "boot degraded two");
    expect(upstream.receivedRequests.length - before).toBe(2);

    //    …and the outage is reported ONCE, not once per request. Three
    //    requests have now been served with a failing cache, each of
    //    which both reads and writes, so an unthrottled gateway has
    //    logged six lines by here and will keep doing so for as long as
    //    the outage lasts — burying everything else in the log. How hard
    //    and how long it is failing is `aisix_redis_failures_total`;
    //    the log line only has to say that it started.
    // A memory-backed policy interleaved with the failing redis ones.
    // Its cache cannot fail, so every one of these used to count as
    // "redis recovered" and re-arm the latch — bringing the per-request
    // flood straight back in any deployment that runs both kinds.
    await timeChat(app.proxyUrl, MEMORY_MODEL, "memory policy one");
    await timeChat(app.proxyUrl, EXACT_MODEL, "boot degraded three");
    await timeChat(app.proxyUrl, MEMORY_MODEL, "memory policy two");
    await timeChat(app.proxyUrl, EXACT_MODEL, "boot degraded four");

    const degradedWarns = app
      .output()
      .split("\n")
      .filter(
        (l) =>
          l.includes("WARN") &&
          (l.includes("cache lookup failed") || l.includes("cache write failed")),
      ).length;
    // ONE, not one per operation: the exact-KV half is a single
    // degradation, and whichever of its read and write gets there first
    // is the one that reports it. The rest of the outage is debug.
    expect(degradedWarns).toBe(1);

    //    A count of one proves throttling only if the requests really
    //    reached the cache gate — one that never got there would read
    //    one too. The counter is what says they did: it is incremented
    //    per failed operation and is deliberately NOT throttled, so it
    //    supplies the lower bound the log line cannot.
    const scrape = await (await fetch(`${app.metricsUrl}/metrics`)).text();
    const cacheGetFailures = Number(
      /aisix_redis_failures_total\{operation="cache_get"\} (\d+)/.exec(scrape)?.[1] ?? 0,
    );
    expect(cacheGetFailures).toBeGreaterThanOrEqual(3);

    // 4. Redis comes up and the cache attaches itself — the degradation
    //    is temporary, not a silent demotion for the life of the process.
    relay.heal();
    const deadline = Date.now() + 30_000;
    while (
      !app.output().includes("cache backend attached") &&
      Date.now() < deadline
    ) {
      await new Promise((r) => setTimeout(r, 200));
    }
    expect(app.output()).toContain("cache backend attached");

    // …and it really caches now: the second identical prompt does not
    //    reach the upstream. A store that had attached in name only would
    //    still forward it.
    const warm = upstream.receivedRequests.length;
    const miss = await timeChat(app.proxyUrl, EXACT_MODEL, "boot attached one");
    expect(miss.status).toBe(200);
    const hit = await timeChat(app.proxyUrl, EXACT_MODEL, "boot attached one");
    expect(hit.status).toBe(200);
    expect(upstream.receivedRequests.length - warm).toBe(1);
  }, 150_000);
});

// The cache half of the credential-refusal case. The full story — that
// the degraded state ends by itself once the credential is corrected on
// the server — is in `ratelimit-cluster-e2e.test.ts`, because both
// subsystems share one connect path and one classification. What is
// asserted here is that the cache reaches the same two conclusions
// through its own call site: it serves, and it says the server REFUSED
// rather than that it timed out.
describe("a cache Redis that refuses the credential degrades the cache, not the boot", () => {
  let app: SpawnedApp | undefined;
  let ready = false;
  const prefix = `/aisix-e2e-cache-refused-${randomUUID()}`;
  const user = `aisix-e2e-cache-${randomUUID().slice(0, 8)}`;

  /** One command over a fresh RESP connection, as bulk strings. */
  async function redisCommand(args: string[]): Promise<void> {
    const m = /^redis:\/\/(?:[^@/]*@)?([^:/]+)(?::(\d+))?/.exec(REDIS_URL);
    if (!m) throw new Error(`not a redis url: ${REDIS_URL}`);
    const host = m[1];
    const port = m[2] ? Number(m[2]) : 6379;
    const payload =
      `*${args.length}\r\n` + args.map((a) => `$${Buffer.byteLength(a)}\r\n${a}\r\n`).join("");
    await new Promise<void>((resolve, reject) => {
      const sock = connect({ host, port }, () => sock.write(payload));
      sock.once("data", () => {
        sock.destroy();
        resolve();
      });
      sock.once("error", (e) => {
        sock.destroy();
        reject(e);
      });
      sock.setTimeout(2000, () => {
        sock.destroy();
        reject(new Error("redis command timed out"));
      });
    });
  }

  beforeAll(async () => {
    ready = await vectorRedisReady();
    if (!ready) return;
    // An ACL user rather than `requirepass`, which is server-wide and
    // would lock every other file in this suite out of the same Redis.
    await redisCommand(["ACL", "SETUSER", user, "on", ">not-the-one-configured", "~*", "+@all"]);
    app = await spawnApp({
      // `info`, so the listening line the first assertion reads is kept.
      logLevel: "info",
      extra: {
        etcd: { endpoints: [ETCD_ENDPOINT], prefix },
        cache: {
          backend: "redis",
          redis: {
            url: REDIS_URL,
            username: user,
            password: "the-one-configured",
            timeout_secs: 5,
          },
        },
      },
    });
  });

  afterAll(async () => {
    await app?.exit();
    if (ready) {
      await redisCommand(["ACL", "DELUSER", user]).catch(() => {});
      await new EtcdClient().deletePrefix(prefix);
    }
  });

  test("it serves, and the WARN names a refusal rather than a timeout", async (ctx) => {
    if (!ready || !app) {
      ctx.skip();
      return;
    }
    expect(app.output()).toContain("aisix listening");

    const warn = app
      .output()
      .split("\n")
      .find((l) => l.includes("cache backend REFUSED"));
    expect(warn).toBeDefined();
    expect(warn).toContain("reason=refused");
    expect(warn).toContain(new URL(REDIS_URL).host);
    // The server's own words: the only part that says WHICH setting.
    expect(warn?.toLowerCase()).toContain("auth");
    expect(warn).not.toContain("timed out");
    expect(warn).not.toContain("unreachable");
    expect(warn).not.toContain("redis://");
  }, 60_000);
});
