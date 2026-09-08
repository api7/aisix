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
async function startRedisBlackhole(upstreamUrl: string): Promise<{
  url: string;
  blackhole(): void;
  close(): Promise<void>;
}> {
  const m = /^redis:\/\/(?:[^@/]*@)?([^:/]+)(?::(\d+))?/.exec(upstreamUrl);
  if (!m) throw new Error(`unparseable redis url: ${upstreamUrl}`);
  const host = m[1];
  const port = m[2] ? Number(m[2]) : 6379;
  let hole = false;
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
    close: () =>
      new Promise<void>((r) => {
        for (const s of live) s.destroy();
        server.close(() => r());
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
  for (const model of [SEMANTIC_MODEL, EXACT_MODEL]) {
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
