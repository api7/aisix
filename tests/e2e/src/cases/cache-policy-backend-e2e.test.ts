import { createHash, randomUUID } from "node:crypto";
import { connect } from "node:net";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: per-policy cache backend dispatch (api7/AISIX-Cloud#519 B.8).
//
// `CachePolicy.backend` selects which cache instance serves a request:
// - `memory` → the in-process cache, always available;
// - `redis`  → the shared redis cache, available iff the bootstrap
//   config carries `cache.redis`.
//
// A `redis` policy on a memory-only DP must DISABLE caching for its
// matching requests — every identical call pays the upstream and no
// `x-aisix-cache` header is emitted. The pre-fix behavior silently
// fell back to the node-local memory cache, which would serve the
// second call from cache while the policy claims shared semantics.

const CALLER_PLAINTEXT = "sk-cache-backend-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const REDIS_URL =
  process.env.AISIX_E2E_REDIS ?? "redis://127.0.0.1:6379";

/** RESP-level PING so the redis-positive suite can skip honestly when
 *  no redis is reachable (CI provisions redis:7-alpine on :6379). */
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

interface SeededApp {
  app: SpawnedApp;
  admin: AdminClient;
}

/**
 * Seed one DP with:
 * - a model `<modelAlias>` plus a `redis`-backend policy scoped to it
 *   (the subject under test), and
 * - a canary model `<canaryAlias>` plus a `memory`-backend policy
 *   scoped to it, created AFTER the redis policy.
 *
 * etcd delivers watch events in revision order, so once the canary
 * policy is observable (its responses carry `x-aisix-cache`), the
 * earlier redis policy is in the snapshot too. Without that positive
 * signal, the "no caching happened" assertions below could pass
 * vacuously while the policy simply hadn't propagated yet.
 */
async function seedApp(
  app: SpawnedApp,
  upstreamBase: string,
  modelAlias: string,
  canaryAlias: string,
): Promise<SeededApp> {
  const admin = new AdminClient(app.adminUrl, app.adminKey);
  const pk = await admin.createProviderKey({
    display_name: `${modelAlias}-pk`,
    secret: "sk-mock",
    api_base: `${upstreamBase}/v1`,
  });
  for (const alias of [modelAlias, canaryAlias]) {
    await admin.createModel({
      display_name: alias,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
  }
  await admin.createApiKey({
    key_hash: CALLER_KEY_HASH,
    allowed_models: [modelAlias, canaryAlias],
  });
  // Order matters: redis policy FIRST, canary memory policy SECOND.
  await admin.json("POST", "/admin/v1/cache_policies", {
    name: `${modelAlias}-redis-policy`,
    enabled: true,
    backend: "redis",
    applies_to: `model:${modelAlias}`,
  });
  await admin.json("POST", "/admin/v1/cache_policies", {
    name: `${canaryAlias}-canary-policy`,
    enabled: true,
    backend: "memory",
    applies_to: `model:${canaryAlias}`,
  });
  return { app, admin };
}

function chatRequest(
  proxyUrl: string,
  model: string,
  prompt: string,
): Promise<Response> {
  return fetch(`${proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model,
      messages: [{ role: "user", content: prompt }],
    }),
  });
}

/** Wait until the canary policy (created after the redis policy) is
 *  live: a canary chat carries `x-aisix-cache` (miss or hit). */
async function waitCanaryPolicyLive(
  proxyUrl: string,
  canaryAlias: string,
): Promise<void> {
  await waitConfigPropagation(async () => {
    try {
      const resp = await chatRequest(proxyUrl, canaryAlias, "canary-probe");
      await resp.text();
      return resp.status === 200 && resp.headers.get("x-aisix-cache") !== null;
    } catch {
      return false;
    }
  });
}

describe("cache policy backend=redis on a memory-only DP disables caching", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    // Default harness config: `cache.backend = memory`, no
    // `cache.redis` block — a memory-only DP.
    app = await spawnApp();
    await seedApp(app, upstream.baseUrl, "cache-redis-only", "cache-canary");
    await waitCanaryPolicyLive(app.proxyUrl, "cache-canary");
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("identical requests ALL reach the upstream; no x-aisix-cache header", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const baseline = upstream.receivedRequests.length;
    const prompt = `redis-unavailable ${randomUUID()}`;

    const first = await chatRequest(app.proxyUrl, "cache-redis-only", prompt);
    expect(first.status).toBe(200);
    expect(first.headers.get("x-aisix-cache")).toBeNull();
    await first.text();
    expect(upstream.receivedRequests.length).toBe(baseline + 1);

    // Pre-fix, this second identical call was served from the
    // node-local memory cache (`x-aisix-cache: hit`, upstream count
    // unchanged). With per-policy dispatch it must pay the upstream.
    const second = await chatRequest(app.proxyUrl, "cache-redis-only", prompt);
    expect(second.status).toBe(200);
    expect(second.headers.get("x-aisix-cache")).toBeNull();
    await second.text();
    expect(upstream.receivedRequests.length).toBe(baseline + 2);
  });
});

describe("cache policy backend=redis with a configured redis is shared across DPs", () => {
  let appA: SpawnedApp | undefined;
  let appB: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let infraReady = false;

  beforeAll(async () => {
    infraReady =
      (await new EtcdClient().ping()) && (await redisPing(REDIS_URL));
    if (!infraReady) return;

    upstream = await startOpenAiUpstream();
    const redisExtra = {
      cache: { backend: "memory", redis: { url: REDIS_URL } },
    };
    // Two DP instances sharing one redis. A hit on the instance that
    // never served the original request proves the entry really lives
    // in redis — an (incorrect) memory-cache write could only produce
    // hits on the same instance.
    appA = await spawnApp({ extra: redisExtra });
    appB = await spawnApp({ extra: redisExtra });
    await seedApp(appA, upstream.baseUrl, "cache-redis-shared", "canary-a");
    await seedApp(appB, upstream.baseUrl, "cache-redis-shared", "canary-b");
    await waitCanaryPolicyLive(appA.proxyUrl, "canary-a");
    await waitCanaryPolicyLive(appB.proxyUrl, "canary-b");
  });

  afterAll(async () => {
    await appA?.exit();
    await appB?.exit();
    await upstream?.close();
  });

  test("miss on DP A, hit on DP B without re-hitting the upstream", async (ctx) => {
    if (!infraReady || !appA || !appB || !upstream) {
      ctx.skip();
      return;
    }

    const baseline = upstream.receivedRequests.length;
    // Unique per run — redis outlives the test, identical prompts
    // from a previous run would already be cached.
    const prompt = `redis-shared ${randomUUID()}`;

    const first = await chatRequest(appA.proxyUrl, "cache-redis-shared", prompt);
    expect(first.status).toBe(200);
    expect(first.headers.get("x-aisix-cache")).toBe("miss");
    const firstBody = (await first.json()) as {
      choices: Array<{ message: { content: string } }>;
    };
    expect(upstream.receivedRequests.length).toBe(baseline + 1);

    const second = await chatRequest(appB.proxyUrl, "cache-redis-shared", prompt);
    expect(second.status).toBe(200);
    expect(second.headers.get("x-aisix-cache")).toBe("hit");
    const secondBody = (await second.json()) as {
      choices: Array<{ message: { content: string } }>;
    };
    expect(upstream.receivedRequests.length).toBe(baseline + 1);

    // The replay must be byte-equivalent content — DP B never talked
    // to the upstream for this fingerprint.
    expect(secondBody.choices[0]?.message.content).toBe(
      firstBody.choices[0]?.message.content,
    );
  });
});
