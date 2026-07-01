import { createHash, randomUUID } from "node:crypto";
import { connect } from "node:net";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  ProxyClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: the response cache is isolated per environment on a shared Redis
// (api7/AISIX-Cloud#788, P1-1).
//
// The cache key is a content-only fingerprint (model + messages + params,
// no env/caller). Redis is user-provided infrastructure our chart does
// not manage, so a user can point the DPs of two different environments
// at ONE Redis. Without env-scoping, a byte-identical request in env-b
// would hit env-a's cached entry and receive env-a's answer.
// `with_env_namespace` scopes the redis key prefix by `env_id`.
//
// Repro: two apps on one shared etcd base + one shared Redis, but with
// distinct `env_id` (so `effective_prefix` isolates their config too).
// Each has a redis-backed `applies_to:all` cache policy. An identical
// request is a HIT within its own env yet a MISS in the peer env, and
// the peer env returns its OWN upstream body — proving no cross-env
// leak. Before the fix the peer request would be a HIT with env-a's body.

const CALLER_PLAINTEXT = "sk-cache-envns-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const ETCD_ENDPOINT = process.env.AISIX_E2E_ETCD ?? "http://127.0.0.1:2379";
const REDIS_URL = process.env.AISIX_E2E_REDIS ?? "redis://127.0.0.1:6379";

// Both envs use the SAME model alias so the request fingerprints collide
// across environments — the env namespace is the only thing keeping their
// cache entries apart.
const MODEL = "cache-envns";

/** RESP-level PING so the suite skips honestly when no redis is reachable
 *  (CI provisions redis:7-alpine on :6379). */
async function redisPing(url: string): Promise<boolean> {
  const m = /^redis:\/\/(?:[^@/]*@)?([^:/]+)(?::(\d+))?/.exec(url);
  if (!m) return false;
  const host = m[1];
  const port = m[2] ? Number(m[2]) : 6379;
  return new Promise((resolve) => {
    const sock = connect({ host, port });
    const done = (ok: boolean) => {
      sock.destroy();
      resolve(ok);
    };
    sock.once("connect", () => sock.write("PING\r\n"));
    sock.once("data", (buf: Buffer) => done(buf.toString().startsWith("+PONG")));
    sock.once("error", () => done(false));
    sock.setTimeout(1000, () => done(false));
  });
}

/** A shared etcd base prefix, but each app carries its own `env_id`, so
 *  `effective_prefix` (`<prefix>/<env_id>/`) keeps their config namespaces
 *  apart while both still point at ONE Redis. */
function envEtcd(prefix: string, envId: string) {
  return {
    endpoints: [ETCD_ENDPOINT],
    prefix,
    env_id: envId,
    dial_timeout_ms: 5000,
    request_timeout_ms: 5000,
  };
}

/** An OpenAI chat.completion body with a distinguishable assistant content,
 *  so a cross-env cache hit is caught by comparing the returned text. */
function completionBody(content: string) {
  return {
    id: `mock-${content}`,
    object: "chat.completion",
    created: 0,
    model: "mock-model",
    choices: [
      { index: 0, message: { role: "assistant", content }, finish_reason: "stop" },
    ],
    usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
  };
}

function chatRequest(proxyUrl: string): Promise<Response> {
  return fetch(`${proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: MODEL,
      messages: [{ role: "user", content: "hello" }],
    }),
  });
}

/** Seed one model (→ this env's own upstream), a permissive ApiKey, and a
 *  redis-backed `applies_to:all` cache policy via this app's admin API. */
async function seed(app: SpawnedApp, upstreamBase: string) {
  const admin = new AdminClient(app.adminUrl, app.adminKey);
  const pk = await admin.createProviderKey({
    display_name: `${MODEL}-pk`,
    secret: "sk-mock",
    api_base: `${upstreamBase}/v1`,
  });
  await admin.createModel({
    display_name: MODEL,
    provider: "openai",
    model_name: "gpt-4o-mini",
    provider_key_id: pk.id,
  });
  await admin.createApiKey({
    key_hash: CALLER_KEY_HASH,
    allowed_models: [MODEL],
  });
  await admin.json("POST", "/admin/v1/cache_policies", {
    name: `${MODEL}-redis-policy`,
    enabled: true,
    backend: "redis",
    ttl_seconds: 300,
    applies_to: "all",
  });
}

/** Wait until `MODEL` is visible on `proxyUrl` (config propagation). */
async function waitModelLive(proxyUrl: string) {
  const probe = new ProxyClient(proxyUrl, CALLER_PLAINTEXT);
  await waitConfigPropagation(async () => {
    const res = await probe.listModels();
    if (res.status !== 200) return false;
    const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
    return data.some((m) => m.id === MODEL);
  });
}

describe("response cache is isolated per env on a shared Redis (#788 P1-1)", () => {
  let appA: SpawnedApp | undefined;
  let appB: SpawnedApp | undefined;
  let upstreamA: OpenAiUpstream | undefined;
  let upstreamB: OpenAiUpstream | undefined;
  let infraReady = false;
  const prefix = `/aisix-e2e-cache-envns-${randomUUID()}`;
  // Unique env ids per run so the redis namespace never collides with a
  // prior run on a shared CI redis.
  const envA = `env-a-${randomUUID()}`;
  const envB = `env-b-${randomUUID()}`;

  beforeAll(async () => {
    infraReady = (await new EtcdClient().ping()) && (await redisPing(REDIS_URL));
    if (!infraReady) return;

    upstreamA = await startOpenAiUpstream({
      nonStreamBody: completionBody("answer-from-env-a"),
    });
    upstreamB = await startOpenAiUpstream({
      nonStreamBody: completionBody("answer-from-env-b"),
    });

    const cache = { backend: "memory", redis: { url: REDIS_URL } };
    appA = await spawnApp({ extra: { etcd: envEtcd(prefix, envA), cache } });
    appB = await spawnApp({ extra: { etcd: envEtcd(prefix, envB), cache } });

    await seed(appA, upstreamA.baseUrl);
    await seed(appB, upstreamB.baseUrl);
    await waitModelLive(appA.proxyUrl);
    await waitModelLive(appB.proxyUrl);
  });

  afterAll(async () => {
    await appA?.exit();
    await appB?.exit();
    await upstreamA?.close();
    await upstreamB?.close();
    // The harness cleans the unique prefixes it generated, not our shared
    // override — drop it ourselves (range delete covers both env scopes).
    if (infraReady) await new EtcdClient().deletePrefix(prefix);
  });

  test("identical request hits within its env but misses in the peer env", async (ctx) => {
    if (!infraReady || !appA || !appB || !upstreamA || !upstreamB) {
      ctx.skip();
      return;
    }

    const baseA = upstreamA.receivedRequests.length;
    const baseB = upstreamB.receivedRequests.length;

    // env-a: first request → MISS, served by env-a's upstream.
    const a1 = await chatRequest(appA.proxyUrl);
    expect(a1.status).toBe(200);
    expect(a1.headers.get("x-aisix-cache")).toBe("miss");
    const a1body = (await a1.json()) as {
      choices: Array<{ message: { content: string } }>;
    };
    expect(a1body.choices[0].message.content).toBe("answer-from-env-a");
    expect(upstreamA.receivedRequests.length).toBe(baseA + 1);

    // env-a: same request again → HIT (proves caching is active in env-a,
    // so a peer MISS below is due to isolation, not caching being off).
    const a2 = await chatRequest(appA.proxyUrl);
    expect(a2.status).toBe(200);
    expect(a2.headers.get("x-aisix-cache")).toBe("hit");
    await a2.body?.cancel();
    expect(upstreamA.receivedRequests.length).toBe(baseA + 1);

    // env-b: byte-identical request on the SAME shared Redis → MISS, because
    // the cache namespace is env-scoped. It returns env-b's OWN upstream
    // body — NOT env-a's. Before the fix this was a HIT with env-a's body
    // and env-b's upstream would never be called.
    const b1 = await chatRequest(appB.proxyUrl);
    expect(b1.status).toBe(200);
    expect(b1.headers.get("x-aisix-cache")).toBe("miss");
    const b1body = (await b1.json()) as {
      choices: Array<{ message: { content: string } }>;
    };
    expect(b1body.choices[0].message.content).toBe("answer-from-env-b");
    expect(upstreamB.receivedRequests.length).toBe(baseB + 1);

    // env-b: same request again → HIT within env-b's own namespace.
    const b2 = await chatRequest(appB.proxyUrl);
    expect(b2.status).toBe(200);
    expect(b2.headers.get("x-aisix-cache")).toBe("hit");
    await b2.body?.cancel();
    expect(upstreamB.receivedRequests.length).toBe(baseB + 1);
  });
});
