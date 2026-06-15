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

const ETCD_ENDPOINT = process.env.AISIX_E2E_ETCD ?? "http://127.0.0.1:2379";
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
    dial_timeout_ms: 5000,
    request_timeout_ms: 5000,
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

/** Seed one model + an RPM=1 ApiKey via this app's admin API. The peer
 *  replica picks the same config up over the shared etcd watch. */
async function seed(app: SpawnedApp, upstreamBase: string, model: string) {
  const admin = new AdminClient(app.adminUrl, app.adminKey);
  const pk = await admin.createProviderKey({
    display_name: `${model}-pk`,
    secret: "sk-mock",
    api_base: `${upstreamBase}/v1`,
  });
  await admin.createModel({
    display_name: model,
    provider: "openai",
    model_name: "gpt-4o-mini",
    provider_key_id: pk.id,
  });
  await admin.createApiKey({
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
    await seed(appA, upstream.baseUrl, model);
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
    etcdReady = await new EtcdClient().ping();
    if (!etcdReady) return;

    upstream = await startOpenAiUpstream();
    // Shared etcd (same ApiKey entry id) but default memory backend — the
    // counters live per-process, so the cap does NOT span replicas.
    const extra = { etcd: sharedEtcd(prefix) };
    appA = await spawnApp({ extra });
    appB = await spawnApp({ extra });
    await seed(appA, upstream.baseUrl, model);
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
