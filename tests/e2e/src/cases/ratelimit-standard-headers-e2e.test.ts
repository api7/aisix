import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  ProxyClient,
  spawnApp,
  startOpenAiUpstream,
  awaitWindowHeadroom,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";
import { harnessRequest } from "../harness/http.js";

// E2E: the standard rate-limit headers on a gateway-produced 429.
//
// A caller that hits a limit needs to learn, from the response alone,
// what the cap was, that it has nothing left, and when to come back.
// This spec reads the RAW response rather than going through an SDK:
// an SDK normalises header access and silently retries, so an
// SDK-level assertion cannot show what is actually on the wire — which
// is the artifact being pinned here.
//
// Contract, per dimension that refused:
//   x-ratelimit-limit      the cap of the dimension that refused
//   x-ratelimit-remaining  headroom left under it (0 on a refusal)
//   x-ratelimit-reset      delta-seconds until it lets the caller back
//   retry-after            same wait, so either header can drive back-off
//   x-ratelimit-scope      AISIX extension: WHICH dimension refused —
//                          `x-ratelimit-limit: 1` is otherwise
//                          indistinguishable between a request cap, a
//                          token cap and a concurrency cap, whose units
//                          all differ
//
// The trio is emitted ONLY when the gateway's own limiter refused. A
// 200, an upstream 429 and a budget rejection all leave it absent, so
// its presence is what tells a caller whose limit it hit. The 200 half
// of that is asserted here; the other two are pinned on the renderer
// (`crates/aisix-proxy/src/error.rs`), which is where the decision
// lives.

const RPM_PLAINTEXT = "sk-rlh-rpm";
const TPM_PLAINTEXT = "sk-rlh-tpm";
const CONC_PLAINTEXT = "sk-rlh-conc";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

const FAST_MODEL = "rlh-fast";
const SLOW_MODEL = "rlh-slow";

/** Upstream stall for the concurrency case, in ms. */
const SLOW_UPSTREAM_MS = 2_000;

interface RawResponse {
  status: number;
  headers: Record<string, string>;
}

async function rawChat(
  proxyUrl: string,
  apiKey: string,
  model: string,
): Promise<RawResponse> {
  const res = await harnessRequest(`${proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${apiKey}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model,
      messages: [{ role: "user", content: "hi" }],
    }),
  });
  // Drain so the connection is not left half-read between calls.
  await res.body.text();
  const headers: Record<string, string> = {};
  for (const [k, v] of Object.entries(res.headers)) {
    headers[k.toLowerCase()] = Array.isArray(v) ? v.join(", ") : String(v ?? "");
  }
  return { status: res.statusCode, headers };
}

/**
 * Assert the four contracted headers plus the scope extension, and
 * return the parsed reset so a caller can bound it against the window
 * the refusing dimension actually has.
 */
function expectStandardHeaders(
  res: RawResponse,
  expected: { limit: number; scope: string },
): number {
  expect(res.status).toBe(429);
  expect(res.headers["x-ratelimit-limit"]).toBe(String(expected.limit));
  expect(res.headers["x-ratelimit-remaining"]).toBe("0");
  expect(res.headers["x-ratelimit-scope"]).toBe(expected.scope);

  const reset = Number.parseInt(res.headers["x-ratelimit-reset"] ?? "", 10);
  const retryAfter = Number.parseInt(res.headers["retry-after"] ?? "", 10);
  expect(Number.isNaN(reset)).toBe(false);
  expect(Number.isNaN(retryAfter)).toBe(false);
  // Both headers answer "when may I retry", so they must agree — a
  // client is free to read whichever one it already supports.
  expect(reset).toBe(retryAfter);
  // Zero would tell the client to retry immediately, which is the
  // retry storm the hint exists to prevent.
  expect(reset).toBeGreaterThanOrEqual(1);
  return reset;
}

describe("rate limit e2e: standard 429 headers", () => {
  let app: SpawnedApp | undefined;
  let fastUpstream: OpenAiUpstream | undefined;
  let slowUpstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    fastUpstream = await startOpenAiUpstream();
    // The concurrency cap can only be observed while a request is still
    // in flight, so that dimension needs an upstream that holds the
    // gateway's slot open long enough for a second call to arrive.
    slowUpstream = await startOpenAiUpstream({
      responseDelayMs: SLOW_UPSTREAM_MS,
    });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const fastPk = await seed.createProviderKey({
      display_name: "rlh-fast-pk",
      secret: "sk-mock",
      api_base: `${fastUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: FAST_MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: fastPk.id,
    });
    const slowPk = await seed.createProviderKey({
      display_name: "rlh-slow-pk",
      secret: "sk-mock",
      api_base: `${slowUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: SLOW_MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: slowPk.id,
    });

    // One key per dimension, each carrying ONLY that dimension, so a
    // refusal cannot be attributed to the wrong one.
    await seed.createApiKey({
      key_hash: hash(RPM_PLAINTEXT),
      allowed_models: [FAST_MODEL],
      rate_limit: { rpm: 1 },
    });
    // tpm is checked-but-not-incremented before dispatch: the first
    // call is admitted and commits the mock's usage, which overruns
    // this cap, so the SECOND call is the one refused.
    await seed.createApiKey({
      key_hash: hash(TPM_PLAINTEXT),
      allowed_models: [FAST_MODEL],
      rate_limit: { tpm: 1 },
    });
    await seed.createApiKey({
      key_hash: hash(CONC_PLAINTEXT),
      allowed_models: [SLOW_MODEL],
      rate_limit: { concurrency: 1 },
    });
  });

  afterAll(async () => {
    await app?.exit();
    await fastUpstream?.close();
    await slowUpstream?.close();
  });

  test("a request-cap 429 carries the four headers on the raw response", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    // listModels does not consume the rpm=1 slot, so it is a readiness
    // probe the test can afford.
    const probe = new ProxyClient(app.proxyUrl, RPM_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      if (res.status !== 200) return false;
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return data.some((m) => m.id === FAST_MODEL);
    });

    // The limiter buckets on fixed wall-clock minutes; a burst that
    // straddles a boundary would get a fresh allowance.
    await awaitWindowHeadroom();

    const first = await rawChat(app.proxyUrl, RPM_PLAINTEXT, FAST_MODEL);
    expect(first.status).toBe(200);
    // A 200 reports the per-dimension `x-ratelimit-*-requests` family
    // and NOT the unsuffixed trio: the trio's presence is what marks a
    // response as the gateway's own rate-limit refusal.
    expect(first.headers["x-ratelimit-limit-requests"]).toBe("1");
    expect(first.headers["x-ratelimit-limit"]).toBeUndefined();
    expect(first.headers["x-ratelimit-scope"]).toBeUndefined();

    const refused = await rawChat(app.proxyUrl, RPM_PLAINTEXT, FAST_MODEL);
    const reset = expectStandardHeaders(refused, { limit: 1, scope: "rpm" });
    // rpm is a minute window, so the wait cannot exceed one.
    expect(reset).toBeLessThanOrEqual(60);
  });

  test("a token-cap 429 reports the token dimension, not the request one", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const probe = new ProxyClient(app.proxyUrl, TPM_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      return res.status === 200;
    });
    await awaitWindowHeadroom();

    const first = await rawChat(app.proxyUrl, TPM_PLAINTEXT, FAST_MODEL);
    expect(first.status).toBe(200);

    const refused = await rawChat(app.proxyUrl, TPM_PLAINTEXT, FAST_MODEL);
    const reset = expectStandardHeaders(refused, { limit: 1, scope: "tpm" });
    expect(reset).toBeLessThanOrEqual(60);
  });

  test("a concurrency 429 carries all four headers and a fixed hint", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const probe = new ProxyClient(app.proxyUrl, CONC_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return res.status === 200 && data.some((m) => m.id === SLOW_MODEL);
    });

    // Hold the only slot with a call the upstream stalls, then race a
    // second one into it.
    const held = rawChat(app.proxyUrl, CONC_PLAINTEXT, SLOW_MODEL);
    await new Promise((r) => setTimeout(r, 300));
    const refused = await rawChat(app.proxyUrl, CONC_PLAINTEXT, SLOW_MODEL);

    // A concurrency slot frees when some in-flight request finishes,
    // which the gateway cannot predict — so unlike a window this
    // reports a fixed hint rather than a countdown. Before this
    // contract the same 429 carried no retry hint at all.
    const reset = expectStandardHeaders(refused, { limit: 1, scope: "concurrency" });
    expect(reset).toBe(60);

    expect((await held).status).toBe(200);
  }, 30_000);
});
