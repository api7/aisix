import { createHash } from "node:crypto";
import OpenAI, { APIError } from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: same-target retries back off before re-hitting the upstream instead
// of hammering it immediately (issue 788 P2). A routing model with a single
// target and `retries: 2` against an always-503 upstream makes three
// attempts; the two inter-retry backoffs have a guaranteed exponential floor
// (250ms then 500ms — additive jitter only, never full-jitter-to-zero), so
// the whole request must take at least ~750ms. Without backoff the three
// attempts complete in single-digit milliseconds.
//
// The assertion is a LOWER bound on elapsed time, which the exponential
// floor makes non-flaky.
//
// The same router is exercised twice — non-streaming and `stream: true` —
// because the two dispatch loops are written separately in chat.rs and the
// streaming one used to ignore `retries` outright (AISIX-Cloud#1119).

const CALLER_PLAINTEXT = "sk-retry-backoff-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

// 250ms (retry 1) + 500ms (retry 2) exponential floor, minus a safety margin
// so jitter/scheduling noise can't push a correct run under the threshold.
const MIN_EXPECTED_MS = 600;

describe("retry backoff e2e: same-target retries wait before re-hitting upstream", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // Every request to this upstream returns a retryable 503.
    upstream = await startOpenAiUpstream({
      status: 503,
      errorBody: { error: { message: "always down", type: "server_error" } },
    });
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "retry-backoff-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "retry-backoff-target",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    // Single-target router with retries=2: the same target is attempted
    // three times, so the two inter-retry backoffs are the only delay.
    await seed.createModel({
      display_name: "retry-backoff-router",
      routing: {
        strategy: "failover",
        targets: [{ model: "retry-backoff-target" }],
        retries: 2,
      },
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["retry-backoff-router"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("retries=2 against an always-503 upstream waits for the backoff floor", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // Wait until the router is active: a probe actually reaches the upstream
    // (status-agnostic — before the router/model/key propagate, the probe
    // 404s at the gateway and never hits the upstream).
    await waitConfigPropagation(async () => {
      const before = upstream!.receivedRequests.length;
      try {
        await client.chat.completions.create({
          model: "retry-backoff-router",
          messages: [{ role: "user", content: "probe" }],
        });
      } catch {
        // expected: upstream is always 503.
      }
      return upstream!.receivedRequests.length > before;
    });

    const hitsBefore = upstream.receivedRequests.length;
    const start = Date.now();
    let caught: unknown;
    try {
      await client.chat.completions.create({
        model: "retry-backoff-router",
        messages: [{ role: "user", content: "drive the retries" }],
      });
    } catch (e) {
      caught = e;
    }
    const elapsed = Date.now() - start;

    // The request exhausts its retries and fails.
    expect(caught).toBeInstanceOf(APIError);
    // Three attempts to the single target (initial + 2 retries).
    expect(upstream.receivedRequests.length - hitsBefore).toBe(3);
    // ...and the two inter-retry backoffs make it take at least the floor.
    expect(elapsed).toBeGreaterThanOrEqual(MIN_EXPECTED_MS);
  });

  // AISIX-Cloud#1119: the streaming dispatch loop walked the target list
  // once and never read `routing.retries`, so a retryable failure went
  // straight to fail-over — and with a single target (this router) the
  // request just failed. Same router, same upstream, `stream: true`: the
  // attempt count and the backoff floor must match the non-streaming case.
  test("retries=2 applies to streaming requests too", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    await waitConfigPropagation(async () => {
      const before = upstream!.receivedRequests.length;
      try {
        await client.chat.completions.create({
          model: "retry-backoff-router",
          messages: [{ role: "user", content: "probe" }],
        });
      } catch {
        // expected: upstream is always 503.
      }
      return upstream!.receivedRequests.length > before;
    });

    const hitsBefore = upstream.receivedRequests.length;
    const start = Date.now();
    let caught: unknown;
    try {
      const stream = await client.chat.completions.create({
        model: "retry-backoff-router",
        messages: [{ role: "user", content: "drive the streaming retries" }],
        stream: true,
      });
      for await (const _chunk of stream) {
        // Drain; the upstream never gets far enough to emit one.
      }
    } catch (e) {
      caught = e;
    }
    const elapsed = Date.now() - start;

    expect(caught).toBeInstanceOf(APIError);
    // Pre-fix this was 1: streaming attempted the single target once.
    expect(upstream.receivedRequests.length - hitsBefore).toBe(3);
    expect(elapsed).toBeGreaterThanOrEqual(MIN_EXPECTED_MS);
  });
});
