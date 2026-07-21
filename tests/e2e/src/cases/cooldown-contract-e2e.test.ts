import { createHash } from "node:crypto";
import OpenAI from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

/**
 * E2E contract tests for the post-#268 cooldown / filter behavior.
 *
 * These tests cover the audit findings from issue #264 that the
 * initial PR landed:
 *  - H1: 401 (auth failure) cools down even though it is non-retryable.
 *  - H2: Retry-After header from upstream drives the cooldown TTL.
 *  - H3: When every routing candidate is filtered out, the proxy
 *        fails fast with 503 + Retry-After (default policy).
 *  - H3 escape hatch: `when_all_unavailable: try_anyway` preserves the
 *        legacy "send to known-bad" behavior for operators that
 *        explicitly opt in.
 *  - M1: 429 cools down regardless of `retry_on_429` — cooldown and
 *        retry are independent layers.
 *
 * Every test exercises the real backend with mock upstreams (no
 * stubbing of the bridge or network layer) and asserts both client-
 * visible outcome and per-upstream request evidence.
 */

const CALLER_PLAINTEXT = "sk-cooldown-contract-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("cooldown contract (H1) — 401 cools down despite being non-retryable", () => {
  let app: SpawnedApp | undefined;
  let admin: AdminClient | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  let authFailUpstream: OpenAiUpstream | undefined;
  let stableUpstream: OpenAiUpstream | undefined;
  let authFailModelID = "";

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    authFailUpstream = await startOpenAiUpstream({
      status: 401,
      errorBody: {
        error: { message: "Incorrect API key provided", type: "invalid_request_error" },
      },
    });
    stableUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-h1-stable",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "h1 stable fallback" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    });

    // Admin listener off; `admin` reads only /status/models on the metrics
    // listener, and every resource is seeded to etcd via `seed`.
    app = await spawnApp({ admin: false });
    admin = new AdminClient(app.adminUrl, app.adminKey, app.metricsUrl);
    seed = new SeedClient(etcd, app.etcdPrefix);

    const failPk = await seed.createProviderKey({
      display_name: "h1-401-pk",
      secret: "sk-mock",
      api_base: `${authFailUpstream.baseUrl}/v1`,
    });
    const stablePk = await seed.createProviderKey({
      display_name: "h1-stable-pk",
      secret: "sk-mock",
      api_base: `${stableUpstream.baseUrl}/v1`,
    });

    authFailModelID = (
      await seed.createModel({
        display_name: "h1-401-model",
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: failPk.id,
      })
    ).id;
    await seed.createModel({
      display_name: "h1-stable-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: stablePk.id,
    });
    await seed.createModel({
      display_name: "h1-router",
      routing: {
        strategy: "failover",
        targets: [{ model: "h1-401-model" }, { model: "h1-stable-model" }],
        max_fallbacks: 1,
      },
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["h1-router", "h1-stable-model"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await authFailUpstream?.close();
    await stableUpstream?.close();
  });

  test("a 401 surfaces as 401 to the caller (non-retryable) AND cools down the target; next request skips the cooled target", async (ctx) => {
    if (!etcdReachable || !app || !admin || !authFailUpstream || !stableUpstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "h1-stable-model",
          messages: [{ role: "user", content: "ready-h1-stable" }],
        });
        return probe.choices[0]?.message.content === "h1 stable fallback";
      } catch {
        return false;
      }
    });

    // First request: 401 is non-retryable, so the dispatch loop
    // does NOT failover within this request — the 401 propagates
    // to the caller. This matches PR #263's retry semantics where
    // non-retryable 4xx stops the loop entirely. The new behavior
    // is purely that COOLDOWN is recorded anyway.
    let firstError: unknown;
    try {
      await client.chat.completions.create({
        model: "h1-router",
        messages: [{ role: "user", content: "trip 401 cooldown" }],
      });
    } catch (err) {
      firstError = err;
    }
    expect(firstError).toBeTruthy();
    expect((firstError as { status?: number }).status).toBe(401);
    expect(authFailUpstream.receivedRequests.length).toBeGreaterThanOrEqual(1);

    // The H1 contract: even though 401 is non-retryable, cooldown
    // was set on the failing target.
    const statuses = await admin.listModelStatuses();
    const failed = statuses.find((row) => row.id === authFailModelID)!;
    expect(failed.status).toBe("cooldown");
    expect(failed.status_reason).toBe("upstream_auth_failure");
    expect(failed.cooldown_until).toBeTruthy();

    // Second request: routing filter sees the 401 target is in
    // cooldown, picks the stable fallback as the first attempt.
    const failBaseline = authFailUpstream.receivedRequests.length;
    const stableBaseline = stableUpstream.receivedRequests.length;

    const second = await client.chat.completions.create({
      model: "h1-router",
      messages: [{ role: "user", content: "skip cooled-down 401 target" }],
    });
    expect(second.choices[0]?.message.content).toBe("h1 stable fallback");
    expect(authFailUpstream.receivedRequests.length - failBaseline).toBe(0);
    expect(stableUpstream.receivedRequests.length - stableBaseline).toBe(1);
  });
});

describe("cooldown contract (M1) — 429 cools down even when retry_on_429=false", () => {
  let app: SpawnedApp | undefined;
  let admin: AdminClient | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  let rateLimitedUpstream: OpenAiUpstream | undefined;
  let stableUpstream: OpenAiUpstream | undefined;
  let rateLimitedModelID = "";

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    rateLimitedUpstream = await startOpenAiUpstream({
      status: 429,
      errorBody: {
        error: { message: "Rate limit reached", type: "rate_limit_error" },
      },
    });
    stableUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-m1-stable",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "m1 stable fallback" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    });

    // Admin listener off; `admin` reads only /status/models on the metrics
    // listener, and every resource is seeded to etcd via `seed`.
    app = await spawnApp({ admin: false });
    admin = new AdminClient(app.adminUrl, app.adminKey, app.metricsUrl);
    seed = new SeedClient(etcd, app.etcdPrefix);

    const rateLimitedPk = await seed.createProviderKey({
      display_name: "m1-429-pk",
      secret: "sk-mock",
      api_base: `${rateLimitedUpstream.baseUrl}/v1`,
    });
    const stablePk = await seed.createProviderKey({
      display_name: "m1-stable-pk",
      secret: "sk-mock",
      api_base: `${stableUpstream.baseUrl}/v1`,
    });

    rateLimitedModelID = (
      await seed.createModel({
        display_name: "m1-429-model",
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: rateLimitedPk.id,
      })
    ).id;
    await seed.createModel({
      display_name: "m1-stable-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: stablePk.id,
    });
    // Note: retry_on_429 defaults to false here. The pre-#268-fix
    // behavior would NOT cooldown a 429 in this configuration; the
    // M1 fix decouples cooldown from retry so the 429 must still
    // mark the target for backoff.
    await seed.createModel({
      display_name: "m1-router",
      routing: {
        strategy: "failover",
        targets: [{ model: "m1-429-model" }, { model: "m1-stable-model" }],
        max_fallbacks: 1,
      },
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["m1-router", "m1-stable-model"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await rateLimitedUpstream?.close();
    await stableUpstream?.close();
  });

  test("429 surfaces to caller (retry_on_429=false → non-retryable) AND cools down; next request skips the cooled target", async (ctx) => {
    if (!etcdReachable || !app || !admin || !rateLimitedUpstream || !stableUpstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "m1-stable-model",
          messages: [{ role: "user", content: "ready-m1-stable" }],
        });
        return probe.choices[0]?.message.content === "m1 stable fallback";
      } catch {
        return false;
      }
    });

    // First request: with retry_on_429 unset (default false), the
    // 429 is non-retryable so the dispatch loop does not failover
    // — the 429 propagates to the caller. This is the same
    // retry-vs-cooldown decoupling the M1 contract pins.
    let firstError: unknown;
    try {
      await client.chat.completions.create({
        model: "m1-router",
        messages: [{ role: "user", content: "trip 429 cooldown" }],
      });
    } catch (err) {
      firstError = err;
    }
    expect(firstError).toBeTruthy();
    expect((firstError as { status?: number }).status).toBe(429);
    expect(rateLimitedUpstream.receivedRequests.length).toBeGreaterThanOrEqual(1);

    // The M1 contract: 429 cools down regardless of retry_on_429.
    const statuses = await admin.listModelStatuses();
    const rateLimited = statuses.find((row) => row.id === rateLimitedModelID)!;
    expect(rateLimited.status).toBe("cooldown");
    expect(rateLimited.status_reason).toBe("upstream_rate_limited");

    // Second request: routing filter skips the cooled target.
    const rateLimitedBaseline = rateLimitedUpstream.receivedRequests.length;
    const stableBaseline = stableUpstream.receivedRequests.length;

    const second = await client.chat.completions.create({
      model: "m1-router",
      messages: [{ role: "user", content: "skip 429 target after cooldown" }],
    });
    expect(second.choices[0]?.message.content).toBe("m1 stable fallback");
    expect(rateLimitedUpstream.receivedRequests.length - rateLimitedBaseline).toBe(0);
    expect(stableUpstream.receivedRequests.length - stableBaseline).toBe(1);
  });
});

describe("cooldown contract (H2) — Retry-After header from upstream drives TTL", () => {
  let app: SpawnedApp | undefined;
  let admin: AdminClient | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  let upstream: OpenAiUpstream | undefined;
  let stableUpstream: OpenAiUpstream | undefined;
  let rateLimitedModelID = "";

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      status: 429,
      errorBody: { error: { message: "slow down", type: "rate_limit_error" } },
      // Force the upstream to advertise a long Retry-After. The
      // gateway must honor it instead of using the default 30s.
      // Test asserts cooldown_until lands well beyond the default.
      responseHeaders: { "retry-after": "180" },
    });
    stableUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-h2-stable",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "h2 stable fallback" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    });

    // Admin listener off; `admin` reads only /status/models on the metrics
    // listener, and every resource is seeded to etcd via `seed`.
    app = await spawnApp({ admin: false });
    admin = new AdminClient(app.adminUrl, app.adminKey, app.metricsUrl);
    seed = new SeedClient(etcd, app.etcdPrefix);

    const upstreamPk = await seed.createProviderKey({
      display_name: "h2-upstream-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    const stablePk = await seed.createProviderKey({
      display_name: "h2-stable-pk",
      secret: "sk-mock",
      api_base: `${stableUpstream.baseUrl}/v1`,
    });

    rateLimitedModelID = (
      await seed.createModel({
        display_name: "h2-429-with-retry-after",
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: upstreamPk.id,
      })
    ).id;
    await seed.createModel({
      display_name: "h2-stable-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: stablePk.id,
    });
    await seed.createModel({
      display_name: "h2-router",
      routing: {
        strategy: "failover",
        targets: [
          { model: "h2-429-with-retry-after" },
          { model: "h2-stable-model" },
        ],
        max_fallbacks: 1,
      },
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["h2-router", "h2-stable-model"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await stableUpstream?.close();
  });

  test("Retry-After: 180 from upstream produces a cooldown_until far beyond the 30s default", async (ctx) => {
    if (!etcdReachable || !app || !admin || !upstream || !stableUpstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "h2-stable-model",
          messages: [{ role: "user", content: "ready-h2-stable" }],
        });
        return probe.choices[0]?.message.content === "h2 stable fallback";
      } catch {
        return false;
      }
    });

    const before = Date.now();
    // 429 with retry_on_429 unset (default false) → non-retryable
    // → propagates to caller. We don't care about the response
    // path here, only about the cooldown TTL the gateway derived
    // from the upstream's Retry-After header.
    let firstError: unknown;
    try {
      await client.chat.completions.create({
        model: "h2-router",
        messages: [{ role: "user", content: "trip 429 retry-after" }],
      });
    } catch (err) {
      firstError = err;
    }
    expect((firstError as { status?: number }).status).toBe(429);

    const statuses = await admin.listModelStatuses();
    const row = statuses.find((r) => r.id === rateLimitedModelID)!;
    expect(row.status).toBe("cooldown");
    expect(row.cooldown_until).toBeTruthy();

    // The reported cooldown_until should be ~180s in the future.
    // 30s default is the wrong answer — that would prove H2 isn't
    // wired. `cooldown_until` is serialized via SystemTime's default
    // serde shape: `{secs_since_epoch, nanos_since_epoch}`. Assert
    // the wire shape first so a future change to the admin
    // serialization (e.g. RFC3339 strings) fails this test with a
    // clear message rather than a silent NaN parse.
    const cooldownUntil = row.cooldown_until as { secs_since_epoch?: unknown };
    expect(
      typeof cooldownUntil.secs_since_epoch,
      "cooldown_until shape changed — update this test to match the new admin wire format",
    ).toBe("number");
    const cooldownUntilMs = (cooldownUntil.secs_since_epoch as number) * 1000;
    const horizonMs = cooldownUntilMs - before;
    expect(horizonMs).toBeGreaterThan(120_000);
    expect(horizonMs).toBeLessThan(240_000);
  });
});

describe("filter contract (H3) — all candidates unhealthy returns 503", () => {
  let app: SpawnedApp | undefined;
  let admin: AdminClient | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  let downAUpstream: OpenAiUpstream | undefined;
  let downBUpstream: OpenAiUpstream | undefined;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    downAUpstream = await startOpenAiUpstream({
      status: 503,
      errorBody: { error: { message: "all-down A", type: "server_error" } },
    });
    downBUpstream = await startOpenAiUpstream({
      status: 503,
      errorBody: { error: { message: "all-down B", type: "server_error" } },
    });

    // Admin listener off; `admin` reads only /status/models on the metrics
    // listener, and every resource is seeded to etcd via `seed`.
    app = await spawnApp({ admin: false });
    admin = new AdminClient(app.adminUrl, app.adminKey, app.metricsUrl);
    seed = new SeedClient(etcd, app.etcdPrefix);

    const aPk = await seed.createProviderKey({
      display_name: "h3-down-a-pk",
      secret: "sk-mock",
      api_base: `${downAUpstream.baseUrl}/v1`,
    });
    const bPk = await seed.createProviderKey({
      display_name: "h3-down-b-pk",
      secret: "sk-mock",
      api_base: `${downBUpstream.baseUrl}/v1`,
    });

    await seed.createModel({
      display_name: "h3-down-a",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: aPk.id,
      background_model_check: {
        enabled: true,
        interval_seconds: 5,
        timeout_seconds: 10,
        prompt: "Respond with OK",
        max_tokens: 8,
        ignore_statuses: [408, 429],
        stale_after_seconds: 120,
      },
    });
    await seed.createModel({
      display_name: "h3-down-b",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: bPk.id,
      background_model_check: {
        enabled: true,
        interval_seconds: 5,
        timeout_seconds: 10,
        prompt: "Respond with OK",
        max_tokens: 8,
        ignore_statuses: [408, 429],
        stale_after_seconds: 120,
      },
    });
    // Default when_all_unavailable policy is "fail" — explicit here for
    // clarity even though it's the default.
    await seed.createModel({
      display_name: "h3-router-fail",
      routing: {
        strategy: "failover",
        targets: [{ model: "h3-down-a" }, { model: "h3-down-b" }],
        max_fallbacks: 1,
        when_all_unavailable: "fail",
      },
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["h3-router-fail"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await downAUpstream?.close();
    await downBUpstream?.close();
  });

  test("when both candidates are background-unhealthy, the proxy fails fast with 503 + Retry-After", async (ctx) => {
    if (!etcdReachable || !app || !admin || !downAUpstream || !downBUpstream) {
      ctx.skip();
      return;
    }

    // Wait for the background probe to mark both candidates unhealthy.
    await waitConfigPropagation(async () => {
      const statuses = await admin!.listModelStatuses();
      const a = statuses.find((row) => row.display_name === "h3-down-a");
      const b = statuses.find((row) => row.display_name === "h3-down-b");
      return a?.status === "unhealthy" && b?.status === "unhealthy";
    });

    const aBaseline = downAUpstream.receivedRequests.length;
    const bBaseline = downBUpstream.receivedRequests.length;

    // Raw fetch instead of the OpenAI SDK so we can inspect the
    // 503 response shape and Retry-After header directly. The SDK
    // would convert 503 to an APIError and obscure the headers.
    const resp = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
      },
      body: JSON.stringify({
        model: "h3-router-fail",
        messages: [{ role: "user", content: "should fail fast" }],
      }),
    });

    expect(resp.status).toBe(503);
    expect(resp.headers.get("retry-after")).toBeTruthy();
    const body = (await resp.json()) as { error?: { type?: string } };
    expect(body.error?.type).toBe("all_candidates_unavailable");

    // Crucially: the proxy must NOT have sent a request to either
    // upstream during the fail-fast path. This is the whole point
    // of H3 — don't pile on known-bad targets.
    expect(downAUpstream.receivedRequests.length - aBaseline).toBe(0);
    expect(downBUpstream.receivedRequests.length - bBaseline).toBe(0);
  });
});

describe("filter contract (H3 escape hatch) — try_anyway sends to known-bad", () => {
  let app: SpawnedApp | undefined;
  let admin: AdminClient | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  let downUpstream: OpenAiUpstream | undefined;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    downUpstream = await startOpenAiUpstream({
      status: 503,
      errorBody: { error: { message: "still down", type: "server_error" } },
    });

    // Admin listener off; `admin` reads only /status/models on the metrics
    // listener, and every resource is seeded to etcd via `seed`.
    app = await spawnApp({ admin: false });
    admin = new AdminClient(app.adminUrl, app.adminKey, app.metricsUrl);
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "h3-escape-pk",
      secret: "sk-mock",
      api_base: `${downUpstream.baseUrl}/v1`,
    });

    await seed.createModel({
      display_name: "h3-escape-down",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
      background_model_check: {
        enabled: true,
        interval_seconds: 5,
        timeout_seconds: 10,
        prompt: "Respond with OK",
        max_tokens: 8,
        ignore_statuses: [408, 429],
        stale_after_seconds: 120,
      },
    });
    await seed.createModel({
      display_name: "h3-router-escape",
      routing: {
        strategy: "failover",
        targets: [{ model: "h3-escape-down" }],
        max_fallbacks: 0,
        when_all_unavailable: "try_anyway",
      },
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["h3-router-escape"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await downUpstream?.close();
  });

  test("with when_all_unavailable=try_anyway, the proxy still tries the unhealthy target (legacy opt-in)", async (ctx) => {
    if (!etcdReachable || !app || !admin || !downUpstream) {
      ctx.skip();
      return;
    }

    await waitConfigPropagation(async () => {
      const statuses = await admin!.listModelStatuses();
      const row = statuses.find((r) => r.display_name === "h3-escape-down");
      return row?.status === "unhealthy";
    });

    const baseline = downUpstream.receivedRequests.length;

    // With try_anyway, the request goes out to the unhealthy
    // target. The upstream still returns 503 and the SDK throws,
    // but the key point is that a request *was sent*.
    const resp = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
      },
      body: JSON.stringify({
        model: "h3-router-escape",
        messages: [{ role: "user", content: "send anyway please" }],
      }),
    });

    // 503 surfaces from the upstream (collapsed to 502 by the bridge
    // mapping in BridgeError::http_status, but the upstream request
    // is what we're verifying here).
    expect([502, 503]).toContain(resp.status);
    expect(downUpstream.receivedRequests.length - baseline).toBe(1);
  });
});

describe("cooldown observability — a cooldown transition emits aisix_deployment_* metrics", () => {
  let app: SpawnedApp | undefined;
  let admin: AdminClient | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  let failUpstream: OpenAiUpstream | undefined;
  let stableUpstream: OpenAiUpstream | undefined;
  let failModelID = "";

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // 500 is a default cooldown trigger status → the failing target
    // cools down; the router then falls over to the stable target.
    failUpstream = await startOpenAiUpstream({
      status: 500,
      errorBody: { error: { message: "boom", type: "server_error" } },
    });
    stableUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-metrics-stable",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "metrics stable fallback" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    });

    // Admin listener off; `admin` reads only /status/models on the metrics
    // listener, and every resource is seeded to etcd via `seed`.
    app = await spawnApp({ admin: false });
    admin = new AdminClient(app.adminUrl, app.adminKey, app.metricsUrl);
    seed = new SeedClient(etcd, app.etcdPrefix);

    const failPk = await seed.createProviderKey({
      display_name: "cd-metrics-fail-pk",
      secret: "sk-mock",
      api_base: `${failUpstream.baseUrl}/v1`,
    });
    const stablePk = await seed.createProviderKey({
      display_name: "cd-metrics-stable-pk",
      secret: "sk-mock",
      api_base: `${stableUpstream.baseUrl}/v1`,
    });

    failModelID = (
      await seed.createModel({
        display_name: "cd-metrics-500-model",
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: failPk.id,
      })
    ).id;
    await seed.createModel({
      display_name: "cd-metrics-stable-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: stablePk.id,
    });
    await seed.createModel({
      display_name: "cd-metrics-router",
      routing: {
        strategy: "failover",
        targets: [
          { model: "cd-metrics-500-model" },
          { model: "cd-metrics-stable-model" },
        ],
        max_fallbacks: 1,
      },
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["cd-metrics-router", "cd-metrics-stable-model"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await failUpstream?.close();
    await stableUpstream?.close();
  });

  test("the deployment cooldown counter and state gauge appear on the scrape with the cooled target's labels", async (ctx) => {
    if (!etcdReachable || !app || !admin || !failUpstream || !stableUpstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "cd-metrics-stable-model",
          messages: [{ role: "user", content: "ready-metrics-stable" }],
        });
        return probe.choices[0]?.message.content === "metrics stable fallback";
      } catch {
        return false;
      }
    });

    // Trip the 500 target: it cools down (and the router fails over to
    // the stable target, so the caller still gets a 200).
    const res = await client.chat.completions.create({
      model: "cd-metrics-router",
      messages: [{ role: "user", content: "trip cooldown for metrics" }],
    });
    expect(res.choices[0]?.message.content).toBe("metrics stable fallback");

    // Confirm the cooldown actually landed before scraping.
    const statuses = await admin.listModelStatuses();
    expect(statuses.find((r) => r.id === failModelID)?.status).toBe("cooldown");

    const scrape = await fetch(`${app.metricsUrl}/metrics`);
    expect(scrape.status).toBe(200);
    const text = await scrape.text();

    // The cooldown counter is emitted for the cooled target, labelled by
    // its resolved deployment identity (not model-id-only "unknown").
    const cooledLine = text
      .split("\n")
      .find(
        (l) =>
          l.startsWith("aisix_deployment_cooled_down_total{") &&
          l.includes('model="cd-metrics-500-model"'),
      );
    expect(cooledLine, `cooldown counter missing:\n${text}`).toBeTruthy();
    expect(cooledLine).toContain('provider="openai"');
    expect(cooledLine).toContain('upstream_model="gpt-4o-mini"');
    expect(Number(cooledLine!.trim().split(/\s+/).pop())).toBeGreaterThanOrEqual(1);

    // The state gauge reflects the cooled target as out-of-rotation (Down=2).
    const stateLine = text
      .split("\n")
      .find(
        (l) =>
          l.startsWith("aisix_deployment_state{") &&
          l.includes('model="cd-metrics-500-model"'),
      );
    expect(stateLine, `deployment state gauge missing:\n${text}`).toBeTruthy();
    expect(Number(stateLine!.trim().split(/\s+/).pop())).toBe(2);
  });
});

/** Value of the `aisix_deployment_state` series carrying `model="<model>"`. */
async function scrapeDeploymentState(
  metricsUrl: string,
  model: string,
): Promise<number | undefined> {
  const text = await (await fetch(`${metricsUrl}/metrics`)).text();
  const line = text
    .split("\n")
    .find(
      (l) =>
        l.startsWith("aisix_deployment_state{") &&
        l.includes(`model="${model}"`),
    );
  return line ? Number(line.trim().split(/\s+/).pop()) : undefined;
}

describe("cooldown observability — the state gauge follows a target back into rotation", () => {
  let app: SpawnedApp | undefined;
  let admin: AdminClient | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  let flakyUpstream: OpenAiUpstream | undefined;
  let stableUpstream: OpenAiUpstream | undefined;
  let flakyModelID = "";

  const RECOVERED_REPLY = "recovered after cooldown";

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // First request 500s (trips the cooldown); every later request
    // succeeds, so the target is genuinely healthy again once the TTL
    // lapses. `scriptedResponses` runs before the static opts.
    flakyUpstream = await startOpenAiUpstream({
      scriptedResponses: [
        {
          status: 500,
          errorBody: { error: { message: "boom", type: "server_error" } },
        },
      ],
      nonStreamBody: {
        id: "cmpl-recovered",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: RECOVERED_REPLY },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    });
    stableUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-recover-stable",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "recover stable fallback" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    });

    // Admin listener off; `admin` reads only /status/models on the metrics
    // listener, and every resource is seeded to etcd via `seed`.
    app = await spawnApp({ admin: false });
    admin = new AdminClient(app.adminUrl, app.adminKey, app.metricsUrl);
    seed = new SeedClient(etcd, app.etcdPrefix);

    const flakyPk = await seed.createProviderKey({
      display_name: "cd-recover-flaky-pk",
      secret: "sk-mock",
      api_base: `${flakyUpstream.baseUrl}/v1`,
    });
    const stablePk = await seed.createProviderKey({
      display_name: "cd-recover-stable-pk",
      secret: "sk-mock",
      api_base: `${stableUpstream.baseUrl}/v1`,
    });

    flakyModelID = (
      await seed.createModel({
        display_name: "cd-recover-flaky-model",
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: flakyPk.id,
        // A 1s TTL keeps the test honest: it must observe the cooldown
        // lapsing on its own, which is what the router does in production.
        cooldown: { default_seconds: 1 },
      })
    ).id;
    await seed.createModel({
      display_name: "cd-recover-stable-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: stablePk.id,
    });
    await seed.createModel({
      display_name: "cd-recover-router",
      routing: {
        strategy: "failover",
        targets: [
          { model: "cd-recover-flaky-model" },
          { model: "cd-recover-stable-model" },
        ],
        max_fallbacks: 1,
      },
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [
        "cd-recover-router",
        "cd-recover-flaky-model",
        "cd-recover-stable-model",
      ],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await flakyUpstream?.close();
    await stableUpstream?.close();
  });

  // The router filters cooled targets out of rotation, so nothing touches
  // the tracker while the TTL runs down — the cooldown simply lapses. An
  // edge-triggered gauge has no transition to observe when the next success
  // finally arrives and pins the target at Down(2) for the rest of the
  // process's life. The gauge must instead follow the target's actual
  // serving state.
  test("after the cooldown lapses on its own, the next success drives the gauge back to Healthy(0)", async (ctx) => {
    if (!etcdReachable || !app || !admin || !flakyUpstream || !stableUpstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "cd-recover-stable-model",
          messages: [{ role: "user", content: "ready-recover-stable" }],
        });
        return probe.choices[0]?.message.content === "recover stable fallback";
      } catch {
        return false;
      }
    });

    // Trip the cooldown on the flaky target (router falls over → caller 200).
    const tripped = await client.chat.completions.create({
      model: "cd-recover-router",
      messages: [{ role: "user", content: "trip the cooldown" }],
    });
    expect(tripped.choices[0]?.message.content).toBe("recover stable fallback");

    expect(
      (await admin.listModelStatuses()).find((r) => r.id === flakyModelID)
        ?.status,
    ).toBe("cooldown");
    expect(
      await scrapeDeploymentState(app.metricsUrl, "cd-recover-flaky-model"),
    ).toBe(2);

    // Let the 1s TTL lapse. Nothing calls back into the tracker here — that
    // is the whole point.
    await expect
      .poll(
        async () =>
          (await admin!.listModelStatuses()).find((r) => r.id === flakyModelID)
            ?.status,
        { timeout: 10_000, interval: 200 },
      )
      .toBe("healthy");

    // The target is back in rotation and serving. The gauge must agree.
    const recovered = await client.chat.completions.create({
      model: "cd-recover-router",
      messages: [{ role: "user", content: "after the cooldown lapsed" }],
    });
    expect(recovered.choices[0]?.message.content).toBe(RECOVERED_REPLY);

    expect(
      await scrapeDeploymentState(app.metricsUrl, "cd-recover-flaky-model"),
      "the recovered target must not stay pinned at Down(2)",
    ).toBe(0);
  });
});
