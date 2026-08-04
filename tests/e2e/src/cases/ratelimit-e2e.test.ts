import { createHash } from "node:crypto";
import OpenAI, { APIError } from "openai";
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

// E2E: per-ApiKey RPM=1 rate limit. The first chat completion in a
// minute window succeeds; the second surfaces to the OpenAI SDK as
// `APIError` with `.status === 429` and a populated `Retry-After`
// header (the load-bearing contract for SDK exponential back-off).
//
// Reference: OpenAI Chat Completions API spec
// (https://platform.openai.com/docs/api-reference/chat/create) and
// RFC 7231 §7.1.3 for the `Retry-After` header semantics
// (https://datatracker.ietf.org/doc/html/rfc7231#section-7.1.3).

const CALLER_PLAINTEXT = "sk-rl-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("rate limit e2e: RPM=1 second call gets 429", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "rl-e2e-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "rl-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    // Rate limit is per-ApiKey here (matching the unit-level
    // `seed_snapshot_with_limits` pattern). RPM=1 means the first
    // call inside a 60s window succeeds; the second is rejected.
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["rl-e2e"],
      rate_limit: { rpm: 1 },
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("second call within RPM=1 window surfaces as APIError 429", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    // Use ProxyClient.listModels as the readiness probe — it does not
    // consume the RPM=1 slot, leaving the test its full quota.
    const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      if (res.status !== 200) return false;
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return data.some((m) => m.id === "rl-e2e");
    });

    // maxRetries=0 keeps the SDK from silently retrying around the
    // 429 — without this, the test could falsely pass because the SDK
    // sleeps long enough for the next minute window to open.
    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // The limiter buckets on fixed wall-clock minutes, so a burst that
    // straddles a boundary gets a fresh allowance and the 429 assertion
    // below flaps. Keep the whole burst inside one window.
    await awaitWindowHeadroom();
    // First call burns the only allowed slot.
    const ok = await client.chat.completions.create({
      model: "rl-e2e",
      messages: [{ role: "user", content: "first" }],
    });
    expect(ok.choices[0]?.message.role).toBe("assistant");

    // Second call within the minute → APIError with status 429 AND a
    // populated `Retry-After` header. The header is the load-bearing
    // contract for SDK back-off — drop the assertion and the test
    // would still pass on a gateway that returned 429 with no
    // `Retry-After`, breaking every SDK that relies on it for
    // exponential back-off.
    let caught: unknown;
    try {
      await client.chat.completions.create({
        model: "rl-e2e",
        messages: [{ role: "user", content: "second" }],
      });
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(APIError);
    if (!(caught instanceof APIError)) {
      // Narrows TS; the expect above already failed if this hits.
      throw new Error("unreachable: caught is not APIError");
    }
    expect(caught.status).toBe(429);
    // OpenAI Node SDK 4.x's APIError.headers is a Proxy that lowercases
    // lookups (createResponseHeaders in core.js), so the lowercase form
    // is the canonical access path.
    const retryAfter = caught.headers?.["retry-after"];
    expect(retryAfter).toBeDefined();
    const retryAfterSeconds = Number.parseInt(String(retryAfter), 10);
    // RPM=1 means the window is at most 60 seconds; a value above
    // that is either a unit confusion or a wall-clock leak. Below 1
    // would tell the SDK to retry immediately, defeating the limit.
    expect(retryAfterSeconds).toBeGreaterThan(0);
    expect(retryAfterSeconds).toBeLessThanOrEqual(60);
  });
});

// E2E: RateLimitPolicy.schedules — recurring suspension windows
// (AISIX-Cloud#1104), driven through etcd exactly as cp-api writes
// them. Windows are picked relative to "now" so the test is
// deterministic without waiting for wall-clock boundaries:
// - an all-week 00:00–24:00 window is always active → suspended
// - a fixed past date (2000-01-01) never matches → enforced
// Also pins the upgrade contract: a pre-`schedules` row (field absent)
// enforces unchanged, and toggling schedules never touches the bucket,
// so the window's burned count survives a suspend/resume cycle.
const SCHED_CALLER = "sk-rlp-sched-e2e-caller";
const SCHED_KEY_ID = "c0000000-0000-0000-0000-000000000011";
const SCHED_POLICY_ID = "d0000000-0000-0000-0000-000000000011";

describe("rate limit policy schedules e2e (AISIX-Cloud#1104)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  const policyDoc = (
    schedules?: Array<Record<string, unknown>>,
  ): Record<string, unknown> => ({
    name: "sched-cap",
    scope: "api_key",
    scope_ref: SCHED_KEY_ID,
    window: "minute",
    max_requests: 1,
    // cp-api omits `schedules` when empty so schedule-less rows stay
    // parseable by pre-`schedules` strict data planes.
    ...(schedules ? { schedules } : {}),
  });

  const alwaysOn = [
    {
      timezone: "UTC",
      days_of_week: ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
      start_time: "00:00",
      end_time: "24:00",
    },
  ];
  const neverOn = [
    {
      timezone: "UTC",
      dates: ["2000-01-01"],
      start_time: "00:00",
      end_time: "24:00",
    },
  ];

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "rlp-sched-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "rlp-sched",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    // api_key scope matches on the key's etcd entry id, so the key
    // needs a fixed id — seed it straight to etcd.
    await etcd.put(
      `${app.etcdPrefix}/api_keys/${SCHED_KEY_ID}`,
      JSON.stringify({
        key_hash: createHash("sha256").update(SCHED_CALLER).digest("hex"),
        allowed_models: ["rlp-sched"],
      }),
    );
    // Start from the pre-`schedules` shape (field absent).
    await seed.update("rate_limit_policies", SCHED_POLICY_ID, policyDoc());
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("suspension window pauses the policy; leaving it resumes the same bucket", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    const probe = new ProxyClient(app.proxyUrl, SCHED_CALLER);
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      if (res.status !== 200) return false;
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return data.some((m) => m.id === "rlp-sched");
    });

    const client = new OpenAI({
      apiKey: SCHED_CALLER,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });
    const callStatus = async (): Promise<number> => {
      try {
        await client.chat.completions.create({
          model: "rlp-sched",
          messages: [{ role: "user", content: "hi" }],
        });
        return 200;
      } catch (e) {
        if (e instanceof APIError) return e.status ?? -1;
        throw e;
      }
    };

    // Keep the burn → suspend → resume sequence inside one fixed
    // minute window so the final 429 provably reuses the burned count.
    await awaitWindowHeadroom(30);

    // Pre-`schedules` row enforces: first call burns the single slot.
    expect(await callStatus()).toBe(200);
    expect(await callStatus()).toBe(429);

    // Enter a suspension window → propagation lands when a call passes
    // again. Suspended probes reserve nothing, so counts stay intact.
    await seed.update(
      "rate_limit_policies",
      SCHED_POLICY_ID,
      policyDoc(alwaysOn),
    );
    await waitConfigPropagation(async () => (await callStatus()) === 200);

    // Leave the window (schedule no longer matches). The bucket still
    // holds the burned slot from this minute, so enforcement resumes
    // as 429 — suspension must not reset quotas.
    await seed.update(
      "rate_limit_policies",
      SCHED_POLICY_ID,
      policyDoc(neverOn),
    );
    await waitConfigPropagation(async () => (await callStatus()) === 429);
  });
});
