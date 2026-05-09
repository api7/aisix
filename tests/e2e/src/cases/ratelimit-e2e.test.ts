import { createHash } from "node:crypto";
import OpenAI, { APIError } from "openai";
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

// E2E: per-model RPM=1 rate limit. The first chat completion in a
// minute window succeeds; the second surfaces to the OpenAI SDK as
// `APIError` with `.status === 429`. Pinned end-to-end (real binary,
// real etcd watch, real OpenAI SDK with auto-retry disabled) — the
// existing in-process `rate_limit_rpm_returns_429_with_retry_after_header`
// covers the unit-level path; this case ensures the wire contract
// holds for a real SDK client.
//
// Reference: OpenAI Chat Completions API spec
// (https://platform.openai.com/docs/api-reference/chat/create); the
// gateway's RateLimit schema lives at
// `crates/aisix-core/src/models/rate_limit.rs`.

const CALLER_PLAINTEXT = "sk-rl-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("rate limit e2e: RPM=1 second call gets 429", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "rl-e2e-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "rl-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    // Rate limit is per-ApiKey here (matching the unit-level
    // `seed_snapshot_with_limits` pattern). RPM=1 means the first
    // call inside a 60s window succeeds; the second is rejected.
    await admin.createApiKey({
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

    // First call burns the only allowed slot.
    const ok = await client.chat.completions.create({
      model: "rl-e2e",
      messages: [{ role: "user", content: "first" }],
    });
    expect(ok.choices[0]?.message.role).toBe("assistant");

    // Second call within the minute → APIError with status 429 AND a
    // populated `Retry-After` header. The header is the load-bearing
    // contract for SDK back-off (in-process counterpart is named
    // `rate_limit_rpm_returns_429_with_retry_after_header` for a
    // reason — drop the assertion and the test would still pass on a
    // gateway that returned 429 with no `Retry-After`, breaking every
    // SDK that relies on it for exponential back-off.
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
    expect(Number.parseInt(String(retryAfter), 10)).toBeGreaterThan(0);
  });
});
