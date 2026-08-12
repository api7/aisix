import { createHash } from "node:crypto";
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

// E2E for #771: `window: "day"` on a RateLimitPolicy. Day was the one
// window whose token counter (`tpd`) is native end to end but that
// `PolicyWindow` did not offer — so a per-day token quota on a TEAM
// (the only scope kind policies can target and keys cannot) was
// inexpressible. Pre-#771 a policy row with `window: "day"` failed enum
// deserialization and the whole row was dropped, so both cases below
// answered 200 on every call.
//
//   1. team scope × day window × max_tokens: two keys on one team share
//      the day bucket — the first call commits its usage, the second
//      key's call is rejected once the pool is exhausted.
//   2. day window × max_requests maps to the native rpd counter (no
//      upscaling): the second request inside the same day is 429.

const KEY_A_PLAINTEXT = "sk-day-window-a";
const KEY_B_PLAINTEXT = "sk-day-window-b";
const KEY_C_PLAINTEXT = "sk-day-window-c";
const SENTINEL_PLAINTEXT = "sk-day-window-sentinel";
const TEAM_ID = "team-day-e2e";

// Upstream-reported usage per call: 16 tokens > the 10-token day pool,
// so ONE call exhausts it for the whole team.
const CHAT_BODY = {
  id: "chatcmpl-mock",
  object: "chat.completion",
  created: 0,
  model: "gpt-4o-mini",
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: "hi" },
      finish_reason: "stop",
    },
  ],
  usage: { prompt_tokens: 8, completion_tokens: 8, total_tokens: 16 },
};

/** Day buckets roll at UTC midnight; a burst that starts seconds before
 * it would split across two buckets and the 429 assertions would flap. */
async function awaitDayWindowHeadroom(headroomSecs = 15): Promise<void> {
  const secsLeft = 86_400 - (Math.floor(Date.now() / 1000) % 86_400);
  if (secsLeft >= headroomSecs) return;
  await new Promise((r) => setTimeout(r, secsLeft * 1000 + 100));
}

describe("rate limit e2e: day window (#771)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({ nonStreamBody: CHAT_BODY });
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "day-window-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "day-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });

    // The team policy precedes every key so any key authenticating
    // proves it applied (watch events apply in revision order).
    await seed.createRateLimitPolicy({
      name: "team-day-tokens",
      scope: "team",
      scope_ref: TEAM_ID,
      window: "day",
      max_tokens: 10,
    });

    // Keys A and B pool on one team; the day-token policy targets it.
    // (The standalone Admin API accepts team_id directly — the CP writes
    // it when managed.)
    for (const [plaintext, user] of [
      [KEY_A_PLAINTEXT, "user-a"],
      [KEY_B_PLAINTEXT, "user-b"],
    ] as const) {
      await seed.createApiKey({
        key_hash: createHash("sha256").update(plaintext).digest("hex"),
        allowed_models: ["day-model"],
        team_id: TEAM_ID,
        user_id: user,
      });
    }

    // Key C is teamless; its own day policy caps requests, not tokens.
    // (api_key scope matches on the resource entry id, so the policy
    // must follow the key.)
    const keyC = await seed.createApiKey({
      key_hash: createHash("sha256").update(KEY_C_PLAINTEXT).digest("hex"),
      allowed_models: ["day-model"],
    });
    await seed.createRateLimitPolicy({
      name: "key-day-requests",
      scope: "api_key",
      scope_ref: keyC.id,
      window: "day",
      max_requests: 1,
    });

    // Readiness sentinel seeded LAST: once it authenticates, every seed
    // above — both policies included — is live on the DP snapshot.
    await seed.createApiKey({
      key_hash: createHash("sha256").update(SENTINEL_PLAINTEXT).digest("hex"),
      allowed_models: [],
    });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${SENTINEL_PLAINTEXT}` },
      });
      await res.text();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  async function chat(plaintext: string): Promise<Response> {
    return fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${plaintext}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "day-model",
        messages: [{ role: "user", content: "hi" }],
      }),
    });
  }

  test("team day token pool: one member's usage exhausts it for the next", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    // Propagation is gated in beforeAll on the last-seeded sentinel key.
    await awaitDayWindowHeadroom();

    // Key A succeeds and commits 16 tokens against the team's 10/day.
    const first = await chat(KEY_A_PLAINTEXT);
    expect(first.status).toBe(200);

    // Key B is a DIFFERENT key on the SAME team: the pool is shared, so
    // the counter (16 >= 10) rejects it. Pre-#771 the policy row was
    // dropped at load and this returned 200.
    const second = await chat(KEY_B_PLAINTEXT);
    expect(second.status).toBe(429);
  });

  test("day request cap maps to the native rpd counter", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    await awaitDayWindowHeadroom();

    const first = await chat(KEY_C_PLAINTEXT);
    expect(first.status).toBe(200);

    const second = await chat(KEY_C_PLAINTEXT);
    expect(second.status).toBe(429);
    // Retry-After stays within one day — above 86400 is a unit
    // confusion, 0 tells SDKs to hammer.
    const retryAfter = Number.parseInt(
      second.headers.get("retry-after") ?? "0",
      10,
    );
    expect(retryAfter).toBeGreaterThan(0);
    expect(retryAfter).toBeLessThanOrEqual(86_400);
  });
});
