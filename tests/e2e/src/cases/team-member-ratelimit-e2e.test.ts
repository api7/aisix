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

// E2E: a `team_member` rate-limit policy is a per-member DEFAULT for a
// team — it matches every key whose `team_id` equals the policy's
// `scope_ref`, but each member counts against an INDEPENDENT bucket
// keyed by the API key's `user_id`. Contrast with `team` scope, which
// pools one bucket across the whole team.
//
// With rpm=1 we assert three things:
//   1. member A's 2nd call → 429 (their own bucket is exhausted)
//   2. member B's 1st call → 200 (independent bucket; A doesn't throttle B)
//   3. a SECOND key owned by member A → 429 (the bucket is per user_id,
//      not per API key, so A can't dodge the cap by minting more keys)

const TEAM_ID = "team-tm-e2e";
const POLICY_ID = "11111111-1111-1111-1111-111111111111";

const KEY_A1 = "sk-tm-a1";
const KEY_A2 = "sk-tm-a2";
const KEY_B = "sk-tm-b";

const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

describe("rate limit e2e: team_member per-member default buckets", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    const admin = new AdminClient(app.adminUrl, app.adminKey);
    const etcd = new EtcdClient();

    // Seed the policy FIRST (lower etcd revision) so that once the
    // probe sees the model (created below, higher revision) the policy
    // is guaranteed already applied — events arrive in revision order.
    await etcd.put(
      `${app.etcdPrefix}/rate_limit_policies/${POLICY_ID}`,
      JSON.stringify({
        name: "tm-default",
        scope: "team_member",
        scope_ref: TEAM_ID,
        window: "minute",
        max_requests: 1,
      }),
    );

    const pk = await admin.createProviderKey({
      display_name: "tm-e2e-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "tm-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    // Members A and B both belong to TEAM_ID; A owns two keys. The
    // standalone Admin API omits team_id/user_id (CP writes those in
    // production), so seed the keys straight to etcd with the full shape.
    const seedKey = (id: string, plaintext: string, userId: string) =>
      etcd.put(
        `${app!.etcdPrefix}/api_keys/${id}`,
        JSON.stringify({
          key_hash: sha256(plaintext),
          allowed_models: ["tm-e2e"],
          team_id: TEAM_ID,
          user_id: userId,
        }),
      );
    await seedKey("a0000000-0000-0000-0000-000000000001", KEY_A1, "user-a");
    await seedKey("a0000000-0000-0000-0000-000000000002", KEY_A2, "user-a");
    await seedKey("b0000000-0000-0000-0000-000000000001", KEY_B, "user-b");
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("per-member buckets are independent and per-user, not per-key", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    // listModels doesn't consume an rpm slot — safe readiness probe.
    const probe = new ProxyClient(app.proxyUrl, KEY_A1);
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      if (res.status !== 200) return false;
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return data.some((m) => m.id === "tm-e2e");
    });

    const chat = (apiKey: string) =>
      new OpenAI({ apiKey, baseURL: `${app!.proxyUrl}/v1`, maxRetries: 0 });

    const callStatus = async (apiKey: string): Promise<number> => {
      try {
        await chat(apiKey).chat.completions.create({
          model: "tm-e2e",
          messages: [{ role: "user", content: "hi" }],
        });
        return 200;
      } catch (e) {
        if (e instanceof APIError) return e.status ?? -1;
        throw e;
      }
    };

    // Member A burns their single slot, then is throttled.
    expect(await callStatus(KEY_A1)).toBe(200);
    expect(await callStatus(KEY_A1)).toBe(429);

    // Member B is on the same team + same policy but a separate bucket,
    // so their first call still succeeds — the per-member isolation.
    expect(await callStatus(KEY_B)).toBe(200);

    // A second key owned by member A shares A's (already-exhausted)
    // bucket → 429. Proves the counter keys on user_id, not key id.
    expect(await callStatus(KEY_A2)).toBe(429);
  });
});
