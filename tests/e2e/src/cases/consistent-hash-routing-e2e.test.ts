import { createHash } from "node:crypto";
import OpenAI from "openai";
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

// E2E: `strategy: consistent_hash` + per-target `priority` tiers — the
// AISIX-Cloud#1206 two-pool shape. Contracts pinned here:
//
//   1. The same hash key keeps landing on the same target; distinct keys
//      spread across the tier (ketama ring, weight-scaled).
//   2. The `hash_on` chain is honored in order (cookie first here), with
//      the caller's API key as the configured fallback.
//   3. Priority tiers: the backup tier receives ZERO traffic while the
//      active tier has a healthy target; a fully-down active tier shifts
//      traffic to the backup within the SAME request (in-request spill);
//      an active target leaving cooldown takes its traffic back.
//   4. A single failed member redistributes within its own tier — never
//      to the backup tier.
//
// Reference: OpenAI Chat Completions shape the caller sees
// (https://platform.openai.com/docs/api-reference/chat).

const CALLER_PLAINTEXT = "sk-chash-routing-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

function okBody(content: string) {
  return {
    id: `cmpl-${content}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: "gpt-4o-mini",
    choices: [
      {
        index: 0,
        message: { role: "assistant", content },
        finish_reason: "stop",
      },
    ],
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
  };
}

const err503 = {
  status: 503,
  errorBody: { error: { message: "instance down", type: "server_error" } },
};

describe("consistent-hash routing + priority tiers e2e", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  async function createMember(
    displayName: string,
    upstream: OpenAiUpstream,
    extra: Record<string, unknown> = {},
  ): Promise<void> {
    if (!seed) throw new Error("seed client not initialized");
    const providerKey = await seed.createProviderKey({
      display_name: `${displayName}-pk`,
      api_key: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: displayName,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: providerKey.id,
      ...extra,
    });
  }

  function client(): OpenAI {
    return new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app?.proxyUrl}/v1`,
      maxRetries: 0,
    });
  }

  async function ask(
    model: string,
    headers: Record<string, string>,
  ): Promise<string | null> {
    const completion = await client().chat.completions.create(
      { model, messages: [{ role: "user", content: "hi" }] },
      { headers },
    );
    return completion.choices[0]?.message.content ?? null;
  }

  // Gate on the DP snapshot via /v1/models — authenticates only once the
  // caller key has propagated, lists the members only once the snapshot
  // has them, and dispatches to no target (which would warm cooldowns and
  // skew the per-target counts the assertions rely on).
  async function waitMembersVisible(members: string[]): Promise<void> {
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      if (res.status !== 200) return false;
      const ids =
        ((await res.json()) as { data?: Array<{ id?: string }> }).data?.map(
          (m) => m.id,
        ) ?? [];
      return members.every((m) => ids.includes(m));
    });
  }

  test("same key sticks to one target while distinct keys spread", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    const one = await startOpenAiUpstream({ nonStreamBody: okBody("one-served") });
    const two = await startOpenAiUpstream({ nonStreamBody: okBody("two-served") });
    upstreams.push(one, two);
    // Router BEFORE its targets: watch events apply in revision order, so
    // once /v1/models lists both targets the router is in the snapshot too.
    await seed.createModel({
      display_name: "ch-basic",
      routing: {
        strategy: "consistent_hash",
        targets: [
          { model: "ch-basic-one", weight: 50 },
          { model: "ch-basic-two", weight: 50 },
        ],
      },
    });
    await createMember("ch-basic-one", one);
    await createMember("ch-basic-two", two);
    await waitMembersVisible(["ch-basic-one", "ch-basic-two"]);

    // Same key → same target on every request.
    const repeated = await Promise.all(
      Array.from({ length: 6 }, () =>
        ask("ch-basic", { "x-aisix-routing-key": "user-A" }),
      ),
    );
    expect(new Set(repeated).size).toBe(1);

    // Distinct keys spread across both targets. Deterministic hashing
    // keeps this stable across runs.
    const served = new Set<string | null>();
    for (let i = 0; i < 32; i++) {
      served.add(await ask("ch-basic", { "x-aisix-routing-key": `user-${i}` }));
    }
    expect(served).toEqual(new Set(["one-served", "two-served"]));
  });

  test("hash_on chain reads the cookie first and falls back to the api key", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    const one = await startOpenAiUpstream({ nonStreamBody: okBody("ck-one") });
    const two = await startOpenAiUpstream({ nonStreamBody: okBody("ck-two") });
    upstreams.push(one, two);
    await seed.createModel({
      display_name: "ch-cookie",
      routing: {
        strategy: "consistent_hash",
        hash_on: [
          { type: "cookie", name: "sid" },
          { type: "api_key" },
        ],
        targets: [{ model: "ch-cookie-one" }, { model: "ch-cookie-two" }],
      },
    });
    await createMember("ch-cookie-one", one);
    await createMember("ch-cookie-two", two);
    await waitMembersVisible(["ch-cookie-one", "ch-cookie-two"]);

    // A cookie-keyed session is stable across requests…
    const viaCookie = await Promise.all(
      Array.from({ length: 5 }, () =>
        ask("ch-cookie", { cookie: "theme=dark; sid=sess-42" }),
      ),
    );
    expect(new Set(viaCookie).size).toBe(1);

    // …and distinct cookie values are what spreads the traffic — proving
    // the cookie (not the shared caller key) is the operative source.
    const spread = new Set<string | null>();
    for (let i = 0; i < 32; i++) {
      spread.add(await ask("ch-cookie", { cookie: `sid=sess-${i}` }));
    }
    expect(spread).toEqual(new Set(["ck-one", "ck-two"]));

    // Without the cookie every request falls back to the caller's API
    // key — one shared key, one consistent target.
    const viaApiKey = await Promise.all(
      Array.from({ length: 5 }, () => ask("ch-cookie", {})),
    );
    expect(new Set(viaApiKey).size).toBe(1);
  });

  test("backup tier idles while the active tier is healthy", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    const a1 = await startOpenAiUpstream({ nonStreamBody: okBody("a1-served") });
    const a2 = await startOpenAiUpstream({ nonStreamBody: okBody("a2-served") });
    const b1 = await startOpenAiUpstream({ nonStreamBody: okBody("b1-served") });
    upstreams.push(a1, a2, b1);
    await seed.createModel({
      display_name: "ch-pools",
      routing: {
        strategy: "consistent_hash",
        targets: [
          { model: "ch-pools-a1" },
          { model: "ch-pools-a2" },
          { model: "ch-pools-b1", priority: -1 },
        ],
      },
    });
    await createMember("ch-pools-a1", a1);
    await createMember("ch-pools-a2", a2);
    await createMember("ch-pools-b1", b1);
    await waitMembersVisible(["ch-pools-a1", "ch-pools-a2", "ch-pools-b1"]);

    const b1Baseline = b1.receivedRequests.length;
    const served = new Set<string | null>();
    for (let i = 0; i < 16; i++) {
      served.add(await ask("ch-pools", { "x-aisix-routing-key": `user-${i}` }));
    }
    // Every response came from the active tier, and the backup upstream
    // never saw a single request.
    expect([...served].every((s) => s === "a1-served" || s === "a2-served")).toBe(
      true,
    );
    expect(served.size).toBe(2); // both active members participate
    expect(b1.receivedRequests.length - b1Baseline).toBe(0);
  });

  test("a fully-down active tier spills to the backup tier within one request, with hash affinity there", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    const a1 = await startOpenAiUpstream(err503);
    const a2 = await startOpenAiUpstream(err503);
    const b1 = await startOpenAiUpstream({ nonStreamBody: okBody("bk-one") });
    const b2 = await startOpenAiUpstream({ nonStreamBody: okBody("bk-two") });
    upstreams.push(a1, a2, b1, b2);
    await seed.createModel({
      display_name: "ch-down",
      routing: {
        strategy: "consistent_hash",
        targets: [
          { model: "ch-down-a1" },
          { model: "ch-down-a2" },
          { model: "ch-down-b1", priority: -1 },
          { model: "ch-down-b2", priority: -1 },
        ],
      },
    });
    await createMember("ch-down-a1", a1);
    await createMember("ch-down-a2", a2);
    await createMember("ch-down-b1", b1);
    await createMember("ch-down-b2", b2);
    await waitMembersVisible([
      "ch-down-a1",
      "ch-down-a2",
      "ch-down-b1",
      "ch-down-b2",
    ]);

    // The FIRST request discovers both active members down and still
    // succeeds — the walk crosses the tier boundary inside one request.
    const first = await ask("ch-down", { "x-aisix-routing-key": "user-A" });
    expect(first === "bk-one" || first === "bk-two").toBe(true);
    expect(
      a1.receivedRequests.length + a2.receivedRequests.length,
    ).toBeGreaterThan(0);

    // While the active tier cools down, the same key stays on the same
    // backup target (hash affinity holds inside the backup tier too).
    const repeated = await Promise.all(
      Array.from({ length: 6 }, () =>
        ask("ch-down", { "x-aisix-routing-key": "user-A" }),
      ),
    );
    expect(new Set(repeated)).toEqual(new Set([first]));

    // Distinct keys spread across BOTH backup members.
    const served = new Set<string | null>();
    for (let i = 0; i < 32; i++) {
      served.add(await ask("ch-down", { "x-aisix-routing-key": `u-${i}` }));
    }
    expect(served).toEqual(new Set(["bk-one", "bk-two"]));
  });

  test("an active target leaving cooldown takes its traffic back from the backup", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    // The active member fails exactly once, then serves; its cooldown is
    // 1s so recovery is observable within the test.
    const a1 = await startOpenAiUpstream({
      scriptedResponses: [{ ...err503 }],
      nonStreamBody: okBody("active-served"),
    });
    const b1 = await startOpenAiUpstream({ nonStreamBody: okBody("backup-served") });
    upstreams.push(a1, b1);
    await seed.createModel({
      display_name: "ch-recover",
      routing: {
        strategy: "consistent_hash",
        targets: [
          { model: "ch-recover-a1" },
          { model: "ch-recover-b1", priority: -1 },
        ],
      },
    });
    await createMember("ch-recover-a1", a1, {
      cooldown: { default_seconds: 1 },
    });
    await createMember("ch-recover-b1", b1);
    await waitMembersVisible(["ch-recover-a1", "ch-recover-b1"]);

    // 1st request: the active member fails (scripted 503), the backup
    // absorbs it in-request.
    expect(await ask("ch-recover", { "x-aisix-routing-key": "s1" })).toBe(
      "backup-served",
    );
    const a1AfterFailure = a1.receivedRequests.length;

    // While the active member cools down, traffic goes STRAIGHT to the
    // backup — the cooled member is not even attempted.
    expect(await ask("ch-recover", { "x-aisix-routing-key": "s1" })).toBe(
      "backup-served",
    );
    expect(a1.receivedRequests.length).toBe(a1AfterFailure);

    // After the 1s cooldown expires the recovered member takes back its
    // traffic (poll rather than sleep a fixed amount — config watches and
    // timers make exact timing environment-dependent).
    await waitConfigPropagation(async () => {
      return (await ask("ch-recover", { "x-aisix-routing-key": "s1" })) ===
        "active-served";
    });
  });

  test("a single failed member redistributes within its tier, never to the backup", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    const a1 = await startOpenAiUpstream(err503);
    const a2 = await startOpenAiUpstream({ nonStreamBody: okBody("peer-served") });
    const b1 = await startOpenAiUpstream({ nonStreamBody: okBody("backup-served") });
    upstreams.push(a1, a2, b1);
    await seed.createModel({
      display_name: "ch-partial",
      routing: {
        strategy: "consistent_hash",
        targets: [
          { model: "ch-partial-a1" },
          { model: "ch-partial-a2" },
          { model: "ch-partial-b1", priority: -1 },
        ],
      },
    });
    await createMember("ch-partial-a1", a1);
    await createMember("ch-partial-a2", a2);
    await createMember("ch-partial-b1", b1);
    await waitMembersVisible(["ch-partial-a1", "ch-partial-a2", "ch-partial-b1"]);

    const b1Baseline = b1.receivedRequests.length;
    for (let i = 0; i < 12; i++) {
      // Keys whose first choice is the dead member fail over to its tier
      // peer; keys mapped to the healthy peer are untouched. Either way
      // the answer comes from inside the active tier.
      expect(await ask("ch-partial", { "x-aisix-routing-key": `k-${i}` })).toBe(
        "peer-served",
      );
    }
    expect(b1.receivedRequests.length - b1Baseline).toBe(0);
  });
});
