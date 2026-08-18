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

// E2E for AISIX-Cloud#892: the CONDITIONAL form of rate_limit_policies —
// a lua-resty-expr-style `conditions` tree (leaves + explicit AND/OR
// groups, negate), `group_by` bucket splitting, and the full 7-field
// `limits`. Covers:
//
//   1. OR-group matching: `team ∈ {T} AND (model_name ~~ ^gpt-4 OR
//      provider == anthropic)` throttles exactly the matched
//      (key, model) pairs; unmatched team / unmatched model pass. The
//      429 body carries `error.policy {id, name}` attribution.
//   2. Leaf `negate` (lua-resty-expr `!in`): the excluded value escapes
//      the policy, everything else the tree matches is throttled.
//   3. `group_by: [member]` — per-member independent buckets keyed by
//      user_id, not by API key (the conditional generalization of the
//      classic `team_member` scope).
//   4. Token settlement: a `limits.tpm` policy sees committed usage
//      (check-only at acquire, actuals committed post-response).
//   5. Routing: a model-property policy (`model_name ~~`) with
//      `group_by: [model]` reserves PER TARGET — an over-limit target
//      becomes a failed attempt that fails over; all targets over →
//      429; the group parent's own name never consumes a bucket.
//
// Every policy pins a test-private team or model-name prefix so the
// suites cannot throttle each other (policies are env-global).

const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

const TEAM_OR = "team-892-or";
const TEAM_NEG = "team-892-neg";
const TEAM_MEMBER = "team-892-member";
const TEAM_TPM = "team-892-tpm";

const POLICY_OR = "892e0000-0000-0000-0000-00000000000a";
const POLICY_NEG = "892e0000-0000-0000-0000-00000000000b";
const POLICY_MEMBER = "892e0000-0000-0000-0000-00000000000c";
const POLICY_TPM = "892e0000-0000-0000-0000-00000000000d";
const POLICY_ROUTE = "892e0000-0000-0000-0000-00000000000e";

const KEY_OR_TEAM = "sk-892-or-team";
const KEY_OR_FREE = "sk-892-or-free";
const KEY_NEG = "sk-892-neg";
const KEY_MEMBER_A1 = "sk-892-m-a1";
const KEY_MEMBER_A2 = "sk-892-m-a2";
const KEY_MEMBER_B = "sk-892-m-b";
const KEY_TPM = "sk-892-tpm";
const KEY_ROUTE = "sk-892-route";
// Readiness probe for the routing group: this key is allowed to access
// NOTHING, so its /v1/models listing is always empty and cannot gate on
// the group — probe with a chat call instead, 404 while the group is
// absent from the snapshot and 403 once it propagated. The 403 fires at
// the ACL gate, before any rate-limit reservation, so probing never
// consumes the buckets under test.
const KEY_PROBE = "sk-892-probe";

function chatBody(content: string, totalTokens = 8) {
  return {
    id: "cmpl-892",
    object: "chat.completion",
    created: 0,
    model: "gpt-4o-mini",
    choices: [
      {
        index: 0,
        message: { role: "assistant", content },
        finish_reason: "stop",
      },
    ],
    usage: {
      prompt_tokens: totalTokens - 3,
      completion_tokens: 3,
      total_tokens: totalTokens,
    },
  };
}

type ChatResult = {
  status: number;
  body: {
    choices?: Array<{ message?: { content?: string } }>;
    error?: { message?: string; type?: string; policy?: { id?: string; name?: string } };
  };
};

describe("conditional rate limit policies e2e (AISIX-Cloud#892)", () => {
  let app: SpawnedApp | undefined;
  let etcd: EtcdClient | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    // Seed every policy FIRST (lowest etcd revisions): once a later
    // resource (model/key) is visible through the proxy, its policies
    // are guaranteed applied — watch events arrive in revision order.
    const putPolicy = (id: string, policy: Record<string, unknown>) =>
      etcd!.put(
        `${app!.etcdPrefix}/rate_limit_policies/${id}`,
        JSON.stringify(policy),
      );

    await putPolicy(POLICY_OR, {
      name: "premium-family",
      conditions: [
        { dimension: "team", operator: "in", value: [TEAM_OR] },
        {
          logic: "or",
          children: [
            { dimension: "model_name", operator: "~~", value: "^gpt-4" },
            { dimension: "provider", operator: "==", value: "anthropic" },
          ],
        },
      ],
      limits: { rpm: 1 },
    });
    await putPolicy(POLICY_NEG, {
      name: "neg-family",
      conditions: [
        { dimension: "team", operator: "in", value: [TEAM_NEG] },
        {
          dimension: "model_name",
          operator: "in",
          negate: true,
          value: ["mdrl-neg-free"],
        },
      ],
      limits: { rpm: 1 },
    });
    await putPolicy(POLICY_MEMBER, {
      name: "per-member-default",
      conditions: [{ dimension: "team", operator: "in", value: [TEAM_MEMBER] }],
      group_by: ["member"],
      limits: { rpm: 1 },
    });
    await putPolicy(POLICY_TPM, {
      name: "team-token-pool",
      conditions: [{ dimension: "team", operator: "in", value: [TEAM_TPM] }],
      limits: { tpm: 10 },
    });
    await putPolicy(POLICY_ROUTE, {
      name: "per-target-cap",
      conditions: [{ dimension: "model_name", operator: "~~", value: "^mdrl-rt-" }],
      group_by: ["model"],
      limits: { rpm: 1 },
    });

    // Caller keys. The standalone Admin API omits team_id/user_id (the
    // CP writes those in production), so seed keys straight to etcd.
    const seedKey = (
      id: string,
      plaintext: string,
      extra: Record<string, unknown> = {},
    ) =>
      etcd!.put(
        `${app!.etcdPrefix}/api_keys/${id}`,
        JSON.stringify({
          key_hash: sha256(plaintext),
          allowed_models: ["*"],
          ...extra,
        }),
      );
    await seedKey("892e0001-0000-0000-0000-000000000001", KEY_OR_TEAM, {
      team_id: TEAM_OR,
    });
    await seedKey("892e0001-0000-0000-0000-000000000002", KEY_OR_FREE);
    await seedKey("892e0001-0000-0000-0000-000000000003", KEY_NEG, {
      team_id: TEAM_NEG,
    });
    await seedKey("892e0001-0000-0000-0000-000000000004", KEY_MEMBER_A1, {
      team_id: TEAM_MEMBER,
      user_id: "user-892-a",
    });
    await seedKey("892e0001-0000-0000-0000-000000000005", KEY_MEMBER_A2, {
      team_id: TEAM_MEMBER,
      user_id: "user-892-a",
    });
    await seedKey("892e0001-0000-0000-0000-000000000006", KEY_MEMBER_B, {
      team_id: TEAM_MEMBER,
      user_id: "user-892-b",
    });
    await seedKey("892e0001-0000-0000-0000-000000000007", KEY_TPM, {
      team_id: TEAM_TPM,
    });
    await seedKey("892e0001-0000-0000-0000-000000000008", KEY_ROUTE);
    await seedKey("892e0001-0000-0000-0000-000000000009", KEY_PROBE, {
      allowed_models: ["__probe-none__"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  async function newUpstream(
    opts: Parameters<typeof startOpenAiUpstream>[0],
  ): Promise<OpenAiUpstream> {
    const u = await startOpenAiUpstream(opts);
    upstreams.push(u);
    return u;
  }

  async function createOpenAiModel(
    displayName: string,
    upstream: OpenAiUpstream,
    extra: Record<string, unknown> = {},
  ): Promise<void> {
    if (!seed) throw new Error("seed client not initialized");
    const pk = await seed.createProviderKey({
      display_name: `${displayName}-pk`,
      secret: "sk-openai-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: displayName,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
      ...extra,
    });
  }

  // Readiness: list models with a key until every name shows up.
  // Listing consumes no rpm slot, so probing never burns the quota
  // under test.
  async function waitModelsListed(apiKey: string, names: string[]): Promise<void> {
    if (!app) throw new Error("app not initialized");
    const probe = new ProxyClient(app.proxyUrl, apiKey);
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      if (res.status !== 200) return false;
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return names.every((n) => data.some((m) => m.id === n));
    });
  }

  // The no-access probe key lists nothing, so wait until a chat call with
  // it flips from 404 (not propagated) to 403 (in snapshot, ACL-rejected
  // before any reservation).
  async function waitGroupPropagated(name: string): Promise<void> {
    if (!app) throw new Error("app not initialized");
    const probe = new ProxyClient(app.proxyUrl, KEY_PROBE);
    await waitConfigPropagation(async () => {
      const res = await probe.chat({
        model: name,
        messages: [{ role: "user", content: "probe" }],
      });
      return res.status === 403;
    });
  }

  async function chatRaw(apiKey: string, model: string): Promise<ChatResult> {
    if (!app) throw new Error("app not initialized");
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${apiKey}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: "hello" }],
      }),
    });
    return { status: res.status, body: (await res.json()) as ChatResult["body"] };
  }

  function servedContent(r: ChatResult): string {
    expect(r.status).toBe(200);
    return r.body.choices?.[0]?.message?.content ?? "";
  }

  test("OR group matches by model_name branch; unmatched team/model pass; 429 carries policy attribution", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const gpt = await newUpstream({ nonStreamBody: chatBody("served-gpt4") });
    const sonnet = await newUpstream({ nonStreamBody: chatBody("served-sonnet") });
    // display_name is the model_name dimension: one matches ^gpt-4,
    // the control model matches neither OR branch (provider openai).
    await createOpenAiModel("gpt-4.1-e2e", gpt);
    await createOpenAiModel("sonnet-e2e", sonnet);
    await waitModelsListed(KEY_OR_TEAM, ["gpt-4.1-e2e", "sonnet-e2e"]);
    await awaitWindowHeadroom(5);

    // Team key on the gpt-4 family: 1 allowed, 2nd throttled.
    expect(servedContent(await chatRaw(KEY_OR_TEAM, "gpt-4.1-e2e"))).toBe("served-gpt4");
    const throttled = await chatRaw(KEY_OR_TEAM, "gpt-4.1-e2e");
    expect(throttled.status).toBe(429);
    expect(throttled.body.error?.type).toBe("rate_limit_exceeded");
    // Attribution: with several policies live, the caller can tell
    // WHICH one rejected them.
    expect(throttled.body.error?.policy).toEqual({
      id: POLICY_OR,
      name: "premium-family",
    });

    // A team-less key on the same model: the `team in` leaf fails →
    // policy inapplicable, request passes even though the shared
    // window would be exhausted if it matched.
    expect(servedContent(await chatRaw(KEY_OR_FREE, "gpt-4.1-e2e"))).toBe("served-gpt4");

    // The team key on a model matching NEITHER or-branch passes.
    expect(servedContent(await chatRaw(KEY_OR_TEAM, "sonnet-e2e"))).toBe("served-sonnet");
  });

  test("leaf negate (!in) exempts the excluded model only", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const capped = await newUpstream({ nonStreamBody: chatBody("served-capped") });
    const free = await newUpstream({ nonStreamBody: chatBody("served-free") });
    await createOpenAiModel("mdrl-neg-capped", capped);
    await createOpenAiModel("mdrl-neg-free", free);
    await waitModelsListed(KEY_NEG, ["mdrl-neg-capped", "mdrl-neg-free"]);
    await awaitWindowHeadroom(5);

    // `model_name !in ["mdrl-neg-free"]` matches every other model.
    expect(servedContent(await chatRaw(KEY_NEG, "mdrl-neg-capped"))).toBe("served-capped");
    expect((await chatRaw(KEY_NEG, "mdrl-neg-capped")).status).toBe(429);

    // The excluded model escapes the policy even with the bucket hot.
    expect(servedContent(await chatRaw(KEY_NEG, "mdrl-neg-free"))).toBe("served-free");
  });

  test("group_by [member] buckets per user_id, not per API key", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const u = await newUpstream({ nonStreamBody: chatBody("served-member") });
    await createOpenAiModel("mdrl-member", u);
    await waitModelsListed(KEY_MEMBER_A1, ["mdrl-member"]);
    await awaitWindowHeadroom(5);

    // Member A burns their slot; their 2nd call throttles.
    expect(servedContent(await chatRaw(KEY_MEMBER_A1, "mdrl-member"))).toBe("served-member");
    expect((await chatRaw(KEY_MEMBER_A1, "mdrl-member")).status).toBe(429);

    // Member B has an independent bucket.
    expect(servedContent(await chatRaw(KEY_MEMBER_B, "mdrl-member"))).toBe("served-member");

    // A's SECOND key shares A's exhausted bucket → per-user, not per-key.
    expect((await chatRaw(KEY_MEMBER_A2, "mdrl-member")).status).toBe(429);
  });

  test("limits.tpm sees committed token usage", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    // Each response commits 8 tokens against tpm=10. TPM is check-only
    // at acquire (cost unknown pre-upstream): call 1 checks 0<10 then
    // commits 8; call 2 checks 8<10 then commits 16; call 3 sees the
    // window exhausted.
    const u = await newUpstream({ nonStreamBody: chatBody("served-tpm", 8) });
    await createOpenAiModel("mdrl-tpm", u);
    await waitModelsListed(KEY_TPM, ["mdrl-tpm"]);
    await awaitWindowHeadroom(5);

    expect(servedContent(await chatRaw(KEY_TPM, "mdrl-tpm"))).toBe("served-tpm");
    expect(servedContent(await chatRaw(KEY_TPM, "mdrl-tpm"))).toBe("served-tpm");
    const third = await chatRaw(KEY_TPM, "mdrl-tpm");
    expect(third.status).toBe(429);
    expect(third.body.error?.message ?? "").toContain("token");
  });

  test("model-property policy with group_by [model] reserves per routing target", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const a = await newUpstream({ nonStreamBody: chatBody("served-rt-a") });
    const b = await newUpstream({ nonStreamBody: chatBody("served-rt-b") });
    await createOpenAiModel("mdrl-rt-a", a);
    await createOpenAiModel("mdrl-rt-b", b);
    if (!seed) throw new Error("seed client not initialized");
    // The group's own display_name also matches ^mdrl-rt-, which pins
    // the routing-parent rule: the request gate must NOT burn a bucket
    // for the parent — only concrete targets reserve.
    await seed.createModel({
      display_name: "mdrl-rt-group",
      routing: {
        strategy: "failover",
        targets: [{ model: "mdrl-rt-a" }, { model: "mdrl-rt-b" }],
      },
    });
    await waitModelsListed(KEY_ROUTE, ["mdrl-rt-a", "mdrl-rt-b"]);
    await waitGroupPropagated("mdrl-rt-group");
    await awaitWindowHeadroom(5);

    // 1st call lands on target a and consumes bucket model=a.
    expect(servedContent(await chatRaw(KEY_ROUTE, "mdrl-rt-group"))).toBe("served-rt-a");
    // 2nd call: target a is over ITS bucket → failed attempt → fails
    // over to b (LiteLLM semantics: rate-limited deployments filtered).
    expect(servedContent(await chatRaw(KEY_ROUTE, "mdrl-rt-group"))).toBe("served-rt-b");
    // 3rd call: both target buckets exhausted → 429 that STILL carries
    // the structured policy attribution (the routing loop must not
    // flatten the rejection into an anonymous upstream error).
    const third = await chatRaw(KEY_ROUTE, "mdrl-rt-group");
    expect(third.status).toBe(429);
    expect(third.body.error?.policy).toEqual({
      id: POLICY_ROUTE,
      name: "per-target-cap",
    });

    // Direct dispatch to an exhausted target hits the same bucket at
    // the request gate (direct = model known pre-dispatch).
    const direct = await chatRaw(KEY_ROUTE, "mdrl-rt-a");
    expect(direct.status).toBe(429);
    expect(direct.body.error?.policy).toEqual({
      id: POLICY_ROUTE,
      name: "per-target-cap",
    });
  });
});
