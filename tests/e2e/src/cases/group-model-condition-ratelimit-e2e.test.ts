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

// E2E for AISIX-Cloud#1267: conditional rate-limit policies whose
// `model` / `model_name` conditions reference a ROUTING GROUP. The
// per-target gate evaluates the {dispatched target, requested parent}
// pair, so a group's own id/alias selects every request addressed to
// the group. Covers:
//
//   1. The reported scenario: `team ∈ {T} AND model ∈ {group uuid}`
//      with `group_by: [member]` — throttles per member THROUGH the
//      group, with policy attribution; a direct call to the member is
//      NOT captured by the group condition even with the bucket hot.
//   2. `model_name == <group alias>` matches through the group and not
//      on the member's own alias.
//   3. Leaf negate: `model !in [group uuid]` EXCLUDES requests routed
//      via the group (previously the member id missed the set, so the
//      negated leaf absurdly matched them) while direct dispatch to
//      the member stays matched.
//   4. The AISIX-Cloud#1087 principle survives: a MEMBER-id condition
//      keeps matching when the member is reached via the group, an
//      over-limit member fails over, and the same bucket throttles the
//      member's direct alias.
//   5. `/v1/messages` drives the same per-target gate (handler-family
//      coverage beyond chat).
//
// Policies reference model UUIDs, so models are seeded FIRST, then the
// policies, then a canary model: etcd watch events arrive in revision
// order, so once the canary lists, every earlier policy row is applied.

const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

const TEAM_GROUP = "team-1267-group";
const TEAM_NAME = "team-1267-name";
const TEAM_NEG = "team-1267-neg";
const TEAM_MEMBER = "team-1267-member";
const TEAM_SIB = "team-1267-sib";
const TEAM_STREAM = "team-1267-stream";
const TEAM_RESP = "team-1267-resp";

const POLICY_GROUP = "12670000-0000-0000-0000-00000000000a";
const POLICY_NAME = "12670000-0000-0000-0000-00000000000b";
const POLICY_NEG = "12670000-0000-0000-0000-00000000000c";
const POLICY_MEMBER = "12670000-0000-0000-0000-00000000000d";
const POLICY_SIB = "12670000-0000-0000-0000-00000000000e";
const POLICY_STREAM = "12670000-0000-0000-0000-00000000000f";
const POLICY_RESP = "12670000-0000-0000-0000-000000000010";

const KEY_G1 = "sk-1267-g1";
const KEY_G2 = "sk-1267-g2";
const KEY_NAME = "sk-1267-name";
const KEY_NEG = "sk-1267-neg";
const KEY_MEMBER = "sk-1267-member";
const KEY_SIB = "sk-1267-sib";
const KEY_STREAM = "sk-1267-stream";
const KEY_RESP = "sk-1267-resp";
const KEY_FREE = "sk-1267-free";

function chatBody(content: string) {
  return {
    id: "cmpl-1267",
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
    usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
  };
}

type ChatResult = {
  status: number;
  body: {
    choices?: Array<{ message?: { content?: string } }>;
    error?: { message?: string; type?: string; policy?: { id?: string; name?: string } };
  };
};

describe("group-referencing model conditions e2e (AISIX-Cloud#1267)", () => {
  let app: SpawnedApp | undefined;
  let etcd: EtcdClient | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];

  // Seeded model ids the policies pin.
  let memberAId = "";
  let groupMainId = "";
  let groupNegId = "";
  let groupStreamId = "";

  async function newUpstream(body: string): Promise<OpenAiUpstream> {
    const u = await startOpenAiUpstream({ nonStreamBody: chatBody(body) });
    upstreams.push(u);
    return u;
  }

  async function seedOpenAiModel(
    displayName: string,
    upstream: OpenAiUpstream,
  ): Promise<string> {
    if (!seed) throw new Error("seed client not initialized");
    const pk = await seed.createProviderKey({
      display_name: `${displayName}-pk`,
      secret: "sk-openai-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    const m = await seed.createModel({
      display_name: displayName,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    return m.id;
  }

  beforeAll(async () => {
    etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    // Members + groups first: the policies below embed their ids.
    const a = await newUpstream("served-a");
    const b = await newUpstream("served-b");
    const c = await newUpstream("served-c");
    memberAId = await seedOpenAiModel("m-1267-a", a);
    await seedOpenAiModel("m-1267-b", b);
    await seedOpenAiModel("m-1267-c", c);
    groupMainId = (
      await seed.createModel({
        display_name: "grp-1267-main",
        routing: {
          strategy: "failover",
          targets: [{ model: "m-1267-a" }, { model: "m-1267-b" }],
        },
      })
    ).id;
    groupNegId = (
      await seed.createModel({
        display_name: "grp-1267-neg",
        routing: { strategy: "failover", targets: [{ model: "m-1267-c" }] },
      })
    ).id;
    // The streaming case gets its own member + group: the mock serves
    // SSE or JSON per FIXTURE (not per request), so the shared members
    // must stay non-streaming while this one answers real SSE.
    const sse = await startOpenAiUpstream({
      streamEvents: [
        JSON.stringify({
          id: "mock-1267-s",
          object: "chat.completion.chunk",
          model: "gpt-4o-mini",
          choices: [{ index: 0, delta: { role: "assistant" } }],
        }),
        JSON.stringify({
          id: "mock-1267-s",
          object: "chat.completion.chunk",
          model: "gpt-4o-mini",
          choices: [{ index: 0, delta: { content: "served-s" }, finish_reason: "stop" }],
          usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
        }),
        "[DONE]",
      ],
    });
    upstreams.push(sse);
    const sPk = await seed.createProviderKey({
      display_name: "m-1267-s-pk",
      secret: "sk-openai-mock",
      api_base: `${sse.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "m-1267-s",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: sPk.id,
    });
    groupStreamId = (
      await seed.createModel({
        display_name: "grp-1267-stream",
        routing: { strategy: "failover", targets: [{ model: "m-1267-s" }] },
      })
    ).id;

    const putPolicy = (id: string, policy: Record<string, unknown>) =>
      etcd!.put(
        `${app!.etcdPrefix}/rate_limit_policies/${id}`,
        JSON.stringify(policy),
      );
    await putPolicy(POLICY_GROUP, {
      name: "group-cap-1267",
      conditions: [
        { dimension: "team", operator: "in", value: [TEAM_GROUP] },
        { dimension: "model", operator: "in", value: [groupMainId] },
      ],
      group_by: ["member"],
      limits: { rpm: 1 },
    });
    await putPolicy(POLICY_NAME, {
      name: "group-alias-1267",
      conditions: [
        { dimension: "team", operator: "in", value: [TEAM_NAME] },
        { dimension: "model_name", operator: "==", value: "grp-1267-main" },
      ],
      limits: { rpm: 1 },
    });
    await putPolicy(POLICY_NEG, {
      name: "all-but-group-1267",
      conditions: [
        { dimension: "team", operator: "in", value: [TEAM_NEG] },
        { dimension: "model", operator: "in", negate: true, value: [groupNegId] },
      ],
      limits: { rpm: 1 },
    });
    await putPolicy(POLICY_MEMBER, {
      name: "member-cap-1267",
      conditions: [
        { dimension: "team", operator: "in", value: [TEAM_MEMBER] },
        { dimension: "model", operator: "in", value: [memberAId] },
      ],
      limits: { rpm: 1 },
    });
    await putPolicy(POLICY_SIB, {
      name: "sibling-cap-1267",
      conditions: [
        { dimension: "team", operator: "in", value: [TEAM_SIB] },
        { dimension: "model", operator: "in", value: [groupMainId] },
      ],
      limits: { rpm: 1 },
    });
    await putPolicy(POLICY_STREAM, {
      name: "stream-cap-1267",
      conditions: [
        { dimension: "team", operator: "in", value: [TEAM_STREAM] },
        { dimension: "model", operator: "in", value: [groupStreamId] },
      ],
      limits: { rpm: 1 },
    });
    await putPolicy(POLICY_RESP, {
      name: "responses-cap-1267",
      conditions: [
        { dimension: "team", operator: "in", value: [TEAM_RESP] },
        { dimension: "model", operator: "in", value: [groupMainId] },
      ],
      limits: { rpm: 1 },
    });

    // Caller keys (raw etcd: team_id/user_id are CP-written fields the
    // standalone Admin API omits).
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
    await seedKey("12670001-0000-0000-0000-000000000001", KEY_G1, {
      team_id: TEAM_GROUP,
      user_id: "user-1267-g1",
    });
    await seedKey("12670001-0000-0000-0000-000000000002", KEY_G2, {
      team_id: TEAM_GROUP,
      user_id: "user-1267-g2",
    });
    await seedKey("12670001-0000-0000-0000-000000000003", KEY_NAME, {
      team_id: TEAM_NAME,
    });
    await seedKey("12670001-0000-0000-0000-000000000004", KEY_NEG, {
      team_id: TEAM_NEG,
    });
    await seedKey("12670001-0000-0000-0000-000000000005", KEY_MEMBER, {
      team_id: TEAM_MEMBER,
    });
    await seedKey("12670001-0000-0000-0000-000000000006", KEY_SIB, {
      team_id: TEAM_SIB,
    });
    await seedKey("12670001-0000-0000-0000-000000000007", KEY_FREE);
    await seedKey("12670001-0000-0000-0000-000000000008", KEY_STREAM, {
      team_id: TEAM_STREAM,
    });
    await seedKey("12670001-0000-0000-0000-000000000009", KEY_RESP, {
      team_id: TEAM_RESP,
    });

    // Canary AFTER every policy/key write: revision order means its
    // visibility proves the rows above are applied.
    const canary = await newUpstream("served-canary");
    await seedOpenAiModel("m-1267-canary", canary);
    await waitModelsListed(KEY_FREE, [
      "m-1267-a",
      "m-1267-b",
      "m-1267-c",
      "m-1267-s",
      "grp-1267-main",
      "grp-1267-neg",
      "grp-1267-stream",
      "m-1267-canary",
    ]);
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  // Listing consumes no rpm slot, so probing never burns the buckets
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

  test("group-id condition throttles per member through the group; direct member calls escape it", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    await awaitWindowHeadroom(5);

    // Member g1 burns their slot through the group; the 2nd call 429s
    // with attribution to the group policy.
    expect(servedContent(await chatRaw(KEY_G1, "grp-1267-main"))).toBe("served-a");
    const throttled = await chatRaw(KEY_G1, "grp-1267-main");
    expect(throttled.status).toBe(429);
    expect(throttled.body.error?.type).toBe("rate_limit_exceeded");
    expect(throttled.body.error?.policy).toEqual({
      id: POLICY_GROUP,
      name: "group-cap-1267",
    });

    // Same team, different member: independent bucket.
    expect(servedContent(await chatRaw(KEY_G2, "grp-1267-main"))).toBe("served-a");

    // Direct dispatch to the member is NOT addressed to the group, so
    // the group condition must not capture it — even with g1's group
    // bucket exhausted.
    expect(servedContent(await chatRaw(KEY_G1, "m-1267-a"))).toBe("served-a");
  });

  test("model_name == group alias matches through the group only", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    await awaitWindowHeadroom(5);

    expect(servedContent(await chatRaw(KEY_NAME, "grp-1267-main"))).toBe("served-a");
    const throttled = await chatRaw(KEY_NAME, "grp-1267-main");
    expect(throttled.status).toBe(429);
    expect(throttled.body.error?.policy).toEqual({
      id: POLICY_NAME,
      name: "group-alias-1267",
    });

    // The member's own alias is not the group alias: passes.
    expect(servedContent(await chatRaw(KEY_NAME, "m-1267-a"))).toBe("served-a");
  });

  test("negated group condition excludes via-group requests and keeps direct dispatch matched", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    await awaitWindowHeadroom(5);

    // Direct dispatch to the member matches `model !in [group]` and
    // burns the shared bucket.
    expect(servedContent(await chatRaw(KEY_NEG, "m-1267-c"))).toBe("served-c");
    expect((await chatRaw(KEY_NEG, "m-1267-c")).status).toBe(429);

    // Via the excluded group the SAME member escapes the policy even
    // with the bucket hot — negate flips the pair result, so "everything
    // except this group" no longer (absurdly) matches the group's own
    // traffic.
    expect(servedContent(await chatRaw(KEY_NEG, "grp-1267-neg"))).toBe("served-c");
  });

  test("member-id condition still matches via the group, fails over, and shares its bucket with the direct alias", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    await awaitWindowHeadroom(5);

    // 1st via group lands on member a (failover order) and burns the
    // member policy's bucket.
    expect(servedContent(await chatRaw(KEY_MEMBER, "grp-1267-main"))).toBe("served-a");
    // 2nd via group: a is over the member policy → failed attempt →
    // fails over to b, which the member condition does not match.
    expect(servedContent(await chatRaw(KEY_MEMBER, "grp-1267-main"))).toBe("served-b");
    // Direct dispatch to a hits the same exhausted bucket at the
    // request gate.
    const direct = await chatRaw(KEY_MEMBER, "m-1267-a");
    expect(direct.status).toBe(429);
    expect(direct.body.error?.policy).toEqual({
      id: POLICY_MEMBER,
      name: "member-cap-1267",
    });
  });

  test("/v1/messages drives the same per-target gate for group-id conditions", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    await awaitWindowHeadroom(5);

    const messagesRaw = async (): Promise<number> => {
      const res = await fetch(`${app!.proxyUrl}/v1/messages`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${KEY_SIB}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: "grp-1267-main",
          max_tokens: 32,
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      await res.text();
      return res.status;
    };

    expect(await messagesRaw()).toBe(200);
    expect(await messagesRaw()).toBe(429);
  });

  test("streaming chat drives the same per-target gate for group-id conditions", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    await awaitWindowHeadroom(5);

    const streamRaw = async (): Promise<{ status: number; contentType: string }> => {
      const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${KEY_STREAM}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: "grp-1267-stream",
          messages: [{ role: "user", content: "hello" }],
          stream: true,
        }),
      });
      // Drain so the connection finalises before the next call.
      await res.text();
      return { status: res.status, contentType: res.headers.get("content-type") ?? "" };
    };

    const first = await streamRaw();
    expect(first.status).toBe(200);
    // Real SSE, so the streaming loop (not a JSON fallback) held the
    // per-target reservation.
    expect(first.contentType).toContain("text/event-stream");
    expect((await streamRaw()).status).toBe(429);
  });

  test("/v1/responses drives the same per-target gate for group-id conditions", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    await awaitWindowHeadroom(5);

    const responsesRaw = async (): Promise<number> => {
      const res = await fetch(`${app!.proxyUrl}/v1/responses`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${KEY_RESP}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({ model: "grp-1267-main", input: "hello" }),
      });
      await res.text();
      return res.status;
    };

    expect(await responsesRaw()).toBe(200);
    expect(await responsesRaw()).toBe(429);
  });
});
