import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  awaitWindowHeadroom,
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForSlsLog,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";
import { startMockOtlp, type CapturedSpan, type MockOtlp } from "../harness/otlp-mock.js";

// E2E: cost-attribution tags. An `api_key`, a `team` and a `user` document
// each carry `tags` (key -> value). A request is attributed, per tag key,
// to the nearest value: the key's own, else its team's, else its member's.
// The resolved set — with the key's `team_id` — rides every usage row of
// the request (each attempt and the gateway's own terminal refusal), is
// exported, and can be matched and split on by a conditional rate-limit
// policy through the `tag:<key>` dimension.
//
// Asserted off a real `aisix` binary's exports: the OTLP span a trace
// backend receives and the SLS row a log consumer reads.

const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

const CREDENTIAL_REF = "mock";
const LOGSTORE = "cost-attribution-tags";

const TEAM = randomUUID();
const MEMBER = randomUUID();
const GROUPED_TEAM = randomUUID();

// Owned by TEAM and MEMBER, with one tag of its own.
const FULL_KEY = "sk-cat-full";
// Owned by MEMBER only, with no tags of its own.
const MEMBER_ONLY_KEY = "sk-cat-member-only";
// Owned by nobody, with no tags.
const BARE_KEY = "sk-cat-bare";
// Matched by a `tag:` condition policy with rpm 1.
const LIMITED_KEY = "sk-cat-limited";
const LIMITED_PEER_KEY = "sk-cat-limited-peer";
// Members of GROUPED_TEAM, split per department by a `group_by` policy.
const DEPT_A_KEY = "sk-cat-dept-a";
const DEPT_A_OTHER_KEY = "sk-cat-dept-a-other";
const DEPT_B_KEY = "sk-cat-dept-b";
const NO_DEPT_KEY = "sk-cat-no-dept";

const MODEL = "cat-model";
const FAILOVER_MODEL = "cat-failover";

function attemptSpans(otlp: MockOtlp, requestId: string): CapturedSpan[] {
  return otlp.spans
    .filter(
      (s) =>
        s.attributes["aisix.request_id"] === requestId &&
        s.attributes["aisix.attempt_index"] !== undefined,
    )
    .sort(
      (a, b) =>
        Number(a.attributes["aisix.attempt_index"]) - Number(b.attributes["aisix.attempt_index"]),
    );
}

async function waitForAttemptSpans(
  otlp: MockOtlp,
  requestId: string,
  count: number,
): Promise<CapturedSpan[]> {
  const deadline = Date.now() + 10_000;
  while (Date.now() < deadline) {
    const spans = attemptSpans(otlp, requestId);
    if (spans.length >= count) return spans;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(
    `expected ${count} attempt spans for ${requestId}, saw ${attemptSpans(otlp, requestId).length}`,
  );
}

/** The `aisix.tags.<key>` attributes of a span, as a plain object. */
function spanTags(span: CapturedSpan): Record<string, string> {
  const tags: Record<string, string> = {};
  for (const [k, v] of Object.entries(span.attributes)) {
    if (k.startsWith("aisix.tags.")) tags[k.slice("aisix.tags.".length)] = String(v);
  }
  return tags;
}

describe("cost-attribution tags e2e", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let otlp: MockOtlp | undefined;
  let sls: MockSls | undefined;
  const upstreams: OpenAiUpstream[] = [];

  async function chat(
    credential: string,
    content: string,
    model = MODEL,
  ): Promise<{ status: number; requestId: string }> {
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${credential}`, "content-type": "application/json" },
      body: JSON.stringify({ model, messages: [{ role: "user", content }] }),
    });
    await res.text();
    return { status: res.status, requestId: res.headers.get("x-aisix-call-id") ?? res.headers.get("x-aisix-request-id") ?? "" };
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    otlp = await startMockOtlp();
    sls = await startMockSls();
    const ok = await startOpenAiUpstream();
    const down = await startOpenAiUpstream({
      status: 502,
      errorBody: { error: { message: "primary down", type: "server_error" } },
    });
    upstreams.push(ok, down);

    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "LTAI_mock_ak",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock_ak_secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);

    await seed.createObservabilityExporter({
      name: "cat-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });
    await seed.createObservabilityExporter({
      name: "cat-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "aisix-e2e-obs",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
    });

    const okPk = await seed.createProviderKey({
      display_name: "cat-ok-pk",
      secret: "sk-mock",
      api_base: `${ok.baseUrl}/v1`,
    });
    const downPk = await seed.createProviderKey({
      display_name: "cat-down-pk",
      secret: "sk-mock",
      api_base: `${down.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: okPk.id,
    });
    await seed.createModel({
      display_name: "cat-primary",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: downPk.id,
    });
    await seed.createModel({
      display_name: FAILOVER_MODEL,
      routing: {
        strategy: "failover",
        targets: [{ model: "cat-primary" }, { model: MODEL }],
        retries: 0,
        max_fallbacks: 1,
      },
    });

    await seed.update("teams", TEAM, {
      name: "Platform",
      tags: { department: "team-dept", bg: "team-bg" },
    });
    await seed.update("users", MEMBER, {
      name: "Alice",
      tags: { department: "user-dept", bg: "user-bg", subsidiary: "user-sub" },
    });
    await seed.update("teams", GROUPED_TEAM, { name: "Grouped", tags: { bg: "grouped" } });

    // Each key belongs to exactly one of the two policies below, by its
    // `bg` tag, so neither policy's counters see another case's traffic.
    await seed.createRateLimitPolicy({
      name: "cat-limited-bg",
      conditions: [{ dimension: "tag:bg", operator: "==", value: "limited" }],
      limits: { rpm: 1 },
    });
    await seed.createRateLimitPolicy({
      name: "cat-per-department",
      conditions: [{ dimension: "tag:bg", operator: "==", value: "grouped" }],
      group_by: ["tag:department"],
      limits: { rpm: 1 },
    });

    const models = [MODEL, FAILOVER_MODEL];
    await seed.createApiKey({
      key_hash: sha256(MEMBER_ONLY_KEY),
      allowed_models: models,
      user_id: MEMBER,
    });
    await seed.createApiKey({ key_hash: sha256(BARE_KEY), allowed_models: models });
    await seed.createApiKey({
      key_hash: sha256(LIMITED_KEY),
      allowed_models: models,
      tags: { bg: "limited" },
    });
    // Same `department` as LIMITED_KEY but not the matched `bg`.
    await seed.createApiKey({
      key_hash: sha256(LIMITED_PEER_KEY),
      allowed_models: models,
      tags: { bg: "unlimited" },
    });
    for (const [plaintext, department] of [
      [DEPT_A_KEY, "a"],
      [DEPT_A_OTHER_KEY, "a"],
      [DEPT_B_KEY, "b"],
    ]) {
      await seed.createApiKey({
        key_hash: sha256(plaintext),
        allowed_models: models,
        team_id: GROUPED_TEAM,
        tags: { department },
      });
    }
    await seed.createApiKey({
      key_hash: sha256(NO_DEPT_KEY),
      allowed_models: models,
      team_id: GROUPED_TEAM,
    });
    // Seeded last: once it authenticates, everything above is loaded.
    await seed.createApiKey({
      key_hash: sha256(FULL_KEY),
      allowed_models: models,
      team_id: TEAM,
      user_id: MEMBER,
      tags: { department: "key-dept" },
    });
    const proxy = new ProxyClient(app.proxyUrl, FULL_KEY);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
    await otlp?.close();
    await sls?.close();
  });

  test("each tag key takes the key's value, else the team's, else the member's", async (ctx) => {
    if (!etcdReachable || !app || !otlp || !sls) {
      ctx.skip();
      return;
    }
    const full = await chat(FULL_KEY, "cat-precedence-full");
    expect(full.status).toBe(200);
    const [span] = await waitForAttemptSpans(otlp, full.requestId, 1);
    // department from the key, bg from the team over the member, and
    // subsidiary from the member — three keys from three documents.
    expect(spanTags(span)).toEqual({
      department: "key-dept",
      bg: "team-bg",
      subsidiary: "user-sub",
    });
    expect(span.attributes["aisix.team_id"]).toBe(TEAM);

    // The same row as a log consumer reads it: the usage event's own field
    // names, the tags as one JSON object.
    const log = await waitForSlsLog(
      sls,
      LOGSTORE,
      (l) => l.get("request_id") === full.requestId,
      "full-key usage row",
    );
    expect(JSON.parse(log.get("tags") ?? "{}")).toEqual({
      department: "key-dept",
      bg: "team-bg",
      subsidiary: "user-sub",
    });
    expect(log.get("team_id")).toBe(TEAM);

    const memberOnly = await chat(MEMBER_ONLY_KEY, "cat-precedence-member");
    expect(memberOnly.status).toBe(200);
    const [memberSpan] = await waitForAttemptSpans(otlp, memberOnly.requestId, 1);
    expect(spanTags(memberSpan)).toEqual({
      department: "user-dept",
      bg: "user-bg",
      subsidiary: "user-sub",
    });
    expect(memberSpan.attributes["aisix.team_id"]).toBeUndefined();

    const bare = await chat(BARE_KEY, "cat-precedence-bare");
    expect(bare.status).toBe(200);
    const [bareSpan] = await waitForAttemptSpans(otlp, bare.requestId, 1);
    expect(spanTags(bareSpan)).toEqual({});
    expect(bareSpan.attributes["aisix.team_id"]).toBeUndefined();
    const bareLog = await waitForSlsLog(
      sls,
      LOGSTORE,
      (l) => l.get("request_id") === bare.requestId,
      "bare-key usage row",
    );
    expect(bareLog.get("tags")).toBeUndefined();
    expect(bareLog.get("team_id")).toBeUndefined();
  });

  test("every attempt of a failed-over request carries the same tags and team", async (ctx) => {
    if (!etcdReachable || !app || !otlp) {
      ctx.skip();
      return;
    }
    const res = await chat(FULL_KEY, "cat-failover", FAILOVER_MODEL);
    expect(res.status).toBe(200);
    const spans = await waitForAttemptSpans(otlp, res.requestId, 2);
    expect(spans.map((s) => s.attributes["aisix.attempt_kind"])).toEqual(["initial", "fallback"]);
    expect(spans[0].attributes["http.response.status_code"]).toBe(502);
    for (const span of spans) {
      expect(spanTags(span)).toEqual({
        department: "key-dept",
        bg: "team-bg",
        subsidiary: "user-sub",
      });
      expect(span.attributes["aisix.team_id"]).toBe(TEAM);
    }
  });

  test("a tag condition limits the keys carrying the tag and no others", async (ctx) => {
    if (!etcdReachable || !app || !otlp) {
      ctx.skip();
      return;
    }
    await awaitWindowHeadroom();
    expect((await chat(LIMITED_KEY, "cat-limited-1")).status).toBe(200);
    const refused = await chat(LIMITED_KEY, "cat-limited-2");
    expect(refused.status).toBe(429);
    // A key without the matched value is outside the policy.
    for (let i = 0; i < 3; i += 1) {
      expect((await chat(LIMITED_PEER_KEY, `cat-limited-peer-${i}`)).status).toBe(200);
    }

    // The gateway's own refusal is a synthesized terminal row — no upstream
    // attempt behind it — and it is attributed like any other.
    const [span] = await waitForAttemptSpans(otlp, refused.requestId, 1);
    expect(span.attributes["http.response.status_code"]).toBe(429);
    expect(span.attributes["aisix.error_class"]).toBe("rate_limit_exceeded");
    expect(spanTags(span)).toEqual({ bg: "limited" });
  });

  test("group_by a tag gives each tag value its own counter", async (ctx) => {
    if (!etcdReachable || !app || !otlp) {
      ctx.skip();
      return;
    }
    await awaitWindowHeadroom();
    expect((await chat(DEPT_A_KEY, "cat-dept-a-1")).status).toBe(200);
    // The counter is the department's, not the key's: another key in
    // department a is already out of budget.
    expect((await chat(DEPT_A_OTHER_KEY, "cat-dept-a-other")).status).toBe(429);
    const refused = await chat(DEPT_A_KEY, "cat-dept-a-2");
    expect(refused.status).toBe(429);
    // Department b counts separately.
    expect((await chat(DEPT_B_KEY, "cat-dept-b-1")).status).toBe(200);
    expect((await chat(DEPT_B_KEY, "cat-dept-b-2")).status).toBe(429);
    // A matched request with no department is not subject to the policy.
    for (let i = 0; i < 3; i += 1) {
      expect((await chat(NO_DEPT_KEY, `cat-no-dept-${i}`)).status).toBe(200);
    }

    // The refusal carries the team the inherited `bg` came from.
    const [span] = await waitForAttemptSpans(otlp, refused.requestId, 1);
    expect(span.attributes["http.response.status_code"]).toBe(429);
    expect(spanTags(span)).toEqual({ department: "a", bg: "grouped" });
    expect(span.attributes["aisix.team_id"]).toBe(GROUPED_TEAM);
  });
});
