import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: the control plane pushes every hard-stop budget into the gateway's
// configuration as a `budgets` document, marked `exhausted` once spent, and
// the gateway decides each request from those documents alone.
//
// What a caller sees is the long-standing budget refusal: 429, `error.type`
// `billing_error`, `error.code` `budget_exceeded`, and the budget's `scope`,
// `scope_ref` and amounts beside them in the error object (OpenAI error envelope:
// https://platform.openai.com/docs/guides/error-codes).

const hash = (plaintext: string) => createHash("sha256").update(plaintext).digest("hex");
const rfc3339 = (ms: number) => new Date(ms).toISOString().replace(".000Z", "Z");

const now = new Date();
const DAY_START = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate());
const MONTH_START = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), 1);
const YESTERDAY = DAY_START - 86_400_000;

const reason = (scope: string, scopeRef: string) => ({
  type: "billing_error",
  code: "budget_exceeded",
  message: `${scope.replace("_", " ")} budget exceeded`,
  scope,
  scope_ref: scopeRef,
  limit_usd: "10.00",
  spent_usd: "10.25",
  remaining_usd: "-0.25",
  period: "month",
});

const KEYS = {
  apiKey: "sk-bd-e2e-api-key",
  member: "sk-bd-e2e-member",
  team: "sk-bd-e2e-team",
  tmBlocked: "sk-bd-e2e-tm-blocked",
  tmOther: "sk-bd-e2e-tm-other",
  tagTeam: "sk-bd-e2e-tag-team",
  tagOwn: "sk-bd-e2e-tag-own",
  tagMember: "sk-bd-e2e-tag-member",
  lifted: "sk-bd-e2e-lifted",
  plain: "sk-bd-e2e-plain",
  sentinel: "sk-bd-e2e-sentinel",
};

const IDS = {
  apiKey: randomUUID(),
  member: randomUUID(),
  team: randomUUID(),
  tmTeam: randomUUID(),
  tmBlockedUser: randomUUID(),
  tmOtherUser: randomUUID(),
  tagTeam: randomUUID(),
  tagUser: randomUUID(),
  lifted: randomUUID(),
};

describe("budgets pushed as documents decide requests locally", () => {
  let app: SpawnedApp | undefined;
  let upA: OpenAiUpstream | undefined;
  let upB: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  const chat = (key: string, model = "bd-direct") =>
    new ProxyClient(app!.proxyUrl, key).chat({
      model,
      messages: [{ role: "user", content: "hello" }],
    });

  const refusal = (res: { status: number; body: unknown }) => {
    expect(res.status).toBe(429);
    const error = (res.body as { error: Record<string, unknown> }).error;
    expect(error.type).toBe("billing_error");
    expect(error.code).toBe("budget_exceeded");
    return error as Record<string, unknown> & { message: string };
  };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upA = await startOpenAiUpstream();
    upB = await startOpenAiUpstream();
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = (name: string, up: OpenAiUpstream) =>
      seed!.createProviderKey({ display_name: name, secret: "sk-mock", api_base: `${up.baseUrl}/v1` });
    const pkA = await pk("bd-pk-a", upA);
    const pkB = await pk("bd-pk-b", upB);
    const pkOut1 = await pk("bd-pk-out-1", upA);
    const pkOut2 = await pk("bd-pk-out-2", upA);
    const model = (display_name: string, provider_key_id: string) =>
      seed!.createModel({ display_name, provider: "openai", model_name: "gpt-4o-mini", provider_key_id });
    await model("bd-direct", pkA.id);
    await model("bd-on-a", pkA.id);
    await model("bd-on-b", pkB.id);
    await model("bd-out-1", pkOut1.id);
    await model("bd-out-2", pkOut2.id);
    // max_fallbacks 0: one attempt only, so the exhausted first target must
    // be dropped before the cut for the request to reach the second.
    await seed.createModel({
      display_name: "bd-group",
      routing: {
        strategy: "failover",
        targets: [
          { model: "bd-out-1", priority: 2 },
          { model: "bd-on-b", priority: 1 },
        ],
        max_fallbacks: 0,
      },
    });
    await seed.createModel({
      display_name: "bd-group-out",
      routing: { strategy: "failover", targets: [{ model: "bd-out-1" }, { model: "bd-out-2" }] },
    });

    for (const [id, tags] of [
      [IDS.tagTeam, { dept: "team-value" }],
      [IDS.team, {}],
      [IDS.tmTeam, {}],
    ] as const) {
      await seed.update("teams", id, { name: `team-${id}`, tags });
    }
    await seed.update("users", IDS.tagUser, { name: "tag member", tags: { dept: "member-value" } });

    const budget = (doc: Record<string, unknown>) => seed!.update("budgets", randomUUID(), doc);
    // Exhausted with no period start: the limit-zero shape, in force in
    // every period. The key's team budget is exhausted too; the more
    // specific api_key one is reported.
    await budget({
      scope: "api_key", scope_ref: IDS.apiKey, period: "month",
      exhausted: { reason: reason("api_key", IDS.apiKey) },
    });
    await budget({
      scope: "team", scope_ref: IDS.team, period: "month",
      exhausted: { period_start: rfc3339(MONTH_START), reason: reason("team", IDS.team) },
    });
    await budget({
      scope: "member", scope_ref: IDS.member, period: "month",
      exhausted: { period_start: rfc3339(MONTH_START), reason: reason("member", IDS.member) },
    });
    await budget({
      scope: "team_member", scope_ref: IDS.tmTeam, period: "day",
      exhausted_members: {
        [IDS.tmBlockedUser]: { period_start: rfc3339(DAY_START), reason: reason("team_member", IDS.tmTeam) },
      },
    });
    await budget({
      scope: "tag", tag: { key: "dept", value: "team-value" }, period: "month",
      exhausted: { period_start: rfc3339(MONTH_START), reason: { ...reason("tag", "dept=team-value"), message: "tag team-value out" } },
    });
    await budget({
      scope: "tag", tag: { key: "dept", value: "member-value" }, period: "month",
      exhausted: { period_start: rfc3339(MONTH_START), reason: { ...reason("tag", "dept=member-value"), message: "tag member-value out" } },
    });
    // Exhausted yesterday on a daily budget: the block has lifted.
    await budget({
      scope: "api_key", scope_ref: IDS.lifted, period: "day",
      exhausted: { period_start: rfc3339(YESTERDAY), reason: reason("api_key", IDS.lifted) },
    });
    for (const id of [pkOut1.id, pkOut2.id]) {
      await budget({
        scope: "provider_key", scope_ref: id, period: "month",
        exhausted: { period_start: rfc3339(MONTH_START), reason: reason("provider_key", id) },
      });
    }

    const key = (id: string, plaintext: string, extra: Record<string, unknown> = {}) =>
      seed!.update("api_keys", id, { key_hash: hash(plaintext), allowed_models: ["*"], ...extra });
    await key(IDS.apiKey, KEYS.apiKey, { team_id: IDS.team });
    await key(randomUUID(), KEYS.member, { user_id: IDS.member });
    await key(randomUUID(), KEYS.team, { team_id: IDS.team });
    await key(randomUUID(), KEYS.tmBlocked, { team_id: IDS.tmTeam, user_id: IDS.tmBlockedUser });
    await key(randomUUID(), KEYS.tmOther, { team_id: IDS.tmTeam, user_id: IDS.tmOtherUser });
    await key(randomUUID(), KEYS.tagTeam, { team_id: IDS.tagTeam, user_id: IDS.tagUser });
    await key(randomUUID(), KEYS.tagOwn, { team_id: IDS.tagTeam, user_id: IDS.tagUser, tags: { dept: "own" } });
    await key(randomUUID(), KEYS.tagMember, { user_id: IDS.tagUser });
    await key(IDS.lifted, KEYS.lifted);
    await key(randomUUID(), KEYS.plain);
    await key(randomUUID(), KEYS.sentinel);

    await waitConfigPropagation(
      async () => (await new ProxyClient(app!.proxyUrl, KEYS.sentinel).listModels()).status === 200,
    );
  });

  afterAll(async () => {
    await app?.exit();
    await upA?.close();
    await upB?.close();
  });

  test("an exhausted api_key budget refuses the key, reported over its team's", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const before = upA!.receivedRequests.length;
    const error = refusal(await chat(KEYS.apiKey));
    expect(error.scope).toBe("api_key");
    expect(error.scope_ref).toBe(IDS.apiKey);
    expect(error.limit_usd).toBe("10.00");
    expect(error.spent_usd).toBe("10.25");
    expect(upA!.receivedRequests.length).toBe(before);
  });

  test("member and team budgets refuse the keys they name", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    expect(refusal(await chat(KEYS.member)).scope).toBe("member");
    expect(refusal(await chat(KEYS.team)).scope).toBe("team");
    expect((await chat(KEYS.plain)).status).toBe(200);
  });

  test("a team_member budget refuses the exhausted member only", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    expect(refusal(await chat(KEYS.tmBlocked)).scope).toBe("team_member");
    expect((await chat(KEYS.tmOther)).status).toBe(200);
  });

  test("a tag budget follows the key's effective tag: key over team over member", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    // The team's value wins over the member's.
    expect(refusal(await chat(KEYS.tagTeam)).message).toBe("tag team-value out");
    // The member's value applies when no team sets the key.
    expect(refusal(await chat(KEYS.tagMember)).message).toBe("tag member-value out");
    // The key's own value wins over both.
    expect((await chat(KEYS.tagOwn)).status).toBe(200);
  });

  test("a block from an earlier period has lifted", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    expect((await chat(KEYS.lifted)).status).toBe(200);
  });

  test("the period reset is the next UTC boundary, as body and Retry-After", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const raw = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${KEYS.member}`, "content-type": "application/json" },
      body: JSON.stringify({ model: "bd-direct", messages: [{ role: "user", content: "hello" }] }),
    });
    const res = { status: raw.status, body: await raw.json(), headers: raw.headers };
    const nextMonth = Date.UTC(now.getUTCFullYear(), now.getUTCMonth() + 1, 1);
    const budget = refusal(res);
    expect(budget.period_resets_at).toBe(rfc3339(nextMonth));
    const retryAfter = Number(res.headers.get("retry-after"));
    expect(retryAfter).toBe(budget.retry_after_seconds);
    expect(Math.abs(retryAfter - (nextMonth - Date.now()) / 1000)).toBeLessThan(60);
  });

  test("/v1/messages refuses with the Anthropic envelope", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await fetch(`${app.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: { "x-api-key": KEYS.member, "content-type": "application/json", "anthropic-version": "2023-06-01" },
      body: JSON.stringify({ model: "bd-direct", max_tokens: 8, messages: [{ role: "user", content: "hi" }] }),
    });
    expect(res.status).toBe(429);
    const body = (await res.json()) as { type: string; error: { type: string } };
    expect(body.type).toBe("error");
    expect(body.error.type).toBe("rate_limit_error");
  });

  test("a routing target on an exhausted provider key is skipped without spending a fallback", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const [beforeA, beforeB] = [upA!.receivedRequests.length, upB!.receivedRequests.length];
    const res = await chat(KEYS.plain, "bd-group");
    expect(res.status).toBe(200);
    // Served by bd-on-b; bd-out-1 was never attempted.
    expect(upB!.receivedRequests.length).toBe(beforeB + 1);
    expect(upA!.receivedRequests.length).toBe(beforeA);
  });

  test("every target exhausted refuses with the provider key's budget", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const before = upA!.receivedRequests.length;
    expect(refusal(await chat(KEYS.plain, "bd-group-out")).scope).toBe("provider_key");
    // A direct model on an exhausted provider key is refused the same way.
    expect(refusal(await chat(KEYS.plain, "bd-out-1")).scope).toBe("provider_key");
    expect(upA!.receivedRequests.length).toBe(before);
  });

  test("/v1/responses refuses on both a key budget and a provider key budget", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const responses = async (key: string, model: string) => {
      const res = await fetch(`${app!.proxyUrl}/v1/responses`, {
        method: "POST",
        headers: { authorization: `Bearer ${key}`, "content-type": "application/json" },
        body: JSON.stringify({ model, input: "hi" }),
      });
      return { status: res.status, body: await res.json() };
    };
    const before = upA!.receivedRequests.length;
    expect(refusal(await responses(KEYS.member, "bd-direct")).scope).toBe("member");
    expect(refusal(await responses(KEYS.plain, "bd-group-out")).scope).toBe("provider_key");
    expect(upA!.receivedRequests.length).toBe(before);
  });

  // Last: it refuses every key in the environment until removed.
  test("an environment budget refuses every key until it is withdrawn", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const id = randomUUID();
    await seed!.update("budgets", id, {
      scope: "environment", scope_ref: "env", period: "week",
      exhausted: { reason: reason("environment", "env") },
    });
    await waitConfigPropagation(async () => (await chat(KEYS.plain)).status === 429);
    expect(refusal(await chat(KEYS.plain)).scope).toBe("environment");
    // A more specific exhausted budget still wins.
    expect(refusal(await chat(KEYS.member)).scope).toBe("member");
    await seed!.delete("budgets", id);
    await waitConfigPropagation(async () => (await chat(KEYS.plain)).status === 200);
  });
});
