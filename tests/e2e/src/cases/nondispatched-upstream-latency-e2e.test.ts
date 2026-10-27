import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
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

// A request that never reached an upstream reports no upstream latency.
//
// `upstream_latency_ms` times one upstream attempt, and 0 means "not
// measured" all the way to the console. A request turned down before its
// first attempt (an input guardrail block, an unknown model) and a response
// served from the cache contacted no upstream, so their rows must carry 0
// there — not the gateway's own handling time dressed up as an upstream
// call. What the caller waited for (`downstream_latency_ms`) is unaffected:
// a cache hit still reports it, and a refused request records it exactly as
// a request whose upstream failed does.
//
// Read off the Aliyun SLS exporter, which writes every UsageEvent field.

const CALLER = "sk-nondispatched-latency";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const CREDENTIAL_REF = "mock";
const LOGSTORE = "nondispatched-latency";
const FORBIDDEN = "forbidden-nondispatched-word";

const upstreamLatency = (row: Map<string, string>) => Number(row.get("upstream_latency_ms") ?? 0);
const downstreamLatency = (row: Map<string, string>) =>
  Number(row.get("downstream_latency_ms") ?? 0);

describe("rows that reached no upstream carry no upstream latency", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let okUpstream: OpenAiUpstream | undefined;
  let failingUpstream: OpenAiUpstream | undefined;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    okUpstream = await startOpenAiUpstream({ responseDelayMs: 20 });
    failingUpstream = await startOpenAiUpstream({ status: 500, responseDelayMs: 20 });
    sls = await startMockSls();
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "LTAI_mock_ak",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock_ak_secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "nd-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "aisix-e2e-obs",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
    });
    for (const [name, upstream] of [
      ["nd-ok", okUpstream],
      ["nd-failing", failingUpstream],
    ] as const) {
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      await seed.createModel({
        display_name: name,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
    }
    await seed.createCachePolicy({ name: "nd-cache", enabled: true, applies_to: "all" });
    await seed.createGuardrail({
      name: "nd-keyword",
      enabled: true,
      hook_point: "input",
      kind: "keyword",
      patterns: [{ kind: "literal", value: FORBIDDEN }],
    });
    // Last, so its authenticating means every row above is live.
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER}` },
      });
      await res.text();
      return res.status === 200;
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await sls?.close();
    await okUpstream?.close();
    await failingUpstream?.close();
  });

  async function post(path: string, body: unknown) {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    await res.text();
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    expect(requestId).not.toBe("");
    return { status: res.status, requestId, cache: res.headers.get("x-aisix-cache") ?? "" };
  }

  const row = (requestId: string, what: string) =>
    waitForSlsLog(sls!, LOGSTORE, (l) => l.get("request_id") === requestId, what, 15_000);

  const chat = (model: string, content: string) => ({
    model,
    messages: [{ role: "user", content }],
  });

  test("a request that reached its upstream reports the attempt's latency", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const ok = await post("/v1/chat/completions", chat("nd-ok", "dispatched success"));
    expect(ok.status).toBe(200);
    expect(ok.cache).toBe("miss");
    expect(upstreamLatency(await row(ok.requestId, "dispatched success"))).toBeGreaterThanOrEqual(1);

    const failed = await post("/v1/chat/completions", chat("nd-failing", "dispatched failure"));
    expect(failed.status).toBeGreaterThanOrEqual(500);
    expect(upstreamLatency(await row(failed.requestId, "dispatched failure"))).toBeGreaterThanOrEqual(
      1,
    );
  });

  test("a chat request blocked by an input guardrail", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const failed = await post("/v1/chat/completions", chat("nd-failing", "reference failure"));
    const failedRow = await row(failed.requestId, "reference failure");

    const blocked = await post("/v1/chat/completions", chat("nd-ok", `say ${FORBIDDEN}`));
    expect(blocked.status).toBeGreaterThanOrEqual(400);
    expect(blocked.status).toBeLessThan(500);
    const blockedRow = await row(blocked.requestId, "chat guardrail block");
    expect(blockedRow.get("guardrail_blocked")).toBe("true");
    expect(upstreamLatency(blockedRow), "no upstream was contacted").toBe(0);
    // The caller's wait is recorded the way it is for a request whose
    // upstream failed: refusing earlier does not change that rule.
    expect(downstreamLatency(blockedRow) > 0).toBe(downstreamLatency(failedRow) > 0);
  });

  test("a chat request for a model that does not exist", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const missing = await post("/v1/chat/completions", chat("nd-no-such-model", "hello"));
    expect(missing.status).toBeGreaterThanOrEqual(400);
    expect(missing.status).toBeLessThan(500);
    expect(upstreamLatency(await row(missing.requestId, "unknown model"))).toBe(0);
  });

  test("a chat response served from the exact-match cache", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const body = chat("nd-ok", "cache me");
    const miss = await post("/v1/chat/completions", body);
    expect(miss.cache).toBe("miss");
    expect(upstreamLatency(await row(miss.requestId, "cache miss"))).toBeGreaterThanOrEqual(1);

    const hit = await post("/v1/chat/completions", body);
    expect(hit.status).toBe(200);
    expect(hit.cache).toBe("hit");
    const hitRow = await row(hit.requestId, "cache hit");
    expect(upstreamLatency(hitRow), "a cache hit contacts no upstream").toBe(0);
    expect(downstreamLatency(hitRow), "the caller's wait is still measured").toBeGreaterThanOrEqual(1);
  });

  test("a /v1/messages request blocked by an input guardrail", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const blocked = await post("/v1/messages", {
      model: "nd-ok",
      max_tokens: 16,
      messages: [{ role: "user", content: `say ${FORBIDDEN}` }],
    });
    expect(blocked.status).toBeGreaterThanOrEqual(400);
    expect(blocked.status).toBeLessThan(500);
    const blockedRow = await row(blocked.requestId, "messages guardrail block");
    expect(blockedRow.get("guardrail_blocked")).toBe("true");
    expect(upstreamLatency(blockedRow)).toBe(0);
  });

  test("a /v1/responses request blocked by an input guardrail", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const blocked = await post("/v1/responses", { model: "nd-ok", input: `say ${FORBIDDEN}` });
    expect(blocked.status).toBeGreaterThanOrEqual(400);
    expect(blocked.status).toBeLessThan(500);
    const blockedRow = await row(blocked.requestId, "responses guardrail block");
    expect(blockedRow.get("guardrail_blocked")).toBe("true");
    expect(upstreamLatency(blockedRow)).toBe(0);
  });
});
