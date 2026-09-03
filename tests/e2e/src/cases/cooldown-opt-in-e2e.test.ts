import { createHash } from "node:crypto";
import OpenAI from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

/**
 * Two contracts that live next to each other (AISIX-Cloud#1499).
 *
 * 1. **Cooldown is opt-in.** A direct model whose operator never wrote a
 *    `cooldown` block is never taken out of rotation, however the
 *    upstream fails. It used to be the opposite: an absent block became
 *    a default one that read as enabled, so every such model cooled for
 *    30s on a built-in status list — 401/408/429/500/502/503/504 — while
 *    the console showed cooldown as disabled.
 * 2. **An exclusion is on the record.** When cooldown IS enabled and the
 *    filter drops the cooling target before the dispatch loop starts,
 *    the gateway says so, naming the target and the reason, at a level a
 *    production deployment is already running at. Without it the request
 *    that never tried the target and the group that only ever had one
 *    candidate leave the same trace.
 */

const CALLER_PLAINTEXT = "sk-cooldown-opt-in-caller";
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

describe("cooldown is opt-in — an unconfigured model stays in rotation", () => {
  let app: SpawnedApp | undefined;
  let admin: AdminClient | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  let failing: OpenAiUpstream | undefined;
  let stable: OpenAiUpstream | undefined;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // 503 is in the trigger list the gateway used to apply on its own.
    failing = await startOpenAiUpstream({
      status: 503,
      errorBody: { error: { message: "upstream down", type: "server_error" } },
    });
    stable = await startOpenAiUpstream({ nonStreamBody: okBody("stable-served") });

    app = await spawnApp({ admin: false });
    admin = new AdminClient(app.adminUrl, app.adminKey, app.metricsUrl);
    seed = new SeedClient(etcd, app.etcdPrefix);

    const failPk = await seed.createProviderKey({
      display_name: "optin-fail-pk",
      secret: "sk-mock",
      api_base: `${failing.baseUrl}/v1`,
    });
    const stablePk = await seed.createProviderKey({
      display_name: "optin-stable-pk",
      secret: "sk-mock",
      api_base: `${stable.baseUrl}/v1`,
    });

    // No `cooldown` block anywhere — exactly what the control plane
    // projects for an operator who left cooldown switched off.
    await seed.createModel({
      display_name: "optin-primary",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: failPk.id,
    });
    await seed.createModel({
      display_name: "optin-stable",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: stablePk.id,
    });
    await seed.createModel({
      display_name: "optin-router",
      routing: {
        strategy: "failover",
        targets: [{ model: "optin-primary" }, { model: "optin-stable" }],
        max_fallbacks: 1,
      },
    });
    // Seeded last: the key authenticating is the gate that implies every
    // resource above is in the snapshot.
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["optin-router"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await failing?.close();
    await stable?.close();
  });

  test("a 503 on a model with no cooldown config leaves it in the candidate list", async (ctx) => {
    if (!etcdReachable || !app || !admin || !failing || !stable) {
      ctx.skip();
      return;
    }
    const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    const first = await client.chat.completions.create({
      model: "optin-router",
      messages: [{ role: "user", content: "first" }],
    });
    expect(first.choices[0]?.message.content).toBe("stable-served");
    const failedOnce = failing.receivedRequests.length;
    expect(failedOnce).toBeGreaterThan(0);

    const second = await client.chat.completions.create({
      model: "optin-router",
      messages: [{ role: "user", content: "second" }],
    });
    expect(second.choices[0]?.message.content).toBe("stable-served");

    // The regression: with the old built-in default the 503 cooled the
    // primary for 30s and the second request never reached it, so this
    // delta was 0.
    expect(failing.receivedRequests.length - failedOnce).toBeGreaterThan(0);

    const row = (await admin.listModelStatuses()).find(
      (r) => r.display_name === "optin-primary",
    );
    expect(row?.status).toBe("healthy");
  });
});

describe("an enabled cooldown records the exclusion it causes", () => {
  let app: SpawnedApp | undefined;
  let admin: AdminClient | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  let failing: OpenAiUpstream | undefined;
  let stable: OpenAiUpstream | undefined;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    failing = await startOpenAiUpstream({
      status: 503,
      errorBody: { error: { message: "upstream down", type: "server_error" } },
    });
    stable = await startOpenAiUpstream({ nonStreamBody: okBody("survivor-served") });

    // Spawned at the harness default level. The lines asserted below
    // have to be visible without an operator having raised verbosity in
    // advance — nobody turns on debug before the incident.
    app = await spawnApp({ admin: false });
    admin = new AdminClient(app.adminUrl, app.adminKey, app.metricsUrl);
    seed = new SeedClient(etcd, app.etcdPrefix);

    const failPk = await seed.createProviderKey({
      display_name: "excl-fail-pk",
      secret: "sk-mock",
      api_base: `${failing.baseUrl}/v1`,
    });
    const stablePk = await seed.createProviderKey({
      display_name: "excl-stable-pk",
      secret: "sk-mock",
      api_base: `${stable.baseUrl}/v1`,
    });

    await seed.createModel({
      display_name: "excl-primary",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: failPk.id,
      cooldown: { enabled: true, default_seconds: 120 },
    });
    await seed.createModel({
      display_name: "excl-stable",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: stablePk.id,
    });
    await seed.createModel({
      display_name: "excl-router",
      routing: {
        strategy: "failover",
        targets: [{ model: "excl-primary" }, { model: "excl-stable" }],
        max_fallbacks: 1,
        fallback_on_statuses: [418],
      },
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["excl-router"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await failing?.close();
    await stable?.close();
  });

  test("the dropped target and its reason reach the log, with the surviving candidate count", async (ctx) => {
    if (!etcdReachable || !app || !admin || !failing || !stable) {
      ctx.skip();
      return;
    }
    const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // Request 1 trips the cooldown; request 2 is the one whose candidate
    // list is short by a target.
    await client.chat.completions.create({
      model: "excl-router",
      messages: [{ role: "user", content: "trip the cooldown" }],
    });
    await waitConfigPropagation(async () => {
      const rows = await admin!.listModelStatuses();
      return rows.some(
        (r) => r.display_name === "excl-primary" && r.status === "cooldown",
      );
    });
    const beforeSecond = failing.receivedRequests.length;

    const second = await client.chat.completions.create({
      model: "excl-router",
      messages: [{ role: "user", content: "served by the survivor" }],
    });
    expect(second.choices[0]?.message.content).toBe("survivor-served");
    expect(failing.receivedRequests.length - beforeSecond).toBe(0);

    const out = app.output();
    const excluded = out
      .split("\n")
      .filter((l) => l.includes("routing candidate excluded before dispatch"));
    expect(
      excluded.length,
      `no exclusion line in gateway output:\n${out}`,
    ).toBeGreaterThan(0);
    const line = excluded[0]!;
    expect(line).toContain("excl-primary");
    expect(line).toContain("excl-router");
    expect(line).toMatch(/reason="?cooling"?/);
    // The list the dispatch loop was actually left with.
    expect(line).toContain("candidates=1");

    // The other half of the same question: request 1 above produced a
    // real upstream failure on `excl-primary`, and the line that records
    // it names the fallback status list the retry/failover decision was
    // judged against — so an operator can tell "the status was not in
    // the list" from "the list never reached the gateway". Before this
    // change only /v1/chat/completions wrote such a line at all.
    const failures = out
      .split("\n")
      .filter((l) => l.includes("routing target attempt failed"));
    expect(
      failures.length,
      `no attempt-failure line in gateway output:\n${out}`,
    ).toBeGreaterThan(0);
    expect(failures[0]!).toContain("excl-primary");
    expect(failures[0]!).toContain("fallback_on_statuses=[418]");
  });
});

/**
 * Family lockstep. The failed-attempt WARN existed only on
 * `/v1/chat/completions`; `/v1/messages`, `/v1/messages/count_tokens` and
 * `/v1/responses` walked routing targets and recorded nothing at any
 * level, so the same incident on an Anthropic-SDK or Codex client left no
 * trace at all. All four now share one emitter.
 */
describe("every routing endpoint records a failed target attempt", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  let failing: OpenAiUpstream | undefined;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // Path-agnostic mock: every endpoint the four handlers call gets the
    // same 503.
    failing = await startOpenAiUpstream({
      status: 503,
      errorBody: { error: { message: "upstream down", type: "server_error" } },
    });

    app = await spawnApp({ admin: false });
    seed = new SeedClient(etcd, app.etcdPrefix);

    const openaiPk = await seed.createProviderKey({
      display_name: "fam-openai-pk",
      secret: "sk-mock",
      api_base: `${failing.baseUrl}/v1`,
    });
    const anthropicPk = await seed.createProviderKey({
      display_name: "fam-anthropic-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-mock",
      // The Anthropic bridge appends /v1/messages itself.
      api_base: failing.baseUrl,
    });

    await seed.createModel({
      display_name: "fam-openai-target",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: openaiPk.id,
    });
    await seed.createModel({
      display_name: "fam-anthropic-target",
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: anthropicPk.id,
    });
    await seed.createModel({
      display_name: "fam-openai-router",
      routing: {
        strategy: "failover",
        targets: [{ model: "fam-openai-target" }],
        max_fallbacks: 0,
      },
    });
    await seed.createModel({
      display_name: "fam-anthropic-router",
      routing: {
        strategy: "failover",
        targets: [{ model: "fam-anthropic-target" }],
        max_fallbacks: 0,
      },
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["fam-openai-router", "fam-anthropic-router"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await failing?.close();
  });

  test("/v1/messages, /v1/messages/count_tokens and /v1/responses each name the target that failed", async (ctx) => {
    if (!etcdReachable || !app || !failing) {
      ctx.skip();
      return;
    }
    const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);

    const post = async (path: string, body: unknown) =>
      fetch(`${app!.proxyUrl}${path}`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
          "content-type": "application/json",
        },
        body: JSON.stringify(body),
      });

    await post("/v1/messages", {
      model: "fam-anthropic-router",
      max_tokens: 64,
      messages: [{ role: "user", content: "hello" }],
    });
    await post("/v1/messages/count_tokens", {
      model: "fam-anthropic-router",
      messages: [{ role: "user", content: "hello" }],
    });
    await post("/v1/responses", {
      model: "fam-openai-router",
      input: "hello",
    });

    const failures = app
      .output()
      .split("\n")
      .filter((l) => l.includes("routing target attempt failed"));
    // Each of the three requests failed on its group's only target, so
    // each owes at least one line naming it.
    expect(
      failures.filter((l) => l.includes("fam-anthropic-target")).length,
      `no attempt-failure line for the Anthropic-shape endpoints:\n${app.output()}`,
    ).toBeGreaterThan(0);
    expect(
      failures.filter((l) => l.includes("fam-openai-target")).length,
      `no attempt-failure line for /v1/responses:\n${app.output()}`,
    ).toBeGreaterThan(0);
  });
});
