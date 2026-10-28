import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  scrapeMetrics,
  spawnApp,
  startMockControlPlane,
  startOpenAiUpstream,
  sumMetric,
  waitConfigPropagation,
  type MockControlPlane,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: a managed gateway under `budget_policy.fail_mode: closed`.
//
// The control plane's spend counters are fed only by the usage events this
// gateway delivers, so once delivery has failed for `closed_grace_seconds`
// a budget that reads "not exhausted" can no longer be trusted. Everything a
// budget applies to is then answered 503 `budget_unverified` — a key with a
// budget, an upstream whose provider key has one — while traffic no budget
// covers carries on. A POST that reaches the control plane again lifts it.
//
// It also pins that budgets are decided locally: the stand-in control plane
// never sees a per-request budget query.

const hash = (plaintext: string) => createHash("sha256").update(plaintext).digest("hex");
const GRACE_SECONDS = 8;
const KEYS = { budgeted: "sk-bs-e2e-budgeted", plain: "sk-bs-e2e-plain", sentinel: "sk-bs-e2e-sentinel" };

describe("strict budget mode follows usage delivery", () => {
  let cp: MockControlPlane | undefined;
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  const chat = (key: string, model: string) =>
    new ProxyClient(app!.proxyUrl, key).chat({ model, messages: [{ role: "user", content: "hi" }] });
  const unhealthySeconds = async () =>
    sumMetric(await scrapeMetrics(app!.metricsUrl), "aisix_usage_delivery_unhealthy_seconds");

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    cp = await startMockControlPlane();
    app = await spawnApp({ extra: { managed: cp.managed }, awaitProxyListener: false });
    const seed = new SeedClient(etcd, `${app.etcdPrefix}/${cp.envId}`);

    const pkPlain = await seed.createProviderKey({ display_name: "bs-pk-plain", secret: "sk-mock", api_base: `${upstream.baseUrl}/v1` });
    const pkBudgeted = await seed.createProviderKey({ display_name: "bs-pk-budgeted", secret: "sk-mock", api_base: `${upstream.baseUrl}/v1` });
    const model = (display_name: string, provider_key_id: string) =>
      seed.createModel({ display_name, provider: "openai", model_name: "gpt-4o-mini", provider_key_id });
    await model("bs-plain", pkPlain.id);
    await model("bs-budgeted", pkBudgeted.id);
    await seed.createModel({
      display_name: "bs-group",
      routing: {
        strategy: "failover",
        targets: [
          { model: "bs-budgeted", priority: 2 },
          { model: "bs-plain", priority: 1 },
        ],
        max_fallbacks: 0,
      },
    });
    await seed.update("budget_policy", cp.envId, { fail_mode: "closed", closed_grace_seconds: GRACE_SECONDS });
    const budgetedKeyId = randomUUID();
    // Neither budget is exhausted: only strict mode can refuse them.
    await seed.update("budgets", randomUUID(), { scope: "api_key", scope_ref: budgetedKeyId, period: "month" });
    await seed.update("budgets", randomUUID(), { scope: "provider_key", scope_ref: pkBudgeted.id, period: "month" });
    await seed.update("api_keys", budgetedKeyId, { key_hash: hash(KEYS.budgeted), allowed_models: ["*"] });
    await seed.createApiKey({ key_hash: hash(KEYS.plain), allowed_models: ["*"] });
    await seed.createApiKey({ key_hash: hash(KEYS.sentinel), allowed_models: ["*"] });

    await waitConfigPropagation(async () => {
      try {
        return (await new ProxyClient(app!.proxyUrl, KEYS.sentinel).listModels()).status === 200;
      } catch {
        // The proxy listener binds once the first configuration applies.
        return false;
      }
    });
  });

  afterAll(async () => {
    await app?.exit();
    await cp?.close();
    await upstream?.close();
  });

  test("budgeted traffic is refused once delivery fails past the grace period, and resumes with it", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // Delivery healthy: strict mode changes nothing.
    expect((await chat(KEYS.budgeted, "bs-plain")).status).toBe(200);
    expect((await chat(KEYS.plain, "bs-budgeted")).status).toBe(200);

    // The control plane stops taking usage. Within the grace period the
    // budgeted key is still served.
    cp!.telemetryStatus = 503;
    await waitConfigPropagation(async () => cp!.telemetryAnswers.includes(503), 15_000);
    expect((await chat(KEYS.budgeted, "bs-plain")).status).toBe(200);

    await waitConfigPropagation(
      async () => (await chat(KEYS.budgeted, "bs-plain")).status === 503,
      (GRACE_SECONDS + 10) * 1000,
    );
    const refused = await chat(KEYS.budgeted, "bs-plain");
    expect(refused.status).toBe(503);
    const error = (refused.body as { error: Record<string, unknown> }).error;
    expect(error.code).toBe("budget_unverified");
    expect(error.type).toBe("api_error");
    expect(await unhealthySeconds()).toBeGreaterThanOrEqual(GRACE_SECONDS);

    // No budget covers this key or this upstream: unaffected.
    expect((await chat(KEYS.plain, "bs-plain")).status).toBe(200);
    // An upstream whose provider key has a budget is not used: a direct
    // model on it is refused, and a group falls through to its other target.
    const direct = await chat(KEYS.plain, "bs-budgeted");
    expect(direct.status).toBe(503);
    expect((direct.body as { error: { code: string } }).error.code).toBe("budget_unverified");
    expect((await chat(KEYS.plain, "bs-group")).status).toBe(200);

    // Delivery comes back: the next POST that reaches the control plane
    // lifts strict mode.
    cp!.telemetryStatus = 200;
    await waitConfigPropagation(
      async () => (await chat(KEYS.budgeted, "bs-plain")).status === 200,
      15_000,
    );
    expect(await unhealthySeconds()).toBeLessThan(GRACE_SECONDS);
    expect((await chat(KEYS.plain, "bs-budgeted")).status).toBe(200);
  });

  test("no request asks the control plane for a budget decision", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    expect(cp!.calls).toContain("/dp/telemetry");
    expect(cp!.calls.filter((path) => path.includes("budget"))).toEqual([]);
  });
});
