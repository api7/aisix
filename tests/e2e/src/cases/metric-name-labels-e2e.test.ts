import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient, ProxyClient, SeedClient, spawnApp, startOpenAiUpstream,
  waitConfigPropagation, type OpenAiUpstream, type SpawnedApp,
} from "../harness/index.js";
import { metricDelta, scrapeMetrics } from "../harness/metrics.js";

// A Prometheus user who slices by team or API key should read who it is
// from the scrape, not by mapping ids through the Admin API. Every id label
// therefore has a name label beside it that `observability.metrics.labels`
// can select, resolved from the named resource's own document: `teams` and
// `users` documents for the key's team and member, the key's own
// `display_name`, the ProviderKey's, the rate-limit policy's.

const hash = (s: string) => createHash("sha256").update(s).digest("hex");
const MODEL = "name-labels-model";
const NAMED_KEY = "sk-name-labels-named";
const BARE_KEY = "sk-name-labels-bare";
const POLICY_KEY = "sk-name-labels-policy";
const INLINE_KEY = "sk-name-labels-inline";
const CALLER = ["api_key_id", "api_key_name", "team_id", "team_name", "user_id", "user_name"];
const NAME_LABELS = ["api_key_name", "team_name", "policy_name"];

function chat(app: SpawnedApp, key: string) {
  return new ProxyClient(app.proxyUrl, key).chat({
    model: MODEL, messages: [{ role: "user", content: "hi" }],
  });
}

async function scrapeText(app: SpawnedApp): Promise<string> {
  const res = await fetch(`${app.metricsUrl}/metrics`);
  expect(res.ok).toBe(true);
  return res.text();
}

describe("metric name labels resolve from the named documents", () => {
  let reachable = false;
  let upstream: OpenAiUpstream | undefined;
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  const teamId = randomUUID();
  const userId = randomUUID();
  let pkId = "";
  let policyId = "";
  let named: { id: string; value: Record<string, unknown> } | undefined;
  let bareId = "";

  beforeAll(async () => {
    const etcd = new EtcdClient();
    reachable = await etcd.ping();
    if (!reachable) return;
    upstream = await startOpenAiUpstream();
    app = await spawnApp({ extraEnv: {
      AISIX_OBSERVABILITY__METRICS__LABELS: JSON.stringify({
        aisix_proxy_requests_total: ["endpoint", ...CALLER],
        aisix_llm_input_tokens_total: CALLER,
        aisix_deployment_requests_total: ["model", "provider_key_id", "provider_key_name"],
        aisix_ratelimit_rejections_total: ["layer", "policy_id", "policy_name"],
        aisix_ratelimit_remaining_requests: ["api_key_id", "api_key_name", "model"],
      }),
    } });
    seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.update("teams", teamId, { name: "Platform" });
    await seed.update("users", userId, { name: "Alice From Users" });
    const pk = await seed.createProviderKey({
      display_name: "Primary OpenAI", secret: "sk-mock", api_base: `${upstream.baseUrl}/v1`,
    });
    pkId = pk.id;
    await seed.createModel({
      display_name: MODEL, provider: "openai", model_name: "gpt-4o-mini", provider_key_id: pk.id,
    });
    // Its `user_name` is deliberately not the users document's name: the
    // document is the member's current name and wins.
    named = await seed.createApiKey({
      key_hash: hash(NAMED_KEY), allowed_models: [MODEL], display_name: "Billing key",
      team_id: teamId, user_id: userId, user_name: "Stale Inline Name",
      rate_limit: { rpm: 1000 },
    });
    // No display name, and a team and member no document names.
    const bare = await seed.createApiKey({
      key_hash: hash(BARE_KEY), allowed_models: [MODEL], team_id: randomUUID(), user_id: randomUUID(),
    });
    bareId = bare.id;
    // A member with no users document keeps the key's own name.
    await seed.createApiKey({
      key_hash: hash(INLINE_KEY), allowed_models: [MODEL], user_id: randomUUID(), user_name: "Inline Only",
    });
    const policyKey = await seed.createApiKey({
      key_hash: hash(POLICY_KEY), allowed_models: [MODEL],
    });
    const policy = await seed.createRateLimitPolicy({
      name: "One per minute", scope: "api_key", scope_ref: policyKey.id, window: "minute", max_requests: 1,
    });
    policyId = policy.id;
    // Gate on the key seeded after the policy: watch events apply in
    // revision order, so it authenticating implies the whole seed set.
    const sentinel = "sk-name-labels-sentinel";
    await seed.createApiKey({ key_hash: hash(sentinel), allowed_models: [] });
    await waitConfigPropagation(async () => (await new ProxyClient(app!.proxyUrl, sentinel).listModels()).status === 200);
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("selected name labels carry the names of the caller and the credential", async (ctx) => {
    if (!reachable) return ctx.skip();
    const before = await scrapeMetrics(app!.metricsUrl);
    expect((await chat(app!, NAMED_KEY)).status).toBe(200);
    const after = await scrapeMetrics(app!.metricsUrl);
    const caller = {
      api_key_id: named!.id, api_key_name: "Billing key", team_id: teamId, team_name: "Platform",
      user_id: userId, user_name: "Alice From Users",
    };
    expect(metricDelta(before, after, "aisix_proxy_requests_total", caller)).toBe(1);
    expect(metricDelta(before, after, "aisix_llm_input_tokens_total", caller)).toBeGreaterThan(0);
    expect(metricDelta(before, after, "aisix_deployment_requests_total", {
      model: MODEL, provider_key_id: pkId, provider_key_name: "Primary OpenAI",
    })).toBe(1);
    expect(after.some((s) => s.name === "aisix_ratelimit_remaining_requests"
      && s.labels.api_key_id === named!.id && s.labels.api_key_name === "Billing key")).toBe(true);
  });

  test("a name nothing provides reads unknown, and a member without a users document keeps the key's own name", async (ctx) => {
    if (!reachable) return ctx.skip();
    const before = await scrapeMetrics(app!.metricsUrl);
    expect((await chat(app!, BARE_KEY)).status).toBe(200);
    expect((await chat(app!, INLINE_KEY)).status).toBe(200);
    const after = await scrapeMetrics(app!.metricsUrl);
    expect(metricDelta(before, after, "aisix_proxy_requests_total", {
      api_key_id: bareId, api_key_name: "unknown", team_name: "unknown", user_name: "unknown",
    })).toBe(1);
    expect(metricDelta(before, after, "aisix_proxy_requests_total", {
      api_key_name: "unknown", team_id: "unknown", team_name: "unknown", user_name: "Inline Only",
    })).toBe(1);
  });

  test("a rejection names its policy, and a layer with no policy leaves both labels empty alike", async (ctx) => {
    if (!reachable) return ctx.skip();
    const before = await scrapeMetrics(app!.metricsUrl);
    expect((await chat(app!, POLICY_KEY)).status).toBe(200);
    expect((await chat(app!, POLICY_KEY)).status).toBe(429);
    const after = await scrapeMetrics(app!.metricsUrl);
    expect(metricDelta(before, after, "aisix_ratelimit_rejections_total", {
      layer: "policy", policy_id: policyId, policy_name: "One per minute",
    })).toBe(1);
    for (const sample of after.filter((s) => s.name === "aisix_ratelimit_rejections_total")) {
      expect(sample.labels.policy_name === "", JSON.stringify(sample.labels)).toBe(sample.labels.policy_id === "");
    }
  });

  test("renaming a team relabels the next request without touching the key", async (ctx) => {
    if (!reachable) return ctx.skip();
    await seed!.update("teams", teamId, { name: "Platform Engineering" });
    await waitConfigPropagation(async () => {
      const before = await scrapeMetrics(app!.metricsUrl);
      if ((await chat(app!, NAMED_KEY)).status !== 200) return false;
      const after = await scrapeMetrics(app!.metricsUrl);
      return metricDelta(before, after, "aisix_proxy_requests_total", {
        api_key_id: named!.id, team_name: "Platform Engineering",
      }) === 1;
    });
  });

  test("renaming the key retires the remaining-quota series under its old name", async (ctx) => {
    if (!reachable) return ctx.skip();
    expect((await chat(app!, NAMED_KEY)).status).toBe(200);
    await seed!.update("api_keys", named!.id, { ...named!.value, display_name: "Billing key v2" });
    await waitConfigPropagation(async () => {
      if ((await chat(app!, NAMED_KEY)).status !== 200) return false;
      return (await scrapeText(app!)).includes(`api_key_name="Billing key v2"`);
    });
    const line = (scrape: string, name: string) => scrape.split("\n").find((l) =>
      l.startsWith("aisix_ratelimit_remaining_requests{")
      && l.includes(`api_key_id="${named!.id}"`) && l.includes(`api_key_name="${name}"`));
    // The sweep rides the periodic upkeep task; wait on its tick.
    const deadline = Date.now() + 30_000;
    let old: string | undefined;
    let current: string | undefined;
    for (;;) {
      const scrape = await scrapeText(app!);
      old = line(scrape, "Billing key");
      current = line(scrape, "Billing key v2");
      if (old?.endsWith(" NaN") || Date.now() > deadline) break;
      await new Promise((r) => setTimeout(r, 500));
    }
    expect(old, "the old name's series must still be exported").toBeTruthy();
    expect(old).toMatch(/ NaN$/);
    expect(Number.isFinite(Number(current!.slice(current!.lastIndexOf(" ") + 1)))).toBe(true);
  }, 60_000);
});

describe("name labels are opt-in", () => {
  let reachable = false;
  let upstream: OpenAiUpstream | undefined;
  let app: SpawnedApp | undefined;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    reachable = await etcd.ping();
    if (!reachable) return;
    upstream = await startOpenAiUpstream();
    app = await spawnApp({});
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const teamId = randomUUID();
    await seed.update("teams", teamId, { name: "Platform" });
    const pk = await seed.createProviderKey({
      display_name: "Primary OpenAI", secret: "sk-mock", api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: MODEL, provider: "openai", model_name: "gpt-4o-mini", provider_key_id: pk.id,
    });
    const key = await seed.createApiKey({
      key_hash: hash(NAMED_KEY), allowed_models: [MODEL], display_name: "Billing key",
      team_id: teamId, rate_limit: { rpm: 1000 },
    });
    await seed.createRateLimitPolicy({
      name: "One per minute", scope: "api_key", scope_ref: key.id, window: "minute", max_requests: 1,
    });
    const sentinel = "sk-name-labels-sentinel";
    await seed.createApiKey({ key_hash: hash(sentinel), allowed_models: [] });
    await waitConfigPropagation(async () => (await new ProxyClient(app!.proxyUrl, sentinel).listModels()).status === 200);
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("with no label selection no family renders a name label, and the defaults keep their sets", async (ctx) => {
    if (!reachable) return ctx.skip();
    expect((await chat(app!, NAMED_KEY)).status).toBe(200);
    expect((await chat(app!, NAMED_KEY)).status).toBe(429);
    const scrape = await scrapeText(app!);
    for (const label of NAME_LABELS) expect(scrape, label).not.toContain(`${label}=`);
    const samples = await scrapeMetrics(app!.metricsUrl);
    const keys = (name: string) => Object.keys(samples.find((s) => s.name === name)!.labels);
    expect(keys("aisix_proxy_requests_total")).toEqual([
      "endpoint", "inbound_protocol", "upstream_protocol", "provider", "model", "upstream_model",
      "provider_key_id", "provider_key_name", "api_key_id", "team_id", "user_id", "user_name",
      "stream", "is_fallback", "status", "outcome",
    ]);
    expect(keys("aisix_deployment_requests_total")).toEqual(["provider", "model", "upstream_model", "provider_key_id"]);
    expect(keys("aisix_ratelimit_rejections_total")).toEqual(["scope", "layer", "policy_id"]);
    expect(keys("aisix_ratelimit_remaining_requests")).toEqual(["api_key_id", "model"]);
  });
});
