import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient, ProxyClient, SeedClient, spawnApp, startOpenAiUpstream,
  waitConfigPropagation, type OpenAiUpstream, type SpawnedApp,
} from "../harness/index.js";
import { metricDelta, scrapeMetrics, sumMetric } from "../harness/metrics.js";

const KEY = "sk-metric-label-selection";
const TTFT = "aisix_request_ttft_seconds";
const E2E = "aisix_request_e2e_latency_seconds";
const contexts = [
  { endpoint: "chat/completions", provider: "openai", wire: "chat" },
  { endpoint: "messages", provider: "deepseek", wire: "chat" },
  { endpoint: "responses", provider: "deepseek", wire: "chat" },
  { endpoint: "messages", provider: "anthropic", wire: "anthropic" },
  { endpoint: "responses", provider: "openai", wire: "responses" },
];
const usage = { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 };
const streams: Record<string, unknown[]> = {
  chat: [
    { id: "labels", model: "upstream", choices: [{ index: 0, delta: { content: "ok" }, finish_reason: null }] },
    { id: "labels", model: "upstream", choices: [{ index: 0, delta: {}, finish_reason: "stop" }], usage },
  ],
  responses: [
    { type: "response.created", response: { id: "resp-labels", model: "upstream", status: "in_progress" } },
    { type: "response.completed", response: { id: "resp-labels", model: "upstream", status: "completed", output: [], usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 } } },
  ],
  anthropic: [
    { type: "message_start", message: { id: "msg-labels", model: "upstream", type: "message", role: "assistant", content: [], usage: { input_tokens: 10, output_tokens: 0 } } },
    { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "ok" } },
    { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 5 } },
    { type: "message_stop" },
  ],
};

describe("metric label configuration is applied to real request observations", () => {
  let app: SpawnedApp | undefined;
  const upstreams: OpenAiUpstream[] = [];
  const scenarios: Array<{ endpoint: string; model: string; name: string }> = [];
  let reachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    reachable = await etcd.ping();
    if (!reachable) return;
    app = await spawnApp({ extraEnv: {
      AISIX_OBSERVABILITY__METRICS__LABELS: JSON.stringify({
        [TTFT]: ["provider_key_name", "upstream_model", "api_key_id"],
        [E2E]: ["endpoint", "provider_key_name"],
        aisix_proxy_requests_total: ["endpoint"],
        aisix_requests_total: [],
      }),
    } });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    for (const [index, context] of contexts.entries()) {
      const upstream = await startOpenAiUpstream({
        streamEvents: [...streams[context.wire].map((v) => JSON.stringify(v)), "[DONE]"],
        firstEventDelayMs: 20,
      });
      upstreams.push(upstream);
      for (const suffix of ["a", "b"]) {
        const name = `credential-${index}-${suffix}`;
        const pk = await seed.createProviderKey({
          display_name: name, provider: context.provider,
          adapter: context.wire === "anthropic" ? "anthropic" : "openai",
          secret: "test", api_base: `${upstream.baseUrl}/v1`,
        });
        const model = `labels-${index}-${suffix}`;
        await seed.createModel({
          display_name: model, model_name: "upstream", provider: context.provider, provider_key_id: pk.id,
        });
        scenarios.push({ endpoint: context.endpoint, model, name });
      }
    }
    await seed.createApiKey({ key_hash: createHash("sha256").update(KEY).digest("hex"), allowed_models: scenarios.map((s) => s.model) });
    const client = new ProxyClient(app.proxyUrl, KEY);
    await waitConfigPropagation(async () => (await client.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  test("unsupported label configuration fails startup", async (ctx) => {
    if (!reachable) return ctx.skip();
    await expect(spawnApp({ extraEnv: {
      AISIX_OBSERVABILITY__METRICS__LABELS: JSON.stringify({ [TTFT]: ["request_id"] }),
    } })).rejects.toThrow(/unsupported variable.*request_id/);
  });

  test("credentials partition all histogram components while removed counter labels aggregate", async (ctx) => {
    if (!reachable) return ctx.skip();
    const before = await scrapeMetrics(app!.metricsUrl);
    for (const scenario of scenarios) {
      for (let repeat = 0; repeat < 2; repeat++) {
        const response = await fetch(`${app!.proxyUrl}/v1/${scenario.endpoint}`, {
          method: "POST", headers: { authorization: `Bearer ${KEY}`, "content-type": "application/json" },
          body: JSON.stringify({ model: scenario.model, stream: true, max_tokens: 100,
            ...(scenario.endpoint === "responses" ? { input: "hello" } : { messages: [{ role: "user", content: "hello" }] }),
          }),
        });
        const body = await response.text();
        expect(response.status, `${scenario.model}: ${body}`).toBe(200);
      }
    }
    await expect.poll(async () => sumMetric(await scrapeMetrics(app!.metricsUrl), `${TTFT}_count`), { timeout: 5000 }).toBe(scenarios.length * 2);
    const after = await scrapeMetrics(app!.metricsUrl);
    expect(metricDelta(before, after, "aisix_requests_total")).toBe(scenarios.length * 2);
    expect(after.filter((s) => s.name === "aisix_requests_total")).toHaveLength(1);
    for (const sample of after.filter((s) => s.name === "aisix_requests_total")) expect(sample.labels).toEqual({});
    for (const scenario of scenarios) {
      for (const family of [TTFT, E2E]) {
        const samples = after.filter((s) => s.name.startsWith(`${family}_`) && s.labels.provider_key_name === scenario.name);
        expect(samples.length, scenario.name).toBeGreaterThan(2);
        expect(sumMetric(samples, `${family}_count`)).toBe(2);
        expect(sumMetric(samples, `${family}_bucket`, { le: "+Inf" })).toBe(2);
        expect(sumMetric(samples, `${family}_sum`)).toBeGreaterThan(0);
        for (const sample of samples) {
          const labels = Object.keys(sample.labels).filter((key) => key !== "le").sort();
          expect(labels).toEqual((family === TTFT ? ["provider_key_name", "upstream_model", "api_key_id"] : ["endpoint", "provider_key_name"]).sort());
          if (family === TTFT) {
            expect(sample.labels.upstream_model).toBe("upstream");
            expect(sample.labels.api_key_id).not.toBe("unknown");
          }
        }
      }
    }
    for (const endpoint of new Set(scenarios.map((s) => `/v1/${s.endpoint}`))) {
      expect(metricDelta(before, after, "aisix_proxy_requests_total", { endpoint }))
        .toBe(scenarios.filter((s) => `/v1/${s.endpoint}` === endpoint).length * 2);
    }
    for (const sample of after.filter((s) => s.name === "aisix_proxy_requests_total")) expect(Object.keys(sample.labels)).toEqual(["endpoint"]);
    const defaults = after.find((s) => s.name === "aisix_llm_requests_total")!;
    expect(defaults.labels).toHaveProperty("model");
    expect(defaults.labels).toHaveProperty("provider_key_id");
  });
});
