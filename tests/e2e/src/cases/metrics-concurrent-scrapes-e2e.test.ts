import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  spawnApp,
  startOpenAiUpstream,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";
import { metricDelta, scrapeMetrics } from "../harness/metrics.js";

const MODEL = "concurrent-metrics";
const KEYS = Array.from({ length: 64 }, (_, i) => `sk-concurrent-metrics-${i}`);

function resources(base: string): string {
  return `
_format_version: "1"
provider_keys:
  - display_name: metrics-upstream
    provider: openai
    api_key: sk-mock
    api_base: ${base}/v1
models:
  - display_name: ${MODEL}
    provider: openai
    model_name: mock-model
    provider_key: metrics-upstream
api_keys:
${KEYS.map((key, i) => `  - display_name: metrics-caller-${i}
    key_hash: ${createHash("sha256").update(key).digest("hex")}
    allowed_models: [${MODEL}]`).join("\n")}
`;
}

describe("concurrent and cancelled metric scrapes preserve request observations", () => {
  let app: SpawnedApp;
  let upstream: OpenAiUpstream;

  beforeAll(async () => {
    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "metrics-response",
        object: "chat.completion",
        model: "mock-model",
        choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
        usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
      },
    });
    app = await spawnApp({ resourcesFile: resources(upstream.baseUrl) });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("scrapes leave counters, summaries and histograms complete while traffic continues", async () => {
    const before = await scrapeMetrics(app.metricsUrl);
    const traffic = async () => {
      for (let offset = 0; offset < KEYS.length; offset += 8) {
        await Promise.all(KEYS.slice(offset, offset + 8).map(async (key) => {
          const response = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
            method: "POST",
            headers: { authorization: `Bearer ${key}`, "content-type": "application/json" },
            body: JSON.stringify({ model: MODEL, messages: [{ role: "user", content: "metrics" }] }),
          });
          const body = await response.json();
          expect(response.status, JSON.stringify(body)).toBe(200);
          expect(body.choices[0].message.content).toBe("ok");
        }));
      }
    };
    await traffic();
    await Promise.all([
      traffic(),
      ...Array.from({ length: 6 }, async () => {
        const response = await fetch(`${app.metricsUrl}/metrics`);
        expect(response.status).toBe(200);
        expect(response.headers.get("content-type")).toContain("text/plain; version=0.0.4");
        expect(await response.text()).toContain("aisix_proxy_request_duration_seconds_count");
      }),
      (async () => {
        const response = await fetch(`${app.metricsUrl}/metrics`);
        expect(response.status).toBe(200);
        await response.body!.cancel();
      })(),
    ]);
    const health = await fetch(`${app.proxyUrl}/livez`);
    expect(health.status).toBe(200);
    await health.text();

    const after = await scrapeMetrics(app.metricsUrl);
    const requests = KEYS.length * 2;
    for (const family of [
      "aisix_proxy_requests_total",
      "aisix_llm_requests_total",
      "aisix_proxy_request_duration_seconds_count",
      "aisix_llm_request_duration_seconds_count",
      "aisix_request_e2e_latency_seconds_count",
    ]) {
      expect(metricDelta(before, after, family, { model: MODEL }), family).toBe(requests);
    }
    expect(metricDelta(before, after, "aisix_llm_input_tokens_total", { model: MODEL })).toBe(requests * 3);
    expect(metricDelta(before, after, "aisix_llm_output_tokens_total", { model: MODEL })).toBe(requests * 2);
    expect(metricDelta(before, after, "aisix_request_e2e_latency_seconds_bucket", { model: MODEL, le: "+Inf" })).toBe(requests);

    const durations = after.filter((s) => s.name === "aisix_proxy_request_duration_seconds" && s.labels.model === MODEL);
    expect(new Set(durations.map((s) => s.labels.api_key_id)).size).toBe(KEYS.length);
    expect([...new Set(durations.map((s) => s.labels.quantile))].sort()).toEqual(["0", "0.5", "0.9", "0.95", "0.99", "0.999", "1"]);
  });
});
