import { createHash } from "node:crypto";
import OpenAI from "openai";
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

// An ensemble emits one UsageEvent per panel member and judge, but its
// Prometheus families describe the caller-visible request. They therefore
// carry the panel+judge aggregate under the ensemble alias. Before this
// coverage test, both streaming and non-streaming ensemble requests silently
// emitted zero token metrics, and streaming omitted the detailed TTFT family.

const CALLER = "sk-ensemble-prometheus-caller";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const NONSTREAM_MODEL = "prom-ensemble-nonstream";
const STREAM_MODEL = "prom-ensemble-stream";

const PANEL_USAGE = { prompt_tokens: 2, completion_tokens: 3, total_tokens: 5 };
const JUDGE_USAGE = { prompt_tokens: 7, completion_tokens: 11, total_tokens: 18 };
const AGGREGATE = { input: 11, output: 17, total: 28 };

function chatBody(id: string, content: string, usage: typeof PANEL_USAGE) {
  return {
    id,
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
    usage,
  };
}

function streamEvents() {
  return [
    JSON.stringify({
      id: "chatcmpl-prom-ensemble-stream",
      object: "chat.completion.chunk",
      model: "gpt-4o-mini",
      choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
    }),
    JSON.stringify({
      id: "chatcmpl-prom-ensemble-stream",
      object: "chat.completion.chunk",
      model: "gpt-4o-mini",
      choices: [{ index: 0, delta: { content: "streamed synthesis" }, finish_reason: null }],
    }),
    JSON.stringify({
      id: "chatcmpl-prom-ensemble-stream",
      object: "chat.completion.chunk",
      model: "gpt-4o-mini",
      choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
      usage: JUDGE_USAGE,
    }),
    "[DONE]",
  ];
}

function metricValue(
  scrape: string,
  series: string,
  labels: Record<string, string>,
): number {
  return scrape
    .split("\n")
    .filter(
      (line) =>
        line.startsWith(`${series}{`) &&
        Object.entries(labels).every(([key, value]) => line.includes(`${key}="${value}"`)),
    )
    .map((line) => Number(line.trim().split(/\s+/).at(-1)))
    .filter((value) => !Number.isNaN(value))
    .reduce((sum, value) => sum + value, 0);
}

describe("ensemble Prometheus token and TTFT coverage", () => {
  let app: SpawnedApp | undefined;
  const upstreams: OpenAiUpstream[] = [];
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    const panelA = await startOpenAiUpstream({
      nonStreamBody: chatBody("chatcmpl-panel-a", "panel answer A", PANEL_USAGE),
    });
    const panelB = await startOpenAiUpstream({
      nonStreamBody: chatBody("chatcmpl-panel-b", "panel answer B", PANEL_USAGE),
    });
    const judgeNonstream = await startOpenAiUpstream({
      nonStreamBody: chatBody("chatcmpl-judge-ns", "buffered synthesis", JUDGE_USAGE),
    });
    const judgeStream = await startOpenAiUpstream({
      streamEvents: streamEvents(),
      firstEventDelayMs: 25,
    });
    upstreams.push(panelA, panelB, judgeNonstream, judgeStream);

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const seedDirect = async (displayName: string, upstream: OpenAiUpstream) => {
      const pk = await seed.createProviderKey({
        display_name: `${displayName}-pk`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      await seed.createModel({
        display_name: displayName,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
    };
    await seedDirect("prom-ensemble-panel-a", panelA);
    await seedDirect("prom-ensemble-panel-b", panelB);
    await seedDirect("prom-ensemble-judge-ns", judgeNonstream);
    await seedDirect("prom-ensemble-judge-stream", judgeStream);

    await seed.createModel({
      display_name: NONSTREAM_MODEL,
      ensemble: {
        panel: [{ model: "prom-ensemble-panel-a" }, { model: "prom-ensemble-panel-b" }],
        judge: { model: "prom-ensemble-judge-ns" },
        min_responses: 2,
      },
    });
    await seed.createModel({
      display_name: STREAM_MODEL,
      ensemble: {
        panel: [{ model: "prom-ensemble-panel-a" }, { model: "prom-ensemble-panel-b" }],
        judge: { model: "prom-ensemble-judge-stream" },
        min_responses: 2,
      },
    });
    await seed.createApiKey({
      key_hash: CALLER_HASH,
      allowed_models: [
        NONSTREAM_MODEL,
        STREAM_MODEL,
        "prom-ensemble-panel-a",
        "prom-ensemble-panel-b",
        "prom-ensemble-judge-ns",
        "prom-ensemble-judge-stream",
      ],
    });

    const proxy = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => {
      const res = await proxy.listModels();
      if (res.status !== 200) return false;
      const models = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return [NONSTREAM_MODEL, STREAM_MODEL].every((name) =>
        models.some((model) => model.id === name),
      );
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((upstream) => upstream.close()));
  });

  const scrape = async (): Promise<string> => {
    const res = await fetch(`${app!.metricsUrl}/metrics`);
    expect(res.status).toBe(200);
    return res.text();
  };

  const tokenLabels = (model: string) => ({
    endpoint: "/v1/chat/completions",
    provider: "ensemble",
    model,
    upstream_model: "unknown",
    provider_key_id: "unknown",
    provider_key_name: "unknown",
  });

  test("non-streaming request records the aggregate token counters under the ensemble alias", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const before = await scrape();
    const client = new OpenAI({ apiKey: CALLER, baseURL: `${app.proxyUrl}/v1`, maxRetries: 0 });
    const response = await client.chat.completions.create({
      model: NONSTREAM_MODEL,
      messages: [{ role: "user", content: "synthesize" }],
    });
    expect(response.usage?.prompt_tokens).toBe(AGGREGATE.input);
    expect(response.usage?.completion_tokens).toBe(AGGREGATE.output);
    expect(response.usage?.total_tokens).toBe(AGGREGATE.total);

    const after = await scrape();
    const labels = tokenLabels(NONSTREAM_MODEL);
    expect(
      metricValue(after, "aisix_llm_input_tokens_total", labels) -
        metricValue(before, "aisix_llm_input_tokens_total", labels),
    ).toBe(AGGREGATE.input);
    expect(
      metricValue(after, "aisix_llm_output_tokens_total", labels) -
        metricValue(before, "aisix_llm_output_tokens_total", labels),
    ).toBe(AGGREGATE.output);
    expect(
      metricValue(after, "aisix_llm_total_tokens_total", labels) -
        metricValue(before, "aisix_llm_total_tokens_total", labels),
    ).toBe(AGGREGATE.total);
  });

  test("streaming request records aggregate tokens and both TTFT families once", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const before = await scrape();
    const client = new OpenAI({ apiKey: CALLER, baseURL: `${app.proxyUrl}/v1`, maxRetries: 0 });
    const stream = await client.chat.completions.create({
      model: STREAM_MODEL,
      messages: [{ role: "user", content: "stream a synthesis" }],
      stream: true,
    });
    let content = "";
    for await (const chunk of stream) {
      content += chunk.choices[0]?.delta?.content ?? "";
    }
    expect(content).toBe("streamed synthesis");

    const labels = tokenLabels(STREAM_MODEL);
    let after = "";
    const deadline = Date.now() + 10_000;
    for (;;) {
      after = await scrape();
      if (
        metricValue(after, "aisix_llm_total_tokens_total", labels) -
          metricValue(before, "aisix_llm_total_tokens_total", labels) ===
          AGGREGATE.total &&
        metricValue(after, "aisix_request_ttft_seconds_count", {
          endpoint: "/v1/chat/completions",
          provider: "ensemble",
          model: STREAM_MODEL,
          streaming: "true",
        }) -
          metricValue(before, "aisix_request_ttft_seconds_count", {
            endpoint: "/v1/chat/completions",
            provider: "ensemble",
            model: STREAM_MODEL,
            streaming: "true",
          }) ===
          1
      ) {
        break;
      }
      if (Date.now() > deadline) break;
      await new Promise((resolve) => setTimeout(resolve, 100));
    }

    expect(
      metricValue(after, "aisix_llm_input_tokens_total", labels) -
        metricValue(before, "aisix_llm_input_tokens_total", labels),
    ).toBe(AGGREGATE.input);
    expect(
      metricValue(after, "aisix_llm_output_tokens_total", labels) -
        metricValue(before, "aisix_llm_output_tokens_total", labels),
    ).toBe(AGGREGATE.output);
    expect(
      metricValue(after, "aisix_llm_total_tokens_total", labels) -
        metricValue(before, "aisix_llm_total_tokens_total", labels),
    ).toBe(AGGREGATE.total);

    const lowCardLabels = {
      endpoint: "/v1/chat/completions",
      provider: "ensemble",
      model: STREAM_MODEL,
      streaming: "true",
    };
    expect(
      metricValue(after, "aisix_request_ttft_seconds_count", lowCardLabels) -
        metricValue(before, "aisix_request_ttft_seconds_count", lowCardLabels),
    ).toBe(1);
    expect(
      metricValue(after, "aisix_llm_time_to_first_token_seconds_count", {
        ...labels,
        inbound_protocol: "openai",
        upstream_protocol: "unknown",
      }) -
        metricValue(before, "aisix_llm_time_to_first_token_seconds_count", {
          ...labels,
          inbound_protocol: "openai",
          upstream_protocol: "unknown",
        }),
    ).toBe(1);
  });
});
