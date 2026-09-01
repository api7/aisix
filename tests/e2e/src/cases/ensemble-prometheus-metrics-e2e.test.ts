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
const NONSTREAM_ESTIMATED_MODEL = "prom-ensemble-nonstream-estimated";
const STREAM_ESTIMATED_MODEL = "prom-ensemble-stream-estimated";
const NONSTREAM_FAILED_ESTIMATED_MODEL = "prom-ensemble-nonstream-failed-estimated";
const STREAM_FAILED_ESTIMATED_MODEL = "prom-ensemble-stream-failed-estimated";
const PANEL_MEMBER_QUOTA_MODEL = "prom-ensemble-panel-member-quota";
const JUDGE_MEMBER_QUOTA_MODEL = "prom-ensemble-judge-member-quota";
const NONSTREAM_FAILURE_CALLER = "sk-ensemble-failed-nonstream";
const STREAM_FAILURE_CALLER = "sk-ensemble-failed-stream";

const PANEL_USAGE = { prompt_tokens: 2, completion_tokens: 3, total_tokens: 5 };
const JUDGE_USAGE = { prompt_tokens: 7, completion_tokens: 11, total_tokens: 18 };
const AGGREGATE = { input: 11, output: 17, total: 28 };

function chatBody(id: string, content: string, usage?: typeof PANEL_USAGE) {
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
    ...(usage ? { usage } : {}),
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
    const panelNoUsageA = await startOpenAiUpstream({
      nonStreamBody: chatBody("chatcmpl-panel-est-a", "estimated panel answer A"),
    });
    const panelNoUsageB = await startOpenAiUpstream({
      nonStreamBody: chatBody("chatcmpl-panel-est-b", "estimated panel answer B"),
    });
    const judgeNoUsage = await startOpenAiUpstream({
      nonStreamBody: chatBody("chatcmpl-judge-est", "estimated buffered synthesis"),
    });
    const judgeFailure = await startOpenAiUpstream({
      status: 500,
      errorBody: { error: { message: "judge unavailable", type: "server_error" } },
    });
    const panelFailure = await startOpenAiUpstream({
      status: 500,
      errorBody: { error: { message: "panel unavailable", type: "server_error" } },
    });
    const quotaPanel = await startOpenAiUpstream({
      nonStreamBody: chatBody("chatcmpl-panel-quota", "estimated quota panel answer"),
    });
    const quotaJudge = await startOpenAiUpstream({
      nonStreamBody: chatBody("chatcmpl-judge-quota", "estimated quota synthesis"),
    });
    upstreams.push(
      panelA,
      panelB,
      judgeNonstream,
      judgeStream,
      panelNoUsageA,
      panelNoUsageB,
      judgeNoUsage,
      judgeFailure,
      panelFailure,
      quotaPanel,
      quotaJudge,
    );

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const seedDirect = async (
      displayName: string,
      upstream: OpenAiUpstream,
      extra: Record<string, unknown> = {},
    ) => {
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
        ...extra,
      });
    };
    await seedDirect("prom-ensemble-panel-a", panelA);
    await seedDirect("prom-ensemble-panel-b", panelB);
    await seedDirect("prom-ensemble-judge-ns", judgeNonstream);
    await seedDirect("prom-ensemble-judge-stream", judgeStream);
    await seedDirect("prom-ensemble-panel-est-a", panelNoUsageA);
    await seedDirect("prom-ensemble-panel-est-b", panelNoUsageB);
    await seedDirect("prom-ensemble-judge-est", judgeNoUsage);
    await seedDirect("prom-ensemble-judge-fail", judgeFailure);
    await seedDirect("prom-ensemble-panel-fail", panelFailure);
    await seedDirect("prom-ensemble-panel-quota", quotaPanel, {
      rate_limit: { tpd: 1 },
    });
    await seedDirect("prom-ensemble-judge-quota", quotaJudge, {
      rate_limit: { tpd: 1 },
    });

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
    await seed.createModel({
      display_name: NONSTREAM_ESTIMATED_MODEL,
      ensemble: {
        panel: [
          { model: "prom-ensemble-panel-est-a" },
          { model: "prom-ensemble-panel-est-b" },
        ],
        judge: { model: "prom-ensemble-judge-est" },
        min_responses: 2,
      },
    });
    await seed.createModel({
      display_name: STREAM_ESTIMATED_MODEL,
      ensemble: {
        panel: [
          { model: "prom-ensemble-panel-est-a" },
          { model: "prom-ensemble-panel-est-b" },
        ],
        judge: { model: "prom-ensemble-judge-stream" },
        min_responses: 2,
      },
    });
    await seed.createModel({
      display_name: NONSTREAM_FAILED_ESTIMATED_MODEL,
      ensemble: {
        panel: [
          { model: "prom-ensemble-panel-est-a" },
          { model: "prom-ensemble-panel-fail" },
        ],
        judge: { model: "prom-ensemble-judge-fail" },
        min_responses: 2,
      },
    });
    await seed.createModel({
      display_name: STREAM_FAILED_ESTIMATED_MODEL,
      ensemble: {
        panel: [
          { model: "prom-ensemble-panel-est-a" },
          { model: "prom-ensemble-panel-est-b" },
        ],
        judge: { model: "prom-ensemble-judge-fail" },
        min_responses: 2,
      },
    });
    await seed.createModel({
      display_name: PANEL_MEMBER_QUOTA_MODEL,
      ensemble: {
        panel: [
          { model: "prom-ensemble-panel-quota" },
          { model: "prom-ensemble-panel-a" },
        ],
        judge: { model: "prom-ensemble-judge-ns" },
        min_responses: 2,
      },
    });
    await seed.createModel({
      display_name: JUDGE_MEMBER_QUOTA_MODEL,
      ensemble: {
        panel: [{ model: "prom-ensemble-panel-a" }, { model: "prom-ensemble-panel-b" }],
        judge: { model: "prom-ensemble-judge-quota" },
        min_responses: 2,
      },
    });
    await seed.createApiKey({
      key_hash: CALLER_HASH,
      allowed_models: [
        NONSTREAM_MODEL,
        STREAM_MODEL,
        NONSTREAM_ESTIMATED_MODEL,
        STREAM_ESTIMATED_MODEL,
        NONSTREAM_FAILED_ESTIMATED_MODEL,
        STREAM_FAILED_ESTIMATED_MODEL,
        PANEL_MEMBER_QUOTA_MODEL,
        JUDGE_MEMBER_QUOTA_MODEL,
        "prom-ensemble-panel-a",
        "prom-ensemble-panel-b",
        "prom-ensemble-judge-ns",
        "prom-ensemble-judge-stream",
        "prom-ensemble-panel-est-a",
        "prom-ensemble-panel-est-b",
        "prom-ensemble-judge-est",
        "prom-ensemble-panel-quota",
        "prom-ensemble-judge-quota",
      ],
    });
    for (const [caller, model] of [
      [NONSTREAM_FAILURE_CALLER, NONSTREAM_FAILED_ESTIMATED_MODEL],
      [STREAM_FAILURE_CALLER, STREAM_FAILED_ESTIMATED_MODEL],
    ] as const) {
      await seed.createApiKey({
        key_hash: createHash("sha256").update(caller).digest("hex"),
        allowed_models: [
          model,
          "prom-ensemble-panel-est-a",
          "prom-ensemble-panel-est-b",
          "prom-ensemble-panel-fail",
          "prom-ensemble-judge-fail",
        ],
        rate_limit: { tpd: 1 },
      });
    }

    const proxy = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => {
      const res = await proxy.listModels();
      if (res.status !== 200) return false;
      const models = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return [
        NONSTREAM_MODEL,
        STREAM_MODEL,
        NONSTREAM_ESTIMATED_MODEL,
        STREAM_ESTIMATED_MODEL,
        NONSTREAM_FAILED_ESTIMATED_MODEL,
        STREAM_FAILED_ESTIMATED_MODEL,
        PANEL_MEMBER_QUOTA_MODEL,
        JUDGE_MEMBER_QUOTA_MODEL,
      ].every((name) =>
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

  const ensembleRequest = async (
    model: string,
    caller: string,
    stream: boolean,
  ): Promise<Response> =>
    fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${caller}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: "bill surviving panel members" }],
        stream,
      }),
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
      metricValue(after, "aisix_request_ttft_seconds_sum", lowCardLabels) -
        metricValue(before, "aisix_request_ttft_seconds_sum", lowCardLabels),
    ).toBeGreaterThan(0);
    const detailedLabels = {
      ...labels,
      inbound_protocol: "openai",
      upstream_protocol: "unknown",
    };
    expect(
      metricValue(after, "aisix_llm_time_to_first_token_seconds_count", detailedLabels) -
        metricValue(before, "aisix_llm_time_to_first_token_seconds_count", detailedLabels),
    ).toBe(1);
    expect(
      metricValue(after, "aisix_llm_time_to_first_token_seconds_sum", detailedLabels) -
        metricValue(before, "aisix_llm_time_to_first_token_seconds_sum", detailedLabels),
    ).toBeGreaterThan(0);
  });

  test("locally estimated subcalls contribute to non-streaming and streaming aggregates", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const before = await scrape();
    const client = new OpenAI({ apiKey: CALLER, baseURL: `${app.proxyUrl}/v1`, maxRetries: 0 });

    const nonstream = await client.chat.completions.create({
      model: NONSTREAM_ESTIMATED_MODEL,
      messages: [{ role: "user", content: "estimate every subcall" }],
    });
    expect(nonstream.choices[0]?.message?.content).toContain("estimated buffered synthesis");

    const stream = await client.chat.completions.create({
      model: STREAM_ESTIMATED_MODEL,
      messages: [{ role: "user", content: "estimate the panel" }],
      stream: true,
    });
    for await (const _chunk of stream) {
      // Drain the stream so on_complete records its aggregate.
    }

    let after = "";
    const deadline = Date.now() + 10_000;
    for (;;) {
      after = await scrape();
      const nonstreamTotal =
        metricValue(
          after,
          "aisix_llm_total_tokens_total",
          tokenLabels(NONSTREAM_ESTIMATED_MODEL),
        ) -
        metricValue(
          before,
          "aisix_llm_total_tokens_total",
          tokenLabels(NONSTREAM_ESTIMATED_MODEL),
        );
      const streamTotal =
        metricValue(
          after,
          "aisix_llm_total_tokens_total",
          tokenLabels(STREAM_ESTIMATED_MODEL),
        ) -
        metricValue(
          before,
          "aisix_llm_total_tokens_total",
          tokenLabels(STREAM_ESTIMATED_MODEL),
        );
      if (nonstreamTotal > 0 && streamTotal > JUDGE_USAGE.total_tokens) break;
      if (Date.now() > deadline) break;
      await new Promise((resolve) => setTimeout(resolve, 100));
    }

    for (const metric of [
      "aisix_llm_input_tokens_total",
      "aisix_llm_output_tokens_total",
      "aisix_llm_total_tokens_total",
    ]) {
      expect(
        metricValue(after, metric, tokenLabels(NONSTREAM_ESTIMATED_MODEL)) -
          metricValue(before, metric, tokenLabels(NONSTREAM_ESTIMATED_MODEL)),
        `${metric} non-streaming estimate`,
      ).toBeGreaterThan(0);
    }
    expect(
      metricValue(
        after,
        "aisix_llm_input_tokens_total",
        tokenLabels(STREAM_ESTIMATED_MODEL),
      ) -
        metricValue(
          before,
          "aisix_llm_input_tokens_total",
          tokenLabels(STREAM_ESTIMATED_MODEL),
        ),
    ).toBeGreaterThan(JUDGE_USAGE.prompt_tokens);
    expect(
      metricValue(
        after,
        "aisix_llm_output_tokens_total",
        tokenLabels(STREAM_ESTIMATED_MODEL),
      ) -
        metricValue(
          before,
          "aisix_llm_output_tokens_total",
          tokenLabels(STREAM_ESTIMATED_MODEL),
        ),
    ).toBeGreaterThan(JUDGE_USAGE.completion_tokens);
    expect(
      metricValue(
        after,
        "aisix_llm_total_tokens_total",
        tokenLabels(STREAM_ESTIMATED_MODEL),
      ) -
        metricValue(
          before,
          "aisix_llm_total_tokens_total",
          tokenLabels(STREAM_ESTIMATED_MODEL),
        ),
    ).toBeGreaterThan(JUDGE_USAGE.total_tokens);
  });

  for (const [name, model, caller, stream] of [
    [
      "non-streaming",
      NONSTREAM_FAILED_ESTIMATED_MODEL,
      NONSTREAM_FAILURE_CALLER,
      false,
    ],
    ["streaming", STREAM_FAILED_ESTIMATED_MODEL, STREAM_FAILURE_CALLER, true],
  ] as const) {
    test(`${name} failure commits estimated usage for successful panel members`, async (ctx) => {
      if (!etcdReachable || !app) {
        ctx.skip();
        return;
      }

      const first = await ensembleRequest(model, caller, stream);
      await first.text();
      expect(first.status).toBe(502);

      const second = await ensembleRequest(model, caller, stream);
      await second.text();
      expect(second.status).toBe(429);
    });
  }

  test("estimated panel usage is committed to the panel member model quota", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const first = await ensembleRequest(PANEL_MEMBER_QUOTA_MODEL, CALLER, false);
    await first.text();
    expect(first.status).toBe(200);

    const second = await ensembleRequest(PANEL_MEMBER_QUOTA_MODEL, CALLER, false);
    await second.text();
    expect(second.status).toBe(502);
  });

  test("estimated judge usage is committed to the judge member model quota", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const first = await ensembleRequest(JUDGE_MEMBER_QUOTA_MODEL, CALLER, false);
    await first.text();
    expect(first.status).toBe(200);

    const second = await ensembleRequest(JUDGE_MEMBER_QUOTA_MODEL, CALLER, false);
    await second.text();
    expect(second.status).toBe(429);
  });
});
