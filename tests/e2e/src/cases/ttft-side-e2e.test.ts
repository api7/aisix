import { createServer, type Server } from "node:http";
import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  pickFreePort,
  scrapeMetrics,
  spawnApp,
  startOpenAiUpstream,
  sumMetric,
  waitConfigPropagation,
  type MetricSample,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: the two TTFT metrics carry `side`. `upstream` is the attempt's wait
// for the upstream's first frame; `downstream` is what the caller waited
// for, from the gateway receiving the request to the first frame handed to
// the client. An input guardrail runs before the attempt starts, so its cost
// must land on the downstream side only — that is what this spec drives,
// on every streaming endpoint that records TTFT.

const KEY = "sk-ttft-side-e2e";
const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

/** How long the input guardrail's provider takes to answer. */
const GUARDRAIL_DELAY_MS = 600;
const D = GUARDRAIL_DELAY_MS / 1000;

const REQUEST_TTFT = "aisix_request_ttft_seconds";
const DETAILED_TTFT = "aisix_llm_time_to_first_token_seconds";

/** green-cip stand-in that passes everything, GUARDRAIL_DELAY_MS late. */
async function startSlowAliyun(): Promise<{ baseUrl: string; close(): Promise<void> }> {
  const server: Server = createServer((req, res) => {
    req.resume();
    req.on("end", () => {
      setTimeout(() => {
        res.statusCode = 200;
        res.setHeader("content-type", "application/json");
        res.end(
          JSON.stringify({
            Code: 200,
            Message: "OK",
            RequestId: "slow",
            Data: { RiskLevel: "none", Result: [] },
          }),
        );
      }, GUARDRAIL_DELAY_MS);
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    close: () =>
      new Promise<void>((resolve, reject) =>
        server.close((err) => (err ? reject(err) : resolve())),
      ),
  };
}

const CHAT_EVENTS = [
  '{"id":"c1","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
  '{"id":"c1","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}',
  '{"id":"c1","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}',
  "[DONE]",
];

const MESSAGES_EVENTS = [
  JSON.stringify({
    type: "message_start",
    message: {
      id: "msg_side",
      type: "message",
      role: "assistant",
      content: [],
      model: "claude-3-5-haiku-20241022",
      stop_reason: null,
      usage: { input_tokens: 3, output_tokens: 1 },
    },
  }),
  JSON.stringify({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
  JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "hi" } }),
  JSON.stringify({ type: "content_block_stop", index: 0 }),
  JSON.stringify({ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 1 } }),
  JSON.stringify({ type: "message_stop" }),
];

const RESPONSES_EVENTS = [
  JSON.stringify({ type: "response.created", response: { id: "resp-side", model: "gpt-4o-mini" } }),
  JSON.stringify({ type: "response.output_text.delta", delta: "hi" }),
  JSON.stringify({
    type: "response.completed",
    response: {
      id: "resp-side",
      status: "completed",
      model: "gpt-4o-mini",
      usage: { input_tokens: 3, output_tokens: 1, total_tokens: 4 },
    },
  }),
  "[DONE]",
];

const CASES = [
  {
    endpoint: "/v1/chat/completions",
    model: "ttft-side-chat",
    provider: "openai",
    modelName: "gpt-4o-mini",
    events: CHAT_EVENTS,
    body: (model: string) => ({ model, stream: true, messages: [{ role: "user", content: "go" }] }),
  },
  {
    endpoint: "/v1/messages",
    model: "ttft-side-messages",
    provider: "anthropic",
    modelName: "claude-3-5-haiku-20241022",
    events: MESSAGES_EVENTS,
    body: (model: string) => ({
      model,
      stream: true,
      max_tokens: 16,
      messages: [{ role: "user", content: "go" }],
    }),
  },
  {
    endpoint: "/v1/responses",
    model: "ttft-side-responses",
    provider: "openai",
    modelName: "gpt-4o-mini",
    events: RESPONSES_EVENTS,
    body: (model: string) => ({ model, stream: true, input: "go" }),
  },
] as const;

describe("TTFT metrics split the wait by side", () => {
  let app: SpawnedApp | undefined;
  let aliyun: Awaited<ReturnType<typeof startSlowAliyun>> | undefined;
  const upstreams: OpenAiUpstream[] = [];
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    aliyun = await startSlowAliyun();
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    for (const c of CASES) {
      const up = await startOpenAiUpstream({ streamEvents: [...c.events] });
      upstreams.push(up);
      const pk = await seed.createProviderKey({
        display_name: `${c.model}-pk`,
        secret: "sk-mock",
        api_base: c.provider === "anthropic" ? up.baseUrl : `${up.baseUrl}/v1`,
        provider: c.provider,
      });
      await seed.createModel({
        display_name: c.model,
        provider: c.provider,
        model_name: c.modelName,
        provider_key_id: pk.id,
      });
    }
    await seed.createGuardrail({
      name: "ttft-side-slow-input",
      enabled: true,
      hook_point: "input",
      fail_open: false,
      kind: "aliyun_text_moderation",
      region: "cn-shanghai",
      endpoint: aliyun.baseUrl,
      access_key_id: "LTAI_E2E",
      access_key_secret: "e2e-secret",
      risk_level_threshold: "high",
      timeout_ms: GUARDRAIL_DELAY_MS * 10,
    });

    await seed.createApiKey({ key_hash: sha256(KEY), allowed_models: ["*"] });
    const proxy = new ProxyClient(app.proxyUrl, KEY);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  }, 90_000);

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
    await aliyun?.close();
  });

  const delta = (
    before: MetricSample[],
    after: MetricSample[],
    name: string,
    labels: Record<string, string>,
  ): number => sumMetric(after, name, labels) - sumMetric(before, name, labels);

  for (const c of CASES) {
    test(`${c.endpoint}: input-guardrail time is on the downstream side only`, async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();

      const before = await scrapeMetrics(app.metricsUrl);
      const res = await fetch(`${app.proxyUrl}${c.endpoint}`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${KEY}`,
          "x-api-key": KEY,
          "anthropic-version": "2023-06-01",
        },
        body: JSON.stringify(c.body(c.model)),
      });
      const text = await res.text();
      expect(res.status, text).toBe(200);

      const labels = { endpoint: c.endpoint, model: c.model };
      let after: MetricSample[] = [];
      await expect
        .poll(async () => {
          after = await scrapeMetrics(app!.metricsUrl);
          return delta(before, after, `${REQUEST_TTFT}_count`, labels);
        })
        .toBe(2);

      for (const name of [REQUEST_TTFT, DETAILED_TTFT]) {
        for (const side of ["upstream", "downstream"]) {
          expect(
            delta(before, after, `${name}_count`, { ...labels, side }),
            `${name} ${side} count`,
          ).toBe(1);
        }
        const upstream = delta(before, after, `${name}_sum`, { ...labels, side: "upstream" });
        const downstream = delta(before, after, `${name}_sum`, { ...labels, side: "downstream" });
        // The guardrail ran before the attempt started: the caller waited
        // for it, the upstream never saw it.
        expect(downstream, `${name} downstream includes the guardrail`).toBeGreaterThanOrEqual(D);
        expect(upstream, `${name} upstream excludes the guardrail`).toBeLessThan(D);
      }
    });
  }
});
