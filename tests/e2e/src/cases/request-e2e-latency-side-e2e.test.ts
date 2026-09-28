import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  scrapeMetrics,
  spawnApp,
  startOpenAiUpstream,
  sumMetric,
  waitConfigPropagation,
  type MetricSample,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: `aisix_request_e2e_latency_seconds` carries `side`. `downstream` is
// the request as the caller experienced it — the observation this metric
// has always recorded — and `upstream` is how long the upstream took on the
// attempt that produced the response. The two observations of one request
// carry the same labels apart from `side`, and a request that never reached
// an upstream (a cache hit) has no upstream side at all.

const KEY = "sk-e2e-latency-side";
const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");
const E2E = "aisix_request_e2e_latency_seconds";

/** How long every upstream below takes before it answers. */
const UPSTREAM_DELAY_MS = 300;
const U = UPSTREAM_DELAY_MS / 1000;

const CHAT_COMPLETION = {
  id: "chatcmpl-side",
  object: "chat.completion",
  model: "gpt-4o-mini",
  choices: [{ index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "stop" }],
  usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
};
const CHAT_EVENTS = [
  '{"id":"c1","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}',
  '{"id":"c1","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}',
  "[DONE]",
];
const ANTHROPIC_MESSAGE = {
  id: "msg_side",
  type: "message",
  role: "assistant",
  model: "claude-3-5-haiku-20241022",
  content: [{ type: "text", text: "hi" }],
  stop_reason: "end_turn",
  usage: { input_tokens: 3, output_tokens: 1 },
};
const MESSAGES_EVENTS = [
  JSON.stringify({
    type: "message_start",
    message: { ...ANTHROPIC_MESSAGE, content: [], stop_reason: null, usage: { input_tokens: 3, output_tokens: 0 } },
  }),
  JSON.stringify({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
  JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "hi" } }),
  JSON.stringify({ type: "content_block_stop", index: 0 }),
  JSON.stringify({ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 1 } }),
  JSON.stringify({ type: "message_stop" }),
];
const RESPONSE = {
  id: "resp-side",
  object: "response",
  status: "completed",
  model: "gpt-4o-mini",
  output: [{ type: "message", role: "assistant", content: [{ type: "output_text", text: "hi" }] }],
  usage: { input_tokens: 3, output_tokens: 1, total_tokens: 4 },
};
const RESPONSES_EVENTS = [
  JSON.stringify({ type: "response.created", response: { ...RESPONSE, status: "in_progress", output: [] } }),
  JSON.stringify({ type: "response.output_text.delta", delta: "hi" }),
  JSON.stringify({ type: "response.completed", response: RESPONSE }),
  "[DONE]",
];

type Case = {
  name: string;
  endpoint: string;
  provider: string;
  modelName: string;
  stream: boolean;
  upstream: Parameters<typeof startOpenAiUpstream>[0];
  body: (model: string) => unknown;
  status: number;
};

const chatBody = (stream: boolean) => (model: string) => ({ model, stream, messages: [{ role: "user", content: "go" }] });
const messagesBody = (stream: boolean) => (model: string) => ({
  model,
  stream,
  max_tokens: 16,
  messages: [{ role: "user", content: "go" }],
});
const responsesBody = (stream: boolean) => (model: string) => ({ model, stream, input: "go" });

const CASES: Case[] = [
  { name: "side-chat", endpoint: "/v1/chat/completions", provider: "openai", modelName: "gpt-4o-mini", stream: false,
    upstream: { responseDelayMs: UPSTREAM_DELAY_MS, nonStreamBody: CHAT_COMPLETION }, body: chatBody(false), status: 200 },
  { name: "side-chat-stream", endpoint: "/v1/chat/completions", provider: "openai", modelName: "gpt-4o-mini", stream: true,
    upstream: { responseDelayMs: UPSTREAM_DELAY_MS, streamEvents: CHAT_EVENTS }, body: chatBody(true), status: 200 },
  { name: "side-messages", endpoint: "/v1/messages", provider: "anthropic", modelName: "claude-3-5-haiku-20241022", stream: false,
    upstream: { responseDelayMs: UPSTREAM_DELAY_MS, nonStreamBody: ANTHROPIC_MESSAGE }, body: messagesBody(false), status: 200 },
  { name: "side-messages-stream", endpoint: "/v1/messages", provider: "anthropic", modelName: "claude-3-5-haiku-20241022", stream: true,
    upstream: { responseDelayMs: UPSTREAM_DELAY_MS, streamEvents: MESSAGES_EVENTS }, body: messagesBody(true), status: 200 },
  { name: "side-responses", endpoint: "/v1/responses", provider: "openai", modelName: "gpt-4o-mini", stream: false,
    upstream: { responseDelayMs: UPSTREAM_DELAY_MS, nonStreamBody: RESPONSE }, body: responsesBody(false), status: 200 },
  { name: "side-responses-stream", endpoint: "/v1/responses", provider: "openai", modelName: "gpt-4o-mini", stream: true,
    upstream: { responseDelayMs: UPSTREAM_DELAY_MS, streamEvents: RESPONSES_EVENTS }, body: responsesBody(true), status: 200 },
  // The upstream answered with an error: its attempt was still dispatched,
  // so the request has an upstream side to observe.
  { name: "side-chat-upstream-error", endpoint: "/v1/chat/completions", provider: "openai", modelName: "gpt-4o-mini", stream: false,
    upstream: { responseDelayMs: UPSTREAM_DELAY_MS, status: 400, errorBody: { error: { message: "bad", type: "invalid_request_error" } } },
    body: chatBody(false), status: 400 },
];

/** The label set of one sample without `side` and `le`, as a stable string. */
const labelsBesidesSide = (s: MetricSample) =>
  JSON.stringify(Object.entries(s.labels).filter(([k]) => k !== "side" && k !== "le").sort());

describe("aisix_request_e2e_latency_seconds splits each request by side", () => {
  let app: SpawnedApp | undefined;
  const upstreams: OpenAiUpstream[] = [];
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const cached = await startOpenAiUpstream({ responseDelayMs: UPSTREAM_DELAY_MS, nonStreamBody: CHAT_COMPLETION });
    upstreams.push(cached);
    for (const c of [...CASES, { name: "side-cached", provider: "openai", modelName: "gpt-4o-mini", upstream: null }]) {
      const up = c.upstream === null ? cached : await startOpenAiUpstream(c.upstream);
      if (c.upstream !== null) upstreams.push(up);
      const pk = await seed.createProviderKey({
        display_name: `${c.name}-pk`,
        secret: "sk-mock",
        provider: c.provider,
        api_base: c.provider === "anthropic" ? up.baseUrl : `${up.baseUrl}/v1`,
      });
      await seed.createModel({
        display_name: c.name,
        provider: c.provider,
        model_name: c.modelName,
        provider_key_id: pk.id,
      });
    }
    await seed.createModel({
      display_name: "side-ensemble",
      ensemble: { panel: [{ model: "side-chat" }], judge: { model: "side-chat" }, min_responses: 1 },
    });
    await seed.createCachePolicy({ name: "side-cache", enabled: true, applies_to: "all" });
    await seed.createApiKey({ key_hash: sha256(KEY), allowed_models: ["*"] });
    const proxy = new ProxyClient(app.proxyUrl, KEY);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  }, 90_000);

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  async function call(endpoint: string, body: unknown): Promise<{ status: number; text: string; cache: string }> {
    const res = await fetch(`${app!.proxyUrl}${endpoint}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${KEY}`,
        "x-api-key": KEY,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify(body),
    });
    return { status: res.status, text: await res.text(), cache: res.headers.get("x-aisix-cache") ?? "" };
  }

  for (const c of CASES) {
    test(`${c.name}: one observation per side, under the same labels`, async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const r = await call(c.endpoint, c.body(c.name));
      expect(r.status, r.text).toBe(c.status);

      let samples: MetricSample[] = [];
      await expect
        .poll(async () => {
          samples = await scrapeMetrics(app!.metricsUrl);
          return sumMetric(samples, `${E2E}_count`, { model: c.name });
        })
        .toBe(2);
      const counts = samples.filter((s) => s.name === `${E2E}_count` && s.labels.model === c.name);
      const bySide = new Map(counts.map((s) => [s.labels.side, s]));
      expect([...bySide.keys()].sort()).toEqual(["downstream", "upstream"]);
      for (const s of counts) expect(s.value, s.labels.side).toBe(1);
      expect(labelsBesidesSide(bySide.get("upstream")!)).toBe(labelsBesidesSide(bySide.get("downstream")!));
      expect(bySide.get("downstream")!.labels.streaming).toBe(String(c.stream));
      expect(bySide.get("downstream")!.labels.endpoint).toBe(c.endpoint);

      const upstream = sumMetric(samples, `${E2E}_sum`, { model: c.name, side: "upstream" });
      const downstream = sumMetric(samples, `${E2E}_sum`, { model: c.name, side: "downstream" });
      // The upstream took its delay; the caller waited for that and more.
      expect(upstream).toBeGreaterThanOrEqual(U);
      expect(downstream).toBeGreaterThanOrEqual(upstream);
    });
  }

  test("an ensemble observes the downstream side only", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    // Its response comes from no single upstream attempt: the judge's call
    // is only the last of several.
    const r = await call("/v1/chat/completions", chatBody(false)("side-ensemble"));
    expect(r.status, r.text).toBe(200);
    let samples: MetricSample[] = [];
    await expect
      .poll(async () => {
        samples = await scrapeMetrics(app!.metricsUrl);
        return sumMetric(samples, `${E2E}_count`, { model: "side-ensemble", side: "downstream" });
      })
      .toBe(1);
    expect(sumMetric(samples, `${E2E}_count`, { model: "side-ensemble", side: "upstream" })).toBe(0);
  });

  test("a cache hit observes the downstream side only", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const body = chatBody(false)("side-cached");
    const miss = await call("/v1/chat/completions", body);
    expect(miss.status, miss.text).toBe(200);
    expect(miss.cache).toBe("miss");
    const hit = await call("/v1/chat/completions", body);
    expect(hit.status, hit.text).toBe(200);
    expect(hit.cache).toBe("hit");

    let samples: MetricSample[] = [];
    await expect
      .poll(async () => {
        samples = await scrapeMetrics(app!.metricsUrl);
        return sumMetric(samples, `${E2E}_count`, { model: "side-cached", side: "downstream" });
      })
      .toBe(2);
    // Only the miss reached the upstream.
    expect(sumMetric(samples, `${E2E}_count`, { model: "side-cached", side: "upstream" })).toBe(1);
  });
});
