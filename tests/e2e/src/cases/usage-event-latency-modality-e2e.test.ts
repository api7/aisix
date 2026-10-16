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
import { startMockOtlp, type CapturedSpan, type MockOtlp } from "../harness/otlp-mock.js";

// E2E for ticket-analysis#1051, items 2–5, read off real exports of a real
// `aisix` binary:
//
// - `api_key_name`: the authenticating key's display name on every event,
//   whatever the content mode.
// - `gateway_overhead_ms`: on the request's terminal event, the caller's
//   wait (`downstream_latency_ms`) minus the time spent waiting on an
//   upstream — every failed attempt whole, the winner up to its first
//   streamed frame or its whole answer when buffered.
// - `upstream_tpot_ms`: per streamed, successful attempt,
//   `(upstream_latency_ms - upstream_ttft_ms) / (output tokens - 1)`.
// - audio / image token counts as the upstream reported them.

const CALLER = "sk-latency-modality";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const KEY_NAME = "team-a-batch-key";
const CREDENTIAL_REF = "mock";
const LOGSTORE = "latency-modality";

const FIRST_TRY_MS = 400;
const SECOND_TRY_MS = 300;
const FIRST_FRAME_MS = 400;
const FRAME_GAP_MS = 80;

function chunk(delta: Record<string, unknown>, extra: Record<string, unknown> = {}): string {
  return JSON.stringify({
    id: "chatcmpl-lm",
    object: "chat.completion.chunk",
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta, finish_reason: null }],
    ...extra,
  });
}

const STREAM = [
  chunk({ role: "assistant", content: "a" }),
  chunk({ content: "b" }),
  chunk({ content: "c" }),
  chunk({ content: "d" }),
  chunk({ content: "e" }),
  JSON.stringify({
    id: "chatcmpl-lm",
    object: "chat.completion.chunk",
    model: "gpt-4o-mini",
    choices: [],
    usage: { prompt_tokens: 7, completion_tokens: 5, total_tokens: 12 },
  }),
  "[DONE]",
];

const AUDIO_BODY = {
  id: "chatcmpl-audio",
  object: "chat.completion",
  model: "gpt-4o-audio-preview",
  choices: [{ index: 0, message: { role: "assistant", content: "heard you" }, finish_reason: "stop" }],
  usage: {
    prompt_tokens: 60,
    completion_tokens: 30,
    total_tokens: 90,
    prompt_tokens_details: { cached_tokens: 0, audio_tokens: 40 },
    completion_tokens_details: { reasoning_tokens: 0, audio_tokens: 25 },
  },
};

const IMAGE_BODY = {
  created: 1_700_000_000,
  data: [{ b64_json: "aGk=" }],
  usage: {
    input_tokens: 79,
    input_tokens_details: { text_tokens: 13, image_tokens: 66 },
    output_tokens: 272,
    total_tokens: 351,
  },
};

const num = (row: Map<string, string>, field: string) => {
  const raw = row.get(field);
  return raw === undefined ? undefined : Number(raw);
};

describe("usage events carry the key name, the latency breakdown and modality counts", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let otlp: MockOtlp | undefined;
  let sls: MockSls | undefined;
  const upstreams: Record<string, OpenAiUpstream> = {};

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstreams.flaky = await startOpenAiUpstream({
      scriptedResponses: [
        { status: 500, responseDelayMs: FIRST_TRY_MS },
        { responseDelayMs: SECOND_TRY_MS },
      ],
    });
    upstreams.stream = await startOpenAiUpstream({
      streamEvents: STREAM,
      firstEventDelayMs: FIRST_FRAME_MS,
      eventDelayMs: FRAME_GAP_MS,
    });
    upstreams.audio = await startOpenAiUpstream({ nonStreamBody: AUDIO_BODY });
    upstreams.image = await startOpenAiUpstream({ nonStreamBody: IMAGE_BODY });
    otlp = await startMockOtlp();
    sls = await startMockSls();

    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "LTAI_mock_ak",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock_ak_secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "lm-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });
    await seed.createObservabilityExporter({
      name: "lm-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "aisix-e2e-obs",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
    });
    for (const [name, upstream] of Object.entries(upstreams)) {
      const pk = await seed.createProviderKey({
        display_name: `lm-${name}-pk`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      await seed.createModel({
        display_name: `lm-${name}`,
        provider: "openai",
        model_name: name === "image" ? "gpt-image-1" : "gpt-4o-mini",
        provider_key_id: pk.id,
      });
    }
    await seed.createApiKey({ key_hash: CALLER_HASH, display_name: KEY_NAME, allowed_models: ["*"] });
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
    await otlp?.close();
    await sls?.close();
    await Promise.all(Object.values(upstreams).map((u) => u.close()));
  });

  async function post(path: string, body: unknown): Promise<{ status: number; requestId: string }> {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    await res.text();
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    expect(requestId).not.toBe("");
    return { status: res.status, requestId };
  }

  const row = (requestId: string, what: string, pred: (l: Map<string, string>) => boolean = () => true) =>
    waitForSlsLog(sls!, LOGSTORE, (l) => l.get("request_id") === requestId && pred(l), what, 15_000);

  async function span(requestId: string, pred: (s: CapturedSpan) => boolean): Promise<CapturedSpan> {
    const deadline = Date.now() + 15_000;
    while (Date.now() < deadline) {
      const hit = otlp!.spans.find((s) => s.attributes["aisix.request_id"] === requestId && pred(s));
      if (hit) return hit;
      await new Promise((r) => setTimeout(r, 50));
    }
    throw new Error(`no matching span for ${requestId}`);
  }

  test("a retried request's overhead leaves out both upstream waits; the key name rides every event", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const { status, requestId } = await post("/v1/chat/completions", {
      model: "lm-flaky",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(status).toBe(200);
    const failed = await row(requestId, "failed attempt", (l) => l.get("attempt_index") === "0");
    const won = await row(requestId, "winning attempt", (l) => l.get("attempt_index") === "1");

    expect(failed.get("api_key_name")).toBe(KEY_NAME);
    expect(won.get("api_key_name")).toBe(KEY_NAME);
    // Only the terminal event carries the request-scoped figures.
    expect(failed.get("gateway_overhead_ms")).toBeUndefined();
    const downstream = num(won, "downstream_latency_ms")!;
    const overhead = num(won, "gateway_overhead_ms");
    expect(downstream).toBeGreaterThanOrEqual(FIRST_TRY_MS + SECOND_TRY_MS);
    expect(overhead).toBeDefined();
    expect(overhead!).toBeGreaterThanOrEqual(0);
    // Both upstream waits are subtracted — not only the winner's. The
    // naive `downstream - upstream_latency_ms` of the winner would still
    // contain the whole first attempt.
    expect(overhead!).toBeLessThanOrEqual(downstream - (FIRST_TRY_MS + SECOND_TRY_MS) + 50);
    expect(downstream - num(won, "upstream_latency_ms")!).toBeGreaterThanOrEqual(FIRST_TRY_MS);
    // A buffered answer has no time per output token.
    expect(won.get("upstream_tpot_ms")).toBeUndefined();

    const s = await span(requestId, (x) => x.attributes["aisix.attempt_index"] === 1 && "gen_ai.operation.name" in x.attributes);
    expect(s.attributes["aisix.api_key_name"]).toBe(KEY_NAME);
    expect(s.attributes["aisix.gateway_overhead_ms"]).toBe(overhead);
  });

  test("a stream's overhead stops at its first frame, and its TPOT follows the stated formula", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const { status, requestId } = await post("/v1/chat/completions", {
      model: "lm-stream",
      stream: true,
      stream_options: { include_usage: true },
      messages: [{ role: "user", content: "hi" }],
    });
    expect(status).toBe(200);
    const r = await row(requestId, "stream row", (l) => l.get("downstream_latency_ms") !== undefined);
    const downstream = num(r, "downstream_latency_ms")!;
    const overhead = num(r, "gateway_overhead_ms")!;
    const ttft = num(r, "upstream_ttft_ms")!;
    const latency = num(r, "upstream_latency_ms")!;
    expect(downstream).toBeGreaterThanOrEqual(FIRST_FRAME_MS);
    // The wait for the first frame is upstream time; the stream after it
    // is not inside the window at all.
    expect(overhead).toBeLessThanOrEqual(downstream - FIRST_FRAME_MS + 50);
    expect(latency - ttft).toBeGreaterThanOrEqual(4 * FRAME_GAP_MS - 20);
    const tpot = num(r, "upstream_tpot_ms")!;
    expect(tpot).toBeCloseTo((latency - ttft) / (5 - 1), 6);

    const s = await span(requestId, (x) => "aisix.upstream_tpot_ms" in x.attributes);
    expect(s.attributes["aisix.upstream_tpot_ms"]).toBeCloseTo(tpot, 6);
  });

  test("audio counts reported by a chat upstream ride the event and the span", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const { status, requestId } = await post("/v1/chat/completions", {
      model: "lm-audio",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(status).toBe(200);
    const r = await row(requestId, "audio row");
    expect(num(r, "prompt_audio_tokens")).toBe(40);
    expect(num(r, "completion_audio_tokens")).toBe(25);
    expect(r.get("prompt_image_tokens")).toBeUndefined();
    expect(r.get("completion_image_tokens")).toBeUndefined();
    expect(num(r, "prompt_tokens")).toBe(60);

    const s = await span(requestId, (x) => "gen_ai.usage.input_tokens" in x.attributes);
    expect(s.attributes["gen_ai.usage.audio.input_tokens"]).toBe(40);
    expect(s.attributes["gen_ai.usage.audio.output_tokens"]).toBe(25);
    expect("gen_ai.usage.image.input_tokens" in s.attributes).toBe(false);
  });

  test("image tokens reported by an image generation ride the event", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const { status, requestId } = await post("/v1/images/generations", {
      model: "lm-image",
      prompt: "a lighthouse",
    });
    expect(status).toBe(200);
    const r = await row(requestId, "image row");
    expect(num(r, "prompt_image_tokens")).toBe(66);
    expect(r.get("prompt_audio_tokens")).toBeUndefined();
    const s = await span(requestId, (x) => "gen_ai.usage.input_tokens" in x.attributes);
    expect(s.attributes["gen_ai.usage.image.input_tokens"]).toBe(66);
  });
});
