import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForLogLine,
  waitForSlsLog,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// The usage record carries the upstream's counters exactly as reported;
// only what a CLIENT is told is adapted to the client's protocol.
//
// Gemini reports thinking tokens (`thoughtsTokenCount`) in a counter of
// their own, and on some model versions BESIDE `candidatesTokenCount`
// rather than inside it — `totalTokenCount` says which. OpenAI accounting
// has no third output bucket, so an OpenAI- or Anthropic-shape answer
// folds the thoughts into its completion count. The record must not: it
// keeps `candidatesTokenCount` as the completion, `thoughtsTokenCount` as
// the reasoning and `totalTokenCount` as the total, and the control plane
// reads that identity itself.
//
// The record's `total_tokens` is the upstream's own total and nothing
// else: an upstream that reports none records none, because a sum
// computed here that happens to equal prompt + completion + reasoning
// would read as reasoning counted beside the completion.
//
// Every inbound protocol records the same row for the same upstream
// call, `/v1/messages` and `/v1/completions` included.
//
// References:
// - https://ai.google.dev/api/generate-content#UsageMetadata
// - https://platform.openai.com/docs/api-reference/chat/object
// - https://platform.openai.com/docs/api-reference/completions/object

const CALLER_PLAINTEXT = "sk-recorded-usage-raw";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const CREDENTIAL_REF = "e2e";
const LOGSTORE = "recorded-usage-store";

// One Gemini thinking call whose candidates EXCLUDE the thoughts:
// 100 + 20 + 30 == 150.
const PROMPT = 100;
const CANDIDATES = 20;
const THOUGHTS = 30;
const TOTAL = PROMPT + CANDIDATES + THOUGHTS;

const GEMINI_MODEL = "raw-usage-gemini";
const GEMINI_STREAM_MODEL = "raw-usage-gemini-stream";
const OPENAI_MODEL = "raw-usage-openai";
const OPENAI_NO_TOTAL_MODEL = "raw-usage-openai-no-total";
const GEMINI_NO_TOTAL_MODEL = "raw-usage-gemini-no-total";
const ALL_MODELS = [
  GEMINI_MODEL,
  GEMINI_STREAM_MODEL,
  OPENAI_MODEL,
  OPENAI_NO_TOTAL_MODEL,
  GEMINI_NO_TOTAL_MODEL,
];

const GEMINI_BODY = {
  candidates: [{ content: { role: "model", parts: [{ text: "ok" }] }, finishReason: "STOP" }],
  usageMetadata: {
    promptTokenCount: PROMPT,
    candidatesTokenCount: CANDIDATES,
    thoughtsTokenCount: THOUGHTS,
    totalTokenCount: TOTAL,
  },
  modelVersion: "gemini-2.5-flash",
};

// The same call from an upstream that omits `totalTokenCount`: nothing
// says the thoughts sit beside the candidates.
const GEMINI_NO_TOTAL_BODY = {
  ...GEMINI_BODY,
  usageMetadata: {
    promptTokenCount: PROMPT,
    candidatesTokenCount: CANDIDATES,
    thoughtsTokenCount: THOUGHTS,
  },
};

// Gemini stamps cumulative usage on every streamed frame.
const GEMINI_STREAM_FRAMES = [
  `data: ${JSON.stringify({
    candidates: [{ content: { role: "model", parts: [{ text: "o" }] } }],
    usageMetadata: {
      promptTokenCount: PROMPT,
      candidatesTokenCount: 5,
      thoughtsTokenCount: THOUGHTS,
      totalTokenCount: PROMPT + 5 + THOUGHTS,
    },
  })}\n\n`,
  `data: ${JSON.stringify({
    candidates: [{ content: { role: "model", parts: [{ text: "k" }] }, finishReason: "STOP" }],
    usageMetadata: {
      promptTokenCount: PROMPT,
      candidatesTokenCount: CANDIDATES,
      thoughtsTokenCount: THOUGHTS,
      totalTokenCount: TOTAL,
    },
  })}\n\n`,
];

// An OpenAI-compatible reasoning answer: reasoning INSIDE the completion.
const OAI_PROMPT = 40;
const OAI_COMPLETION = 25;
const OAI_REASONING = 10;
const OAI_TOTAL = OAI_PROMPT + OAI_COMPLETION;

function openAiChatBody(withTotal: boolean): unknown {
  return {
    id: "chatcmpl-raw-usage",
    object: "chat.completion",
    created: 1_700_000_000,
    model: "gpt-5-mini",
    choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
    usage: {
      prompt_tokens: OAI_PROMPT,
      completion_tokens: OAI_COMPLETION,
      ...(withTotal ? { total_tokens: OAI_TOTAL } : {}),
      completion_tokens_details: { reasoning_tokens: OAI_REASONING },
    },
  };
}

const OPENAI_COMPLETIONS_BODY = {
  id: "cmpl-raw-usage",
  object: "text_completion",
  created: 1_700_000_000,
  model: "gpt-5-mini",
  choices: [{ index: 0, text: "ok", finish_reason: "stop" }],
  usage: {
    prompt_tokens: OAI_PROMPT,
    completion_tokens: OAI_COMPLETION,
    total_tokens: OAI_TOTAL,
    completion_tokens_details: { reasoning_tokens: OAI_REASONING },
  },
};

interface Called {
  status: number;
  requestId: string;
  text: string;
}

/** A `key=value` field of a log line. */
function field(line: string, name: string): string | undefined {
  const m = line.match(new RegExp(`\\b${name}=(?:"([^"]*)"|([^\\s]+))`));
  if (!m) return undefined;
  return m[1] ?? m[2];
}

/** The last usage object a stream carried, located by `pick`. */
function usageFromSse(text: string, pick: (frame: Record<string, unknown>) => unknown): Record<string, any> {
  let found: Record<string, any> | undefined;
  for (const line of text.split("\n")) {
    if (!line.startsWith("data: ")) continue;
    const data = line.slice(6).trim();
    if (data === "[DONE]") continue;
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(data) as Record<string, unknown>;
    } catch {
      continue;
    }
    const candidate = pick(parsed);
    if (candidate && typeof candidate === "object") found = candidate as Record<string, any>;
  }
  if (!found) throw new Error(`no usage frame in stream:\n${text}`);
  return found;
}

describe("the usage record carries upstream usage as reported", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  const upstreams: OpenAiUpstream[] = [];

  async function call(path: string, body: unknown): Promise<Called> {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify(body),
    });
    return {
      status: res.status,
      requestId: res.headers.get("x-aisix-request-id") ?? "",
      text: await res.text(),
    };
  }

  async function recorded(requestId: string): Promise<Map<string, string>> {
    return waitForSlsLog(
      sls!,
      LOGSTORE,
      (l) => l.get("request_id") === requestId,
      `usage row for ${requestId}`,
      15_000,
    );
  }

  /**
   * The access-log line keeps the gateway's own (folded) numbers, streamed
   * or not — only the usage record is raw.
   */
  async function expectFoldedAccessLog(requestId: string): Promise<void> {
    const line = await waitForLogLine(
      app!,
      (l) => l.includes("proxy request completed") && field(l, "request_id") === requestId,
      `the access-log line of ${requestId}`,
    );
    expect(field(line, "completion_tokens")).toBe(String(CANDIDATES + THOUGHTS));
    expect(field(line, "total_tokens")).toBe(String(TOTAL));
  }

  /** Gemini's own counters, whichever protocol addressed the call. */
  async function expectRawGeminiRecord(requestId: string): Promise<void> {
    const log = await recorded(requestId);
    expect(log.get("prompt_tokens")).toBe(String(PROMPT));
    expect(log.get("completion_tokens")).toBe(String(CANDIDATES));
    expect(log.get("reasoning_tokens")).toBe(String(THOUGHTS));
    expect(log.get("total_tokens")).toBe(String(TOTAL));
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    const gemini = await startOpenAiUpstream({ nonStreamBody: GEMINI_BODY });
    const geminiStream = await startOpenAiUpstream({ rawStreamFrames: GEMINI_STREAM_FRAMES });
    const openai = await startOpenAiUpstream({ nonStreamBody: openAiChatBody(true) });
    const openaiNoTotal = await startOpenAiUpstream({ nonStreamBody: openAiChatBody(false) });
    const geminiNoTotal = await startOpenAiUpstream({ nonStreamBody: GEMINI_NO_TOTAL_BODY });
    upstreams.push(gemini, geminiStream, openai, openaiNoTotal, geminiNoTotal);

    // The access log is an `info` event; the harness defaults to `warn`.
    app = await spawnApp({
      extraEnv: {
        RUST_LOG: "info",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-ak-id",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-ak-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "recorded-usage-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "recorded-usage-proj",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
    });

    for (const [model, upstream] of [
      [GEMINI_MODEL, gemini],
      [GEMINI_STREAM_MODEL, geminiStream],
      [GEMINI_NO_TOTAL_MODEL, geminiNoTotal],
    ] as const) {
      const pk = await seed.createProviderKey({
        display_name: `${model}-pk`,
        provider: "google",
        adapter: "vertex",
        secret: JSON.stringify({
          access_token: "ya29.recorded-usage-e2e",
          project: "proj-e2e",
          region: "us-central1",
        }),
        api_base: upstream.baseUrl,
      });
      await seed.createModel({
        display_name: model,
        provider: "google",
        model_name: "gemini-2.5-flash",
        provider_key_id: pk.id,
      });
    }
    for (const [model, upstream] of [
      [OPENAI_MODEL, openai],
      [OPENAI_NO_TOTAL_MODEL, openaiNoTotal],
    ] as const) {
      const pk = await seed.createProviderKey({
        display_name: `${model}-pk`,
        secret: "sk-openai-mock",
        api_base: `${upstream.baseUrl}/v1`,
        provider: "openai",
        adapter: "openai",
      });
      await seed.createModel({
        display_name: model,
        provider: "openai",
        model_name: "gpt-5-mini",
        provider_key_id: pk.id,
      });
    }

    // Caller key last: once it authenticates, the whole seed set is in.
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ALL_MODELS });
    const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      if (res.status !== 200) return false;
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return ALL_MODELS.every((m) => data.some((row) => row.id === m));
    });
  });

  afterAll(async () => {
    await app?.exit();
    for (const upstream of upstreams) await upstream.close();
    await sls?.close();
  });

  test("chat/completions over Gemini: the client sees thoughts folded in, the record does not", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await call("/v1/chat/completions", {
      model: GEMINI_MODEL,
      messages: [{ role: "user", content: "hi" }],
    });
    expect(res.status, res.text).toBe(200);
    const usage = JSON.parse(res.text).usage;
    expect(usage.prompt_tokens).toBe(PROMPT);
    expect(usage.completion_tokens).toBe(CANDIDATES + THOUGHTS);
    expect(usage.completion_tokens_details.reasoning_tokens).toBe(THOUGHTS);
    expect(usage.total_tokens).toBe(TOTAL);

    await expectRawGeminiRecord(res.requestId);
    await expectFoldedAccessLog(res.requestId);
  });

  test("chat/completions streaming over Gemini records the same raw row", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await call("/v1/chat/completions", {
      model: GEMINI_STREAM_MODEL,
      messages: [{ role: "user", content: "hi" }],
      stream: true,
      stream_options: { include_usage: true },
    });
    expect(res.status, res.text).toBe(200);
    const usage = usageFromSse(res.text, (f) => f.usage);
    expect(usage.prompt_tokens).toBe(PROMPT);
    expect(usage.completion_tokens).toBe(CANDIDATES + THOUGHTS);
    expect(usage.total_tokens).toBe(TOTAL);

    await expectRawGeminiRecord(res.requestId);
    await expectFoldedAccessLog(res.requestId);
  });

  test("messages over Gemini: Anthropic output folds the thoughts, the record keeps them apart", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await call("/v1/messages", {
      model: GEMINI_MODEL,
      max_tokens: 64,
      messages: [{ role: "user", content: "hi" }],
    });
    expect(res.status, res.text).toBe(200);
    const usage = JSON.parse(res.text).usage;
    expect(usage.input_tokens).toBe(PROMPT);
    expect(usage.output_tokens).toBe(CANDIDATES + THOUGHTS);

    await expectRawGeminiRecord(res.requestId);
  });

  test("messages streaming over Gemini records the same raw row", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await call("/v1/messages", {
      model: GEMINI_STREAM_MODEL,
      max_tokens: 64,
      stream: true,
      messages: [{ role: "user", content: "hi" }],
    });
    expect(res.status, res.text).toBe(200);
    const usage = usageFromSse(res.text, (f) => (f.type === "message_delta" ? f.usage : undefined));
    expect(usage.output_tokens).toBe(CANDIDATES + THOUGHTS);

    await expectRawGeminiRecord(res.requestId);
    await expectFoldedAccessLog(res.requestId);
  });

  test("responses over Gemini: output_tokens folds the thoughts, the record keeps them apart", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await call("/v1/responses", { model: GEMINI_MODEL, input: "hi" });
    expect(res.status, res.text).toBe(200);
    const usage = JSON.parse(res.text).usage;
    expect(usage.input_tokens).toBe(PROMPT);
    expect(usage.output_tokens).toBe(CANDIDATES + THOUGHTS);
    expect(usage.output_tokens_details.reasoning_tokens).toBe(THOUGHTS);

    await expectRawGeminiRecord(res.requestId);
  });

  test("responses streaming over Gemini records the same raw row", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await call("/v1/responses", { model: GEMINI_STREAM_MODEL, input: "hi", stream: true });
    expect(res.status, res.text).toBe(200);
    const usage = usageFromSse(res.text, (f) => (f.response as { usage?: unknown } | undefined)?.usage);
    expect(usage.output_tokens).toBe(CANDIDATES + THOUGHTS);

    await expectRawGeminiRecord(res.requestId);
    await expectFoldedAccessLog(res.requestId);
  });

  test("Gemini without a total: the record keeps the folded completion and names no total", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await call("/v1/chat/completions", {
      model: GEMINI_NO_TOTAL_MODEL,
      messages: [{ role: "user", content: "hi" }],
    });
    expect(res.status, res.text).toBe(200);
    expect(JSON.parse(res.text).usage.completion_tokens).toBe(CANDIDATES + THOUGHTS);
    // Without the upstream's total the reasoning reads as a subset of the
    // completion, so an unfolded completion would bill the candidates as
    // nothing.
    const log = await recorded(res.requestId);
    expect(log.get("completion_tokens")).toBe(String(CANDIDATES + THOUGHTS));
    expect(log.get("reasoning_tokens")).toBe(String(THOUGHTS));
    expect(log.has("total_tokens")).toBe(false);
  });

  test("an OpenAI-compatible upstream's total is recorded verbatim", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await call("/v1/chat/completions", {
      model: OPENAI_MODEL,
      messages: [{ role: "user", content: "hi" }],
    });
    expect(res.status, res.text).toBe(200);
    const log = await recorded(res.requestId);
    expect(log.get("completion_tokens")).toBe(String(OAI_COMPLETION));
    expect(log.get("reasoning_tokens")).toBe(String(OAI_REASONING));
    expect(log.get("total_tokens")).toBe(String(OAI_TOTAL));
  });

  test("an upstream that reports no total records none, though the client is given one", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await call("/v1/chat/completions", {
      model: OPENAI_NO_TOTAL_MODEL,
      messages: [{ role: "user", content: "hi" }],
    });
    expect(res.status, res.text).toBe(200);
    expect(JSON.parse(res.text).usage.total_tokens).toBe(OAI_PROMPT + OAI_COMPLETION);
    const log = await recorded(res.requestId);
    expect(log.get("completion_tokens")).toBe(String(OAI_COMPLETION));
    expect(log.has("total_tokens")).toBe(false);
  });

  test("messages over an OpenAI-compatible upstream records reasoning and total like chat does", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await call("/v1/messages", {
      model: OPENAI_MODEL,
      max_tokens: 64,
      messages: [{ role: "user", content: "hi" }],
    });
    expect(res.status, res.text).toBe(200);
    const log = await recorded(res.requestId);
    expect(log.get("completion_tokens")).toBe(String(OAI_COMPLETION));
    expect(log.get("reasoning_tokens")).toBe(String(OAI_REASONING));
    expect(log.get("total_tokens")).toBe(String(OAI_TOTAL));
  });
});

describe("legacy completions record reasoning and total like chat does", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let upstream: OpenAiUpstream | undefined;
  const MODEL = "raw-usage-completions";

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    upstream = await startOpenAiUpstream({ nonStreamBody: OPENAI_COMPLETIONS_BODY });
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-ak-id",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-ak-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "recorded-usage-completions-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "recorded-usage-proj",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
    });
    const pk = await seed.createProviderKey({
      display_name: `${MODEL}-pk`,
      secret: "sk-openai-mock",
      api_base: `${upstream.baseUrl}/v1`,
      provider: "openai",
      adapter: "openai",
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "openai",
      model_name: "gpt-5-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: [MODEL] });
    const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      if (res.status !== 200) return false;
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return data.some((row) => row.id === MODEL);
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await sls?.close();
  });

  test("the record carries the upstream's reasoning count and total", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await fetch(`${app.proxyUrl}/v1/completions`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${CALLER_PLAINTEXT}` },
      body: JSON.stringify({ model: MODEL, prompt: "hi" }),
    });
    const text = await res.text();
    expect(res.status, text).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    const log = await waitForSlsLog(
      sls,
      LOGSTORE,
      (l) => l.get("request_id") === requestId,
      `usage row for ${requestId}`,
      15_000,
    );
    expect(log.get("completion_tokens")).toBe(String(OAI_COMPLETION));
    expect(log.get("reasoning_tokens")).toBe(String(OAI_REASONING));
    expect(log.get("total_tokens")).toBe(String(OAI_TOTAL));
  });
});
