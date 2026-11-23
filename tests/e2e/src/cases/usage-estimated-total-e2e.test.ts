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
  waitForSlsLog,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// Two upstream behaviours that look alike and are not.
//
// An upstream that LEAVES a count OUT of its usage — `{completion_tokens: N,
// total_tokens: N}`, no `prompt_tokens` — gets that count estimated. Its
// total was reported beside the missing count, so it no longer adds up to
// the counts the usage record carries: the record carries `usage_estimated:
// true` and NO `total_tokens` (the control plane derives one), and the
// client is told the same counts as the record, its total recomputed as
// estimated prompt + N. Pinned on every inbound protocol and both response
// modes.
//
// An upstream that REPORTS a count as `0` reported it. Nothing is estimated:
// the record carries the upstream's counts and total as reported, with no
// `usage_estimated`, and the client is told the same.
//
// The record is read off a metadata-only SLS exporter, which serializes the
// usage event field for field (a field the event omits is absent there).

const CALLER_PLAINTEXT = "sk-estimated-total-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const CREDENTIAL_REF = "e2e";
const SLS_PROJECT = "estimated-total-proj";
const LOGSTORE = "estimated-total-store";

const OUT = 2;

const chatBody = (usage: Record<string, number>) => ({
  id: "chatcmpl-est-total",
  object: "chat.completion",
  created: 1_700_000_000,
  model: "relay-compat-x",
  choices: [
    { index: 0, message: { role: "assistant", content: "Hello world" }, finish_reason: "stop" },
  ],
  usage,
});
const chatStream = (usage: Record<string, number>) => [
  '{"id":"chatcmpl-est-total","object":"chat.completion.chunk","model":"relay-compat-x","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
  '{"id":"chatcmpl-est-total","object":"chat.completion.chunk","model":"relay-compat-x","choices":[{"index":0,"delta":{"content":"Hello world"},"finish_reason":null}]}',
  '{"id":"chatcmpl-est-total","object":"chat.completion.chunk","model":"relay-compat-x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
  JSON.stringify({
    id: "chatcmpl-est-total",
    object: "chat.completion.chunk",
    model: "relay-compat-x",
    choices: [],
    usage,
  }),
  "[DONE]",
];

const responseObject = (usage: Record<string, number>) => ({
  id: "resp_est_total",
  object: "response",
  created_at: 1_700_000_000,
  model: "relay-compat-x",
  status: "completed",
  output: [
    {
      type: "message",
      id: "msg_est_total",
      role: "assistant",
      status: "completed",
      content: [{ type: "output_text", text: "Hello world", annotations: [] }],
    },
  ],
  usage,
});
const responsesStream = (usage: Record<string, number>) => [
  JSON.stringify({
    type: "response.created",
    response: { ...responseObject(usage), status: "in_progress", output: [], usage: null },
  }),
  JSON.stringify({ type: "response.output_text.delta", item_id: "msg_est_total", output_index: 0, content_index: 0, delta: "Hello world" }),
  JSON.stringify({ type: "response.completed", response: responseObject(usage) }),
];

const completionsBody = (usage: Record<string, number>) => ({
  id: "cmpl-est-total",
  object: "text_completion",
  created: 1_700_000_000,
  model: "relay-compat-x",
  choices: [{ index: 0, text: "Hello world", finish_reason: "stop" }],
  usage,
});

const ANTHROPIC_BODY_INPUT_ZERO = {
  id: "msg_est_total",
  type: "message",
  role: "assistant",
  model: "relay-claude-x",
  content: [{ type: "text", text: "Hello world" }],
  stop_reason: "end_turn",
  stop_sequence: null,
  usage: { input_tokens: 0, output_tokens: OUT },
};

const ANTHROPIC_BODY_INPUT_LEFT_OUT = {
  ...ANTHROPIC_BODY_INPUT_ZERO,
  usage: { output_tokens: OUT },
};

const EMBEDDINGS_NO_USAGE = {
  object: "list",
  model: "relay-compat-x",
  data: [{ object: "embedding", index: 0, embedding: [0.11, -0.22] }],
};

/** The usage each upstream returns, by the shape its endpoint speaks. */
const PROMPT_LEFT_OUT = { completion_tokens: OUT, total_tokens: OUT };
const PROMPT_ZERO = { prompt_tokens: 0, completion_tokens: OUT, total_tokens: OUT };
const INPUT_LEFT_OUT = { output_tokens: OUT, total_tokens: OUT };
const INPUT_ZERO = { input_tokens: 0, output_tokens: OUT, total_tokens: OUT };
// A reported prompt beside a completion reported as `0`, although the
// stream delivered text.
const COMPLETION_ZERO = { prompt_tokens: 5, completion_tokens: 0, total_tokens: 5 };

/** Model display name → the upstream body it is served, its provider, and
 * whether its provider key declares no Responses surface (so
 * `/v1/responses` bridges to Chat Completions). */
type Served = { nonStreamBody: unknown } | { streamEvents: string[] };
const MODELS: Record<string, { served: Served; provider?: string; bridge?: boolean }> = {
  "est-chat-absent": { served: { nonStreamBody: chatBody(PROMPT_LEFT_OUT) } },
  "est-chat-zero": { served: { nonStreamBody: chatBody(PROMPT_ZERO) } },
  "est-chat-reported": {
    served: { nonStreamBody: chatBody({ prompt_tokens: 5, completion_tokens: OUT, total_tokens: 5 + OUT }) },
  },
  "est-chat-stream-absent": { served: { streamEvents: chatStream(PROMPT_LEFT_OUT) } },
  "est-chat-stream-zero": { served: { streamEvents: chatStream(PROMPT_ZERO) } },
  "est-chat-stream-completion-zero": { served: { streamEvents: chatStream(COMPLETION_ZERO) } },
  "est-resp-absent": { served: { nonStreamBody: responseObject(INPUT_LEFT_OUT) } },
  "est-resp-zero": { served: { nonStreamBody: responseObject(INPUT_ZERO) } },
  "est-resp-stream-absent": { served: { streamEvents: responsesStream(INPUT_LEFT_OUT) } },
  "est-resp-stream-zero": { served: { streamEvents: responsesStream(INPUT_ZERO) } },
  "est-resp-bridge-absent": { served: { nonStreamBody: chatBody(PROMPT_LEFT_OUT) }, bridge: true },
  "est-resp-bridge-stream-absent": {
    served: { streamEvents: chatStream(PROMPT_LEFT_OUT) },
    bridge: true,
  },
  "est-cmpl-absent": { served: { nonStreamBody: completionsBody(PROMPT_LEFT_OUT) } },
  "est-cmpl-zero": { served: { nonStreamBody: completionsBody(PROMPT_ZERO) } },
  "est-anth-zero": { served: { nonStreamBody: ANTHROPIC_BODY_INPUT_ZERO }, provider: "anthropic" },
  "est-anth-absent": { served: { nonStreamBody: ANTHROPIC_BODY_INPUT_LEFT_OUT }, provider: "anthropic" },
  "est-embed-absent": { served: { nonStreamBody: EMBEDDINGS_NO_USAGE } },
};

function sseFrames(text: string): Array<Record<string, any>> {
  const frames: Array<Record<string, any>> = [];
  for (const line of text.split("\n")) {
    if (!line.startsWith("data: ")) continue;
    const data = line.slice(6).trim();
    if (data === "[DONE]") continue;
    try {
      frames.push(JSON.parse(data) as Record<string, any>);
    } catch {
      // not a JSON payload
    }
  }
  return frames;
}

/** The last usage `pick` finds on a stream's frames. */
function lastSseUsage(
  text: string,
  pick: (frame: Record<string, any>) => unknown,
): Record<string, number> {
  let found: Record<string, number> | undefined;
  for (const frame of sseFrames(text)) {
    const usage = pick(frame);
    if (usage && typeof usage === "object") found = usage as Record<string, number>;
  }
  if (!found) throw new Error(`no usage in stream:\n${text}`);
  return found;
}

const chatStreamUsage = (text: string) => lastSseUsage(text, (f) => f.usage);
const messagesStreamUsage = (text: string) =>
  lastSseUsage(text, (f) => (f.type === "message_delta" ? f.usage : undefined));
const responsesStreamUsage = (text: string) =>
  lastSseUsage(text, (f) => (f.type === "response.completed" ? f.response?.usage : undefined));
const bodyUsage = (text: string) => (JSON.parse(text) as { usage: Record<string, number> }).usage;

describe("estimated usage counts only what the upstream left out", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  const upstreams: OpenAiUpstream[] = [];

  async function call(path: string, body: unknown): Promise<{ requestId: string; text: string }> {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
      },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    expect(res.status, text).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    expect(requestId).not.toBe("");
    return { requestId, text };
  }

  async function record(requestId: string) {
    return waitForSlsLog(
      sls!,
      LOGSTORE,
      (l) => l.get("request_id") === requestId,
      `usage row for ${requestId}`,
      15_000,
    );
  }

  /** The request's usage record: estimated, upstream's completion kept, no
   * total. Returns its prompt count. */
  async function expectEstimatedWithoutTotal(requestId: string): Promise<number> {
    const log = await record(requestId);
    expect(log.get("usage_estimated")).toBe("true");
    expect(log.get("completion_tokens")).toBe(String(OUT));
    const prompt = Number(log.get("prompt_tokens") ?? "0");
    expect(prompt).toBeGreaterThan(0);
    expect(log.get("total_tokens")).toBeUndefined();
    return prompt;
  }

  /** The request's usage record carries the upstream's counts as reported,
   * nothing estimated. */
  async function expectAsReported(
    requestId: string,
    want: { prompt: number; completion: number; total: number },
  ) {
    const log = await record(requestId);
    expect(log.get("usage_estimated")).toBeUndefined();
    expect(log.get("prompt_tokens") ?? "0").toBe(String(want.prompt));
    expect(log.get("completion_tokens") ?? "0").toBe(String(want.completion));
    expect(log.get("total_tokens")).toBe(String(want.total));
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-ak-id",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-ak-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "estimated-total-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });

    // Non-OpenAI upstream model names, so the estimator's fallback
    // encoding applies.
    for (const [display, { served, provider, bridge }] of Object.entries(MODELS)) {
      const upstream = await startOpenAiUpstream(served);
      upstreams.push(upstream);
      const anthropic = provider === "anthropic";
      const pk = await seed.createProviderKey({
        display_name: `${display}-pk`,
        api_key: "sk-mock",
        api_base: anthropic ? upstream.baseUrl : `${upstream.baseUrl}/v1`,
        ...(bridge ? { apis: {} } : {}),
      });
      await seed.createModel({
        display_name: display,
        provider: provider ?? "openai",
        model_name: anthropic ? "relay-claude-x" : "relay-compat-x",
        provider_key_id: pk.id,
      });
    }

    // Caller key last: it authenticating implies the whole seed set is in
    // the snapshot (tests/e2e/AGENTS.md).
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: Object.keys(MODELS) });
    const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await sls?.close();
    for (const u of upstreams) await u.close();
  });

  const chatRequest = (model: string, stream = false) => ({
    model,
    messages: [{ role: "user", content: "hi" }],
    ...(stream ? { stream: true, stream_options: { include_usage: true } } : {}),
  });
  const messagesRequest = (model: string, stream = false) => ({
    model,
    max_tokens: 16,
    messages: [{ role: "user", content: "hi" }],
    ...(stream ? { stream: true } : {}),
  });
  const responsesRequest = (model: string, stream = false) => ({
    model,
    input: "hi",
    ...(stream ? { stream: true } : {}),
  });

  describe("a count the upstream left out is estimated, for the record and the client", () => {
    test("/v1/chat/completions, buffered", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const res = await call("/v1/chat/completions", chatRequest("est-chat-absent"));
      const usage = bodyUsage(res.text);
      const prompt = await expectEstimatedWithoutTotal(res.requestId);
      expect(usage).toMatchObject({ prompt_tokens: prompt, completion_tokens: OUT, total_tokens: prompt + OUT });
    });

    test("/v1/chat/completions, streamed with include_usage", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const res = await call("/v1/chat/completions", chatRequest("est-chat-stream-absent", true));
      const usage = chatStreamUsage(res.text);
      const prompt = await expectEstimatedWithoutTotal(res.requestId);
      expect(usage).toMatchObject({ prompt_tokens: prompt, completion_tokens: OUT, total_tokens: prompt + OUT });
    });

    test("/v1/messages over a Chat Completions upstream, buffered", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const res = await call("/v1/messages", messagesRequest("est-chat-absent"));
      const usage = bodyUsage(res.text);
      const prompt = await expectEstimatedWithoutTotal(res.requestId);
      expect(usage).toMatchObject({ input_tokens: prompt, output_tokens: OUT });
    });

    test("/v1/messages over a Chat Completions upstream, streamed", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const res = await call("/v1/messages", messagesRequest("est-chat-stream-absent", true));
      const usage = messagesStreamUsage(res.text);
      const prompt = await expectEstimatedWithoutTotal(res.requestId);
      expect(usage).toMatchObject({ input_tokens: prompt, output_tokens: OUT });
    });

    test("/v1/responses, native, buffered", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const res = await call("/v1/responses", responsesRequest("est-resp-absent"));
      const usage = bodyUsage(res.text);
      const prompt = await expectEstimatedWithoutTotal(res.requestId);
      expect(usage).toMatchObject({ input_tokens: prompt, output_tokens: OUT, total_tokens: prompt + OUT });
    });

    test("/v1/responses, native, streamed", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const res = await call("/v1/responses", responsesRequest("est-resp-stream-absent", true));
      const usage = responsesStreamUsage(res.text);
      const prompt = await expectEstimatedWithoutTotal(res.requestId);
      expect(usage).toMatchObject({ input_tokens: prompt, output_tokens: OUT, total_tokens: prompt + OUT });
    });

    test("/v1/responses bridged to Chat Completions, buffered", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const res = await call("/v1/responses", responsesRequest("est-resp-bridge-absent"));
      const usage = bodyUsage(res.text);
      const prompt = await expectEstimatedWithoutTotal(res.requestId);
      expect(usage).toMatchObject({ input_tokens: prompt, output_tokens: OUT, total_tokens: prompt + OUT });
    });

    test("/v1/responses bridged to Chat Completions, streamed", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const res = await call("/v1/responses", responsesRequest("est-resp-bridge-stream-absent", true));
      const usage = responsesStreamUsage(res.text);
      const prompt = await expectEstimatedWithoutTotal(res.requestId);
      expect(usage).toMatchObject({ input_tokens: prompt, output_tokens: OUT, total_tokens: prompt + OUT });
    });

    test("/v1/messages, native, buffered", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const res = await call("/v1/messages", messagesRequest("est-anth-absent"));
      const usage = bodyUsage(res.text);
      const prompt = await expectEstimatedWithoutTotal(res.requestId);
      expect(usage).toMatchObject({ input_tokens: prompt, output_tokens: OUT });
      expect(usage).not.toHaveProperty("total_tokens");
    });

    test("/v1/embeddings, no usage reported", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const res = await call("/v1/embeddings", { model: "est-embed-absent", input: "hello world" });
      const usage = bodyUsage(res.text);
      const log = await record(res.requestId);
      expect(log.get("usage_estimated")).toBe("true");
      expect(log.get("total_tokens")).toBeUndefined();
      const prompt = Number(log.get("prompt_tokens") ?? "0");
      expect(prompt).toBeGreaterThan(0);
      expect(usage).toEqual({ prompt_tokens: prompt, total_tokens: prompt });
    });

    test("/v1/completions", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const res = await call("/v1/completions", { model: "est-cmpl-absent", prompt: "hi" });
      const usage = bodyUsage(res.text);
      const prompt = await expectEstimatedWithoutTotal(res.requestId);
      expect(usage).toMatchObject({ prompt_tokens: prompt, completion_tokens: OUT, total_tokens: prompt + OUT });
    });
  });

  describe("a count the upstream reported as 0 is recorded and relayed as reported", () => {
    test("control: a fully reported usage keeps the upstream's total", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const res = await call("/v1/chat/completions", chatRequest("est-chat-reported"));
      await expectAsReported(res.requestId, { prompt: 5, completion: OUT, total: 5 + OUT });
    });

    test("/v1/chat/completions, buffered, prompt_tokens 0", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const res = await call("/v1/chat/completions", chatRequest("est-chat-zero"));
      await expectAsReported(res.requestId, { prompt: 0, completion: OUT, total: OUT });
      expect(bodyUsage(res.text)).toMatchObject({ prompt_tokens: 0, completion_tokens: OUT, total_tokens: OUT });
    });

    test("/v1/chat/completions, streamed, prompt_tokens 0", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const res = await call("/v1/chat/completions", chatRequest("est-chat-stream-zero", true));
      await expectAsReported(res.requestId, { prompt: 0, completion: OUT, total: OUT });
      expect(chatStreamUsage(res.text)).toMatchObject({ prompt_tokens: 0, completion_tokens: OUT, total_tokens: OUT });
    });

    test("/v1/chat/completions, streamed, completion_tokens 0 beside delivered text", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const res = await call("/v1/chat/completions", chatRequest("est-chat-stream-completion-zero", true));
      await expectAsReported(res.requestId, { prompt: 5, completion: 0, total: 5 });
      expect(chatStreamUsage(res.text)).toMatchObject({ prompt_tokens: 5, completion_tokens: 0, total_tokens: 5 });
    });

    test("/v1/messages over a Chat Completions upstream, streamed, prompt_tokens 0", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const res = await call("/v1/messages", messagesRequest("est-chat-stream-zero", true));
      await expectAsReported(res.requestId, { prompt: 0, completion: OUT, total: OUT });
      expect(messagesStreamUsage(res.text)).toMatchObject({ input_tokens: 0, output_tokens: OUT });
    });

    test("/v1/messages, native, input_tokens 0", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const res = await call("/v1/messages", messagesRequest("est-anth-zero"));
      const log = await record(res.requestId);
      expect(log.get("usage_estimated")).toBeUndefined();
      expect(log.get("prompt_tokens") ?? "0").toBe("0");
      expect(log.get("completion_tokens")).toBe(String(OUT));
      expect(bodyUsage(res.text)).toMatchObject({ input_tokens: 0, output_tokens: OUT });
    });

    test("/v1/responses, native, buffered, input_tokens 0", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const res = await call("/v1/responses", responsesRequest("est-resp-zero"));
      await expectAsReported(res.requestId, { prompt: 0, completion: OUT, total: OUT });
      expect(bodyUsage(res.text)).toMatchObject({ input_tokens: 0, output_tokens: OUT, total_tokens: OUT });
    });

    test("/v1/responses, native, streamed, input_tokens 0", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const res = await call("/v1/responses", responsesRequest("est-resp-stream-zero", true));
      await expectAsReported(res.requestId, { prompt: 0, completion: OUT, total: OUT });
      expect(responsesStreamUsage(res.text)).toMatchObject({ input_tokens: 0, output_tokens: OUT, total_tokens: OUT });
    });

    test("/v1/completions, prompt_tokens 0", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const res = await call("/v1/completions", { model: "est-cmpl-zero", prompt: "hi" });
      await expectAsReported(res.requestId, { prompt: 0, completion: OUT, total: OUT });
      expect(bodyUsage(res.text)).toMatchObject({ prompt_tokens: 0, completion_tokens: OUT, total_tokens: OUT });
    });
  });
});
