import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  slsLogsFor,
  spawnApp,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForLogLine,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";
import type { OpenAiUpstreamOptions, OpenAiUpstreamStep } from "../harness/upstream-openai.js";

// E2E for AISIX-Cloud#1136: a request whose first attempt ends in a 504 and
// whose same-target retry succeeds must leave BOTH attempts in the usage
// export, under the one request id — the failed attempt with its 504, the
// retry with its 200. Driven for both shapes of 504: one the upstream
// answers itself, and one the gateway raises when the upstream outlives the
// model's timeout. Observed through the SLS exporter, which receives the
// usage rows as the control plane does, and in the plain log, whose
// failed-attempt line must carry the same request id.

const CALLER_PLAINTEXT = "sk-retried-attempt-1136";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const CREDENTIAL_REF = "mock";
const LOGSTORE = "retried-attempt-store";

// Long enough past the model's timeout that the gateway gives up first.
const TIMEOUT_MS = 400;
const SLOW_MS = 2_500;

const CHAT_BODY = {
  id: "chatcmpl-1136",
  object: "chat.completion",
  created: 1,
  model: "upstream-echo",
  choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
  usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
};
const CHAT_STREAM = [
  JSON.stringify({
    id: "chatcmpl-1136",
    object: "chat.completion.chunk",
    model: "upstream-echo",
    choices: [{ index: 0, delta: { role: "assistant", content: "ok" }, finish_reason: null }],
  }),
  JSON.stringify({
    id: "chatcmpl-1136",
    object: "chat.completion.chunk",
    model: "upstream-echo",
    choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
    usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
  }),
  "[DONE]",
];
const ANTHROPIC_BODY = {
  id: "msg_1136",
  type: "message",
  role: "assistant",
  model: "claude-3-5-haiku-20241022",
  content: [{ type: "text", text: "ok" }],
  stop_reason: "end_turn",
  usage: { input_tokens: 3, output_tokens: 2 },
};
// One body every single-shot endpoint's response parser accepts.
const SINGLE_SHOT_BODY = {
  id: "video_1136",
  object: "video",
  status: "queued",
  progress: 0,
  created: 1,
  created_at: 1,
  model: "upstream-echo",
  text: "ok",
  choices: [{ index: 0, text: "ok", finish_reason: "stop" }],
  data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2], url: "https://example.com/i.png" }],
  results: [{ index: 0, relevance_score: 0.9 }],
  usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5, input_tokens: 3, output_tokens: 2 },
};

type Kind = "openai" | "openai-bridged-responses" | "anthropic" | "video";

interface Surface {
  label: string;
  kind: Kind;
  ok: OpenAiUpstreamOptions;
  call: (model: string) => Promise<Response>;
}

const auth = { authorization: `Bearer ${CALLER_PLAINTEXT}` };

describe("a retried 504 attempt stays in the usage export (AISIX-Cloud#1136)", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  const upstreams: OpenAiUpstream[] = [];

  const json =
    (path: string, body: (model: string) => Record<string, unknown>) =>
    (model: string): Promise<Response> =>
      fetch(`${app!.proxyUrl}${path}`, {
        method: "POST",
        headers: { ...auth, "content-type": "application/json" },
        body: JSON.stringify(body(model)),
      });
  const multipart =
    (path: string, file: string, fields: Record<string, string> = {}) =>
    (model: string): Promise<Response> => {
      const form = new FormData();
      form.set("model", model);
      for (const [k, v] of Object.entries(fields)) form.set(k, v);
      form.set(file, new Blob([new Uint8Array([0x49, 0x44, 0x33])], { type: "application/octet-stream" }), "f.bin");
      return fetch(`${app!.proxyUrl}${path}`, { method: "POST", headers: auth, body: form });
    };

  const chat = (stream: boolean) =>
    json("/v1/chat/completions", (model) => ({ model, stream, messages: [{ role: "user", content: "hi" }] }));
  const messages = (stream: boolean) =>
    json("/v1/messages", (model) => ({
      model,
      stream,
      max_tokens: 16,
      messages: [{ role: "user", content: "hi" }],
    }));
  const responses = (stream: boolean) => json("/v1/responses", (model) => ({ model, stream, input: "hi" }));

  const surfaces: Surface[] = [
    { label: "chat", kind: "openai", ok: { nonStreamBody: CHAT_BODY }, call: chat(false) },
    { label: "chat-stream", kind: "openai", ok: { streamEvents: CHAT_STREAM }, call: chat(true) },
    { label: "messages-bridged", kind: "openai", ok: { nonStreamBody: CHAT_BODY }, call: messages(false) },
    { label: "messages-bridged-stream", kind: "openai", ok: { streamEvents: CHAT_STREAM }, call: messages(true) },
    { label: "messages-native", kind: "anthropic", ok: { nonStreamBody: ANTHROPIC_BODY }, call: messages(false) },
    {
      label: "responses-bridged",
      kind: "openai-bridged-responses",
      ok: { nonStreamBody: CHAT_BODY },
      call: responses(false),
    },
    {
      label: "responses-bridged-stream",
      kind: "openai-bridged-responses",
      ok: { streamEvents: CHAT_STREAM },
      call: responses(true),
    },
    {
      label: "count_tokens",
      kind: "anthropic",
      ok: { nonStreamBody: { input_tokens: 3 } },
      call: json("/v1/messages/count_tokens", (model) => ({ model, messages: [{ role: "user", content: "hi" }] })),
    },
    {
      label: "completions",
      kind: "openai",
      ok: { nonStreamBody: SINGLE_SHOT_BODY },
      call: json("/v1/completions", (model) => ({ model, prompt: "hi" })),
    },
    {
      label: "embeddings",
      kind: "openai",
      ok: { nonStreamBody: SINGLE_SHOT_BODY },
      call: json("/v1/embeddings", (model) => ({ model, input: "hi" })),
    },
    {
      label: "rerank",
      kind: "openai",
      ok: { nonStreamBody: SINGLE_SHOT_BODY },
      call: json("/v1/rerank", (model) => ({ model, query: "q", documents: ["a", "b"] })),
    },
    {
      label: "images-generations",
      kind: "openai",
      ok: { nonStreamBody: SINGLE_SHOT_BODY },
      call: json("/v1/images/generations", (model) => ({ model, prompt: "a cat" })),
    },
    {
      label: "images-edits",
      kind: "openai",
      ok: { nonStreamBody: SINGLE_SHOT_BODY },
      call: multipart("/v1/images/edits", "image", { prompt: "a hat" }),
    },
    {
      label: "audio-transcriptions",
      kind: "openai",
      ok: { nonStreamBody: SINGLE_SHOT_BODY },
      call: multipart("/v1/audio/transcriptions", "file"),
    },
    {
      label: "audio-translations",
      kind: "openai",
      ok: { nonStreamBody: SINGLE_SHOT_BODY },
      call: multipart("/v1/audio/translations", "file"),
    },
    {
      label: "audio-speech",
      kind: "openai",
      ok: { rawBody: "ID3-whole-audio", rawContentType: "audio/mpeg" },
      call: json("/v1/audio/speech", (model) => ({ model, input: "hello", voice: "alloy" })),
    },
    {
      label: "videos",
      kind: "video",
      ok: { nonStreamBody: SINGLE_SHOT_BODY },
      call: json("/v1/videos", (model) => ({ model, prompt: "a boat" })),
    },
  ];

  // The first attempt's two shapes of 504.
  // The first attempt's two shapes of 504, and the status its row carries.
  // An upstream's own 5xx reaches the caller as a 502, whatever its code,
  // so that is the attempt's status too; the gateway's own deadline is a 504.
  const failures: Array<[string, string, (ok: OpenAiUpstreamOptions) => OpenAiUpstreamStep]> = [
    [
      "upstream-504",
      "502",
      () => ({ status: 504, errorBody: { error: { message: "upstream gateway timeout" } } }),
    ],
    ["gateway-timeout", "504", (ok) => ({ ...(ok as OpenAiUpstreamStep), responseDelayMs: SLOW_MS })],
  ];

  const cases = surfaces.flatMap((s) =>
    failures.map(([failure, failedStatus, step]) => ({
      ...s,
      failure,
      failedStatus,
      step,
      model: `r1136-${s.label}-${failure}`,
    })),
  );

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    sls = await startMockSls();
    // At the default `info`, so the request span every line inherits its
    // `request_id` from is live — the harness otherwise runs at `warn`.
    app = await spawnApp({
      logLevel: "info",
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "sls-retried-attempt",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "retried-attempt-proj",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });
    for (const c of cases) {
      // First request fails, every later one is served.
      const upstream = await startOpenAiUpstream({ ...c.ok, scriptedResponses: [c.step(c.ok)] });
      upstreams.push(upstream);
      const base =
        c.kind === "anthropic" || c.kind === "video" ? upstream.baseUrl : `${upstream.baseUrl}/v1`;
      const pk = await seed.createProviderKey({
        display_name: `${c.model}-pk`,
        secret: "sk-mock",
        api_base: base,
        ...(c.kind === "anthropic" ? { provider: "anthropic", adapter: "anthropic" } : {}),
        ...(c.kind === "openai-bridged-responses" ? { apis: {} } : {}),
      });
      await seed.createModel({
        display_name: c.model,
        provider: c.kind === "anthropic" ? "anthropic" : "openai",
        model_name: c.kind === "anthropic" ? "claude-3-5-haiku-20241022" : "upstream-echo",
        provider_key_id: pk.id,
        retries: 1,
        timeout: TIMEOUT_MS,
        stream_timeout: TIMEOUT_MS,
      });
    }
    // Seeded last: the key authenticating implies the whole set is live.
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, { headers: auth });
      await res.arrayBuffer();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    for (const u of upstreams) await u.close();
    await sls?.close();
  });

  async function rowsFor(requestId: string, count: number): Promise<Map<string, string>[]> {
    const matching = () =>
      slsLogsFor(sls!, LOGSTORE)
        .filter((l) => l.get("request_id") === requestId)
        .sort((a, b) => Number(a.get("attempt_index") ?? 0) - Number(b.get("attempt_index") ?? 0));
    const deadline = Date.now() + 15_000;
    while (Date.now() < deadline && matching().length < count) {
      await new Promise((r) => setTimeout(r, 100));
    }
    // Settle past the count: a surplus row would ride the same export.
    await new Promise((r) => setTimeout(r, 500));
    return matching();
  }

  test.for(cases)("$label, first attempt $failure: both attempts are exported", { timeout: 60_000 }, async (c, ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await c.call(c.model);
    const text = await res.text();
    expect(res.status, `${c.label}: ${text}`).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    expect(requestId).not.toBe("");

    const rows = await rowsFor(requestId, 2);
    const summary = rows.map((r) => `${r.get("attempt_index")}/${r.get("attempt_kind")}/${r.get("status_code")}`);
    expect(rows, `${c.label}: exported rows ${JSON.stringify(summary)}`).toHaveLength(2);
    const [failed, retry] = rows;
    expect(failed.get("attempt_index") ?? "0").toBe("0");
    expect(failed.get("status_code")).toBe(c.failedStatus);
    expect(failed.get("error_class") ?? "", "the failed attempt names its failure").not.toBe("");
    expect(retry.get("attempt_index")).toBe("1");
    expect(retry.get("attempt_kind")).toBe("retry");
    expect(retry.get("status_code")).toBe("200");

    // The plain log names the failed attempt under the same request id.
    await waitForLogLine(
      app,
      (l) => l.includes("routing target attempt failed") && l.includes(requestId),
      `${c.label}: the failed-attempt log line of ${requestId}`,
    );
  });
});
