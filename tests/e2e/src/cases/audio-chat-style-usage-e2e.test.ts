import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  pickFreePort,
  spawnApp,
  startOpenAiUpstream,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// An OpenAI-compatible transcription upstream may report its usage in
// chat's `prompt_tokens` / `completion_tokens` rather than the
// transcription object's `input_tokens` / `output_tokens`. Those tokens
// must be recorded on every audio route that reads a usage block.

const CALLER_PLAINTEXT = "sk-audio-chat-style-usage";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const PROMPT_TOKENS = 31;
const COMPLETION_TOKENS = 9;
const PROMPT_AUDIO_TOKENS = 25;
const COMPLETION_AUDIO_TOKENS = 4;
const USAGE = {
  prompt_tokens: PROMPT_TOKENS,
  completion_tokens: COMPLETION_TOKENS,
  total_tokens: PROMPT_TOKENS + COMPLETION_TOKENS,
  prompt_tokens_details: { audio_tokens: PROMPT_AUDIO_TOKENS },
  completion_tokens_details: { audio_tokens: COMPLETION_AUDIO_TOKENS },
};
const JSON_BODY = JSON.stringify({ text: "hello world", usage: USAGE });
const FRAME_END = "\r\n\r\n";
const SSE_BODY = [
  `data: ${JSON.stringify({ type: "transcript.text.delta", delta: "hello world" })}`,
  `data: ${JSON.stringify({ type: "transcript.text.done", text: "hello world", usage: USAGE })}`,
  "data: [DONE]",
]
  .join(FRAME_END)
  .concat(FRAME_END);

interface OtlpReceiver {
  url: string;
  spanAttrs: Array<Record<string, string>>;
  close(): Promise<void>;
}

async function startOtlpReceiver(): Promise<OtlpReceiver> {
  const spanAttrs: Array<Record<string, string>> = [];
  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      try {
        const body = JSON.parse(raw);
        for (const rs of body.resourceSpans ?? []) {
          for (const ss of rs.scopeSpans ?? []) {
            for (const span of ss.spans ?? []) {
              const attrs: Record<string, string> = {};
              for (const a of span.attributes ?? []) {
                const v = a.value ?? {};
                attrs[a.key] =
                  v.stringValue ?? String(v.intValue ?? v.boolValue ?? "");
              }
              spanAttrs.push(attrs);
            }
          }
        }
      } catch {
        // ignore malformed bodies — assertions fail on missing spans
      }
      res.statusCode = 200;
      res.end("{}");
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    url: `http://127.0.0.1:${port}/v1/traces`,
    spanAttrs,
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

async function waitUsageSpan(
  recv: OtlpReceiver,
  requestId: string,
  timeoutMs = 10_000,
): Promise<Record<string, string>> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const hit = recv.spanAttrs.find(
      (a) =>
        a["aisix.request_id"] === requestId &&
        a["gen_ai.usage.input_tokens"] !== undefined,
    );
    if (hit) return hit;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(`no usage span for request_id=${requestId}`);
}

describe("audio usage reported in chat-style fields", () => {
  let app: SpawnedApp | undefined;
  let otlp: OtlpReceiver | undefined;
  let etcdReachable = false;
  const up: Record<string, OpenAiUpstream> = {};

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    up.json = await startOpenAiUpstream({ rawBody: JSON_BODY, rawContentType: "application/json" });
    up.sse = await startOpenAiUpstream({ rawBody: SSE_BODY, rawContentType: "text/event-stream" });
    otlp = await startOtlpReceiver();
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "audio-chat-usage-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });
    const model = async (name: string, upstream: OpenAiUpstream) => {
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      await seed.createModel({
        display_name: name,
        provider: "openai",
        model_name: "glm-asr-2512",
        provider_key_id: pk.id,
      });
    };
    await model("asr-json", up.json);
    await model("asr-sse", up.sse);
    // Seeded last: the key authenticating implies the whole seed set is live.
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    const deadline = Date.now() + 30_000;
    for (;;) {
      const res = await fetch(`${app.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      await res.arrayBuffer();
      if (res.status === 200) break;
      if (Date.now() > deadline) throw new Error(`caller key never authenticated: ${res.status}`);
      await new Promise((r) => setTimeout(r, 100));
    }
  });

  afterAll(async () => {
    await app?.exit();
    await otlp?.close();
    await Promise.all(Object.values(up).map((u) => u.close()));
  });

  test.for([
    ["transcriptions", "asr-json", false],
    ["transcriptions", "asr-sse", true],
    ["translations", "asr-json", false],
  ] as const)("/v1/audio/%s (%s) records the reported tokens", async ([route, model, stream], ctx) => {
    if (!etcdReachable || !app || !otlp) return ctx.skip();
    const form = new FormData();
    form.set("model", model);
    if (stream) form.set("stream", "true");
    form.set("file", new Blob([new Uint8Array([0x49, 0x44, 0x33])], { type: "audio/mpeg" }), "a.mp3");
    const res = await fetch(`${app.proxyUrl}/v1/audio/${route}`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      body: form,
    });
    expect(res.status, await res.text()).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    expect(requestId).not.toBe("");
    const span = await waitUsageSpan(otlp, requestId);
    expect(Number(span["gen_ai.usage.input_tokens"])).toBe(PROMPT_TOKENS);
    expect(Number(span["gen_ai.usage.output_tokens"])).toBe(COMPLETION_TOKENS);
    expect(Number(span["gen_ai.usage.audio.input_tokens"])).toBe(PROMPT_AUDIO_TOKENS);
    expect(Number(span["gen_ai.usage.audio.output_tokens"])).toBe(COMPLETION_AUDIO_TOKENS);
  });
});
