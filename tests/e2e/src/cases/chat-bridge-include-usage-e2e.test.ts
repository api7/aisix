import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  waitConfigPropagation,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: a streamed /v1/chat/completions request with the OpenAI SDK's
// `stream_options: {include_usage: true}`, sent to a model whose upstream
// speaks another protocol.
//
// `stream_options` is an OpenAI request parameter. The Anthropic Messages API
// (and Claude on Vertex AI) refuses any top-level field it does not define, so
// it must not reach those upstreams — the request is answered, not refused.
//
// `include_usage` itself is a promise about the response the caller reads:
// one extra chunk, after the finish chunk, whose `choices` is empty and which
// alone carries `usage`. Claude reports usage on the same event that ends the
// message, and Gemini on its last chunk beside the finish reason, so the usage
// has to arrive on a chunk of its own all the same — and once, even when
// the upstream reports running counts on earlier frames too.
//
// The mock upstream accepts only the top-level fields the Messages API
// documents and rejects anything else the way Anthropic does.

const CALLER = "sk-chat-bridge-include-usage";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");

const MODELS = {
  anthropic: "include-usage-anthropic",
  vertexClaude: "include-usage-vertex-claude",
  vertexGemini: "include-usage-vertex-gemini",
} as const;

const ANTHROPIC_FIELDS = new Set([
  "model",
  "messages",
  "max_tokens",
  "system",
  "metadata",
  "stop_sequences",
  "stream",
  "temperature",
  "top_k",
  "top_p",
  "tools",
  "tool_choice",
  "thinking",
  "service_tier",
  "container",
  "mcp_servers",
  "context_management",
  "output_config",
  "inference_geo",
  "cache_control",
]);
// Claude on Vertex AI takes the version in the body and the model in the URL.
const VERTEX_CLAUDE_FIELDS = new Set([...ANTHROPIC_FIELDS, "anthropic_version"]);

function anthropicStream(): string {
  return [
    {
      type: "message_start",
      message: {
        id: "msg_include_usage",
        type: "message",
        role: "assistant",
        content: [],
        model: "claude-sonnet-4-5",
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 21, output_tokens: 1 },
      },
    },
    { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
    { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Paris." } },
    { type: "content_block_stop", index: 0 },
    { type: "message_delta", delta: { stop_reason: "end_turn", stop_sequence: null }, usage: { output_tokens: 5 } },
    { type: "message_stop" },
  ]
    .map((f) => `event: ${f.type}\ndata: ${JSON.stringify(f)}\n\n`)
    .join("");
}

function geminiStream(): string {
  return [
    // Gemini may report running counts before the last frame.
    {
      candidates: [{ content: { role: "model", parts: [{ text: "Par" }] } }],
      usageMetadata: { promptTokenCount: 21, candidatesTokenCount: 2, totalTokenCount: 23 },
      modelVersion: "gemini-2.5-flash",
    },
    {
      candidates: [{ content: { role: "model", parts: [{ text: "is." }] }, finishReason: "STOP" }],
      usageMetadata: { promptTokenCount: 21, candidatesTokenCount: 5, totalTokenCount: 26 },
      modelVersion: "gemini-2.5-flash",
    },
  ]
    .map((f) => `data: ${JSON.stringify(f)}\r\n\r\n`)
    .join("");
}

interface Received {
  path: string;
  body: Record<string, unknown>;
}

function startUpstream(received: Received[]): Server {
  return createServer((req, res) => {
    let raw = "";
    req.on("data", (c) => (raw += c));
    req.on("end", () => {
      const path = req.url ?? "";
      const body = JSON.parse(raw || "{}") as Record<string, unknown>;
      received.push({ path, body });
      if (path.includes(":streamGenerateContent")) {
        res.writeHead(200, { "content-type": "text/event-stream" });
        res.end(geminiStream());
        return;
      }
      const allowed = path.includes(":streamRawPredict") ? VERTEX_CLAUDE_FIELDS : ANTHROPIC_FIELDS;
      const extra = Object.keys(body).find((k) => !allowed.has(k));
      if (extra !== undefined) {
        res.writeHead(400, { "content-type": "application/json" });
        res.end(
          JSON.stringify({
            type: "error",
            error: { type: "invalid_request_error", message: `${extra}: Extra inputs are not permitted` },
          }),
        );
        return;
      }
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.end(anthropicStream());
    });
  });
}

function parseChatSse(raw: string): Array<Record<string, any>> {
  return raw
    .split("\n")
    .filter((l) => l.startsWith("data: ") && l !== "data: [DONE]")
    .map((l) => JSON.parse(l.slice(6)));
}

describe("chat streaming with stream_options.include_usage on bridged upstreams", () => {
  let app: SpawnedApp | undefined;
  let upstream: Server | undefined;
  const received: Received[] = [];
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = startUpstream(received);
    await new Promise<void>((resolve) => upstream!.listen(0, "127.0.0.1", resolve));
    const addr = upstream.address();
    if (addr === null || typeof addr === "string") throw new Error("no port");
    const baseUrl = `http://127.0.0.1:${addr.port}`;

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const anthropicPk = await seed.createProviderKey({
      display_name: `${MODELS.anthropic}-pk`,
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-anth-mock",
      api_base: baseUrl,
    });
    await seed.createModel({
      display_name: MODELS.anthropic,
      provider: "anthropic",
      model_name: "claude-sonnet-4-5",
      provider_key_id: anthropicPk.id,
    });
    const vertexPk = await seed.createProviderKey({
      display_name: "include-usage-vertex-pk",
      provider: "google",
      adapter: "vertex",
      secret: JSON.stringify({ access_token: "ya29.include-usage", project: "p", region: "us-east5" }),
      api_base: baseUrl,
    });
    await seed.createModel({
      display_name: MODELS.vertexClaude,
      provider: "google",
      model_name: "claude-sonnet-4-5",
      provider_key_id: vertexPk.id,
    });
    await seed.createModel({
      display_name: MODELS.vertexGemini,
      provider: "google",
      model_name: "gemini-2.5-flash",
      provider_key_id: vertexPk.id,
    });
    // Seeded last: this key authenticating implies the whole seed set landed.
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: Object.values(MODELS) });
    const probe = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await new Promise<void>((resolve) => (upstream ? upstream.close(() => resolve()) : resolve()));
  });

  async function streamChat(model: string, extra: Record<string, unknown>) {
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify({
        model,
        stream: true,
        messages: [{ role: "user", content: "What is the capital of France?" }],
        ...extra,
      }),
    });
    const raw = await res.text();
    return { status: res.status, raw, chunks: res.status === 200 ? parseChatSse(raw) : [] };
  }

  for (const [label, model] of Object.entries(MODELS)) {
    test(`${label}: include_usage ends the stream with one usage-only chunk after the finish chunk`, async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const before = received.length;
      const out = await streamChat(model, { stream_options: { include_usage: true } });
      expect(out.status, out.raw).toBe(200);

      // The upstream never saw the OpenAI parameter.
      const sent = received.slice(before);
      expect(sent).toHaveLength(1);
      expect(sent[0]!.body).not.toHaveProperty("stream_options");

      const content = out.chunks.map((c) => c.choices?.[0]?.delta?.content ?? "").join("");
      expect(content).toBe("Paris.");

      const withUsage = out.chunks.filter((c) => c.usage != null);
      expect(withUsage, out.raw).toHaveLength(1);
      const last = out.chunks.at(-1)!;
      expect(last.choices, out.raw).toEqual([]);
      expect(last.usage).toMatchObject({ prompt_tokens: 21, completion_tokens: 5, total_tokens: 26 });

      const finishes = out.chunks.filter((c) => c.choices?.[0]?.finish_reason != null);
      expect(finishes).toHaveLength(1);
      expect(finishes[0]!.choices[0].finish_reason).toBe("stop");
      expect(finishes[0]!.usage ?? null).toBeNull();
      expect(out.chunks.indexOf(finishes[0]!)).toBe(out.chunks.length - 2);
    });

    if (label !== "vertexGemini") {
      // OpenAI's `max_completion_tokens` and `stop` are Anthropic's `max_tokens`
      // and `stop_sequences`; sent under their OpenAI names they are refused.
      test(`${label}: max_completion_tokens and stop reach Claude under the Messages API names`, async (ctx) => {
        if (!etcdReachable || !app) return ctx.skip();
        const cases: Array<[Record<string, unknown>, Record<string, unknown>]> = [
          [{ max_completion_tokens: 77, stop: "END" }, { max_tokens: 77, stop_sequences: ["END"] }],
          [{ max_completion_tokens: 77, max_tokens: 33, stop: ["A", "B"] }, { max_tokens: 77, stop_sequences: ["A", "B"] }],
          [{ max_tokens: 33, stop: null }, { max_tokens: 33 }],
          [{ stop: [] }, { max_tokens: 4096 }],
        ];
        for (const [extra, expected] of cases) {
          const before = received.length;
          const out = await streamChat(model, extra);
          expect(out.status, `${JSON.stringify(extra)}: ${out.raw}`).toBe(200);
          const body = received.slice(before)[0]!.body;
          expect(body).toMatchObject(expected);
          expect(body).not.toHaveProperty("max_completion_tokens");
          expect(body).not.toHaveProperty("stop");
          if (!("stop_sequences" in expected)) expect(body).not.toHaveProperty("stop_sequences");
        }
      });
    }

    test(`${label}: without include_usage no usage-only chunk is sent`, async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      for (const extra of [{}, { stream_options: { include_usage: false } }]) {
        const out = await streamChat(model, extra);
        expect(out.status, out.raw).toBe(200);
        expect(out.chunks.every((c) => (c.choices ?? []).length === 1), out.raw).toBe(true);
        const finish = out.chunks.find((c) => c.choices[0].finish_reason != null)!;
        expect(finish.choices[0].finish_reason).toBe("stop");
        // Unchanged: these bridges report usage on the finish chunk.
        expect(finish.usage).toMatchObject({ prompt_tokens: 21, completion_tokens: 5 });
      }
    });
  }
});
