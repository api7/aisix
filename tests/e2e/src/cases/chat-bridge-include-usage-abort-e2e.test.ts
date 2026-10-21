import { createHash } from "node:crypto";
import { createServer, type ServerResponse, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startMockSls,
  waitConfigPropagation,
  waitForSlsLog,
  type MockSls,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: with `stream_options.include_usage`, the Claude and Gemini bridges keep
// the upstream's usage off every chunk and send it on one usage-only chunk at
// the end. A stream that stops before that end — an in-stream upstream error,
// or the client going away — must still account the latest usage the upstream
// reported, in the usage event, as it does without `include_usage`.
//
// Each mock model reports usage, then fails ("error") or stalls ("hang", the
// client then disconnects) before the stream ends.

const CALLER = "sk-chat-bridge-include-usage-abort";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const CREDENTIAL_REF = "e2e";
const SLS_PROJECT = "include-usage-abort-proj";
const LOGSTORE = "include-usage-abort-store";

function anthropicFrames(): string {
  return [
    {
      type: "message_start",
      message: {
        id: "msg_abort",
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
  ]
    .map((f) => `event: ${f.type}\ndata: ${JSON.stringify(f)}\n\n`)
    .join("");
}

const ANTHROPIC_ERROR = `event: error\ndata: ${JSON.stringify({
  type: "error",
  error: { type: "overloaded_error", message: "Overloaded" },
})}\n\n`;

// Gemini reports running counts on a frame before the last one.
const GEMINI_FIRST_FRAME = `data: ${JSON.stringify({
  candidates: [{ content: { role: "model", parts: [{ text: "Par" }] } }],
  usageMetadata: { promptTokenCount: 21, candidatesTokenCount: 2, totalTokenCount: 23 },
  modelVersion: "gemini-2.5-flash",
})}\r\n\r\n`;

const stalled: ServerResponse[] = [];

function startUpstream(): Server {
  return createServer((req, res) => {
    let raw = "";
    req.on("data", (c) => (raw += c));
    req.on("end", () => {
      const path = req.url ?? "";
      const body = JSON.parse(raw || "{}") as Record<string, unknown>;
      const target = path.includes(":stream") ? path : String(body.model ?? "");
      res.writeHead(200, { "content-type": "text/event-stream" });
      if (target.includes("gemini")) {
        res.write(GEMINI_FIRST_FRAME);
        if (target.includes("cut-error")) {
          res.end("data: {not json\r\n\r\n");
        } else {
          stalled.push(res);
        }
        return;
      }
      res.write(anthropicFrames());
      if (target.includes("cut-error")) {
        res.end(ANTHROPIC_ERROR);
      } else {
        stalled.push(res);
      }
    });
  });
}

const CASES = [
  { label: "anthropic", adapter: "anthropic", model: "claude-sonnet-4-5", usage: { prompt: "21", completion: "5" } },
  { label: "vertexClaude", adapter: "vertex", model: "claude-sonnet-4-5", usage: { prompt: "21", completion: "5" } },
  { label: "vertexGemini", adapter: "vertex", model: "gemini-2.5-flash", usage: { prompt: "21", completion: "2" } },
] as const;
const MODES = ["cut-error", "cut-hang"] as const;
const modelName = (label: string, mode: string) => `include-usage-${label}-${mode}`;

describe("include_usage streams that stop early still account the reported usage", () => {
  let app: SpawnedApp | undefined;
  let upstream: Server | undefined;
  let sls: MockSls | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = startUpstream();
    await new Promise<void>((resolve) => upstream!.listen(0, "127.0.0.1", resolve));
    const addr = upstream.address();
    if (addr === null || typeof addr === "string") throw new Error("no port");
    const baseUrl = `http://127.0.0.1:${addr.port}`;

    sls = await startMockSls();
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-ak-id",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-ak-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "include-usage-abort-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
    });
    const anthropicPk = await seed.createProviderKey({
      display_name: "include-usage-abort-anthropic-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-anth-mock",
      api_base: baseUrl,
    });
    const vertexPk = await seed.createProviderKey({
      display_name: "include-usage-abort-vertex-pk",
      provider: "google",
      adapter: "vertex",
      secret: JSON.stringify({ access_token: "ya29.include-usage", project: "p", region: "us-east5" }),
      api_base: baseUrl,
    });
    const names: string[] = [];
    for (const c of CASES) {
      for (const mode of MODES) {
        const name = modelName(c.label, mode);
        names.push(name);
        // The mock reads the mode from the upstream model name.
        await seed.createModel({
          display_name: name,
          provider: c.adapter === "anthropic" ? "anthropic" : "google",
          model_name: `${c.model}-${mode}`,
          provider_key_id: c.adapter === "anthropic" ? anthropicPk.id : vertexPk.id,
        });
      }
    }
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: names });
    const probe = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
  });

  afterAll(async () => {
    for (const res of stalled) res.destroy();
    await app?.exit();
    await sls?.close();
    await new Promise<void>((resolve) => (upstream ? upstream.close(() => resolve()) : resolve()));
  });

  // Reads the stream until `until` matches (then disconnects) or it ends.
  async function streamChat(model: string, until?: RegExp) {
    const abort = new AbortController();
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      signal: abort.signal,
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify({
        model,
        stream: true,
        stream_options: { include_usage: true },
        messages: [{ role: "user", content: "What is the capital of France?" }],
      }),
    });
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let raw = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      raw += decoder.decode(value, { stream: true });
      if (until?.test(raw)) {
        abort.abort();
        break;
      }
    }
    return { status: res.status, requestId, raw };
  }

  for (const c of CASES) {
    for (const mode of MODES) {
      test(`${c.label}: ${mode === "cut-error" ? "an in-stream upstream error" : "a client disconnect"} keeps the usage`, async (ctx) => {
        if (!etcdReachable || !app) return ctx.skip();
        // Disconnect once the last content the upstream sent has arrived.
        const until = mode === "cut-hang" ? (c.label === "vertexGemini" ? /"Par"/ : /"finish_reason"/) : undefined;
        const out = await streamChat(modelName(c.label, mode), until);
        expect(out.status, out.raw).toBe(200);
        expect(out.requestId).not.toBe("");
        if (mode === "cut-error") expect(out.raw).toContain("event: error");
        // Nothing before the end carries usage to the client.
        expect(out.raw, out.raw).not.toContain('"usage"');

        const row = await waitForSlsLog(
          sls!,
          LOGSTORE,
          (l) => l.get("request_id") === out.requestId,
          `usage row for ${out.requestId}`,
          15_000,
        );
        expect(row.get("prompt_tokens"), JSON.stringify([...row])).toBe(c.usage.prompt);
        expect(row.get("completion_tokens"), JSON.stringify([...row])).toBe(c.usage.completion);
      });
    }
  }
});
