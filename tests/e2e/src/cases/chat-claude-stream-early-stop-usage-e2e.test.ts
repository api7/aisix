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

// E2E: a stream to a Claude-protocol upstream that stops before its closing
// `message_delta` — the client goes away, or the upstream fails mid-stream.
// Claude reports the input and cache counts on the first event,
// `message_start`, so the usage event must carry those as reported rather
// than an estimate of the prompt, as `/v1/messages` does. Covered for a chat
// completions stream (with and without `include_usage`) on Anthropic and on
// Claude on Vertex AI, and for a Responses API stream on Anthropic.

const CALLER = "sk-claude-stream-early-stop";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const CREDENTIAL_REF = "e2e";
const LOGSTORE = "claude-early-stop-store";

const INPUT = 1699;
const CACHE_READ = 300;
const CACHE_WRITE = 40;

function startFrames(): string {
  return [
    {
      type: "message_start",
      message: {
        id: "msg_early_stop",
        type: "message",
        role: "assistant",
        content: [],
        model: "claude-haiku-4-5",
        stop_reason: null,
        stop_sequence: null,
        usage: {
          input_tokens: INPUT,
          cache_read_input_tokens: CACHE_READ,
          cache_creation_input_tokens: CACHE_WRITE,
          output_tokens: 1,
        },
      },
    },
    { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
    { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "One, " } },
    { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "two, " } },
    { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "three." } },
  ]
    .map((f) => `event: ${f.type}\ndata: ${JSON.stringify(f)}\n\n`)
    .join("");
}

const UPSTREAM_ERROR = `event: error\ndata: ${JSON.stringify({
  type: "error",
  error: { type: "overloaded_error", message: "Overloaded" },
})}\n\n`;

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
      res.write(startFrames());
      if (target.includes("cut-error")) {
        res.end(UPSTREAM_ERROR);
      } else {
        stalled.push(res);
      }
    });
  });
}

const PLATFORMS = ["anthropic", "vertex"] as const;
const MODES = ["cut-error", "cut-hang"] as const;
const modelName = (platform: string, mode: string) => `early-stop-${platform}-${mode}`;

describe("a Claude stream that stops before message_delta accounts the reported input and cache counts", () => {
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
      name: "claude-early-stop-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "claude-early-stop-proj",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
    });
    const anthropicPk = await seed.createProviderKey({
      display_name: "early-stop-anthropic-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-anth-mock",
      api_base: baseUrl,
    });
    const vertexPk = await seed.createProviderKey({
      display_name: "early-stop-vertex-pk",
      provider: "google",
      adapter: "vertex",
      secret: JSON.stringify({ access_token: "ya29.early-stop", project: "p", region: "us-east5" }),
      api_base: baseUrl,
    });
    const names: string[] = [];
    for (const platform of PLATFORMS) {
      for (const mode of MODES) {
        const name = modelName(platform, mode);
        names.push(name);
        // The mock reads the mode from the upstream model name.
        await seed.createModel({
          display_name: name,
          provider: platform === "anthropic" ? "anthropic" : "google",
          model_name: `claude-haiku-4-5-${mode}`,
          provider_key_id: platform === "anthropic" ? anthropicPk.id : vertexPk.id,
        });
      }
    }
    // Seeded last: this key authenticating implies the whole seed set landed.
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
  async function stream(path: string, body: Record<string, unknown>, until?: RegExp) {
    const abort = new AbortController();
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      signal: abort.signal,
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify({ stream: true, ...body }),
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

  async function expectReportedInput(requestId: string) {
    const row = await waitForSlsLog(
      sls!,
      LOGSTORE,
      (l) => l.get("request_id") === requestId,
      `usage row for ${requestId}`,
      15_000,
    );
    const all = JSON.stringify([...row]);
    expect(row.get("prompt_tokens"), all).toBe(String(INPUT));
    expect(row.get("cache_read_tokens"), all).toBe(String(CACHE_READ));
    expect(row.get("cache_creation_tokens"), all).toBe(String(CACHE_WRITE));
  }

  const MESSAGES = [{ role: "user", content: "Count to three." }];

  for (const platform of PLATFORMS) {
    for (const mode of MODES) {
      for (const includeUsage of [false, true]) {
        const stop = mode === "cut-error" ? "an in-stream upstream error" : "a client disconnect";
        test(`chat ${platform}: ${stop}${includeUsage ? " with include_usage" : ""}`, async (ctx) => {
          if (!etcdReachable || !app) return ctx.skip();
          const out = await stream(
            "/v1/chat/completions",
            {
              model: modelName(platform, mode),
              messages: MESSAGES,
              ...(includeUsage ? { stream_options: { include_usage: true } } : {}),
            },
            mode === "cut-hang" ? /three\./ : undefined,
          );
          expect(out.status, out.raw).toBe(200);
          expect(out.raw).toContain("three.");
          if (mode === "cut-error") expect(out.raw).toContain("event: error");
          expect(out.raw, out.raw).not.toContain('"usage"');
          await expectReportedInput(out.requestId);
        });
      }
    }
  }

  test("responses anthropic: a client disconnect", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const out = await stream(
      "/v1/responses",
      { model: modelName("anthropic", "cut-hang"), input: "Count to three." },
      /three\./,
    );
    expect(out.status, out.raw).toBe(200);
    await expectReportedInput(out.requestId);
  });
});
