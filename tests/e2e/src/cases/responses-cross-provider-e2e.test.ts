import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  pickFreePort,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E for AISIX-Cloud#825: the OpenAI Responses API (`POST /v1/responses`)
// must work against a non-OpenAI backend. The `codex` CLI speaks only the
// Responses API; pointing it at an Anthropic model (e.g. `opus-4.7`) used to
// return a hard 400 ("model ... is not an OpenAI provider; /v1/responses
// requires OpenAI"). The gateway now bridges the Responses request through
// the canonical ChatFormat to the Anthropic Messages upstream and re-encodes
// the reply back into the Responses-API shape — non-streaming JSON and
// streaming SSE — and translates the agent-loop tool history so multi-turn
// tool calls round-trip.

const CALLER_PLAINTEXT = "sk-issue-825-xprovider";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

// Anthropic Messages non-streaming reply (text).
const ANTHROPIC_NONSTREAM = {
  id: "msg_xprov_ns",
  type: "message",
  role: "assistant",
  model: "claude-3-haiku-20240307",
  content: [{ type: "text", text: "Hello from Claude" }],
  stop_reason: "end_turn",
  usage: { input_tokens: 11, output_tokens: 7 },
};

const STREAM_INPUT_TOKENS = 12;
const STREAM_OUTPUT_TOKENS = 5;

// Anthropic Messages streaming wire shape (data-only; the mock writes the
// `data:` line). The DP parses by the JSON `type`.
const ANTHROPIC_STREAM_EVENTS = [
  JSON.stringify({
    type: "message_start",
    message: {
      id: "msg_xprov_s",
      type: "message",
      role: "assistant",
      model: "claude-3-haiku-20240307",
      content: [],
      usage: { input_tokens: STREAM_INPUT_TOKENS, output_tokens: 0 },
    },
  }),
  JSON.stringify({
    type: "content_block_start",
    index: 0,
    content_block: { type: "text", text: "" },
  }),
  JSON.stringify({
    type: "content_block_delta",
    index: 0,
    delta: { type: "text_delta", text: "Hi from Claude" },
  }),
  JSON.stringify({ type: "content_block_stop", index: 0 }),
  JSON.stringify({
    type: "message_delta",
    delta: { stop_reason: "end_turn" },
    usage: { output_tokens: STREAM_OUTPUT_TOKENS },
  }),
  JSON.stringify({ type: "message_stop" }),
];

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
      } catch (err) {
        // Surface malformed OTLP so a decode bug is obvious rather than only
        // showing up later as a missing-span assertion failure.
        console.error("OTLP receiver failed to parse body:", err);
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

async function collectUsageSpans(
  recv: OtlpReceiver,
  requestId: string,
  timeoutMs = 10_000,
): Promise<Array<Record<string, string>>> {
  const matches = () =>
    recv.spanAttrs.filter(
      (a) =>
        a["aisix.request_id"] === requestId &&
        a["gen_ai.usage.input_tokens"] !== undefined,
    );
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (matches().length > 0) break;
    await new Promise((r) => setTimeout(r, 50));
  }
  if (matches().length === 0) {
    throw new Error(`no usage span for request_id=${requestId}`);
  }
  await new Promise((r) => setTimeout(r, 300));
  return matches();
}

function post(app: SpawnedApp, body: unknown): Promise<Response> {
  return fetch(`${app.proxyUrl}/v1/responses`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
      "user-agent": "codex_cli_rs/0.5.0",
    },
    body: JSON.stringify(body),
  });
}

describe("/v1/responses cross-provider → Anthropic (#825)", () => {
  let app: SpawnedApp | undefined;
  let nsUpstream: OpenAiUpstream | undefined;
  let streamUpstream: OpenAiUpstream | undefined;
  let otlp: OtlpReceiver | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    nsUpstream = await startOpenAiUpstream({ nonStreamBody: ANTHROPIC_NONSTREAM });
    streamUpstream = await startOpenAiUpstream({
      streamEvents: ANTHROPIC_STREAM_EVENTS,
      eventDelayMs: 2,
    });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    otlp = await startOtlpReceiver();
    await seed.createObservabilityExporter({
      name: "issue825-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });

    // Two Anthropic-backed models, one per mock (non-streaming vs streaming);
    // api_base is the bare host — the bridge composes `/v1/messages`.
    const nsPk = await seed.createProviderKey({
      display_name: "issue825-ns-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-mock",
      api_base: nsUpstream.baseUrl,
    });
    await seed.createModel({
      display_name: "opus-4.7",
      provider: "anthropic",
      model_name: "claude-3-haiku-20240307",
      provider_key_id: nsPk.id,
    });
    const streamPk = await seed.createProviderKey({
      display_name: "issue825-stream-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-mock",
      api_base: streamUpstream.baseUrl,
    });
    await seed.createModel({
      display_name: "opus-4.7-stream",
      provider: "anthropic",
      model_name: "claude-3-haiku-20240307",
      provider_key_id: streamPk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["opus-4.7", "opus-4.7-stream"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await nsUpstream?.close();
    await streamUpstream?.close();
    await otlp?.close();
  });

  test("non-streaming: Anthropic reply is re-encoded into the Responses shape + usage event", async (ctx) => {
    if (!etcdReachable || !app || !nsUpstream || !otlp) {
      ctx.skip();
      return;
    }
    await waitConfigPropagation(async () => {
      try {
        const r = await post(app!, { model: "opus-4.7", input: "ready" });
        return r.status === 200 && (await r.json()).object === "response";
      } catch {
        return false;
      }
    });

    const res = await post(app, { model: "opus-4.7", input: "hi" });
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id");
    expect(requestId).toBeTruthy();

    const body = await res.json();
    expect(body.object).toBe("response");
    expect(body.status).toBe("completed");
    // Operator-facing model name, not the upstream id.
    expect(body.model).toBe("opus-4.7");
    expect(body.output[0].type).toBe("message");
    expect(body.output[0].content[0].type).toBe("output_text");
    expect(body.output[0].content[0].text).toBe("Hello from Claude");
    expect(body.usage.input_tokens).toBe(11);
    expect(body.usage.output_tokens).toBe(7);

    // The gateway spoke the Anthropic Messages protocol upstream.
    const last = nsUpstream.receivedRequests.at(-1);
    expect(last?.path).toBe("/v1/messages");

    const spans = await collectUsageSpans(otlp, requestId!);
    expect(spans).toHaveLength(1);
    expect(spans[0]["gen_ai.usage.input_tokens"]).toBe("11");
    expect(spans[0]["gen_ai.usage.output_tokens"]).toBe("7");
    expect(spans[0]["http.response.status_code"]).toBe("200");
  });

  test("streaming: codex-style streamed call yields Responses SSE events + usage event", async (ctx) => {
    if (!etcdReachable || !app || !streamUpstream || !otlp) {
      ctx.skip();
      return;
    }
    await waitConfigPropagation(async () => {
      try {
        const r = await post(app!, {
          model: "opus-4.7-stream",
          input: "ready",
          stream: true,
        });
        const t = await r.text();
        return r.status === 200 && t.includes("response.completed");
      } catch {
        return false;
      }
    });

    const res = await post(app, {
      model: "opus-4.7-stream",
      input: "hi",
      stream: true,
    });
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id");
    const body = await res.text();
    // Canonical Responses streaming event set, ending in response.completed.
    expect(body).toContain("event: response.created");
    expect(body).toContain("event: response.output_item.added");
    expect(body).toContain("event: response.output_text.delta");
    expect(body).toContain('"delta":"Hi from Claude"');
    expect(body).toContain("event: response.completed");

    const spans = await collectUsageSpans(otlp, requestId!);
    expect(spans).toHaveLength(1);
    expect(spans[0]["gen_ai.usage.input_tokens"]).toBe(String(STREAM_INPUT_TOKENS));
    expect(spans[0]["gen_ai.usage.output_tokens"]).toBe(String(STREAM_OUTPUT_TOKENS));
  });

  test("multi-turn tool loop: function_call history is sent to Anthropic with alternating roles + tool_use/tool_result", async (ctx) => {
    if (!etcdReachable || !app || !nsUpstream) {
      ctx.skip();
      return;
    }
    await waitConfigPropagation(async () => {
      try {
        const r = await post(app!, { model: "opus-4.7", input: "ready" });
        return r.status === 200;
      } catch {
        return false;
      }
    });

    const baseline = nsUpstream.receivedRequests.length;
    const res = await post(app, {
      model: "opus-4.7",
      // The codex agent-loop history shape: a user turn, the assistant's
      // prior tool call, and the tool result fed back.
      input: [
        { role: "user", content: "run ls" },
        {
          type: "function_call",
          call_id: "call_1",
          name: "shell",
          arguments: '{"cmd":"ls"}',
        },
        { type: "function_call_output", call_id: "call_1", output: "a.txt" },
      ],
      tools: [
        {
          type: "function",
          name: "shell",
          description: "run a shell command",
          parameters: {
            type: "object",
            properties: { cmd: { type: "string" } },
            required: ["cmd"],
          },
        },
      ],
    });
    expect(res.status).toBe(200);

    // Inspect what the gateway actually sent to the Anthropic upstream.
    const sent = nsUpstream.receivedRequests
      .slice(baseline)
      .find((r) => r.path === "/v1/messages");
    expect(sent).toBeDefined();
    const anthropicReq = JSON.parse(sent!.body);

    // Roles strictly alternate user → assistant → user.
    const roles = anthropicReq.messages.map((m: { role: string }) => m.role);
    expect(roles).toEqual(["user", "assistant", "user"]);

    // The assistant turn carries the tool_use translated from function_call.
    const toolUse = anthropicReq.messages[1].content.find(
      (b: { type: string }) => b.type === "tool_use",
    );
    expect(toolUse).toBeDefined();
    expect(toolUse.id).toBe("call_1");
    expect(toolUse.name).toBe("shell");
    expect(toolUse.input).toEqual({ cmd: "ls" });

    // The tool result alternates back as a user tool_result block.
    const toolResult = anthropicReq.messages[2].content.find(
      (b: { type: string }) => b.type === "tool_result",
    );
    expect(toolResult).toBeDefined();
    expect(toolResult.tool_use_id).toBe("call_1");

    // Tools were translated to the Anthropic input_schema shape.
    expect(anthropicReq.tools[0].name).toBe("shell");
    expect(anthropicReq.tools[0].input_schema.type).toBe("object");

    // No OpenAI-only Responses knobs leaked onto the Anthropic wire.
    expect(anthropicReq.reasoning).toBeUndefined();
    expect(anthropicReq.store).toBeUndefined();
  });
});
