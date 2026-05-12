import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: Anthropic Messages client → OpenAI-compatible upstream — tool
// translation (#236).
//
// When a caller sends an Anthropic Messages request (`POST /v1/messages`)
// with `tools` and `tool_choice`, and the upstream Model is
// OpenAI-compatible, the gateway must:
//
//   1. Translate Anthropic `tools` → OpenAI `tools` on the way out
//      (`{name, description, input_schema}` → `{type:"function",
//       function:{name, description, parameters}}`).
//   2. Translate Anthropic `tool_choice` → OpenAI `tool_choice`
//      (`{type:"any"}` → `"required"`, etc.).
//   3. Translate OpenAI `tool_calls` in the response back to Anthropic
//      `content: [{type:"tool_use", …}]` on the way back.
//
// Prior to this fix (#236), tools/tool_choice were passed through
// verbatim in Anthropic shape, which OpenAI upstreams silently ignore
// or reject.

const CALLER_PLAINTEXT = "sk-anth-tools-xprov-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("Anthropic Messages client → OpenAI upstream: tools translation (#236)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    // Mock OpenAI upstream returns a tool_calls response.
    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "chatcmpl-tool-01",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o",
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: null,
              tool_calls: [
                {
                  id: "call_abc123",
                  type: "function",
                  function: {
                    name: "get_time",
                    arguments: '{"timezone":"UTC"}',
                  },
                },
              ],
            },
            finish_reason: "tool_calls",
          },
        ],
        usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
      },
    });

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "anth-tools-xprov-pk",
      secret: "sk-openai-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "anth-tools-xprov",
      provider: "openai",
      model_name: "gpt-4o",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["anth-tools-xprov"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("Anthropic tools/tool_choice translate to OpenAI shape on upstream request", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    // Wait for config propagation using a simple probe via /v1/messages
    await waitConfigPropagation(async () => {
      try {
        const res = await fetch(`${app!.proxyUrl}/v1/messages`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-api-key": CALLER_PLAINTEXT,
          },
          body: JSON.stringify({
            model: "anth-tools-xprov",
            max_tokens: 100,
            messages: [{ role: "user", content: "probe" }],
          }),
        });
        return res.ok;
      } catch {
        return false;
      }
    });

    const baseline = upstream.receivedRequests.length;

    // Send Anthropic-shaped request with tools + tool_choice
    const res = await fetch(`${app.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": CALLER_PLAINTEXT,
      },
      body: JSON.stringify({
        model: "anth-tools-xprov",
        max_tokens: 200,
        tools: [
          {
            name: "get_time",
            description: "Get the current time",
            input_schema: {
              type: "object",
              properties: {
                timezone: { type: "string" },
              },
              required: ["timezone"],
            },
          },
        ],
        tool_choice: { type: "any" },
        messages: [
          { role: "user", content: "What time is it? Use get_time." },
        ],
      }),
    });

    expect(res.ok).toBe(true);
    const body = (await res.json()) as {
      type?: string;
      content?: Array<{
        type?: string;
        id?: string;
        name?: string;
        input?: Record<string, unknown>;
      }>;
      stop_reason?: string;
    };

    // Response should be Anthropic-shaped with tool_use content block
    expect(body.type).toBe("message");
    expect(body.stop_reason).toBe("tool_use");
    expect(body.content).toBeDefined();
    const toolBlock = body.content?.find((b) => b.type === "tool_use");
    expect(toolBlock).toBeDefined();
    expect(toolBlock?.id).toBe("call_abc123");
    expect(toolBlock?.name).toBe("get_time");
    expect(toolBlock?.input).toEqual({ timezone: "UTC" });

    // Verify upstream received OpenAI-shaped tools
    const upstreamReq = upstream.receivedRequests
      .slice(baseline)
      .find((r) => r.path === "/v1/chat/completions");
    expect(upstreamReq).toBeDefined();

    const sentBody = JSON.parse(upstreamReq!.body) as {
      tools?: Array<{
        type?: string;
        function?: {
          name?: string;
          description?: string;
          parameters?: { type?: string; required?: string[] };
        };
      }>;
      tool_choice?: string | { type?: string };
    };

    // tools: Anthropic shape must be translated to OpenAI shape
    expect(sentBody.tools).toHaveLength(1);
    expect(sentBody.tools?.[0]?.type).toBe("function");
    expect(sentBody.tools?.[0]?.function?.name).toBe("get_time");
    expect(sentBody.tools?.[0]?.function?.description).toBe(
      "Get the current time",
    );
    expect(sentBody.tools?.[0]?.function?.parameters?.type).toBe("object");
    expect(sentBody.tools?.[0]?.function?.parameters?.required).toEqual([
      "timezone",
    ]);

    // tool_choice: Anthropic {type:"any"} → OpenAI "required"
    expect(sentBody.tool_choice).toBe("required");
  });
});

// ─── Streaming test ─────────────────────────────────────────────────

const STREAM_CALLER_PLAINTEXT = "sk-anth-tools-stream-xprov";
const STREAM_CALLER_KEY_HASH = createHash("sha256")
  .update(STREAM_CALLER_PLAINTEXT)
  .digest("hex");

describe("Anthropic Messages client → OpenAI upstream: streaming tool_calls (#236)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    // Mock OpenAI upstream returns streaming tool_calls chunks.
    const streamEvents = [
      JSON.stringify({
        id: "cmpl-stream-tool",
        object: "chat.completion.chunk",
        model: "gpt-4o",
        choices: [
          {
            index: 0,
            delta: {
              role: "assistant",
              content: null,
              tool_calls: [
                {
                  index: 0,
                  id: "call_stream_1",
                  type: "function",
                  function: { name: "get_time", arguments: "" },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      }),
      JSON.stringify({
        id: "cmpl-stream-tool",
        object: "chat.completion.chunk",
        model: "gpt-4o",
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [
                {
                  index: 0,
                  function: { arguments: '{"timezone"' },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      }),
      JSON.stringify({
        id: "cmpl-stream-tool",
        object: "chat.completion.chunk",
        model: "gpt-4o",
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [
                {
                  index: 0,
                  function: { arguments: ':"UTC"}' },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      }),
      JSON.stringify({
        id: "cmpl-stream-tool",
        object: "chat.completion.chunk",
        model: "gpt-4o",
        choices: [
          {
            index: 0,
            delta: {},
            finish_reason: "tool_calls",
          },
        ],
        usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
      }),
      "[DONE]",
    ];

    upstream = await startOpenAiUpstream({ streamEvents });

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "anth-tools-stream-xprov-pk",
      secret: "sk-openai-stream-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "anth-tools-stream-xprov",
      provider: "openai",
      model_name: "gpt-4o",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: STREAM_CALLER_KEY_HASH,
      allowed_models: ["anth-tools-stream-xprov"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("Streaming tool_calls are translated to Anthropic SSE tool_use events", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    await waitConfigPropagation(async () => {
      try {
        const res = await fetch(`${app!.proxyUrl}/v1/messages`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-api-key": STREAM_CALLER_PLAINTEXT,
          },
          body: JSON.stringify({
            model: "anth-tools-stream-xprov",
            max_tokens: 100,
            stream: true,
            messages: [{ role: "user", content: "probe" }],
          }),
        });
        return res.ok;
      } catch {
        return false;
      }
    });

    const baseline = upstream.receivedRequests.length;

    const res = await fetch(`${app.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": STREAM_CALLER_PLAINTEXT,
      },
      body: JSON.stringify({
        model: "anth-tools-stream-xprov",
        max_tokens: 200,
        stream: true,
        tools: [
          {
            name: "get_time",
            description: "Get the current time",
            input_schema: {
              type: "object",
              properties: { timezone: { type: "string" } },
              required: ["timezone"],
            },
          },
        ],
        tool_choice: { type: "any" },
        messages: [
          { role: "user", content: "What time is it?" },
        ],
      }),
    });

    expect(res.ok).toBe(true);
    expect(res.headers.get("content-type")).toContain("text/event-stream");

    const text = await res.text();
    const lines = text.split("\n").filter((l) => l.startsWith("data: "));
    const events = lines.map((l) => JSON.parse(l.slice(6)) as Record<string, unknown>);

    // Should contain message_start, content_block_start (tool_use),
    // content_block_delta (input_json_delta), content_block_stop,
    // message_delta, message_stop
    const types = events.map((e) => e.type);
    expect(types).toContain("message_start");
    expect(types).toContain("message_stop");

    // Find the tool_use content_block_start
    const toolStart = events.find(
      (e) =>
        e.type === "content_block_start" &&
        (e.content_block as Record<string, unknown>)?.type === "tool_use",
    );
    expect(toolStart).toBeDefined();
    const cb = toolStart!.content_block as Record<string, unknown>;
    expect(cb.id).toBe("call_stream_1");
    expect(cb.name).toBe("get_time");

    // Find input_json_delta events
    const jsonDeltas = events.filter(
      (e) =>
        e.type === "content_block_delta" &&
        (e.delta as Record<string, unknown>)?.type === "input_json_delta",
    );
    expect(jsonDeltas.length).toBeGreaterThanOrEqual(1);
    const fullJson = jsonDeltas
      .map((d) => (d.delta as Record<string, unknown>).partial_json as string)
      .join("");
    expect(fullJson).toBe('{"timezone":"UTC"}');

    // Find message_delta with stop_reason tool_use
    const msgDelta = events.find(
      (e) =>
        e.type === "message_delta" &&
        (e.delta as Record<string, unknown>)?.stop_reason === "tool_use",
    );
    expect(msgDelta).toBeDefined();

    // Verify upstream received OpenAI-shaped tools
    const upstreamReq = upstream.receivedRequests
      .slice(baseline)
      .find((r) => r.path === "/v1/chat/completions");
    expect(upstreamReq).toBeDefined();
    const sentBody = JSON.parse(upstreamReq!.body) as {
      tools?: Array<{ type?: string; function?: { name?: string } }>;
      tool_choice?: string;
      stream?: boolean;
    };
    expect(sentBody.tools?.[0]?.type).toBe("function");
    expect(sentBody.tools?.[0]?.function?.name).toBe("get_time");
    expect(sentBody.tool_choice).toBe("required");
    expect(sentBody.stream).toBe(true);
  });
});
