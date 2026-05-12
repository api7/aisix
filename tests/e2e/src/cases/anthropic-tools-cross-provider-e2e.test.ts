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
