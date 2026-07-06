import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  spawnApp,
  startOpenAiUpstream,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: /v1/messages cross-provider content-block translation (#722,
// AISIX-Cloud#873 §⑤ "Anthropic Messages 跨厂商 仅文本块").
//
// Pre-#722 the inbound parse dropped every non-text block silently, so a
// Claude-Code-style multi-turn tool loop pointed at a non-Anthropic
// Model lost its tool history (the upstream saw empty user turns), and
// vision inputs vanished. These cases pin the translated UPSTREAM wire
// (what the OpenAI-compatible provider actually receives) for the two
// journeys that matter:
//
//   1. Multi-turn tool loop: assistant `tool_use` history →
//      `tool_calls[]`; user `tool_result` → a `role:"tool"` message
//      that directly follows the assistant tool_calls turn (OpenAI
//      ordering requirement); trailing user text survives.
//   2. Vision: user `image` (base64) block → `image_url` data-URL part
//      alongside the text part.

const CALLER_PLAINTEXT = "sk-blocks-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("anthropic content blocks → OpenAI upstream (#722)", () => {
  let app: SpawnedApp | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  async function modelBackedBy(name: string) {
    const upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "chatcmpl-blocks",
        object: "chat.completion",
        model: "gpt-4o",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "done" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 },
      },
    });
    upstreams.push(upstream);
    const pk = await admin!.createProviderKey({
      display_name: `${name}-pk`,
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin!.createModel({
      display_name: name,
      provider: "openai",
      model_name: "gpt-4o",
      provider_key_id: pk.id,
    });
    return upstream;
  }

  test("multi-turn tool loop history reaches the OpenAI upstream intact", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }
    const upstream = await modelBackedBy("blocks-tools-model");

    const resp = await fetch(`${app.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": CALLER_PLAINTEXT,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: "blocks-tools-model",
        max_tokens: 128,
        tools: [
          {
            name: "get_weather",
            description: "weather lookup",
            input_schema: {
              type: "object",
              properties: { city: { type: "string" } },
            },
          },
        ],
        messages: [
          { role: "user", content: "weather in SF?" },
          {
            role: "assistant",
            content: [
              { type: "text", text: "let me check" },
              {
                type: "tool_use",
                id: "toolu_01",
                name: "get_weather",
                input: { city: "SF" },
              },
            ],
          },
          {
            role: "user",
            content: [
              {
                type: "tool_result",
                tool_use_id: "toolu_01",
                content: "sunny, 21C",
              },
              { type: "text", text: "summarize please" },
            ],
          },
        ],
      }),
    });
    expect(resp.status).toBe(200);

    const seen = upstream.receivedRequests.find((r) =>
      r.path.includes("/chat/completions"),
    );
    expect(seen).toBeDefined();
    const sent = JSON.parse(seen!.body) as {
      messages: {
        role: string;
        content: unknown;
        tool_calls?: { id: string; function: { name: string; arguments: string } }[];
        tool_call_id?: string;
      }[];
      tools?: { type: string; function: { name: string } }[];
    };

    // Tool definitions translate (pre-existing #236 behavior still holds).
    expect(sent.tools?.[0]?.type).toBe("function");
    expect(sent.tools?.[0]?.function.name).toBe("get_weather");

    // The conversation structure: user → assistant(tool_calls) →
    // tool → user. Pre-#722 the assistant turn lost its tool_calls and
    // the tool_result turn collapsed to an empty user message.
    const roles = sent.messages.map((m) => m.role);
    expect(roles).toEqual(["user", "assistant", "tool", "user"]);

    const assistant = sent.messages[1];
    expect(assistant.tool_calls?.length).toBe(1);
    expect(assistant.tool_calls?.[0].id).toBe("toolu_01");
    expect(assistant.tool_calls?.[0].function.name).toBe("get_weather");
    expect(JSON.parse(assistant.tool_calls![0].function.arguments)).toEqual({
      city: "SF",
    });

    const toolMsg = sent.messages[2];
    expect(toolMsg.tool_call_id).toBe("toolu_01");
    expect(toolMsg.content).toBe("sunny, 21C");

    expect(sent.messages[3].content).toContain("summarize please");
  });

  test("vision: base64 image block reaches the upstream as an image_url data URL", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }
    const upstream = await modelBackedBy("blocks-vision-model");

    const resp = await fetch(`${app.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": CALLER_PLAINTEXT,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: "blocks-vision-model",
        max_tokens: 64,
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "what is in this image?" },
              {
                type: "image",
                source: {
                  type: "base64",
                  media_type: "image/png",
                  data: "aGVsbG8=",
                },
              },
            ],
          },
        ],
      }),
    });
    expect(resp.status).toBe(200);

    const seen = upstream.receivedRequests.find((r) =>
      r.path.includes("/chat/completions"),
    );
    expect(seen).toBeDefined();
    const sent = JSON.parse(seen!.body) as {
      messages: { role: string; content: unknown }[];
    };
    const content = sent.messages[0].content as {
      type: string;
      text?: string;
      image_url?: { url: string };
    }[];
    expect(Array.isArray(content)).toBe(true);
    expect(content[0]).toEqual({ type: "text", text: "what is in this image?" });
    expect(content[1].type).toBe("image_url");
    expect(content[1].image_url?.url).toBe("data:image/png;base64,aGVsbG8=");
  });
});
