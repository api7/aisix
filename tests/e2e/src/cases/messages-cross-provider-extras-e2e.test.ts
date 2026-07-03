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

// E2E: Anthropic Messages client → OpenAI-compatible upstream —
// Anthropic-only extras must not leak upstream (AISIX-Cloud#953).
//
// The cross-provider path preserves unknown Anthropic top-level fields
// in ChatFormat.extra, and the OpenAI bridge flattens extras into the
// upstream request body. Anthropic-only fields (`context_management`,
// `thinking`, `top_k`, …) reached the OpenAI/Azure upstream verbatim,
// which rejects them: 400 "Unknown parameter: 'context_management'".
//
// The gateway must translate what maps cleanly onto the OpenAI chat
// shape and drop the rest — never forward Anthropic-only fields.

// Every Anthropic-only /v1/messages field with no OpenAI equivalent,
// including newer SDK fields like context_management (the #953 trigger).
const ANTHROPIC_ONLY_FIELDS: Record<string, unknown> = {
  context_management: {
    edits: [{ type: "clear_tool_uses_20250919" }],
  },
  thinking: { type: "enabled", budget_tokens: 2048 },
  top_k: 40,
  mcp_servers: [
    { type: "url", url: "https://example.com/mcp", name: "example" },
  ],
  container: "container_abc",
  service_tier: "standard_only",
  betas: ["context-management-2025-06-27"],
  anthropic_version: "2023-06-01",
};

function expectNoAnthropicOnlyFields(sentBody: Record<string, unknown>) {
  for (const key of Object.keys(ANTHROPIC_ONLY_FIELDS)) {
    expect(sentBody[key], `\`${key}\` leaked to upstream`).toBeUndefined();
  }
}

const CALLER_PLAINTEXT = "sk-anth-extras-xprov-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("Anthropic Messages → OpenAI upstream: Anthropic-only extras are not forwarded (#953)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "chatcmpl-extras-01",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "hello" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
      },
    });

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "anth-extras-xprov-pk",
      secret: "sk-openai-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "anth-extras-xprov",
      provider: "openai",
      model_name: "gpt-4o",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["anth-extras-xprov"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("non-streaming: Anthropic-only fields are dropped, translatable fields survive", async (ctx) => {
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
            "x-api-key": CALLER_PLAINTEXT,
          },
          body: JSON.stringify({
            model: "anth-extras-xprov",
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

    const res = await fetch(`${app.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": CALLER_PLAINTEXT,
      },
      body: JSON.stringify({
        model: "anth-extras-xprov",
        max_tokens: 200,
        temperature: 0.5,
        metadata: { user_id: "user-123" },
        stop_sequences: ["\\n\\nHuman:"],
        tools: [
          {
            name: "get_time",
            description: "Get the current time",
            input_schema: { type: "object", properties: {} },
          },
        ],
        tool_choice: { type: "auto" },
        messages: [{ role: "user", content: "hello" }],
        ...ANTHROPIC_ONLY_FIELDS,
      }),
    });
    expect(res.ok).toBe(true);

    const upstreamReq = upstream.receivedRequests
      .slice(baseline)
      .find((r) => r.path === "/v1/chat/completions");
    expect(upstreamReq).toBeDefined();
    const sentBody = JSON.parse(upstreamReq!.body) as Record<string, unknown>;

    // The #953 bug: none of these may reach an OpenAI-compatible upstream.
    expectNoAnthropicOnlyFields(sentBody);
    expect(sentBody.metadata, "`metadata` leaked to upstream").toBeUndefined();
    expect(
      sentBody.stop_sequences,
      "`stop_sequences` leaked to upstream",
    ).toBeUndefined();

    // Translatable fields must still flow.
    expect(sentBody.max_tokens).toBe(200);
    expect(sentBody.temperature).toBe(0.5);
    expect(sentBody.stop).toEqual(["\\n\\nHuman:"]);
    expect(sentBody.user).toBe("user-123");
    const tools = sentBody.tools as Array<{
      type?: string;
      function?: { name?: string };
    }>;
    expect(tools).toHaveLength(1);
    expect(tools[0]?.type).toBe("function");
    expect(tools[0]?.function?.name).toBe("get_time");
    expect(sentBody.tool_choice).toBe("auto");
  });
});

// ─── Streaming variant ──────────────────────────────────────────────

const STREAM_CALLER_PLAINTEXT = "sk-anth-extras-stream-xprov";
const STREAM_CALLER_KEY_HASH = createHash("sha256")
  .update(STREAM_CALLER_PLAINTEXT)
  .digest("hex");

describe("Anthropic Messages → OpenAI upstream: streaming extras are not forwarded (#953)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      streamEvents: [
        JSON.stringify({
          id: "cmpl-extras-stream",
          object: "chat.completion.chunk",
          model: "gpt-4o",
          choices: [
            {
              index: 0,
              delta: { role: "assistant", content: "hi" },
              finish_reason: null,
            },
          ],
        }),
        JSON.stringify({
          id: "cmpl-extras-stream",
          object: "chat.completion.chunk",
          model: "gpt-4o",
          choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
          usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
        }),
        "[DONE]",
      ],
    });

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "anth-extras-stream-xprov-pk",
      secret: "sk-openai-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "anth-extras-stream-xprov",
      provider: "openai",
      model_name: "gpt-4o",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: STREAM_CALLER_KEY_HASH,
      allowed_models: ["anth-extras-stream-xprov"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("streaming: Anthropic-only fields are dropped", async (ctx) => {
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
            model: "anth-extras-stream-xprov",
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
        model: "anth-extras-stream-xprov",
        max_tokens: 200,
        stream: true,
        messages: [{ role: "user", content: "hello" }],
        ...ANTHROPIC_ONLY_FIELDS,
      }),
    });
    expect(res.ok).toBe(true);
    expect(res.headers.get("content-type")).toContain("text/event-stream");
    await res.text();

    const upstreamReq = upstream.receivedRequests
      .slice(baseline)
      .find((r) => r.path === "/v1/chat/completions");
    expect(upstreamReq).toBeDefined();
    const sentBody = JSON.parse(upstreamReq!.body) as Record<string, unknown>;

    expectNoAnthropicOnlyFields(sentBody);
    expect(sentBody.stream).toBe(true);
  });
});
