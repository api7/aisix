import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

const API_KEY = "sk-effort-mapping";
const KEY_HASH = createHash("sha256").update(API_KEY).digest("hex");

const chatResponse = {
  id: "chatcmpl-effort-map",
  object: "chat.completion",
  created: 1,
  model: "upstream-model",
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: "ok" },
      finish_reason: "stop",
    },
  ],
  usage: { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 },
};

const anthropicResponse = {
  id: "msg_effort_map",
  type: "message",
  role: "assistant",
  content: [{ type: "text", text: "ok" }],
  model: "upstream-model",
  stop_reason: "end_turn",
  usage: { input_tokens: 2, output_tokens: 1 },
};

const openaiStreamEvents = [
  JSON.stringify({
    id: "chatcmpl-effort-stream",
    object: "chat.completion.chunk",
    model: "upstream-model",
    choices: [
      {
        index: 0,
        delta: { role: "assistant", content: "ok" },
        finish_reason: null,
      },
    ],
  }),
  JSON.stringify({
    id: "chatcmpl-effort-stream",
    object: "chat.completion.chunk",
    model: "upstream-model",
    choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
  }),
  "[DONE]",
];

const anthropicStreamEvents = [
  JSON.stringify({
    type: "message_start",
    message: {
      id: "msg_effort_stream",
      role: "assistant",
      content: [],
      model: "upstream-model",
      stop_reason: null,
      usage: { input_tokens: 2, output_tokens: 1 },
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
    delta: { type: "text_delta", text: "ok" },
  }),
  JSON.stringify({ type: "content_block_stop", index: 0 }),
  JSON.stringify({
    type: "message_delta",
    delta: { stop_reason: "end_turn" },
    usage: { output_tokens: 1 },
  }),
  JSON.stringify({ type: "message_stop" }),
];

describe("direct-model effort mapping", () => {
  let app: SpawnedApp | undefined;
  let openai: OpenAiUpstream | undefined;
  let anthropic: OpenAiUpstream | undefined;
  let openaiStream: OpenAiUpstream | undefined;
  let anthropicStream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    openai = await startOpenAiUpstream({ nonStreamBody: chatResponse });
    anthropic = await startOpenAiUpstream({
      scriptedResponses: [
        { nonStreamBody: anthropicResponse },
        { nonStreamBody: anthropicResponse },
        { nonStreamBody: anthropicResponse },
        { nonStreamBody: { input_tokens: 3 } },
      ],
    });
    openaiStream = await startOpenAiUpstream({
      streamEvents: openaiStreamEvents,
    });
    anthropicStream = await startOpenAiUpstream({
      streamEvents: anthropicStreamEvents,
    });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const openaiKey = await seed.createProviderKey({
      display_name: "effort-map-openai-key",
      secret: "sk-openai-mock",
      api_base: `${openai.baseUrl}/v1`,
    });
    const anthropicKey = await seed.createProviderKey({
      display_name: "effort-map-anthropic-key",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-anthropic-mock",
      api_base: anthropic.baseUrl,
    });
    const openaiStreamKey = await seed.createProviderKey({
      display_name: "effort-map-openai-stream-key",
      secret: "sk-openai-stream-mock",
      api_base: `${openaiStream.baseUrl}/v1`,
    });
    const anthropicStreamKey = await seed.createProviderKey({
      display_name: "effort-map-anthropic-stream-key",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-anthropic-stream-mock",
      api_base: anthropicStream.baseUrl,
    });

    const mapping = { medium: "high", high: "max" };
    await seed.createModel({
      display_name: "effort-map-openai",
      provider: "openai",
      model_name: "glm-openai-wire",
      provider_key_id: openaiKey.id,
      effort_mapping: mapping,
    });
    await seed.createModel({
      display_name: "effort-map-openai-stream",
      provider: "openai",
      model_name: "glm-openai-stream-wire",
      provider_key_id: openaiStreamKey.id,
      effort_mapping: mapping,
    });
    await seed.createModel({
      display_name: "effort-map-anthropic-stream",
      provider: "anthropic",
      model_name: "glm-anthropic-stream-wire",
      provider_key_id: anthropicStreamKey.id,
      effort_mapping: mapping,
    });
    await seed.createModel({
      display_name: "effort-map-anthropic",
      provider: "anthropic",
      model_name: "glm-anthropic-wire",
      provider_key_id: anthropicKey.id,
      effort_mapping: mapping,
    });
    await seed.createModel({
      display_name: "effort-map-group",
      routing: {
        strategy: "failover",
        targets: [{ model: "effort-map-openai" }],
      },
    });

    // The caller key is seeded last, so successful authentication implies
    // every model and provider key above has reached the snapshot.
    await seed.createApiKey({
      key_hash: KEY_HASH,
      allowed_models: [
        "effort-map-openai",
        "effort-map-anthropic",
        "effort-map-openai-stream",
        "effort-map-anthropic-stream",
        "effort-map-group",
      ],
    });
    const proxy = new ProxyClient(app.proxyUrl, API_KEY);
    await waitConfigPropagation(
      async () => (await proxy.listModels()).status === 200,
    );
  });

  afterAll(async () => {
    await app?.exit();
    await openai?.close();
    await anthropic?.close();
    await openaiStream?.close();
    await anthropicStream?.close();
  });

  async function post(
    path: string,
    body: Record<string, unknown>,
    auth: "openai" | "anthropic" = "openai",
  ): Promise<{ body: string; contentType: string | null }> {
    const response = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        ...(auth === "openai"
          ? { authorization: `Bearer ${API_KEY}` }
          : { "x-api-key": API_KEY }),
      },
      body: JSON.stringify(body),
    });
    const responseBody = await response.text();
    expect(response.ok, responseBody).toBe(true);
    return {
      body: responseBody,
      contentType: response.headers.get("content-type"),
    };
  }

  function receivedSince(
    upstream: OpenAiUpstream,
    baseline: number,
    path: string,
  ): Record<string, unknown> {
    const request = upstream.receivedRequests
      .slice(baseline)
      .find((candidate) => candidate.path === path);
    expect(request).toBeDefined();
    return JSON.parse(request!.body) as Record<string, unknown>;
  }

  test("maps every supported request shape on native and translated paths", async (ctx) => {
    if (!etcdReachable || !app || !openai || !anthropic) {
      ctx.skip();
      return;
    }

    let baseline = openai.receivedRequests.length;
    await post("/v1/chat/completions", {
      model: "effort-map-openai",
      messages: [{ role: "user", content: "hello" }],
      reasoning_effort: "medium",
    });
    expect(
      receivedSince(openai, baseline, "/v1/chat/completions")
        .reasoning_effort,
    ).toBe("high");

    baseline = openai.receivedRequests.length;
    await post(
      "/v1/messages",
      {
        model: "effort-map-openai",
        max_tokens: 64,
        messages: [{ role: "user", content: "hello" }],
        output_config: { effort: "medium" },
      },
      "anthropic",
    );
    expect(
      receivedSince(openai, baseline, "/v1/chat/completions")
        .reasoning_effort,
    ).toBe("high");

    baseline = openai.receivedRequests.length;
    await post("/v1/responses", {
      model: "effort-map-openai",
      input: "hello",
      reasoning: { effort: "medium", summary: "auto" },
    });
    expect(
      receivedSince(openai, baseline, "/v1/responses").reasoning,
    ).toEqual({ effort: "high", summary: "auto" });

    baseline = anthropic.receivedRequests.length;
    await post(
      "/v1/messages",
      {
        model: "effort-map-anthropic",
        max_tokens: 64,
        messages: [{ role: "user", content: "hello" }],
        output_config: { effort: "medium" },
      },
      "anthropic",
    );
    expect(
      receivedSince(anthropic, baseline, "/v1/messages").output_config,
    ).toEqual({ effort: "high" });

    baseline = anthropic.receivedRequests.length;
    await post("/v1/chat/completions", {
      model: "effort-map-anthropic",
      messages: [{ role: "user", content: "hello" }],
      reasoning_effort: "medium",
    });
    expect(
      receivedSince(anthropic, baseline, "/v1/messages").output_config,
    ).toEqual({ effort: "high" });

    baseline = anthropic.receivedRequests.length;
    await post("/v1/responses", {
      model: "effort-map-anthropic",
      input: "hello",
      reasoning: { effort: "medium" },
    });
    expect(
      receivedSince(anthropic, baseline, "/v1/messages").output_config,
    ).toEqual({ effort: "high" });

    baseline = anthropic.receivedRequests.length;
    await post(
      "/v1/messages/count_tokens",
      {
        model: "effort-map-anthropic",
        messages: [{ role: "user", content: "hello" }],
        output_config: { effort: "medium" },
      },
      "anthropic",
    );
    expect(
      receivedSince(
        anthropic,
        baseline,
        "/v1/messages/count_tokens",
      ).output_config,
    ).toEqual({ effort: "high" });
  });

  test("uses the dispatched target map once and passes unlisted values through", async (ctx) => {
    if (!etcdReachable || !app || !openai) {
      ctx.skip();
      return;
    }

    let baseline = openai.receivedRequests.length;
    await post("/v1/chat/completions", {
      model: "effort-map-group",
      messages: [{ role: "user", content: "hello" }],
      reasoning_effort: "medium",
    });
    expect(
      receivedSince(openai, baseline, "/v1/chat/completions")
        .reasoning_effort,
    ).toBe("high");

    baseline = openai.receivedRequests.length;
    await post("/v1/chat/completions", {
      model: "effort-map-openai",
      messages: [{ role: "user", content: "hello" }],
      reasoning_effort: "low",
    });
    expect(
      receivedSince(openai, baseline, "/v1/chat/completions")
        .reasoning_effort,
    ).toBe("low");
  });

  test("maps native and translated streaming requests", async (ctx) => {
    if (!etcdReachable || !app || !openaiStream || !anthropicStream) {
      ctx.skip();
      return;
    }

    let baseline = openaiStream.receivedRequests.length;
    const openaiResponse = await post("/v1/chat/completions", {
      model: "effort-map-openai-stream",
      messages: [{ role: "user", content: "hello" }],
      reasoning_effort: "high",
      stream: true,
    });
    expect(openaiResponse.contentType).toContain("text/event-stream");
    expect(openaiResponse.body).toContain("data:");
    expect(
      receivedSince(openaiStream, baseline, "/v1/chat/completions")
        .reasoning_effort,
    ).toBe("max");

    baseline = anthropicStream.receivedRequests.length;
    const translatedResponse = await post("/v1/chat/completions", {
      model: "effort-map-anthropic-stream",
      messages: [{ role: "user", content: "hello" }],
      reasoning_effort: "medium",
      stream: true,
    });
    expect(translatedResponse.contentType).toContain("text/event-stream");
    expect(translatedResponse.body).toContain("data:");
    expect(
      receivedSince(anthropicStream, baseline, "/v1/messages")
        .output_config,
    ).toEqual({ effort: "high" });
  });
});
