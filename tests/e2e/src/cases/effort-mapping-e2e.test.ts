import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
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

const execFileP = promisify(execFile);

// The three ways a request sets no reasoning effort at all. Every one of
// them takes the mapping's `""` entry.
const NO_EFFORT_CARRIERS: Record<string, unknown>[] = [
  {},
  { reasoning_effort: null },
  { reasoning_effort: "" },
];

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
  let ensembleStream: OpenAiUpstream | undefined;
  let tokensOpenai: OpenAiUpstream | undefined;
  let tokensAnthropic: OpenAiUpstream | undefined;
  let tokensCount: OpenAiUpstream | undefined;
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
    ensembleStream = await startOpenAiUpstream({
      scriptedResponses: [
        { nonStreamBody: chatResponse },
        { streamEvents: openaiStreamEvents },
      ],
    });
    // The reserved-token models get upstreams of their own: the ones above
    // answer from a fixed script, so an extra request would run off its end.
    tokensOpenai = await startOpenAiUpstream({ nonStreamBody: chatResponse });
    tokensAnthropic = await startOpenAiUpstream({
      nonStreamBody: anthropicResponse,
    });
    tokensCount = await startOpenAiUpstream({
      nonStreamBody: { input_tokens: 3 },
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
    const ensembleStreamKey = await seed.createProviderKey({
      display_name: "effort-map-ensemble-stream-key",
      secret: "sk-ensemble-stream-mock",
      api_base: `${ensembleStream.baseUrl}/v1`,
    });
    const tokensOpenaiKey = await seed.createProviderKey({
      display_name: "effort-map-tokens-openai-key",
      secret: "sk-tokens-openai-mock",
      api_base: `${tokensOpenai.baseUrl}/v1`,
    });
    const tokensAnthropicKey = await seed.createProviderKey({
      display_name: "effort-map-tokens-anthropic-key",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-tokens-anthropic-mock",
      api_base: tokensAnthropic.baseUrl,
    });
    const tokensCountKey = await seed.createProviderKey({
      display_name: "effort-map-tokens-count-key",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-tokens-count-mock",
      api_base: tokensCount.baseUrl,
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
      display_name: "effort-map-ensemble-panel",
      provider: "openai",
      model_name: "glm-ensemble-panel-wire",
      provider_key_id: openaiKey.id,
      effort_mapping: mapping,
    });
    await seed.createModel({
      display_name: "effort-map-ensemble-judge",
      provider: "openai",
      model_name: "glm-ensemble-judge-wire",
      provider_key_id: openaiKey.id,
      effort_mapping: { medium: "max" },
    });
    await seed.createModel({
      display_name: "effort-map-ensemble",
      ensemble: {
        panel: [{ model: "effort-map-ensemble-panel" }],
        judge: { model: "effort-map-ensemble-judge" },
        min_responses: 1,
      },
    });
    await seed.createModel({
      display_name: "effort-map-ensemble-stream-panel",
      provider: "openai",
      model_name: "glm-ensemble-stream-panel-wire",
      provider_key_id: ensembleStreamKey.id,
      effort_mapping: mapping,
    });
    await seed.createModel({
      display_name: "effort-map-ensemble-stream-judge",
      provider: "openai",
      model_name: "glm-ensemble-stream-judge-wire",
      provider_key_id: ensembleStreamKey.id,
      effort_mapping: { medium: "max" },
    });
    await seed.createModel({
      display_name: "effort-map-ensemble-stream",
      ensemble: {
        panel: [{ model: "effort-map-ensemble-stream-panel" }],
        judge: { model: "effort-map-ensemble-stream-judge" },
        min_responses: 1,
      },
    });
    // One map carrying all three reserved entries: add `high` when the
    // request sets no effort, drop the field for `medium`, and catch every
    // other value with `*`.
    const tokens = { "": "high", medium: null, "*": "low" };
    await seed.createModel({
      display_name: "effort-tokens-openai",
      provider: "openai",
      model_name: "glm-tokens-openai-wire",
      provider_key_id: tokensOpenaiKey.id,
      effort_mapping: tokens,
    });
    await seed.createModel({
      display_name: "effort-tokens-star",
      provider: "openai",
      model_name: "glm-tokens-star-wire",
      provider_key_id: tokensOpenaiKey.id,
      effort_mapping: { "*": "low" },
    });
    await seed.createModel({
      display_name: "effort-tokens-anthropic",
      provider: "anthropic",
      model_name: "glm-tokens-anthropic-wire",
      provider_key_id: tokensAnthropicKey.id,
      effort_mapping: tokens,
    });
    await seed.createModel({
      display_name: "effort-tokens-count",
      provider: "anthropic",
      model_name: "glm-tokens-count-wire",
      provider_key_id: tokensCountKey.id,
      effort_mapping: tokens,
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
        "effort-map-ensemble",
        "effort-map-ensemble-stream",
        "effort-map-group",
        "effort-tokens-openai",
        "effort-tokens-star",
        "effort-tokens-anthropic",
        "effort-tokens-count",
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
    await ensembleStream?.close();
    await tokensOpenai?.close();
    await tokensAnthropic?.close();
    await tokensCount?.close();
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

  function chatRequestsSince(
    upstream: OpenAiUpstream,
    baseline: number,
  ): Record<string, unknown>[] {
    return upstream.receivedRequests
      .slice(baseline)
      .filter((request) => request.path === "/v1/chat/completions")
      .map((request) => JSON.parse(request.body) as Record<string, unknown>);
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

    baseline = openai.receivedRequests.length;
    await post("/v1/chat/completions", {
      model: "effort-map-ensemble",
      messages: [{ role: "user", content: "hello" }],
      reasoning_effort: "medium",
    });
    const ensembleRequests = chatRequestsSince(openai, baseline);
    expect(
      ensembleRequests.find(
        (request) => request.model === "glm-ensemble-panel-wire",
      )?.reasoning_effort,
    ).toBe("high");
    expect(
      ensembleRequests.find(
        (request) => request.model === "glm-ensemble-judge-wire",
      )?.reasoning_effort,
    ).toBe("max");
  });

  test("adds the configured effort when the request sets none", async (ctx) => {
    if (!etcdReachable || !app || !tokensOpenai || !tokensAnthropic || !tokensCount) {
      ctx.skip();
      return;
    }

    // A request sets no effort three ways, and all three take the same
    // entry: the field is absent, it is null, or it is empty.
    for (const carrier of NO_EFFORT_CARRIERS) {
      const baseline = tokensOpenai.receivedRequests.length;
      await post("/v1/chat/completions", {
        model: "effort-tokens-openai",
        messages: [{ role: "user", content: "hello" }],
        ...carrier,
      });
      expect(
        receivedSince(tokensOpenai, baseline, "/v1/chat/completions")
          .reasoning_effort,
        JSON.stringify(carrier),
      ).toBe("high");
    }

    for (const carrier of [
      {},
      { reasoning: null },
      { reasoning: { effort: null } },
      { reasoning: { effort: "" } },
    ]) {
      const baseline = tokensOpenai.receivedRequests.length;
      await post("/v1/responses", {
        model: "effort-tokens-openai",
        input: "hello",
        ...carrier,
      });
      expect(
        receivedSince(tokensOpenai, baseline, "/v1/responses").reasoning,
        JSON.stringify(carrier),
      ).toEqual({ effort: "high" });
    }

    // An existing parent object keeps the keys the caller put beside the
    // effort.
    let baseline = tokensOpenai.receivedRequests.length;
    await post("/v1/responses", {
      model: "effort-tokens-openai",
      input: "hello",
      reasoning: { summary: "auto" },
    });
    expect(
      receivedSince(tokensOpenai, baseline, "/v1/responses").reasoning,
    ).toEqual({ effort: "high", summary: "auto" });

    baseline = tokensAnthropic.receivedRequests.length;
    await post(
      "/v1/messages",
      {
        model: "effort-tokens-anthropic",
        max_tokens: 64,
        messages: [{ role: "user", content: "hello" }],
      },
      "anthropic",
    );
    expect(
      receivedSince(tokensAnthropic, baseline, "/v1/messages").output_config,
    ).toEqual({ effort: "high" });

    // Same model reached through the Responses → Chat bridge.
    baseline = tokensAnthropic.receivedRequests.length;
    await post("/v1/responses", {
      model: "effort-tokens-anthropic",
      input: "hello",
    });
    expect(
      receivedSince(tokensAnthropic, baseline, "/v1/messages").output_config,
    ).toEqual({ effort: "high" });

    baseline = tokensCount.receivedRequests.length;
    await post(
      "/v1/messages/count_tokens",
      {
        model: "effort-tokens-count",
        messages: [{ role: "user", content: "hello" }],
      },
      "anthropic",
    );
    expect(
      receivedSince(tokensCount, baseline, "/v1/messages/count_tokens")
        .output_config,
    ).toEqual({ effort: "high" });
  });

  test("removes the effort field when the entry maps to null", async (ctx) => {
    if (!etcdReachable || !app || !tokensOpenai || !tokensAnthropic || !tokensCount) {
      ctx.skip();
      return;
    }

    let baseline = tokensOpenai.receivedRequests.length;
    await post("/v1/chat/completions", {
      model: "effort-tokens-openai",
      messages: [{ role: "user", content: "hello" }],
      reasoning_effort: "medium",
    });
    expect(
      receivedSince(tokensOpenai, baseline, "/v1/chat/completions"),
    ).not.toHaveProperty("reasoning_effort");

    // Removing the leaf empties `reasoning`, so the parent goes too.
    baseline = tokensOpenai.receivedRequests.length;
    await post("/v1/responses", {
      model: "effort-tokens-openai",
      input: "hello",
      reasoning: { effort: "medium" },
    });
    expect(
      receivedSince(tokensOpenai, baseline, "/v1/responses"),
    ).not.toHaveProperty("reasoning");

    // A sibling key keeps the parent alive and is left as the caller
    // wrote it.
    baseline = tokensOpenai.receivedRequests.length;
    await post("/v1/responses", {
      model: "effort-tokens-openai",
      input: "hello",
      reasoning: { effort: "medium", summary: "auto" },
    });
    expect(
      receivedSince(tokensOpenai, baseline, "/v1/responses").reasoning,
    ).toEqual({ summary: "auto" });

    baseline = tokensAnthropic.receivedRequests.length;
    await post(
      "/v1/messages",
      {
        model: "effort-tokens-anthropic",
        max_tokens: 64,
        messages: [{ role: "user", content: "hello" }],
        output_config: { effort: "medium" },
      },
      "anthropic",
    );
    expect(
      receivedSince(tokensAnthropic, baseline, "/v1/messages"),
    ).not.toHaveProperty("output_config");

    baseline = tokensAnthropic.receivedRequests.length;
    await post("/v1/responses", {
      model: "effort-tokens-anthropic",
      input: "hello",
      reasoning: { effort: "medium" },
    });
    expect(
      receivedSince(tokensAnthropic, baseline, "/v1/messages"),
    ).not.toHaveProperty("output_config");

    baseline = tokensCount.receivedRequests.length;
    await post(
      "/v1/messages/count_tokens",
      {
        model: "effort-tokens-count",
        messages: [{ role: "user", content: "hello" }],
        output_config: { effort: "medium" },
      },
      "anthropic",
    );
    expect(
      receivedSince(tokensCount, baseline, "/v1/messages/count_tokens"),
    ).not.toHaveProperty("output_config");
  });

  test("falls back to the wildcard only for a present unlisted value", async (ctx) => {
    if (!etcdReachable || !app || !tokensOpenai || !tokensAnthropic || !tokensCount) {
      ctx.skip();
      return;
    }

    let baseline = tokensOpenai.receivedRequests.length;
    await post("/v1/chat/completions", {
      model: "effort-tokens-openai",
      messages: [{ role: "user", content: "hello" }],
      reasoning_effort: "xl",
    });
    expect(
      receivedSince(tokensOpenai, baseline, "/v1/chat/completions")
        .reasoning_effort,
    ).toBe("low");

    baseline = tokensOpenai.receivedRequests.length;
    await post("/v1/responses", {
      model: "effort-tokens-openai",
      input: "hello",
      reasoning: { effort: "xl" },
    });
    expect(
      receivedSince(tokensOpenai, baseline, "/v1/responses").reasoning,
    ).toEqual({ effort: "low" });

    baseline = tokensAnthropic.receivedRequests.length;
    await post(
      "/v1/messages",
      {
        model: "effort-tokens-anthropic",
        max_tokens: 64,
        messages: [{ role: "user", content: "hello" }],
        output_config: { effort: "xl" },
      },
      "anthropic",
    );
    expect(
      receivedSince(tokensAnthropic, baseline, "/v1/messages").output_config,
    ).toEqual({ effort: "low" });

    baseline = tokensCount.receivedRequests.length;
    await post(
      "/v1/messages/count_tokens",
      {
        model: "effort-tokens-count",
        messages: [{ role: "user", content: "hello" }],
        output_config: { effort: "xl" },
      },
      "anthropic",
    );
    expect(
      receivedSince(tokensCount, baseline, "/v1/messages/count_tokens")
        .output_config,
    ).toEqual({ effort: "low" });

    // A model whose only entry is the wildcard leaves a request that sets
    // no effort alone: the wildcard stands for a value, not for its
    // absence.
    for (const carrier of NO_EFFORT_CARRIERS) {
      baseline = tokensOpenai.receivedRequests.length;
      await post("/v1/chat/completions", {
        model: "effort-tokens-star",
        messages: [{ role: "user", content: "hello" }],
        ...carrier,
      });
      const sent = receivedSince(
        tokensOpenai,
        baseline,
        "/v1/chat/completions",
      );
      expect(sent.reasoning_effort, JSON.stringify(carrier)).toEqual(
        carrier.reasoning_effort,
      );
    }
  });

  test("passes an unset effort through when no entry matches it", async (ctx) => {
    if (!etcdReachable || !app || !openai) {
      ctx.skip();
      return;
    }

    // `effort-map-openai` maps only `medium` and `high`, so a request that
    // sets no effort reaches the upstream exactly as the caller wrote it —
    // an explicit null or empty string included.
    for (const carrier of NO_EFFORT_CARRIERS) {
      const baseline = openai.receivedRequests.length;
      await post("/v1/chat/completions", {
        model: "effort-map-openai",
        messages: [{ role: "user", content: "hello" }],
        ...carrier,
      });
      const sent = receivedSince(openai, baseline, "/v1/chat/completions");
      expect(sent.reasoning_effort, JSON.stringify(carrier)).toEqual(
        carrier.reasoning_effort,
      );
    }

    let baseline = openai.receivedRequests.length;
    await post("/v1/responses", {
      model: "effort-map-openai",
      input: "hello",
      reasoning: { effort: null, summary: "auto" },
    });
    expect(
      receivedSince(openai, baseline, "/v1/responses").reasoning,
    ).toEqual({ effort: null, summary: "auto" });

    baseline = openai.receivedRequests.length;
    await post("/v1/responses", {
      model: "effort-map-openai",
      input: "hello",
    });
    expect(
      receivedSince(openai, baseline, "/v1/responses"),
    ).not.toHaveProperty("reasoning");
  });

  test("maps native and translated streaming requests", async (ctx) => {
    if (
      !etcdReachable ||
      !app ||
      !openaiStream ||
      !anthropicStream ||
      !ensembleStream
    ) {
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

    baseline = ensembleStream.receivedRequests.length;
    const ensembleResponse = await post("/v1/chat/completions", {
      model: "effort-map-ensemble-stream",
      messages: [{ role: "user", content: "hello" }],
      reasoning_effort: "medium",
      stream: true,
    });
    expect(ensembleResponse.contentType).toContain("text/event-stream");
    expect(ensembleResponse.body).toContain("data:");
    const ensembleRequests = chatRequestsSince(ensembleStream, baseline);
    expect(
      ensembleRequests.find(
        (request) => request.model === "glm-ensemble-stream-panel-wire",
      )?.reasoning_effort,
    ).toBe("high");
    expect(
      ensembleRequests.find(
        (request) => request.model === "glm-ensemble-stream-judge-wire",
      )?.reasoning_effort,
    ).toBe("max");
  });
});

// The empty-string key stands for a request that sets no effort, and a
// null value removes the effort field — so the pair asks to remove a field
// the request never set. A rule that can never do anything is refused at
// the write contract rather than stored and never read.
describe("resources file: the not-set key may not map to null", () => {
  const BIN_PATH =
    process.env.AISIX_BIN ??
    join(process.cwd(), "..", "..", "target", "debug", "aisix");

  const file = (mapping: string) =>
    [
      '_format_version: "1"',
      "provider_keys:",
      "  - display_name: pk",
      "    api_key: sk-x",
      "models:",
      "  - display_name: m",
      "    provider: openai",
      "    model_name: gpt-4o",
      "    provider_key: pk",
      "    effort_mapping:",
      mapping,
    ].join("\n");

  test("`\"\": null` is refused, and the other two tokens validate", async () => {
    const dir = await mkdtemp(join(tmpdir(), "aisix-effort-tokens-"));
    try {
      const bad = join(dir, "bad.yaml");
      await writeFile(bad, file('      "": null'), "utf8");
      let failure: (Error & { code?: number; stderr?: string }) | undefined;
      try {
        await execFileP(BIN_PATH, ["validate", "--resources", bad]);
      } catch (e) {
        failure = e as Error & { code?: number; stderr?: string };
      }
      if (!failure) throw new Error("expected `aisix validate` to fail");
      expect(failure.code).toBe(1);
      expect(String(failure.stderr)).toContain("effort_mapping");

      // The same file with a value on that key, plus the two tokens that
      // do combine, validates — so the refusal is the pair and nothing
      // else in the fixture.
      const good = join(dir, "good.yaml");
      await writeFile(
        good,
        file('      "": high\n      "*": low\n      medium: null'),
        "utf8",
      );
      const ok = await execFileP(BIN_PATH, ["validate", "--resources", good]);
      expect(ok.stdout).toContain("OK:");
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  }, 60_000);
});
