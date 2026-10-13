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

// A Claude model without native structured output answers
// `response_format` through a synthetic JSON tool, which cannot stream:
// a streaming request on that route runs non-streaming upstream and the
// result is replayed to the client as a stream. With thinking enabled
// the upstream reply carries a thinking block beside the tool call, and
// the streaming client must receive that reasoning just as the
// non-streaming one does — on every client protocol.

const CALLER = "sk-structured-fake-stream-reasoning";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");

const REASONING = "The user asked about Ada Lovelace; she was 36.";

const PERSON_SCHEMA = {
  type: "object",
  properties: {
    name: { type: "string" },
    age: { type: "integer" },
  },
};

// Thinking enabled leaves the synthetic tool under the model's own
// `auto`, so the model thinks first and then calls it.
const CLAUDE_REPLY = {
  id: "msg_tool_think",
  type: "message",
  role: "assistant",
  model: "claude-sonnet-4-20250514",
  content: [
    { type: "thinking", thinking: REASONING, signature: "sig-upstream" },
    {
      type: "tool_use",
      id: "toolu_json",
      name: "json_tool_call",
      input: { name: "Ada", age: 36 },
    },
  ],
  stop_reason: "tool_use",
  usage: { input_tokens: 21, output_tokens: 40 },
};

const MODELS = {
  anthropic: "tool-think-anthropic",
  bedrock: "tool-think-bedrock",
} as const;

interface SseEvent {
  event: string;
  data: Record<string, any>;
}

function parseNamedSse(raw: string): SseEvent[] {
  const events: SseEvent[] = [];
  for (const frame of raw.split("\n\n")) {
    const lines = frame.split("\n");
    const event = lines.find((l) => l.startsWith("event: "))?.slice(7);
    const data = lines.find((l) => l.startsWith("data: "))?.slice(6);
    if (!event || !data) continue;
    events.push({ event, data: JSON.parse(data) });
  }
  return events;
}

function parseChatSse(raw: string): Array<Record<string, any>> {
  return raw
    .split("\n")
    .filter((l) => l.startsWith("data: ") && l !== "data: [DONE]")
    .map((l) => JSON.parse(l.slice(6)));
}

describe("streaming tool-route structured output keeps the reasoning", () => {
  let app: SpawnedApp | undefined;
  let anthropic: OpenAiUpstream | undefined;
  let bedrock: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    anthropic = await startOpenAiUpstream({ nonStreamBody: CLAUDE_REPLY });
    // Claude on Bedrock answers a non-streaming call on `/invoke` in the
    // Anthropic Messages envelope.
    bedrock = await startOpenAiUpstream({ nonStreamBody: CLAUDE_REPLY });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const anthropicPk = await seed.createProviderKey({
      display_name: "tool-think-anthropic-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-mock",
      api_base: anthropic.baseUrl,
    });
    await seed.createModel({
      display_name: MODELS.anthropic,
      provider: "anthropic",
      model_name: "claude-sonnet-4-20250514",
      provider_key_id: anthropicPk.id,
    });
    const bedrockPk = await seed.createProviderKey({
      display_name: "tool-think-bedrock-pk",
      provider: "bedrock",
      adapter: "bedrock",
      secret: JSON.stringify({
        access_key_id: "AKIA-tool-think",
        secret_access_key: "sk-tool-think",
        region: "us-west-2",
      }),
      api_base: bedrock.baseUrl,
    });
    await seed.createModel({
      display_name: MODELS.bedrock,
      provider: "bedrock",
      model_name: "anthropic.claude-sonnet-4-20250514-v1:0",
      provider_key_id: bedrockPk.id,
    });
    // Seeded last, so this key authenticating implies the whole seed set
    // has reached the gateway's snapshot.
    await seed.createApiKey({
      key_hash: CALLER_HASH,
      allowed_models: Object.values(MODELS),
    });
    const proxy = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(
      async () => (await proxy.listModels()).status === 200,
    );
  });

  afterAll(async () => {
    await app?.exit();
    await anthropic?.close();
    await bedrock?.close();
  });

  function post(path: string, body: unknown): Promise<Response> {
    return fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        "x-api-key": CALLER,
        authorization: `Bearer ${CALLER}`,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    });
  }

  function lastUpstreamBody(upstream: OpenAiUpstream): Record<string, any> {
    const last = upstream.receivedRequests.at(-1);
    expect(last, "upstream received no request").toBeDefined();
    return JSON.parse(last!.body);
  }

  const chatRequest = (stream: boolean) => ({
    model: MODELS.anthropic,
    messages: [{ role: "user", content: "who is Ada" }],
    response_format: {
      type: "json_schema",
      json_schema: { name: "person", schema: PERSON_SCHEMA },
    },
    reasoning_effort: "low",
    stream,
  });

  test("/v1/chat/completions streaming carries the reasoning the non-streaming reply has", async (ctx) => {
    if (!etcdReachable || !app || !anthropic) return ctx.skip();

    const plain = await post("/v1/chat/completions", chatRequest(false));
    expect(plain.status).toBe(200);
    const message = (await plain.json()).choices[0].message;
    expect(message.reasoning_content).toBe(REASONING);

    const res = await post("/v1/chat/completions", chatRequest(true));
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("text/event-stream");
    const frames = parseChatSse(await res.text());

    // The streaming request reached the upstream as the tool route with
    // thinking on, run non-streaming.
    const sent = lastUpstreamBody(anthropic);
    expect(sent.stream).toBe(false);
    expect(sent.thinking?.type).toBeDefined();
    expect(sent.tools?.[0]?.name).toBe("json_tool_call");

    const deltas = frames.map((f) => f.choices?.[0]?.delta ?? {});
    const reasoning = deltas.map((d) => d.reasoning_content ?? "").join("");
    expect(reasoning).toBe(message.reasoning_content);
    const content = deltas.map((d) => d.content ?? "").join("");
    expect(JSON.parse(content)).toEqual({ name: "Ada", age: 36 });
    // Reasoning precedes the answer, as on a real upstream stream.
    const firstReasoning = deltas.findIndex((d) => d.reasoning_content);
    const firstContent = deltas.findIndex((d) => d.content);
    expect(firstReasoning).toBeLessThan(firstContent);
  });

  test("/v1/responses streaming carries the reasoning as a reasoning item", async (ctx) => {
    if (!etcdReachable || !app || !anthropic) return ctx.skip();

    const res = await post("/v1/responses", {
      model: MODELS.anthropic,
      input: "who is Ada",
      reasoning: { effort: "low" },
      text: {
        format: { type: "json_schema", name: "person", schema: PERSON_SCHEMA },
      },
      stream: true,
    });
    expect(res.status).toBe(200);
    const events = parseNamedSse(await res.text());
    expect(lastUpstreamBody(anthropic).stream).toBe(false);

    const done = events.find((e) => e.event === "response.completed");
    expect(done, "no response.completed event").toBeDefined();
    const output = done!.data.response.output as Array<Record<string, any>>;
    const reasoningItem = output.find((item) => item.type === "reasoning");
    expect(reasoningItem, "no reasoning item in the output").toBeDefined();
    const reasoningText = (reasoningItem!.summary as Array<{ text: string }>)
      .map((s) => s.text)
      .join("");
    expect(reasoningText).toBe(REASONING);
    const messageItem = output.find((item) => item.type === "message");
    expect(JSON.parse(messageItem!.content[0].text)).toEqual({
      name: "Ada",
      age: 36,
    });
    expect(output.indexOf(reasoningItem!)).toBeLessThan(
      output.indexOf(messageItem!),
    );
  });

  test("/v1/messages streaming to Bedrock Claude opens a thinking block before the JSON", async (ctx) => {
    if (!etcdReachable || !app || !bedrock) return ctx.skip();

    const request = (stream: boolean) => ({
      model: MODELS.bedrock,
      max_tokens: 2048,
      thinking: { type: "enabled", budget_tokens: 1024 },
      output_config: {
        format: { type: "json_schema", schema: PERSON_SCHEMA },
      },
      messages: [{ role: "user", content: "who is Ada" }],
      stream,
    });

    const plain = await post("/v1/messages", request(false));
    expect(plain.status).toBe(200);
    const plainBody = (await plain.json()) as {
      content: Array<Record<string, any>>;
    };
    const plainThinking = plainBody.content.find((b) => b.type === "thinking");
    expect(plainThinking?.thinking).toBe(REASONING);

    const res = await post("/v1/messages", request(true));
    expect(res.status).toBe(200);
    const events = parseNamedSse(await res.text());

    const sent = bedrock.receivedRequests.at(-1);
    expect(sent?.path).toMatch(/\/invoke$/);

    const starts = events.filter((e) => e.event === "content_block_start");
    expect(starts.map((e) => e.data.content_block.type)).toEqual([
      "thinking",
      "text",
    ]);
    const thinking = events
      .filter((e) => e.data.delta?.type === "thinking_delta")
      .map((e) => e.data.delta.thinking)
      .join("");
    expect(thinking).toBe(plainThinking!.thinking);
    const text = events
      .filter((e) => e.data.delta?.type === "text_delta")
      .map((e) => e.data.delta.text)
      .join("");
    expect(JSON.parse(text)).toEqual({ name: "Ada", age: 36 });
  });
});
