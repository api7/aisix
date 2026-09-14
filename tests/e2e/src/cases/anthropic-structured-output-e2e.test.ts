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

// Structured outputs on Anthropic-protocol upstreams. A chat caller (or a
// `/v1/responses` caller, whose `text.format` the bridge turns into
// `response_format`) that asks for JSON used to get prose: the field was
// consumed and dropped. It now becomes one of two request shapes, picked
// by the target model's family:
//
//   * Claude 4.5 and later → `output_config.format`, the model's own
//     structured-output control.
//   * everything else (older Claude, and non-Claude models behind
//     Anthropic-compatible endpoints) → a synthetic `json_tool_call` tool
//     under a forced `tool_choice`, whose call the decoder translates back
//     into plain JSON content.

const CALLER_PLAINTEXT = "sk-anthropic-structured-output";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const PERSON_SCHEMA = {
  type: "object",
  properties: {
    name: { type: "string" },
    age: { type: "integer" },
  },
};

const RESPONSE_FORMAT = {
  type: "json_schema",
  json_schema: { name: "person", schema: PERSON_SCHEMA, strict: true },
};

// What a Claude 4.5+ model returns once `output_config.format` constrains
// it: ordinary text that happens to be the JSON document.
const NATIVE_REPLY = {
  id: "msg_native_json",
  type: "message",
  role: "assistant",
  model: "claude-sonnet-4-5-20250929",
  content: [{ type: "text", text: '{"name":"Ada","age":36}' }],
  stop_reason: "end_turn",
  usage: { input_tokens: 14, output_tokens: 9 },
};

// What the tool path gets back: the model calls the synthetic tool, and
// the call's input is the answer.
const TOOL_REPLY = {
  id: "msg_tool_json",
  type: "message",
  role: "assistant",
  model: "claude-3-5-haiku-20241022",
  content: [
    {
      type: "tool_use",
      id: "toolu_json",
      name: "json_tool_call",
      input: { name: "Ada", age: 36 },
    },
  ],
  stop_reason: "tool_use",
  usage: { input_tokens: 21, output_tokens: 12 },
};

function lastBody(upstream: OpenAiUpstream): Record<string, unknown> {
  const last = upstream.receivedRequests.at(-1);
  expect(last?.path).toBe("/v1/messages");
  return JSON.parse(last!.body);
}

function chat(app: SpawnedApp, body: unknown): Promise<Response> {
  return fetch(`${app.proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    },
    body: JSON.stringify(body),
  });
}

describe("chat response_format → Anthropic structured outputs", () => {
  let app: SpawnedApp | undefined;
  let nativeUpstream: OpenAiUpstream | undefined;
  let toolUpstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    nativeUpstream = await startOpenAiUpstream({ nonStreamBody: NATIVE_REPLY });
    toolUpstream = await startOpenAiUpstream({ nonStreamBody: TOOL_REPLY });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const nativePk = await seed.createProviderKey({
      display_name: "structured-native-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-mock",
      api_base: nativeUpstream.baseUrl,
    });
    // The alias is deliberately not the upstream name: the family gate
    // reads the model the gateway dispatches to, not what the caller
    // typed.
    await seed.createModel({
      display_name: "json-native",
      provider: "anthropic",
      model_name: "claude-sonnet-4-5",
      provider_key_id: nativePk.id,
    });
    const toolPk = await seed.createProviderKey({
      display_name: "structured-tool-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-mock",
      api_base: toolUpstream.baseUrl,
    });
    await seed.createModel({
      display_name: "json-legacy",
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: toolPk.id,
    });
    // Seeded last, so this key authenticating implies the whole seed
    // set has reached the gateway's snapshot.
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["json-native", "json-legacy"],
    });
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(
      async () => (await proxy.listModels()).status === 200,
    );
  });

  afterAll(async () => {
    await app?.exit();
    await nativeUpstream?.close();
    await toolUpstream?.close();
  });

  test("claude 4.5+ takes the native path: output_config.format, no response_format", async (ctx) => {
    if (!etcdReachable || !app || !nativeUpstream) {
      ctx.skip();
      return;
    }
    const res = await chat(app, {
      model: "json-native",
      messages: [{ role: "user", content: "who is Ada" }],
      response_format: RESPONSE_FORMAT,
    });
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.choices[0].message.content).toBe('{"name":"Ada","age":36}');

    const sent = lastBody(nativeUpstream);
    const format = (sent.output_config as Record<string, any>)?.format;
    expect(format?.type).toBe("json_schema");
    expect(format?.schema?.properties?.name?.type).toBe("string");
    // Anthropic rejects an open object schema.
    expect(format?.schema?.additionalProperties).toBe(false);
    // The OpenAI spelling never reaches the Anthropic body, and the
    // native path adds no tool.
    expect(sent.response_format).toBeUndefined();
    expect(sent.tools).toBeUndefined();
    expect(sent.tool_choice).toBeUndefined();
  });

  test("older claude takes the tool path and the call comes back as JSON content", async (ctx) => {
    if (!etcdReachable || !app || !toolUpstream) {
      ctx.skip();
      return;
    }
    const res = await chat(app, {
      model: "json-legacy",
      messages: [{ role: "user", content: "who is Ada" }],
      response_format: RESPONSE_FORMAT,
    });
    expect(res.status).toBe(200);

    const sent = lastBody(toolUpstream);
    expect(sent.response_format).toBeUndefined();
    expect(sent.output_config).toBeUndefined();
    const tools = sent.tools as Array<Record<string, any>>;
    expect(tools).toHaveLength(1);
    expect(tools[0].name).toBe("json_tool_call");
    expect(tools[0].input_schema.additionalProperties).toBe(false);
    expect(sent.tool_choice).toEqual({ type: "tool", name: "json_tool_call" });

    // The caller never offered a tool, so it must not be told the model
    // stopped to call one.
    const body = await res.json();
    const message = body.choices[0].message;
    expect(JSON.parse(message.content)).toEqual({ name: "Ada", age: 36 });
    expect(message.tool_calls).toBeUndefined();
    expect(body.choices[0].finish_reason).toBe("stop");
  });

  test("streaming on the tool path fake-streams the JSON and still reports usage", async (ctx) => {
    if (!etcdReachable || !app || !toolUpstream) {
      ctx.skip();
      return;
    }
    const res = await chat(app, {
      model: "json-legacy",
      messages: [{ role: "user", content: "who is Ada" }],
      response_format: RESPONSE_FORMAT,
      stream: true,
      stream_options: { include_usage: true },
    });
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("text/event-stream");
    const raw = await res.text();

    // The upstream leg ran non-streaming — the JSON only exists once the
    // tool call is complete.
    const sent = lastBody(toolUpstream);
    expect(sent.stream).toBe(false);
    expect(sent.tool_choice).toEqual({ type: "tool", name: "json_tool_call" });

    const frames = raw
      .split("\n")
      .filter((l) => l.startsWith("data: ") && !l.includes("[DONE]"))
      .map((l) => JSON.parse(l.slice(6)));
    expect(frames[0].choices[0].delta.role).toBe("assistant");
    const text = frames
      .map((f) => f.choices?.[0]?.delta?.content ?? "")
      .join("");
    expect(JSON.parse(text)).toEqual({ name: "Ada", age: 36 });
    expect(frames.some((f) => f.choices?.[0]?.finish_reason === "stop")).toBe(
      true,
    );
    expect(
      frames.some((f) => f.usage?.completion_tokens === 12),
    ).toBe(true);
  });

  test("/v1/responses text.format reaches Anthropic as output_config.format", async (ctx) => {
    if (!etcdReachable || !app || !nativeUpstream) {
      ctx.skip();
      return;
    }
    const res = await fetch(`${app.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "json-native",
        input: "who is Ada",
        text: {
          format: {
            type: "json_schema",
            name: "person",
            schema: PERSON_SCHEMA,
            strict: true,
          },
        },
      }),
    });
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.output[0].content[0].text).toBe('{"name":"Ada","age":36}');

    const sent = lastBody(nativeUpstream);
    const format = (sent.output_config as Record<string, any>)?.format;
    expect(format?.type).toBe("json_schema");
    expect(format?.schema?.additionalProperties).toBe(false);
    expect(sent.response_format).toBeUndefined();
  });
});
