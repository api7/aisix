import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  waitConfigPropagation,
  type SpawnedApp,
} from "../harness/index.js";

// Structured outputs on the two providers whose chat bridge builds its
// own request shape rather than forwarding an OpenAI body. Both used to
// drop a caller's `response_format` on the floor and answer in prose.
//
//   * Gemini takes `generationConfig.responseMimeType` plus the schema,
//     in `responseJsonSchema` from Gemini 2 onwards and in the older
//     OpenAPI-flavoured `responseSchema` before that.
//   * Bedrock picks by whether the model constrains its own decoding.
//     A Claude 4.5 answers non-streaming over the Anthropic Messages
//     `/invoke` wire, whose control is `output_config.format`. Every
//     other model takes the synthetic `json_tool_call` tool on Converse,
//     and the call it makes is translated back into JSON content.

const CALLER_PLAINTEXT = "sk-provider-structured-output";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const PERSON_SCHEMA = {
  type: "object",
  properties: {
    name: { type: "string" },
    nickname: { type: "string" },
  },
  required: ["name"],
};

const RESPONSE_FORMAT = {
  type: "json_schema",
  json_schema: { name: "person", schema: PERSON_SCHEMA, strict: true },
};

const ANSWER = '{"name":"Ada","nickname":"Countess"}';

interface RecordedRequest {
  path: string;
  body: string;
}

interface RecordingUpstream {
  baseUrl: string;
  received: RecordedRequest[];
  close(): Promise<void>;
}

/** A JSON upstream that answers every route from one reply function. */
async function startJsonUpstream(
  reply: (path: string) => unknown,
): Promise<RecordingUpstream> {
  const received: RecordedRequest[] = [];
  const server: Server = createServer((req, res) => {
    res.on("error", () => {});
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      const path = (req.url ?? "/").split("?")[0];
      received.push({ path, body: Buffer.concat(chunks).toString("utf8") });
      res.statusCode = 200;
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify(reply(path)));
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return {
    baseUrl: `http://127.0.0.1:${addr.port}`,
    received,
    close: () =>
      new Promise<void>((resolve, reject) =>
        server.close((e) => (e ? reject(e) : resolve())),
      ),
  };
}

function lastRequest(upstream: RecordingUpstream): {
  path: string;
  body: Record<string, any>;
} {
  const last = upstream.received.at(-1);
  expect(last, "upstream received no request").toBeDefined();
  return { path: last!.path, body: JSON.parse(last!.body) };
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

describe("chat response_format → Gemini and Bedrock", () => {
  let app: SpawnedApp | undefined;
  let gemini: RecordingUpstream | undefined;
  let bedrock: RecordingUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    gemini = await startJsonUpstream(() => ({
      candidates: [
        {
          content: { role: "model", parts: [{ text: ANSWER }] },
          finishReason: "STOP",
        },
      ],
      usageMetadata: {
        promptTokenCount: 7,
        candidatesTokenCount: 11,
        totalTokenCount: 18,
      },
    }));
    // One Bedrock endpoint serves both routes: the Anthropic Messages
    // envelope on `/invoke`, the Converse envelope on `/converse`.
    bedrock = await startJsonUpstream((path) =>
      path.endsWith("/invoke")
        ? {
            id: "msg_bedrock_json",
            type: "message",
            role: "assistant",
            model: "claude-sonnet-4-5-20250929",
            content: [{ type: "text", text: ANSWER }],
            stop_reason: "end_turn",
            usage: { input_tokens: 7, output_tokens: 11 },
          }
        : {
            output: {
              message: {
                role: "assistant",
                content: [
                  {
                    toolUse: {
                      toolUseId: "tooluse_json",
                      name: "json_tool_call",
                      input: { name: "Ada", nickname: "Countess" },
                    },
                  },
                ],
              },
            },
            stopReason: "tool_use",
            usage: { inputTokens: 7, outputTokens: 11, totalTokens: 18 },
            metrics: { latencyMs: 1 },
          },
    );

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const vertexPk = await seed.createProviderKey({
      display_name: "structured-vertex-pk",
      provider: "google",
      adapter: "vertex",
      secret: JSON.stringify({
        access_token: "ya29.structured-e2e",
        project: "proj-e2e",
        region: "us-central1",
      }),
      api_base: gemini.baseUrl,
    });
    // The aliases are deliberately not the upstream names: every gate
    // here reads the model the gateway dispatches to, never what the
    // caller typed.
    await seed.createModel({
      display_name: "json-gemini",
      provider: "google",
      model_name: "gemini-2.5-flash",
      provider_key_id: vertexPk.id,
    });
    await seed.createModel({
      display_name: "json-gemini-legacy",
      provider: "google",
      model_name: "gemini-1.5-pro",
      provider_key_id: vertexPk.id,
    });

    const bedrockPk = await seed.createProviderKey({
      display_name: "structured-bedrock-pk",
      provider: "bedrock",
      adapter: "bedrock",
      secret: JSON.stringify({
        access_key_id: "AKIA-structured-e2e",
        secret_access_key: "sk-structured-e2e",
        region: "us-west-2",
      }),
      api_base: bedrock.baseUrl,
    });
    await seed.createModel({
      display_name: "json-claude-bedrock",
      provider: "bedrock",
      model_name: "anthropic.claude-sonnet-4-5-20250929-v1:0",
      provider_key_id: bedrockPk.id,
    });
    await seed.createModel({
      display_name: "json-nova",
      provider: "bedrock",
      model_name: "amazon.nova-pro-v1:0",
      provider_key_id: bedrockPk.id,
    });

    // Seeded last, so this key authenticating implies the whole seed set
    // has reached the gateway's snapshot.
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      if (res.status !== 200) {
        await res.text();
        return false;
      }
      const body = (await res.json()) as { data?: Array<{ id?: string }> };
      return (body.data ?? []).some((m) => m.id === "json-nova");
    });
  });

  afterAll(async () => {
    await app?.exit();
    await gemini?.close();
    await bedrock?.close();
  });

  test("gemini 2+ gets responseMimeType and responseJsonSchema", async (ctx) => {
    if (!etcdReachable || !app || !gemini) {
      ctx.skip();
      return;
    }
    const res = await chat(app, {
      model: "json-gemini",
      messages: [{ role: "user", content: "who is Ada" }],
      response_format: RESPONSE_FORMAT,
    });
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(JSON.parse(body.choices[0].message.content)).toEqual({
      name: "Ada",
      nickname: "Countess",
    });

    const sent = lastRequest(gemini);
    expect(sent.path).toContain("gemini-2.5-flash:generateContent");
    expect(sent.body.generationConfig.responseMimeType).toBe(
      "application/json",
    );
    // Ordinary JSON Schema, forwarded as the caller wrote it — the
    // caller's `required` included.
    expect(sent.body.generationConfig.responseJsonSchema).toEqual(
      PERSON_SCHEMA,
    );
    expect(sent.body.generationConfig.responseSchema).toBeUndefined();
    // The OpenAI spelling has no Gemini counterpart; forwarding it 400s.
    expect(sent.body.response_format).toBeUndefined();
  });

  test("gemini 1.x gets the OpenAPI-flavoured responseSchema", async (ctx) => {
    if (!etcdReachable || !app || !gemini) {
      ctx.skip();
      return;
    }
    const res = await chat(app, {
      model: "json-gemini-legacy",
      messages: [{ role: "user", content: "who is Ada" }],
      response_format: RESPONSE_FORMAT,
    });
    expect(res.status).toBe(200);

    const sent = lastRequest(gemini);
    expect(sent.path).toContain("gemini-1.5-pro:generateContent");
    const gc = sent.body.generationConfig;
    expect(gc.responseMimeType).toBe("application/json");
    expect(gc.responseJsonSchema).toBeUndefined();
    expect(gc.responseSchema.type).toBe("OBJECT");
    expect(gc.responseSchema.properties.name.type).toBe("STRING");
    expect(gc.responseSchema.propertyOrdering).toEqual(["name", "nickname"]);
    expect(gc.responseSchema.required).toEqual(["name"]);
  });

  test("a bedrock claude 4.5 gets output_config.format on the messages wire", async (ctx) => {
    if (!etcdReachable || !app || !bedrock) {
      ctx.skip();
      return;
    }
    const res = await chat(app, {
      model: "json-claude-bedrock",
      messages: [{ role: "user", content: "who is Ada" }],
      response_format: RESPONSE_FORMAT,
    });
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.choices[0].message.content).toBe(ANSWER);

    const sent = lastRequest(bedrock);
    expect(sent.path).toMatch(/\/invoke$/);
    expect(sent.body.output_config.format.type).toBe("json_schema");
    // Sealed, because Bedrock rejects an open object — but the caller's
    // optional `nickname` is still optional.
    expect(sent.body.output_config.format.schema.additionalProperties).toBe(
      false,
    );
    expect(sent.body.output_config.format.schema.required).toEqual(["name"]);
    expect(sent.body.response_format).toBeUndefined();
    expect(sent.body.tools).toBeUndefined();
  });

  test("a bedrock nova gets the synthetic tool and its call comes back as JSON", async (ctx) => {
    if (!etcdReachable || !app || !bedrock) {
      ctx.skip();
      return;
    }
    const res = await chat(app, {
      model: "json-nova",
      messages: [{ role: "user", content: "who is Ada" }],
      response_format: RESPONSE_FORMAT,
    });
    expect(res.status).toBe(200);

    const sent = lastRequest(bedrock);
    expect(sent.path).toMatch(/\/converse$/);
    expect(sent.body.outputConfig).toBeUndefined();
    const tools = sent.body.toolConfig.tools;
    expect(tools).toHaveLength(1);
    expect(tools[0].toolSpec.name).toBe("json_tool_call");
    expect(tools[0].toolSpec.inputSchema.json.additionalProperties).toBe(false);
    // Nova is one of the two families whose Converse honours a
    // `toolChoice`, so the tool is forced rather than merely offered.
    expect(sent.body.toolConfig.toolChoice).toEqual({
      tool: { name: "json_tool_call" },
    });

    // The caller never offered a tool, so it must not be told the model
    // stopped to call one.
    const body = await res.json();
    const message = body.choices[0].message;
    expect(JSON.parse(message.content)).toEqual({
      name: "Ada",
      nickname: "Countess",
    });
    expect(message.tool_calls).toBeUndefined();
    expect(body.choices[0].finish_reason).toBe("stop");
  });
});
