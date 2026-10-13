import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  eventStream,
  spawnApp,
  waitConfigPropagation,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: a streamed Claude-on-Bedrock reply carries its reasoning, on every
// client protocol, just as the same reply does when it is not streamed.
//
// Streaming Bedrock chat goes through ConverseStream, whose reasoning
// arrives as `contentBlockDelta.reasoningContent` events with no
// `contentBlockStart` before them. The event sequence below follows one
// captured from `us.anthropic.claude-haiku-4-5-20251001-v1:0` (thinking
// budget 1024): reasoning text deltas on block 0, the last one empty, then
// a signature-only delta. The text is shortened and the signature replaced
// with a marker; a redacted-reasoning block is added so the spec can show
// that ciphertext stays out of the reply, as it does when not streamed.

const CALLER_PLAINTEXT = "sk-bedrock-stream-reasoning";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const MODEL = "bedrock-claude-reasoning";

const REASONING_DELTAS = ["17", " * 23", "\n\nBreak it down:\n17 * 20", " + 17 * 3", "\n= 340 + 51", "\n= 391", ""];
const REASONING = REASONING_DELTAS.join("");
const ANSWER_DELTAS = ["17", " × 23 = 391"];
const ANSWER = ANSWER_DELTAS.join("");
const SIGNATURE = "SIGNATURE_FIXTURE_c41f";
const REDACTED = Buffer.from("REDACTED_FIXTURE_9b2e").toString("base64");

const CONVERSE_STREAM = eventStream([
  ["messageStart", { role: "assistant" }],
  ...REASONING_DELTAS.map(
    (text): [string, unknown] => [
      "contentBlockDelta",
      { contentBlockIndex: 0, delta: { reasoningContent: { text } } },
    ],
  ),
  ["contentBlockDelta", { contentBlockIndex: 0, delta: { reasoningContent: { signature: SIGNATURE } } }],
  ["contentBlockStop", { contentBlockIndex: 0 }],
  ["contentBlockDelta", { contentBlockIndex: 1, delta: { reasoningContent: { redactedContent: REDACTED } } }],
  ["contentBlockStop", { contentBlockIndex: 1 }],
  ...ANSWER_DELTAS.map(
    (text): [string, unknown] => ["contentBlockDelta", { contentBlockIndex: 2, delta: { text } }],
  ),
  ["contentBlockStop", { contentBlockIndex: 2 }],
  ["messageStop", { stopReason: "end_turn" }],
  [
    "metadata",
    { usage: { inputTokens: 46, outputTokens: 119, totalTokens: 165 }, metrics: { latencyMs: 1045 } },
  ],
]);

// The same reply as Bedrock `/invoke` returns it when not streamed.
const INVOKE_REPLY = {
  id: "msg_bdrk_reasoning",
  type: "message",
  role: "assistant",
  model: "claude-haiku-4-5-20251001",
  content: [
    { type: "thinking", thinking: REASONING, signature: SIGNATURE },
    { type: "redacted_thinking", data: REDACTED },
    { type: "text", text: ANSWER },
  ],
  stop_reason: "end_turn",
  usage: { input_tokens: 46, output_tokens: 119 },
};

// The same reply as non-streaming Converse returns it, which is how every
// non-Anthropic publisher answers when not streamed.
const CONVERSE_REPLY = {
  output: {
    message: {
      role: "assistant",
      content: [
        { reasoningContent: { reasoningText: { text: REASONING, signature: SIGNATURE } } },
        { reasoningContent: { redactedContent: REDACTED } },
        { text: ANSWER },
      ],
    },
  },
  stopReason: "end_turn",
  usage: { inputTokens: 46, outputTokens: 119, totalTokens: 165 },
  metrics: { latencyMs: 1045 },
};

const PROMPT = "What is 17 * 23?";
const CONVERSE_MODEL = "bedrock-deepseek-reasoning";

describe("Bedrock: streamed reasoning matches the non-streamed reply", () => {
  let app: SpawnedApp | undefined;
  let upstream: Server | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = createServer((req, res) => {
      req.resume();
      req.on("end", () => {
        if (req.url?.endsWith("/converse-stream")) {
          res.writeHead(200, { "content-type": "application/vnd.amazon.eventstream" });
          res.end(CONVERSE_STREAM);
        } else if (req.url?.endsWith("/converse")) {
          res.writeHead(200, { "content-type": "application/json" });
          res.end(JSON.stringify(CONVERSE_REPLY));
        } else if (req.url?.endsWith("/invoke")) {
          res.writeHead(200, { "content-type": "application/json" });
          res.end(JSON.stringify(INVOKE_REPLY));
        } else {
          res.writeHead(404, { "content-type": "application/json" });
          res.end(JSON.stringify({ message: `unexpected path ${req.url}` }));
        }
      });
    });
    await new Promise<void>((resolve) => upstream!.listen(0, "127.0.0.1", resolve));
    const addr = upstream.address();
    if (addr === null || typeof addr === "string") throw new Error("no port");

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "bedrock-stream-reasoning-pk",
      provider: "bedrock",
      adapter: "bedrock",
      secret: JSON.stringify({
        access_key_id: "AKIA-stream-reasoning",
        secret_access_key: "sk-stream-reasoning",
        region: "us-west-2",
      }),
      api_base: `http://127.0.0.1:${addr.port}`,
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "bedrock",
      model_name: "us.anthropic.claude-haiku-4-5-20251001-v1:0",
      provider_key_id: pk.id,
    });
    await seed.createModel({
      display_name: CONVERSE_MODEL,
      provider: "bedrock",
      model_name: "us.deepseek.r1-v1:0",
      provider_key_id: pk.id,
    });
    // Seeded last, so this key authenticating implies the whole seed set
    // has reached the gateway's snapshot.
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      await res.text();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await new Promise<void>((resolve) => (upstream ? upstream.close(() => resolve()) : resolve()));
  });

  async function post(path: string, body: unknown): Promise<string> {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "x-api-key": CALLER_PLAINTEXT,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    expect(res.status, text).toBe(200);
    return text;
  }

  function sseData(text: string): any[] {
    return text
      .split("\n")
      .filter((l) => l.startsWith("data: ") && l !== "data: [DONE]")
      .map((l) => JSON.parse(l.slice("data: ".length)));
  }

  function expectNoOpaqueReasoning(text: string) {
    expect(text).not.toContain(SIGNATURE);
    expect(text).not.toContain(REDACTED);
  }

  test("/v1/chat/completions streams delta.reasoning_content", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const request = {
      model: MODEL,
      reasoning_effort: "low",
      messages: [{ role: "user", content: PROMPT }],
    };

    const plain = JSON.parse(await post("/v1/chat/completions", request));
    expect(plain.choices[0].message.reasoning_content).toBe(REASONING);

    const raw = await post("/v1/chat/completions", { ...request, stream: true });
    const deltas = sseData(raw).map((c) => c.choices?.[0]?.delta ?? {});
    expect(deltas.map((d) => d.reasoning_content ?? "").join("")).toBe(REASONING);
    expect(deltas.map((d) => d.content ?? "").join("")).toBe(ANSWER);
    expectNoOpaqueReasoning(raw);
  });

  test("/v1/messages streams a thinking block", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const request = {
      model: MODEL,
      max_tokens: 2048,
      thinking: { type: "enabled", budget_tokens: 1024 },
      messages: [{ role: "user", content: PROMPT }],
    };

    const plain = JSON.parse(await post("/v1/messages", request));
    const plainThinking = plain.content
      .filter((b: any) => b.type === "thinking")
      .map((b: any) => b.thinking)
      .join("");
    expect(plainThinking).toBe(REASONING);

    const raw = await post("/v1/messages", { ...request, stream: true });
    const events = sseData(raw);
    const thinkingBlocks = new Set(
      events
        .filter((e) => e.type === "content_block_start" && e.content_block?.type === "thinking")
        .map((e) => e.index),
    );
    expect(thinkingBlocks.size).toBe(1);
    const thinking = events
      .filter((e) => e.type === "content_block_delta" && thinkingBlocks.has(e.index))
      .map((e) => e.delta.thinking ?? "")
      .join("");
    expect(thinking).toBe(REASONING);
    const text = events
      .filter((e) => e.type === "content_block_delta" && e.delta?.type === "text_delta")
      .map((e) => e.delta.text)
      .join("");
    expect(text).toBe(ANSWER);
    expect(events.some((e) => e.delta?.type === "signature_delta")).toBe(false);
    expectNoOpaqueReasoning(raw);
  });

  test("/v1/responses streams a reasoning item", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const request = { model: MODEL, reasoning: { effort: "low" }, input: PROMPT };
    const reasoningIn = (output: any[]) =>
      output.filter((i) => i.type === "reasoning").map((i) => JSON.stringify(i)).join("");
    const encoded = JSON.stringify(REASONING).slice(1, -1);

    const plain = JSON.parse(await post("/v1/responses", request));
    expect(reasoningIn(plain.output)).toContain(encoded);

    const raw = await post("/v1/responses", { ...request, stream: true });
    const completed = sseData(raw).find((e) => e.type === "response.completed");
    expect(completed, raw).toBeDefined();
    expect(reasoningIn(completed.response.output)).toContain(encoded);
    expectNoOpaqueReasoning(raw);
  });

  // A non-Anthropic publisher answers through Converse both ways, so its
  // non-streamed reply has to carry the reasoning as well.
  test("a Converse publisher returns reasoning streamed and not streamed", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const request = { model: CONVERSE_MODEL, messages: [{ role: "user", content: PROMPT }] };

    const plainRaw = await post("/v1/chat/completions", request);
    const plain = JSON.parse(plainRaw);
    expect(plain.choices[0].message.reasoning_content).toBe(REASONING);
    expect(plain.choices[0].message.content).toBe(ANSWER);
    expectNoOpaqueReasoning(plainRaw);

    const raw = await post("/v1/chat/completions", { ...request, stream: true });
    const deltas = sseData(raw).map((c) => c.choices?.[0]?.delta ?? {});
    expect(deltas.map((d) => d.reasoning_content ?? "").join("")).toBe(REASONING);
    expect(deltas.map((d) => d.content ?? "").join("")).toBe(ANSWER);

    const messagesRaw = await post("/v1/messages", {
      model: CONVERSE_MODEL,
      max_tokens: 2048,
      messages: [{ role: "user", content: PROMPT }],
    });
    const messages = JSON.parse(messagesRaw);
    expect(
      messages.content.filter((b: any) => b.type === "thinking").map((b: any) => b.thinking).join(""),
    ).toBe(REASONING);
    expectNoOpaqueReasoning(messagesRaw);
  });
});
