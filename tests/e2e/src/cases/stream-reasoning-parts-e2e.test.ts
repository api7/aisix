import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";
import { startMockOtlp, type MockOtlp } from "../harness/otlp-mock.js";

// An OpenAI-compatible upstream may stream a delta whose `content` is an
// array of parts, reasoning parts among them (Mistral's `thinking`,
// Databricks' `reasoning`). The upstream generated and billed that
// reasoning, so a streaming caller must receive it in its own protocol's
// reasoning slot, in the order it was streamed — as it does reasoning that
// arrives in `delta.reasoning_content`.

const CALLER_PLAINTEXT = "sk-stream-reasoning-parts";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const HEADERS = {
  authorization: `Bearer ${CALLER_PLAINTEXT}`,
  "content-type": "application/json",
};

const THINK_A = "The user asks for the capital of France. ";
const THINK_B = "That is Paris.";
const THINKING = THINK_A + THINK_B;
const ANSWER_A = "Paris is the capital ";
const ANSWER_B = "of France.";
const ANSWER = ANSWER_A + ANSWER_B;

type Shape = "mistral" | "databricks";

// The two documented shapes; the second fragment of each is signed.
const reasoningPart = (shape: Shape, text: string, signature?: string) =>
  shape === "mistral"
    ? { type: "thinking", thinking: [{ type: "text", text }], ...(signature ? { signature } : {}) }
    : {
        type: "reasoning",
        summary: [{ type: "summary_text", text, ...(signature ? { signature } : {}) }],
      };

const frame = (choice: Record<string, unknown>) =>
  JSON.stringify({
    id: "chatcmpl-parts",
    object: "chat.completion.chunk",
    created: 1_700_000_000,
    model: "magistral-medium",
    choices: [{ index: 0, finish_reason: null, ...choice }],
  });

const USAGE_FRAME = JSON.stringify({
  id: "chatcmpl-parts",
  object: "chat.completion.chunk",
  created: 1_700_000_000,
  model: "magistral-medium",
  choices: [],
  usage: { prompt_tokens: 9, completion_tokens: 30, total_tokens: 39 },
});

// Reasoning over two frames, the second of which also starts the answer.
const partsStream = (shape: Shape) => [
  frame({ delta: { role: "assistant", content: [reasoningPart(shape, THINK_A)] } }),
  frame({ delta: { content: [reasoningPart(shape, THINK_B, "sig-b"), { type: "text", text: ANSWER_A }] } }),
  frame({ delta: { content: [{ type: "text", text: ANSWER_B }] }, finish_reason: "stop" }),
  USAGE_FRAME,
  "[DONE]",
];

// One delta whose parts put reasoning between two stretches of text.
const INTERLEAVED_STREAM = [
  frame({
    delta: {
      role: "assistant",
      content: [
        { type: "text", text: "Let me check. " },
        reasoningPart("mistral", "Checking the atlas."),
        { type: "text", text: "Paris." },
      ],
    },
    finish_reason: "stop",
  }),
  USAGE_FRAME,
  "[DONE]",
];

// Long enough that its tokens dominate the estimate of a stream that
// reports no usage; the answer text alone is a handful of tokens.
const LONG_THINKING = "Considering every European capital in turn before settling on one. ".repeat(20);
const NO_USAGE_STREAM = [
  frame({ delta: { role: "assistant", content: [reasoningPart("mistral", LONG_THINKING)] } }),
  frame({ delta: { content: [{ type: "text", text: "Paris." }] }, finish_reason: "stop" }),
  "[DONE]",
];

async function stream(app: SpawnedApp, path: string, body: Record<string, unknown>): Promise<string> {
  const res = await fetch(`${app.proxyUrl}${path}`, {
    method: "POST",
    headers: HEADERS,
    body: JSON.stringify({ ...body, stream: true }),
  });
  const text = await res.text();
  expect(res.status, text).toBe(200);
  return text;
}

const chatBody = (model: string, prompt = "What is the capital of France?") => ({
  model,
  messages: [{ role: "user", content: prompt }],
});
const messagesBody = (model: string, prompt = "What is the capital of France?") => ({
  model,
  max_tokens: 256,
  messages: [{ role: "user", content: prompt }],
});
const responsesBody = (model: string, prompt = "What is the capital of France?") => ({ model, input: prompt });

/** Chat SSE as runs of `[kind, text]`, adjacent deltas of one kind joined. */
function chatRuns(raw: string): Array<[string, string]> {
  const runs: Array<[string, string]> = [];
  const push = (kind: string, text: unknown) => {
    if (typeof text !== "string" || text === "") return;
    const last = runs[runs.length - 1];
    if (last && last[0] === kind) last[1] += text;
    else runs.push([kind, text]);
  };
  for (const line of raw.split("\n")) {
    if (!line.startsWith("data: ") || line === "data: [DONE]") continue;
    for (const choice of JSON.parse(line.slice(6)).choices ?? []) {
      push("reasoning", choice.delta?.reasoning_content);
      push("content", choice.delta?.content);
    }
  }
  return runs;
}

function namedEvents(raw: string): Array<{ event: string; data: any }> {
  const events: Array<{ event: string; data: any }> = [];
  for (const block of raw.split("\n\n")) {
    const lines = block.split("\n");
    const event = lines.find((l) => l.startsWith("event: "))?.slice(7);
    const data = lines.find((l) => l.startsWith("data: "))?.slice(6);
    if (event && data) events.push({ event, data: JSON.parse(data) });
  }
  return events;
}

/** The content blocks an Anthropic Messages stream assembles to. */
function messagesBlocks(raw: string): any[] {
  const blocks: any[] = [];
  for (const { data } of namedEvents(raw)) {
    if (data.type === "content_block_start") blocks[data.index] = { ...data.content_block };
    if (data.type !== "content_block_delta") continue;
    const block = blocks[data.index];
    if (data.delta.type === "thinking_delta") block.thinking += data.delta.thinking;
    if (data.delta.type === "signature_delta") block.signature += data.delta.signature;
    if (data.delta.type === "text_delta") block.text += data.delta.text;
  }
  return blocks;
}

describe("streamed reasoning content parts", () => {
  let app: SpawnedApp | undefined;
  let otlp: MockOtlp | undefined;
  let etcdReachable = false;
  const up: Record<string, OpenAiUpstream> = {};

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    up.mistral = await startOpenAiUpstream({ streamEvents: partsStream("mistral") });
    up.databricks = await startOpenAiUpstream({ streamEvents: partsStream("databricks") });
    up.interleaved = await startOpenAiUpstream({ streamEvents: INTERLEAVED_STREAM });
    up.noUsage = await startOpenAiUpstream({ streamEvents: NO_USAGE_STREAM });
    otlp = await startMockOtlp();

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const model = async (name: string, upstream: OpenAiUpstream) => {
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      await seed.createModel({
        display_name: name,
        // Not `openai`: `/v1/responses` reaches it through the chat bridge.
        provider: "mistral",
        model_name: "magistral-medium",
        provider_key_id: pk.id,
      });
    };
    await model("srp-mistral", up.mistral);
    await model("srp-databricks", up.databricks);
    await model("srp-interleaved", up.interleaved);
    await model("srp-no-usage", up.noUsage);
    await seed.createObservabilityExporter({
      name: "srp-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
      content_mode: "full",
      content_max_bytes: 8192,
    });
    // Seeded last: the key authenticating implies the whole seed set is live.
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, { headers: HEADERS });
      await res.arrayBuffer();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await otlp?.close();
    await Promise.all(Object.values(up).map((u) => u.close()));
  });

  test.for(["mistral", "databricks"] as const)(
    "(%s) reach a /v1/chat/completions caller as reasoning_content ahead of the content",
    async (shape, ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const raw = await stream(app, "/v1/chat/completions", chatBody(`srp-${shape}`));
      expect(chatRuns(raw), raw).toEqual([
        ["reasoning", THINKING],
        ["content", ANSWER],
      ]);
    },
  );

  test("keep their place between the text parts of one delta on /v1/chat/completions", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const raw = await stream(app, "/v1/chat/completions", chatBody("srp-interleaved"));
    expect(chatRuns(raw), raw).toEqual([
      ["content", "Let me check. "],
      ["reasoning", "Checking the atlas."],
      ["content", "Paris."],
    ]);
  });

  test("keep their place between the text parts of one delta on /v1/messages", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const raw = await stream(app, "/v1/messages", messagesBody("srp-interleaved"));
    expect(
      namedEvents(raw)
        .filter(({ event }) => event === "content_block_start")
        .map(({ data }) => [data.index, data.content_block.type]),
      raw,
    ).toEqual([
      [0, "text"],
      [1, "thinking"],
      [2, "text"],
    ]);
    expect(messagesBlocks(raw), raw).toEqual([
      { type: "text", text: "Let me check. " },
      { type: "thinking", thinking: "Checking the atlas.", signature: "" },
      { type: "text", text: "Paris." },
    ]);
  });

  test("keep their place between the text parts of one delta on /v1/responses", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const raw = await stream(app, "/v1/responses", responsesBody("srp-interleaved"));
    const events = namedEvents(raw);
    // Each item is finished before the next one is added.
    const open = new Set<number>();
    const added: number[] = [];
    for (const { event, data } of events) {
      if (event === "response.output_item.added") {
        expect(open.size, `item ${data.output_index} added while ${[...open]} open\n${raw}`).toBe(0);
        added.push(data.output_index);
        open.add(data.output_index);
      }
      if (event === "response.output_item.done") {
        expect(open.delete(data.output_index), `item ${data.output_index} done without being open\n${raw}`).toBe(true);
      }
    }
    expect(added, raw).toEqual([0, 1, 2]);
    expect([...open], raw).toEqual([]);
    const completed = events.find((e) => e.event === "response.completed");
    expect(completed, raw).toBeDefined();
    const output = completed!.data.response.output;
    expect(output.map((item: any) => item.type)).toEqual(["message", "reasoning", "message"]);
    expect(output[0].content).toMatchObject([{ type: "output_text", text: "Let me check. " }]);
    expect(output[1].summary).toEqual([{ type: "summary_text", text: "Checking the atlas." }]);
    expect(output[2].content).toMatchObject([{ type: "output_text", text: "Paris." }]);
  });

  test.for(["mistral", "databricks"] as const)(
    "(%s) reach a /v1/messages caller as a thinking block ahead of the text",
    async (shape, ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const raw = await stream(app, "/v1/messages", messagesBody(`srp-${shape}`));
      // The upstream's signature is not a Claude signature; the block
      // carries the empty one, as a non-streamed answer's does.
      expect(messagesBlocks(raw), raw).toEqual([
        { type: "thinking", thinking: THINKING, signature: "" },
        { type: "text", text: ANSWER },
      ]);
    },
  );

  test.for(["mistral", "databricks"] as const)(
    "(%s) reach a /v1/responses caller as a reasoning item ahead of the message",
    async (shape, ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const raw = await stream(app, "/v1/responses", responsesBody(`srp-${shape}`));
      const events = namedEvents(raw);
      const deltas = events
        .filter((e) => e.event === "response.reasoning_summary_text.delta")
        .map((e) => e.data.delta)
        .join("");
      expect(deltas, raw).toBe(THINKING);
      const completed = events.find((e) => e.event === "response.completed");
      expect(completed, raw).toBeDefined();
      const output = completed!.data.response.output;
      expect(output.map((item: any) => item.type)).toEqual(["reasoning", "message"]);
      expect(output[0].summary).toEqual([{ type: "summary_text", text: THINKING }]);
      expect(output[1].content).toMatchObject([{ type: "output_text", text: ANSWER }]);
    },
  );

  test.for([
    ["/v1/chat/completions", chatBody],
    ["/v1/messages", messagesBody],
    ["/v1/responses", responsesBody],
  ] as const)("reach the content exporter as a reasoning part from %s", async ([path, body], ctx) => {
    if (!etcdReachable || !app || !otlp) return ctx.skip();
    const probe = `stream-capture-probe ${path}`;
    await stream(app, path, body("srp-mistral", probe));
    const deadline = Date.now() + 10_000;
    let span: Record<string, unknown> | undefined;
    while (!span && Date.now() < deadline) {
      span = otlp.spans
        .map((s) => s.attributes)
        .find((a) => String(a["gen_ai.input.messages"] ?? "").includes(probe));
      if (!span) await new Promise((r) => setTimeout(r, 50));
    }
    expect(span, `no span for ${probe}; parse failures: ${JSON.stringify(otlp.parseFailures)}`).toBeDefined();
    const output = JSON.parse(String(span!["gen_ai.output.messages"] ?? "null"));
    expect(output?.[0]?.parts, String(span!["gen_ai.output.messages"])).toEqual([
      expect.objectContaining({ type: "reasoning", content: THINKING }),
      expect.objectContaining({ type: "text", content: ANSWER }),
    ]);
  });

  test("count toward the completion tokens estimated for a stream without usage", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const outputTokens = async () => {
      const scrape = await fetch(`${app!.metricsUrl}/metrics`).then((r) => r.text());
      return scrape
        .split("\n")
        .filter((l) => l.startsWith("aisix_llm_output_tokens_total{") && l.includes('model="srp-no-usage"'))
        .reduce((sum, l) => sum + Number(l.slice(l.lastIndexOf(" ") + 1)), 0);
    };
    const before = await outputTokens();
    await stream(app, "/v1/chat/completions", chatBody("srp-no-usage"));
    const deadline = Date.now() + 10_000;
    let after = before;
    while (after === before && Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 100));
      after = await outputTokens();
    }
    // The answer text alone is a couple of tokens; the reasoning is hundreds.
    expect(after - before).toBeGreaterThan(100);
  });
});
