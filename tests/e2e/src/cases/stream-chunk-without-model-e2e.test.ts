import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type SpawnedApp,
} from "../harness/index.js";

// E2E (#533): an OpenAI-compatible upstream whose responses omit the
// top-level `model` must still answer through every inbound protocol.
//
// OpenAI itself always sends `model`, but OpenAI-compatible servers and
// proxies in front of them do not all do so — on streamed chunks in
// particular. The caller is answered with the model it addressed, never
// the upstream's echo, so a response that is otherwise well formed must
// not be failed over it: streamed on `/v1/messages` (Anthropic shape out),
// `/v1/chat/completions`, and `/v1/responses` bridged to a
// chat-completions upstream; and unstreamed on chat and embeddings.

const CALLER_PLAINTEXT = "sk-chunk-no-model-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const chunk = (delta: object, finish: string | null = null) =>
  JSON.stringify({
    id: "chatcmpl-no-model",
    object: "chat.completion.chunk",
    created: 1,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });

const STREAM_EVENTS = [
  chunk({ role: "assistant", content: "Hel" }),
  chunk({ content: "lo" }),
  chunk({}, "stop"),
  "[DONE]",
];

const NON_STREAM_CHAT = {
  id: "chatcmpl-no-model",
  object: "chat.completion",
  created: 1,
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: "Hello" },
      finish_reason: "stop",
    },
  ],
  usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
};

const NON_STREAM_EMBEDDING = {
  object: "list",
  data: [{ object: "embedding", index: 0, embedding: [0.25, 0.5] }],
  usage: { prompt_tokens: 1, total_tokens: 1 },
};

describe("upstream responses without `model` (#533)", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  const closers: Array<() => Promise<void>> = [];

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(closers.map((c) => c()));
  });

  // `provider` other than "openai" keeps `/v1/responses` off the verbatim
  // forward, so it is bridged onto the chat-completions stream.
  async function modelBackedBy(
    name: string,
    nonStreamBody?: unknown,
  ): Promise<void> {
    const upstream = await startOpenAiUpstream(
      nonStreamBody ? { nonStreamBody } : { streamEvents: STREAM_EVENTS },
    );
    closers.push(() => upstream.close());
    const pk = await seed!.createProviderKey({
      display_name: `${name}-pk`,
      provider: "deepseek",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed!.createModel({
      display_name: name,
      provider: "deepseek",
      model_name: "deepseek-chat",
      provider_key_id: pk.id,
    });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      if (res.status !== 200) return false;
      const body = (await res.json()) as { data?: Array<{ id?: string }> };
      return (body.data ?? []).some((m) => m.id === name);
    });
  }

  async function streamText(path: string, body: unknown): Promise<string> {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "x-api-key": CALLER_PLAINTEXT,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify(body),
    });
    expect(res.status).toBe(200);
    return res.text();
  }

  function dataPayloads(sse: string): Array<Record<string, unknown>> {
    return sse
      .split("\n")
      .filter((l) => l.startsWith("data: ") && l !== "data: [DONE]")
      .map((l) => JSON.parse(l.slice("data: ".length)));
  }

  const skip = (ctx: { skip: () => void }) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return true;
    }
    return false;
  };

  test("/v1/messages streams the text to message_stop", async (ctx) => {
    if (skip(ctx)) return;
    await modelBackedBy("no-model-chunk-messages");
    const sse = await streamText("/v1/messages", {
      model: "no-model-chunk-messages",
      max_tokens: 16,
      stream: true,
      messages: [{ role: "user", content: "hi" }],
    });
    const events = dataPayloads(sse);
    expect(events.filter((e) => e.type === "error")).toEqual([]);
    const text = events
      .filter((e) => e.type === "content_block_delta")
      .map((e) => (e.delta as { text?: string }).text ?? "")
      .join("");
    expect(text).toBe("Hello");
    expect(events.at(-1)?.type).toBe("message_stop");
  });

  test("/v1/chat/completions streams the text to the finish reason", async (ctx) => {
    if (skip(ctx)) return;
    await modelBackedBy("no-model-chunk-chat");
    const sse = await streamText("/v1/chat/completions", {
      model: "no-model-chunk-chat",
      stream: true,
      messages: [{ role: "user", content: "hi" }],
    });
    const events = dataPayloads(sse);
    expect(events.filter((e) => "error" in e)).toEqual([]);
    type Choice = { delta?: { content?: string }; finish_reason?: string };
    const choices = events.flatMap((e) => (e.choices as Choice[]) ?? []);
    expect(choices.map((c) => c.delta?.content ?? "").join("")).toBe("Hello");
    expect(choices.some((c) => c.finish_reason === "stop")).toBe(true);
  });

  test("/v1/responses (bridged) streams the text to response.completed", async (ctx) => {
    if (skip(ctx)) return;
    await modelBackedBy("no-model-chunk-responses");
    const sse = await streamText("/v1/responses", {
      model: "no-model-chunk-responses",
      stream: true,
      input: "hi",
    });
    const events = dataPayloads(sse);
    const types = events.map((e) => e.type);
    expect(types).not.toContain("response.failed");
    expect(types).not.toContain("error");
    const text = events
      .filter((e) => e.type === "response.output_text.delta")
      .map((e) => e.delta as string)
      .join("");
    expect(text).toBe("Hello");
    expect(types.at(-1)).toBe("response.completed");
  });

  test("/v1/chat/completions answers an unstreamed body", async (ctx) => {
    if (skip(ctx)) return;
    await modelBackedBy("no-model-body-chat", NON_STREAM_CHAT);
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
      },
      body: JSON.stringify({
        model: "no-model-body-chat",
        messages: [{ role: "user", content: "hi" }],
      }),
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      model?: string;
      choices?: Array<{ message?: { content?: string } }>;
    };
    expect(body.choices?.[0]?.message?.content).toBe("Hello");
    expect(body.model).toBe("no-model-body-chat");
  });

  test("/v1/embeddings answers an unstreamed body", async (ctx) => {
    if (skip(ctx)) return;
    await modelBackedBy("no-model-body-embed", NON_STREAM_EMBEDDING);
    const res = await fetch(`${app!.proxyUrl}/v1/embeddings`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
      },
      body: JSON.stringify({ model: "no-model-body-embed", input: "hi" }),
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      model?: string;
      data?: Array<{ embedding?: number[] }>;
    };
    expect(body.data?.[0]?.embedding).toEqual([0.25, 0.5]);
    expect(body.model).toBe("no-model-body-embed");
  });
});
