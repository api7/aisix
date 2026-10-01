import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  pickFreePort,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// A real AISIX gateway relays Chat output to a local upstream and sends its
// selected output text to a separate OpenAI Moderation-compatible peer. The
// peer records exactly what AISIX sends across the external guardrail boundary.

const CALLER = "sk-passthrough-chat-media";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const BUFFERED_IMAGE = "buffered-image-media-sentinel";
const BUFFERED_AUDIO = "buffered-audio-media-sentinel";
const BUFFERED_FILE = "buffered-file-media-sentinel";
const BUFFERED_OPAQUE = "buffered-opaque-part-sentinel";
const BUFFERED_REASONING = "buffered-reasoning-sentinel";
const BUFFERED_VISIBLE = "buffered-visible-text-sentinel";
const BUFFERED_TOOL = "buffered-tool-arguments-sentinel";
const STREAM_IMAGE = "stream-image-media-sentinel";
const STREAM_AUDIO = "stream-audio-media-sentinel";
const STREAM_FILE = "stream-file-media-sentinel";
const STREAM_OPAQUE = "stream-opaque-part-sentinel";
const STREAM_REASONING = "stream-reasoning-sentinel";
const STREAM_VISIBLE = "stream-visible-text-sentinel";
const STREAM_TOOL = "stream-tool-arguments-sentinel";
const BUFFERED_MESSAGE_REFUSAL = "buffered-message-refusal-BLOCKME";
const BUFFERED_CONTENT_REFUSAL = "buffered-content-refusal-BLOCKME";
const STREAM_REFUSAL = "stream-refusal-BLOCKME";
// `openai_moderation` uses the default whole-stream hold cap (256 KiB).
const STREAM_REFUSAL_CAP = 262_144;
const STREAM_OVERSIZED_REFUSAL = `stream-oversized-refusal-${"x".repeat(STREAM_REFUSAL_CAP + 1)}`;

interface ModerationSink {
  baseUrl: string;
  inputs: string[];
  close(): Promise<void>;
}

async function startModerationSink(): Promise<ModerationSink> {
  const inputs: string[] = [];
  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (chunk: Buffer) => (raw += chunk.toString("utf8")));
    req.on("end", () => {
      let input: string | undefined;
      try {
        const body = JSON.parse(raw) as { input?: unknown };
        if (typeof body.input === "string") {
          input = body.input;
          inputs.push(input);
        }
      } catch {
        // Reply normally so a malformed moderation request remains visible
        // through the gateway's own output-policy behavior.
      }
      const flagged = input?.includes("BLOCKME") ?? false;
      res.statusCode = 200;
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          results: [{ flagged, categories: flagged ? { refusal: true } : {} }],
        }),
      );
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, "127.0.0.1", resolve);
  });
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    inputs,
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

const bufferedResponse = {
  id: "chat_media_buffered",
  object: "chat.completion",
  model: "gpt-4o-mini",
  choices: [
    {
      index: 0,
      message: {
        role: "assistant",
        content: [
          { type: "image_url", image_url: { url: BUFFERED_IMAGE } },
          { type: "input_audio", input_audio: { data: BUFFERED_AUDIO } },
          { type: "file", file: { file_data: BUFFERED_FILE } },
          { type: "future_media", text: BUFFERED_OPAQUE },
          { type: "text", text: BUFFERED_VISIBLE },
        ],
        reasoning_content: BUFFERED_REASONING,
        tool_calls: [
          {
            id: "call_media_buffered",
            type: "function",
            function: { name: "lookup", arguments: BUFFERED_TOOL },
          },
        ],
      },
      finish_reason: "tool_calls",
    },
  ],
};

const streamedResponse = [
  `data: ${JSON.stringify({
    id: "chat_media_stream",
    object: "chat.completion.chunk",
    model: "gpt-4o-mini",
    choices: [
      {
        index: 0,
        delta: {
          role: "assistant",
          content: [
            { index: 0, type: "image_url", image_url: { url: STREAM_IMAGE } },
            { index: 1, type: "input_audio", input_audio: { data: STREAM_AUDIO } },
            { index: 2, type: "file", file: { file_data: STREAM_FILE } },
            { index: 3, type: "future_media", text: STREAM_OPAQUE },
            { index: 4, type: "text", text: STREAM_VISIBLE },
          ],
          reasoning_content: STREAM_REASONING,
          tool_calls: [
            {
              index: 0,
              id: "call_media_stream",
              type: "function",
              function: { name: "lookup", arguments: STREAM_TOOL },
            },
          ],
        },
      },
    ],
  })}\n\n`,
  "data: [DONE]\n\n",
];

const bufferedMessageRefusalResponse = {
  id: "chat_message_refusal_buffered",
  object: "chat.completion",
  model: "gpt-4o-mini",
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: null, refusal: BUFFERED_MESSAGE_REFUSAL },
      finish_reason: "content_filter",
    },
  ],
};

const bufferedContentRefusalResponse = {
  id: "chat_content_refusal_buffered",
  object: "chat.completion",
  model: "gpt-4o-mini",
  choices: [
    {
      index: 0,
      message: {
        role: "assistant",
        content: [{ type: "refusal", refusal: BUFFERED_CONTENT_REFUSAL }],
      },
      finish_reason: "content_filter",
    },
  ],
};

const streamedRefusalResponse = [
  `data: ${JSON.stringify({
    id: "chat_refusal_stream",
    object: "chat.completion.chunk",
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta: { role: "assistant", refusal: STREAM_REFUSAL } }],
  })}\n\n`,
  "data: [DONE]\n\n",
];

const streamedOversizedRefusalResponse = [
  `data: ${JSON.stringify({
    id: "chat_oversized_refusal_stream",
    object: "chat.completion.chunk",
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta: { role: "assistant", refusal: STREAM_OVERSIZED_REFUSAL } }],
  })}\n\n`,
  "data: [DONE]\n\n",
];

const streamedOversizedContentRefusalResponse = [
  `data: ${JSON.stringify({
    id: "chat_oversized_content_refusal_stream",
    object: "chat.completion.chunk",
    model: "gpt-4o-mini",
    choices: [
      {
        index: 0,
        delta: {
          role: "assistant",
          content: [{ index: 0, type: "refusal", refusal: STREAM_OVERSIZED_REFUSAL }],
        },
      },
    ],
  })}\n\n`,
  "data: [DONE]\n\n",
];

const streamedOversizedLegacyToolResponse = [
  `data: ${JSON.stringify({
    id: "chat_oversized_legacy_tool_stream",
    object: "chat.completion.chunk",
    model: "gpt-4o-mini",
    choices: [
      {
        index: 0,
        delta: {
          role: "assistant",
          function_call: { name: "lookup", arguments: STREAM_OVERSIZED_REFUSAL },
        },
      },
    ],
  })}\n\n`,
  "data: [DONE]\n\n",
];

describe("Chat passthrough keeps media out of external output guardrails", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let bufferedUpstream: OpenAiUpstream | undefined;
  let streamUpstream: OpenAiUpstream | undefined;
  let bufferedMessageRefusalUpstream: OpenAiUpstream | undefined;
  let bufferedContentRefusalUpstream: OpenAiUpstream | undefined;
  let streamRefusalUpstream: OpenAiUpstream | undefined;
  let streamOversizedRefusalUpstream: OpenAiUpstream | undefined;
  let streamOversizedContentRefusalUpstream: OpenAiUpstream | undefined;
  let streamOversizedLegacyToolUpstream: OpenAiUpstream | undefined;
  let moderation: ModerationSink | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    moderation = await startModerationSink();
    bufferedUpstream = await startOpenAiUpstream({ nonStreamBody: bufferedResponse });
    streamUpstream = await startOpenAiUpstream({ rawStreamFrames: streamedResponse });
    bufferedMessageRefusalUpstream = await startOpenAiUpstream({
      nonStreamBody: bufferedMessageRefusalResponse,
    });
    bufferedContentRefusalUpstream = await startOpenAiUpstream({
      nonStreamBody: bufferedContentRefusalResponse,
    });
    streamRefusalUpstream = await startOpenAiUpstream({ rawStreamFrames: streamedRefusalResponse });
    streamOversizedRefusalUpstream = await startOpenAiUpstream({
      rawStreamFrames: streamedOversizedRefusalResponse,
    });
    streamOversizedContentRefusalUpstream = await startOpenAiUpstream({
      rawStreamFrames: streamedOversizedContentRefusalResponse,
    });
    streamOversizedLegacyToolUpstream = await startOpenAiUpstream({
      rawStreamFrames: streamedOversizedLegacyToolResponse,
    });
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const providerKey = await seed.createProviderKey({
      display_name: "passthrough-chat-media-pk",
      secret: "sk-mock",
      api_base: bufferedUpstream.baseUrl,
    });
    await seed.createPassthroughRoute({
      name: "passthrough-chat-media-buffered",
      path_prefix: "/chat-media-buffered",
      target_url: bufferedUpstream.baseUrl,
      provider_key_id: providerKey.id,
    });
    await seed.createPassthroughRoute({
      name: "passthrough-chat-media-stream",
      path_prefix: "/chat-media-stream",
      target_url: streamUpstream.baseUrl,
      provider_key_id: providerKey.id,
    });
    await seed.createPassthroughRoute({
      name: "passthrough-chat-refusal-message",
      path_prefix: "/chat-refusal-message",
      target_url: bufferedMessageRefusalUpstream.baseUrl,
      provider_key_id: providerKey.id,
    });
    await seed.createPassthroughRoute({
      name: "passthrough-chat-refusal-content",
      path_prefix: "/chat-refusal-content",
      target_url: bufferedContentRefusalUpstream.baseUrl,
      provider_key_id: providerKey.id,
    });
    await seed.createPassthroughRoute({
      name: "passthrough-chat-refusal-stream",
      path_prefix: "/chat-refusal-stream",
      target_url: streamRefusalUpstream.baseUrl,
      provider_key_id: providerKey.id,
    });
    await seed.createPassthroughRoute({
      name: "passthrough-chat-refusal-stream-oversized",
      path_prefix: "/chat-refusal-stream-oversized",
      target_url: streamOversizedRefusalUpstream.baseUrl,
      provider_key_id: providerKey.id,
    });
    await seed.createPassthroughRoute({
      name: "passthrough-chat-refusal-stream-oversized-content",
      path_prefix: "/chat-refusal-stream-oversized-content",
      target_url: streamOversizedContentRefusalUpstream.baseUrl,
      provider_key_id: providerKey.id,
    });
    await seed.createPassthroughRoute({
      name: "passthrough-chat-legacy-tool-stream-oversized",
      path_prefix: "/chat-legacy-tool-stream-oversized",
      target_url: streamOversizedLegacyToolUpstream.baseUrl,
      provider_key_id: providerKey.id,
    });
    await seed.createGuardrail({
      name: "passthrough-chat-media-output",
      enabled: true,
      hook_point: "output",
      kind: "openai_moderation",
      api_key: "sk-local-moderation",
      endpoint: moderation.baseUrl,
      output_fail_open: false,
    });
    // Seed the authentication gate last, then wait for it through the real
    // models endpoint so the routes and output guardrail are already active.
    await seed.createApiKey({
      key_hash: CALLER_HASH,
      allowed_models: [],
      allowed_routes: ["*"],
    });
    const proxy = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  }, 120_000);

  afterAll(async () => {
    await app?.exit();
    await bufferedUpstream?.close();
    await streamUpstream?.close();
    await bufferedMessageRefusalUpstream?.close();
    await bufferedContentRefusalUpstream?.close();
    await streamRefusalUpstream?.close();
    await streamOversizedRefusalUpstream?.close();
    await streamOversizedContentRefusalUpstream?.close();
    await streamOversizedLegacyToolUpstream?.close();
    await moderation?.close();
  });

  const request = (route: string, stream: boolean) =>
    fetch(`${app!.proxyUrl}/${route}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "go" }],
        stream,
      }),
    });

  const expectExternalGuardrailText = (
    inputs: string[],
    visible: string,
    tool: string,
    opaque: string[],
  ) => {
    expect(inputs.length, "the external output guardrail was invoked").toBeGreaterThan(0);
    expect(inputs.some((input) => input.includes(visible)), inputs.join("\n")).toBe(true);
    expect(inputs.some((input) => input.includes(tool)), inputs.join("\n")).toBe(true);
    const visibleOccurrences = inputs.reduce(
      (count, input) => count + input.split(visible).length - 1,
      0,
    );
    expect(visibleOccurrences, `external guardrail input: ${inputs.join("\n")}`).toBe(1);
    for (const value of opaque) {
      expect(inputs.every((input) => !input.includes(value)), inputs.join("\n")).toBe(true);
    }
  };

  const expectBlockedRefusal = async (
    route: string,
    stream: boolean,
    upstream: OpenAiUpstream,
    refusal: string,
  ) => {
    const upstreamBefore = upstream.receivedRequests.length;
    const moderationBefore = moderation!.inputs.length;
    const response = await request(route, stream);
    const body = await response.text();
    expect(response.status, body).toBe(stream ? 200 : 422);
    expect(body).toContain("content_filter");
    if (stream) expect(body).toContain("event: error");
    expect(body).not.toContain(refusal);
    expect(upstream.receivedRequests.length).toBe(upstreamBefore + 1);
    const inputs = moderation!.inputs.slice(moderationBefore);
    expect(inputs.some((input) => input.includes(refusal)), inputs.join("\n")).toBe(true);
  };

  test("buffered Chat media stays out of the external guardrail while text and tools are scanned", async (ctx) => {
    if (!etcdReachable || !app || !bufferedUpstream || !moderation) {
      ctx.skip();
      return;
    }
    const upstreamBefore = bufferedUpstream.receivedRequests.length;
    const moderationBefore = moderation.inputs.length;
    const response = await request("chat-media-buffered", false);
    const body = await response.text();
    expect(response.status, body).toBe(200);
    for (const value of [
      BUFFERED_IMAGE,
      BUFFERED_AUDIO,
      BUFFERED_FILE,
      BUFFERED_OPAQUE,
      BUFFERED_REASONING,
      BUFFERED_VISIBLE,
      BUFFERED_TOOL,
    ]) {
      expect(body).toContain(value);
    }
    expect(bufferedUpstream.receivedRequests.length).toBe(upstreamBefore + 1);
    expectExternalGuardrailText(
      moderation.inputs.slice(moderationBefore),
      BUFFERED_VISIBLE,
      BUFFERED_TOOL,
      [BUFFERED_IMAGE, BUFFERED_AUDIO, BUFFERED_FILE, BUFFERED_OPAQUE, BUFFERED_REASONING],
    );
  });

  test("streamed Chat media stays out of the external guardrail while text and tools are scanned", async (ctx) => {
    if (!etcdReachable || !app || !streamUpstream || !moderation) {
      ctx.skip();
      return;
    }
    const upstreamBefore = streamUpstream.receivedRequests.length;
    const moderationBefore = moderation.inputs.length;
    const response = await request("chat-media-stream", true);
    const body = await response.text();
    expect(response.status, body).toBe(200);
    for (const value of [
      STREAM_IMAGE,
      STREAM_AUDIO,
      STREAM_FILE,
      STREAM_OPAQUE,
      STREAM_REASONING,
      STREAM_VISIBLE,
      STREAM_TOOL,
    ]) {
      expect(body).toContain(value);
    }
    expect(streamUpstream.receivedRequests.length).toBe(upstreamBefore + 1);
    expectExternalGuardrailText(
      moderation.inputs.slice(moderationBefore),
      STREAM_VISIBLE,
      STREAM_TOOL,
      [STREAM_IMAGE, STREAM_AUDIO, STREAM_FILE, STREAM_OPAQUE, STREAM_REASONING],
    );
  });

  test("buffered Chat refusals are blocked by the external output guardrail", async (ctx) => {
    if (
      !etcdReachable ||
      !app ||
      !bufferedMessageRefusalUpstream ||
      !bufferedContentRefusalUpstream ||
      !moderation
    ) {
      ctx.skip();
      return;
    }
    await expectBlockedRefusal(
      "chat-refusal-message",
      false,
      bufferedMessageRefusalUpstream,
      BUFFERED_MESSAGE_REFUSAL,
    );
    await expectBlockedRefusal(
      "chat-refusal-content",
      false,
      bufferedContentRefusalUpstream,
      BUFFERED_CONTENT_REFUSAL,
    );
  });

  test("streamed Chat refusal deltas are blocked by the external output guardrail", async (ctx) => {
    if (!etcdReachable || !app || !streamRefusalUpstream || !moderation) {
      ctx.skip();
      return;
    }
    await expectBlockedRefusal("chat-refusal-stream", true, streamRefusalUpstream, STREAM_REFUSAL);
  });

  test("streamed Chat refusal deltas count toward the output hold cap", async (ctx) => {
    if (!etcdReachable || !app || !streamOversizedRefusalUpstream || !moderation) {
      ctx.skip();
      return;
    }
    const upstreamBefore = streamOversizedRefusalUpstream.receivedRequests.length;
    const moderationBefore = moderation.inputs.length;
    const response = await request("chat-refusal-stream-oversized", true);
    const body = await response.text();
    const bodyExcerpt = body.slice(0, 512);
    expect(response.status, bodyExcerpt).toBe(200);
    expect(body.includes("output_buffer_exceeded"), bodyExcerpt).toBe(true);
    expect(body.includes(STREAM_OVERSIZED_REFUSAL), bodyExcerpt).toBe(false);
    expect(streamOversizedRefusalUpstream.receivedRequests.length).toBe(upstreamBefore + 1);
    expect(moderation.inputs.slice(moderationBefore)).toHaveLength(0);
  });

  test("streamed typed Chat refusals count toward the output hold cap", async (ctx) => {
    if (!etcdReachable || !app || !streamOversizedContentRefusalUpstream || !moderation) {
      ctx.skip();
      return;
    }
    const upstreamBefore = streamOversizedContentRefusalUpstream.receivedRequests.length;
    const moderationBefore = moderation.inputs.length;
    const response = await request("chat-refusal-stream-oversized-content", true);
    const body = await response.text();
    const bodyExcerpt = body.slice(0, 512);
    expect(response.status, bodyExcerpt).toBe(200);
    expect(body.includes("output_buffer_exceeded"), bodyExcerpt).toBe(true);
    expect(body.includes(STREAM_OVERSIZED_REFUSAL), bodyExcerpt).toBe(false);
    expect(streamOversizedContentRefusalUpstream.receivedRequests.length).toBe(upstreamBefore + 1);
    expect(moderation.inputs.slice(moderationBefore)).toHaveLength(0);
  });

  test("streamed legacy Chat tool arguments count toward the output hold cap", async (ctx) => {
    if (!etcdReachable || !app || !streamOversizedLegacyToolUpstream || !moderation) {
      ctx.skip();
      return;
    }
    const upstreamBefore = streamOversizedLegacyToolUpstream.receivedRequests.length;
    const moderationBefore = moderation.inputs.length;
    const response = await request("chat-legacy-tool-stream-oversized", true);
    const body = await response.text();
    const bodyExcerpt = body.slice(0, 512);
    expect(response.status, bodyExcerpt).toBe(200);
    expect(body.includes("output_buffer_exceeded"), bodyExcerpt).toBe(true);
    expect(body.includes(STREAM_OVERSIZED_REFUSAL), bodyExcerpt).toBe(false);
    expect(streamOversizedLegacyToolUpstream.receivedRequests.length).toBe(upstreamBefore + 1);
    expect(moderation.inputs.slice(moderationBefore)).toHaveLength(0);
  });
});
