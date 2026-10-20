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

// Image, audio and file content parts on the providers whose bridge builds
// its own request shape. Each part reaches the upstream in that upstream's
// own media shape, or the request is refused with a 400 that names the
// part — it is never dropped, which would have the model answer a vision
// request without the image while the caller sees a 200. The gateway never
// fetches a caller's URL: a URL goes up as a reference the upstream reads
// itself, or is refused.
//
// Reference:
//   - <https://platform.openai.com/docs/api-reference/chat/create>
//   - <https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/Content>
//   - <https://docs.anthropic.com/en/docs/build-with-claude/vision>
//   - <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ContentBlock.html>

const CALLER_PLAINTEXT = "sk-chat-media-parts-native";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const PNG_B64 = "iVBORw0KGgo=";
const PDF_B64 = "JVBERi0xLjQ=";
const WAV_B64 = "UklGRiQAAABXQVZF";

const ANTHROPIC_REPLY = {
  id: "msg_media",
  type: "message",
  role: "assistant",
  model: "claude",
  content: [{ type: "text", text: "ok" }],
  stop_reason: "end_turn",
  usage: { input_tokens: 7, output_tokens: 1 },
};
const GEMINI_REPLY = {
  candidates: [
    { content: { role: "model", parts: [{ text: "ok" }] }, finishReason: "STOP" },
  ],
  usageMetadata: { promptTokenCount: 7, candidatesTokenCount: 1, totalTokenCount: 8 },
};
const CONVERSE_REPLY = {
  output: { message: { role: "assistant", content: [{ text: "ok" }] } },
  stopReason: "end_turn",
  usage: { inputTokens: 7, outputTokens: 1, totalTokens: 8 },
  metrics: { latencyMs: 1 },
};

interface RecordingUpstream {
  baseUrl: string;
  received: Array<{ path: string; body: string }>;
  close(): Promise<void>;
}

/** One upstream for every adapter, answering in the shape the path asks
 * for. */
async function startUpstream(): Promise<RecordingUpstream> {
  const received: Array<{ path: string; body: string }> = [];
  const server: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      const path = (req.url ?? "/").split("?")[0];
      received.push({ path, body: Buffer.concat(chunks).toString("utf8") });
      const reply = path.endsWith(":generateContent")
        ? GEMINI_REPLY
        : path.endsWith("/converse")
          ? CONVERSE_REPLY
          : ANTHROPIC_REPLY;
      res.statusCode = 200;
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify(reply));
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

const text = (t: string) => ({ type: "text", text: t });
const image = (url: string, extra: Record<string, unknown> = {}) => ({
  type: "image_url",
  image_url: { url, ...extra },
});
const audio = { type: "input_audio", input_audio: { data: WAV_B64, format: "wav" } };
const pdf = {
  type: "file",
  file: { file_data: `data:application/pdf;base64,${PDF_B64}`, filename: "a.pdf" },
};

describe("chat media content parts → native provider media shapes", () => {
  let app: SpawnedApp | undefined;
  let upstream: RecordingUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startUpstream();
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const anthropicPk = await seed.createProviderKey({
      display_name: "media-anthropic-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-mock",
      api_base: upstream.baseUrl,
    });
    await seed.createModel({
      display_name: "media-claude",
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: anthropicPk.id,
    });

    const vertexPk = await seed.createProviderKey({
      display_name: "media-vertex-pk",
      provider: "google",
      adapter: "vertex",
      secret: JSON.stringify({
        access_token: "ya29.media-e2e",
        project: "proj-e2e",
        region: "us-central1",
      }),
      api_base: upstream.baseUrl,
    });
    await seed.createModel({
      display_name: "media-gemini",
      provider: "google",
      model_name: "gemini-2.5-flash",
      provider_key_id: vertexPk.id,
    });
    await seed.createModel({
      display_name: "media-vertex-claude",
      provider: "google",
      model_name: "claude-3-5-haiku@20241022",
      provider_key_id: vertexPk.id,
    });

    const bedrockPk = await seed.createProviderKey({
      display_name: "media-bedrock-pk",
      provider: "bedrock",
      adapter: "bedrock",
      secret: JSON.stringify({
        access_key_id: "AKIA-media-e2e",
        secret_access_key: "sk-media-e2e",
        region: "us-west-2",
      }),
      api_base: upstream.baseUrl,
    });
    await seed.createModel({
      display_name: "media-nova",
      provider: "bedrock",
      model_name: "amazon.nova-pro-v1:0",
      provider_key_id: bedrockPk.id,
    });
    await seed.createModel({
      display_name: "media-bedrock-claude",
      provider: "bedrock",
      model_name: "anthropic.claude-3-5-haiku-20241022-v1:0",
      provider_key_id: bedrockPk.id,
    });

    // Seeded last, so this key authenticating implies the whole seed set
    // has reached the gateway's snapshot.
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [
        "media-claude",
        "media-gemini",
        "media-vertex-claude",
        "media-nova",
        "media-bedrock-claude",
      ],
    });
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
    await upstream?.close();
  });

  async function post(path: string, body: unknown): Promise<{ status: number; text: string }> {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    });
    return { status: res.status, text: await res.text() };
  }

  /** Sends one user turn through `path` and returns the body the
   * upstream received. */
  async function forwarded(path: string, body: unknown): Promise<Record<string, any>> {
    const before = upstream!.received.length;
    const res = await post(path, body);
    expect(res.status, res.text).toBe(200);
    expect(upstream!.received.length - before).toBe(1);
    return JSON.parse(upstream!.received.at(-1)!.body);
  }

  const chat = (model: string, content: unknown[]) =>
    forwarded("/v1/chat/completions", { model, messages: [{ role: "user", content }] });

  /** Sends a request that must be refused before it reaches the upstream,
   * and returns the error message. */
  async function refused(model: string, messages: unknown[], stream = false): Promise<string> {
    const before = upstream!.received.length;
    const res = await post("/v1/chat/completions", { model, messages, stream });
    expect(res.status, res.text).toBe(400);
    expect(upstream!.received.length - before).toBe(0);
    return JSON.parse(res.text).error.message as string;
  }

  const ready = (ctx: { skip: () => void }) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return false;
    }
    return true;
  };

  test("gemini: inline data and gs:// / https references become inlineData and fileData", async (ctx) => {
    if (!ready(ctx)) return;
    const body = await chat("media-gemini", [
      text("What color?"),
      image(`data:image/png;base64,${PNG_B64}`),
      image("gs://bucket/cat.jpg"),
      image("https://img.example/cat", { format: "image/webp" }),
      audio,
      pdf,
    ]);
    expect(body.contents).toEqual([
      {
        role: "user",
        parts: [
          { text: "What color?" },
          { inlineData: { mimeType: "image/png", data: PNG_B64 } },
          { fileData: { mimeType: "image/jpeg", fileUri: "gs://bucket/cat.jpg" } },
          { fileData: { mimeType: "image/webp", fileUri: "https://img.example/cat" } },
          { inlineData: { mimeType: "audio/wav", data: WAV_B64 } },
          { inlineData: { mimeType: "application/pdf", data: PDF_B64 } },
        ],
      },
    ]);
  });

  test("gemini: an image reaches it through /v1/messages and /v1/responses too", async (ctx) => {
    if (!ready(ctx)) return;
    const viaMessages = await forwarded("/v1/messages", {
      model: "media-gemini",
      max_tokens: 16,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "What color?" },
            { type: "image", source: { type: "base64", media_type: "image/png", data: PNG_B64 } },
          ],
        },
      ],
    });
    const viaResponses = await forwarded("/v1/responses", {
      model: "media-gemini",
      input: [
        {
          role: "user",
          content: [
            { type: "input_text", text: "What color?" },
            { type: "input_image", image_url: `data:image/png;base64,${PNG_B64}` },
          ],
        },
      ],
    });
    for (const body of [viaMessages, viaResponses]) {
      expect(body.contents[0].parts).toContainEqual({
        inlineData: { mimeType: "image/png", data: PNG_B64 },
      });
    }
  });

  test("gemini: parts it cannot take are refused, not dropped", async (ctx) => {
    if (!ready(ctx)) return;
    const user = (part: unknown) => [{ role: "user", content: [text("hi"), part] }];
    expect(await refused("media-gemini", user(image("https://img.example/cat")))).toMatch(
      /^messages\[0\]\.content\[1\]: .*image_url\.format/,
    );
    expect(await refused("media-gemini", user(image("http://img.example/cat.png")))).toMatch(
      /does not fetch/,
    );
    expect(
      await refused("media-gemini", user({ type: "video_url", video_url: { url: "x" } })),
    ).toMatch(/"video_url" is not supported/);
    // Bare base64 where a URL belongs is quoted back cut short, not whole.
    const echoed = await refused("media-gemini", user(image("A".repeat(100_000))));
    expect(echoed.length).toBeLessThan(400);
    expect(
      await refused("media-gemini", [
        { role: "user", content: "look" },
        {
          role: "assistant",
          content: null,
          tool_calls: [
            { id: "call_1", type: "function", function: { name: "snap", arguments: "{}" } },
          ],
        },
        { role: "tool", tool_call_id: "call_1", content: [image(`data:image/png;base64,${PNG_B64}`)] },
      ]),
    ).toMatch(/^messages\[2\]\.content\[0\]: a tool result can carry only text/);
  });

  test("anthropic: images and PDFs become base64 or url sources", async (ctx) => {
    if (!ready(ctx)) return;
    const body = await chat("media-claude", [
      text("What color?"),
      image(`data:image/png;base64,${PNG_B64}`),
      image("https://img.example/cat.png"),
      pdf,
    ]);
    expect(body.messages[0].content).toEqual([
      { type: "text", text: "What color?" },
      { type: "image", source: { type: "base64", media_type: "image/png", data: PNG_B64 } },
      { type: "image", source: { type: "url", url: "https://img.example/cat.png" } },
      { type: "document", source: { type: "base64", media_type: "application/pdf", data: PDF_B64 } },
    ]);
    expect(
      await refused("media-claude", [{ role: "user", content: [text("hi"), audio] }]),
    ).toMatch(/do not accept audio/);
  });

  for (const [model, label] of [
    ["media-vertex-claude", "Claude on Vertex AI"],
    ["media-bedrock-claude", "Claude on Amazon Bedrock"],
  ] as const) {
    test(`${label}: inline images go up as base64, URLs are refused`, async (ctx) => {
      if (!ready(ctx)) return;
      const body = await chat(model, [text("What color?"), image(`data:image/png;base64,${PNG_B64}`)]);
      expect(body.messages[0].content).toEqual([
        { type: "text", text: "What color?" },
        { type: "image", source: { type: "base64", media_type: "image/png", data: PNG_B64 } },
      ]);
      expect(
        await refused(model, [
          { role: "user", content: [text("hi"), image("https://img.example/cat.png")] },
        ]),
      ).toContain(`${label} does not read media from a URL`);
    });
  }

  test("Claude on Amazon Bedrock refuses audio on the streaming (Converse) rail too", async (ctx) => {
    if (!ready(ctx)) return;
    expect(
      await refused("media-bedrock-claude", [{ role: "user", content: [text("hi"), audio] }], true),
    ).toMatch(/do not accept audio/);
  });

  test("bedrock converse: inline media become bytes blocks, URLs are refused", async (ctx) => {
    if (!ready(ctx)) return;
    const body = await chat("media-nova", [
      text("What color?"),
      image(`data:image/png;base64,${PNG_B64}`),
      pdf,
    ]);
    expect(body.messages[0].content).toEqual([
      { text: "What color?" },
      { image: { format: "png", source: { bytes: PNG_B64 } } },
      { document: { format: "pdf", name: "document-0-2", source: { bytes: PDF_B64 } } },
    ]);
    expect(
      await refused("media-nova", [
        { role: "user", content: [text("hi"), image("https://img.example/cat.png")] },
      ]),
    ).toContain("Amazon Bedrock does not read media from a URL");
  });

  test("bedrock converse: a document-only turn gets the text block Converse requires", async (ctx) => {
    if (!ready(ctx)) return;
    const body = await chat("media-nova", [pdf]);
    expect(body.messages[0].content).toEqual([
      { document: { format: "pdf", name: "document-0-0", source: { bytes: PDF_B64 } } },
      { text: "." },
    ]);
  });
});
