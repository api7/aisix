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
// request without the image while the caller sees a 200. A URL goes up as a
// reference where the upstream reads that URL itself; otherwise the gateway
// downloads it and sends the bytes inline. The download reaches only
// publicly routable addresses unless the host is in
// `upstream.media_fetch.allowed_hosts`: the media server below listens on
// loopback, so `localhost:<port>` is allow-listed and `127.0.0.1:<port>`,
// the same server, is not.
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

const PNG = Buffer.from(PNG_B64, "base64");
const PDF = Buffer.from(PDF_B64, "base64");
const MAX_BYTES = 1024;

interface MediaServer {
  port: number;
  hits: Map<string, number>;
  close(): Promise<void>;
}

/** Serves media over plain http on every loopback address. */
async function startMediaServer(): Promise<MediaServer> {
  const hits = new Map<string, number>();
  let port = 0;
  const server: Server = createServer((req, res) => {
    const path = (req.url ?? "/").split("?")[0];
    hits.set(path, (hits.get(path) ?? 0) + 1);
    const send = (status: number, type: string, body: Buffer | string) => {
      res.statusCode = status;
      res.setHeader("content-type", type);
      res.end(body);
    };
    switch (path) {
      case "/cat.png":
        return send(200, "image/png", PNG);
      case "/untyped":
        return send(200, "application/octet-stream", PNG);
      case "/doc.pdf":
        return send(200, "application/pdf", PDF);
      case "/big.png":
        return send(200, "image/png", Buffer.alloc(MAX_BYTES * 2, 1));
      case "/to-private.png":
        res.statusCode = 302;
        res.setHeader("location", `http://127.0.0.1:${port}/cat.png`);
        return res.end();
      case "/to-allowed.png":
        res.statusCode = 302;
        res.setHeader("location", `http://localhost:${port}/cat.png`);
        return res.end();
      default:
        return send(404, "text/plain", "not found");
    }
  });
  await new Promise<void>((resolve) => server.listen(0, resolve));
  const addr = server.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  port = addr.port;
  return {
    port,
    hits,
    close: () =>
      new Promise<void>((resolve, reject) =>
        server.close((e) => (e ? reject(e) : resolve())),
      ),
  };
}

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
  let media: MediaServer | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startUpstream();
    media = await startMediaServer();
    app = await spawnApp({
      extra: {
        upstream: {
          media_fetch: { allowed_hosts: [`localhost:${media.port}`], max_bytes: MAX_BYTES },
        },
      },
    });
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
    await media?.close();
  });

  /** A URL on the media server through the allow-listed name. */
  const allowed = (path: string) => `http://localhost:${media!.port}${path}`;
  /** The same server through an address that is not allow-listed. */
  const unlisted = (path: string) => `http://127.0.0.1:${media!.port}${path}`;
  const hits = (path: string) => media!.hits.get(path) ?? 0;

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
    if (!etcdReachable || !app || !upstream || !media) {
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
    expect(await refused("media-gemini", user(image("gs://bucket/cat")))).toMatch(
      /^messages\[0\]\.content\[1\]: .*image_url\.format/,
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
    test(`${label}: inline images and downloaded URLs go up as base64`, async (ctx) => {
      if (!ready(ctx)) return;
      const body = await chat(model, [
        text("What color?"),
        image(`data:image/png;base64,${PNG_B64}`),
        image(allowed("/cat.png")),
        { type: "file", file: { file_id: allowed("/doc.pdf") } },
      ]);
      expect(body.messages[0].content).toEqual([
        { type: "text", text: "What color?" },
        { type: "image", source: { type: "base64", media_type: "image/png", data: PNG_B64 } },
        { type: "image", source: { type: "base64", media_type: "image/png", data: PNG_B64 } },
        { type: "document", source: { type: "base64", media_type: "application/pdf", data: PDF_B64 } },
      ]);
      expect(
        await refused(model, [
          { role: "user", content: [text("hi"), image("gs://bucket/cat.png")] },
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
        { role: "user", content: [text("hi"), image("gs://bucket/cat.png")] },
      ]),
    ).toContain("Amazon Bedrock does not read media from a URL");
  });

  test("bedrock converse: a URL is downloaded and sent as bytes, on every entry", async (ctx) => {
    if (!ready(ctx)) return;
    const viaChat = await chat("media-nova", [text("What color?"), image(allowed("/cat.png"))]);
    const viaMessages = await forwarded("/v1/messages", {
      model: "media-nova",
      max_tokens: 16,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "What color?" },
            { type: "image", source: { type: "url", url: allowed("/cat.png") } },
          ],
        },
      ],
    });
    const viaResponses = await forwarded("/v1/responses", {
      model: "media-nova",
      input: [
        {
          role: "user",
          content: [
            { type: "input_text", text: "What color?" },
            { type: "input_image", image_url: allowed("/cat.png") },
          ],
        },
      ],
    });
    for (const body of [viaChat, viaMessages, viaResponses]) {
      expect(body.messages[0].content).toContainEqual({
        image: { format: "png", source: { bytes: PNG_B64 } },
      });
    }
  });

  test("bedrock converse: the media type comes from the body when the URL and server give none", async (ctx) => {
    if (!ready(ctx)) return;
    const body = await chat("media-nova", [text("What color?"), image(allowed("/untyped"))]);
    expect(body.messages[0].content).toContainEqual({
      image: { format: "png", source: { bytes: PNG_B64 } },
    });
  });

  test("bedrock converse: a URL repeated in one request is downloaded once", async (ctx) => {
    if (!ready(ctx)) return;
    const path = "/cat.png";
    const before = hits(path);
    const url = `${allowed(path)}?dedupe=1`;
    const body = await chat("media-nova", [text("twice"), image(url), image(url)]);
    const images = body.messages[0].content.filter((b: any) => b.image);
    expect(images).toHaveLength(2);
    expect(hits(path) - before).toBe(1);
  });

  test("Claude on Amazon Bedrock: the streaming (Converse) rail downloads a URL too", async (ctx) => {
    if (!ready(ctx)) return;
    const before = upstream!.received.length;
    await post("/v1/chat/completions", {
      model: "media-bedrock-claude",
      stream: true,
      messages: [{ role: "user", content: [text("What color?"), image(allowed("/cat.png"))] }],
    });
    // The mock cannot answer in the event-stream framing, so the gateway
    // may retry; what matters is what reached the upstream.
    expect(upstream!.received.length).toBeGreaterThan(before);
    const sent = upstream!.received[before];
    expect(sent.path).toMatch(/converse-stream$/);
    expect(JSON.parse(sent.body).messages[0].content).toContainEqual({
      image: { format: "png", source: { bytes: PNG_B64 } },
    });
  });

  for (const [model, label, pick] of [
    ["media-claude", "anthropic", (b: any) => b.messages[0].content],
    ["media-vertex-claude", "Claude on Vertex AI", (b: any) => b.messages[0].content],
    ["media-gemini", "gemini", (b: any) => b.contents[0].parts],
  ] as const) {
    test(`${label}: the streaming rail downloads a URL too`, async (ctx) => {
      if (!ready(ctx)) return;
      const before = upstream!.received.length;
      await post("/v1/chat/completions", {
        model,
        stream: true,
        messages: [{ role: "user", content: [text("What color?"), image(allowed("/cat.png"))] }],
      });
      // The mock does not answer in SSE framing, so the gateway may retry;
      // what matters is what reached the upstream.
      expect(upstream!.received.length).toBeGreaterThan(before);
      const sent = JSON.parse(upstream!.received[before].body);
      const parts = JSON.stringify(pick(sent));
      expect(parts).toContain(PNG_B64);
      expect(parts).not.toContain(`localhost:${media!.port}`);
    });
  }

  test("anthropic and gemini: an http:// URL is downloaded, an https:// one still goes up as a reference", async (ctx) => {
    if (!ready(ctx)) return;
    const claude = await chat("media-claude", [text("What color?"), image(allowed("/cat.png"))]);
    expect(claude.messages[0].content[1]).toEqual({
      type: "image",
      source: { type: "base64", media_type: "image/png", data: PNG_B64 },
    });
    const gemini = await chat("media-gemini", [
      text("What color?"),
      image(allowed("/cat.png")),
      image(allowed("/untyped")),
    ]);
    expect(gemini.contents[0].parts.slice(1)).toEqual([
      { inlineData: { mimeType: "image/png", data: PNG_B64 } },
      { inlineData: { mimeType: "image/png", data: PNG_B64 } },
    ]);
  });

  test("download: a non-public address is refused unless allow-listed, on every hop", async (ctx) => {
    if (!ready(ctx)) return;
    const user = (url: string) => [{ role: "user", content: [text("hi"), image(url)] }];
    const before = hits("/cat.png");
    for (const model of ["media-nova", "media-bedrock-claude", "media-vertex-claude", "media-claude", "media-gemini"]) {
      const message = await refused(model, user(unlisted("/cat.png")));
      expect(message).toMatch(/^messages\[0\]\.content\[1\]: /);
      expect(message).toContain("upstream.media_fetch.allowed_hosts");
    }
    // Refused before any request left the gateway.
    expect(hits("/cat.png")).toBe(before);

    expect(await refused("media-nova", user(allowed("/to-private.png")))).toContain(
      "upstream.media_fetch.allowed_hosts",
    );
    const followed = await chat("media-nova", [text("hi"), image(allowed("/to-allowed.png"))]);
    expect(followed.messages[0].content).toContainEqual({
      image: { format: "png", source: { bytes: PNG_B64 } },
    });
  });

  test("download: too large, an error status and an unfetchable scheme are 400s", async (ctx) => {
    if (!ready(ctx)) return;
    const user = (url: string) => [{ role: "user", content: [text("hi"), image(url)] }];
    expect(await refused("media-nova", user(allowed("/big.png")))).toMatch(/larger than the 1024-byte limit/);

    const before = hits("/gone.png");
    const missing = await refused("media-nova", user(`${allowed("/gone.png")}?X-Amz-Signature=sekrit`));
    expect(missing).toMatch(/^messages\[0\]\.content\[1\]: .*HTTP 404/);
    // A signed URL's query never comes back.
    expect(missing).not.toContain("sekrit");
    // An answer, not a transport failure: asked once.
    expect(hits("/gone.png") - before).toBe(1);

    expect(await refused("media-nova", user("ftp://files.example/cat.png"))).toContain(
      "does not read media from a URL",
    );
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

describe("chat media URLs with upstream.media_fetch.enabled: false", () => {
  let app: SpawnedApp | undefined;
  let media: MediaServer | undefined;
  let etcdReachable = false;
  const CALLER = "sk-chat-media-fetch-off";

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    media = await startMediaServer();
    app = await spawnApp({
      extra: {
        upstream: {
          media_fetch: { enabled: false, allowed_hosts: [`localhost:${media.port}`] },
        },
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "media-off-bedrock-pk",
      provider: "bedrock",
      adapter: "bedrock",
      secret: JSON.stringify({
        access_key_id: "AKIA-media-e2e",
        secret_access_key: "sk-media-e2e",
        region: "us-west-2",
      }),
      api_base: "http://127.0.0.1:9",
    });
    await seed.createModel({
      display_name: "media-off-nova",
      provider: "bedrock",
      model_name: "amazon.nova-pro-v1:0",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: createHash("sha256").update(CALLER).digest("hex"),
      allowed_models: ["media-off-nova"],
    });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER}` },
      });
      await res.text();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await media?.close();
  });

  test("a URL the upstream cannot read is refused and never downloaded", async (ctx) => {
    if (!etcdReachable || !app || !media) return ctx.skip();
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify({
        model: "media-off-nova",
        messages: [
          {
            role: "user",
            content: [text("hi"), image(`http://localhost:${media.port}/cat.png`)],
          },
        ],
      }),
    });
    const body = await res.text();
    expect(res.status, body).toBe(400);
    expect(JSON.parse(body).error.message).toContain("Amazon Bedrock does not read media from a URL");
    expect(media.hits.get("/cat.png") ?? 0).toBe(0);
  });
});
