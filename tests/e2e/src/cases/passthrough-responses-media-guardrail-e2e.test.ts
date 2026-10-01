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

// E2E (AISIX-Cloud#1262): a detected Responses passthrough may relay image
// bytes verbatim, but it must not copy them to an external output guardrail.
// This starts the real AISIX binary and etcd plus two real local HTTP peers:
// the passthrough upstream and an OpenAI Moderation-compatible guardrail
// endpoint. The latter records the body AISIX actually sends it.

const CALLER = "sk-passthrough-responses-media";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const BUFFERED_MEDIA = "buffered-media-sentinel-not-for-guardrail";
const BUFFERED_VISIBLE = "buffered-visible-text-for-guardrail";
const STREAM_MEDIA = "stream-media-sentinel-not-for-guardrail";
const STREAM_VISIBLE = "stream-visible-text-for-guardrail";

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
      try {
        const body = JSON.parse(raw) as { input?: unknown };
        if (typeof body.input === "string") inputs.push(body.input);
      } catch {
        // The test asserts only requests that follow the moderation wire
        // contract. A malformed request still receives a well-formed answer
        // so the gateway's own failure path remains observable in CI.
      }
      res.statusCode = 200;
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ results: [{ flagged: false }] }));
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
  id: "resp_media_buffered",
  object: "response",
  output: [
    {
      type: "image_generation_call",
      id: "ig_media_buffered",
      status: "completed",
      result: BUFFERED_MEDIA,
    },
    {
      type: "message",
      content: [{ type: "output_text", text: BUFFERED_VISIBLE }],
    },
  ],
};

const streamedResponse = [
  `data: ${JSON.stringify({
    type: "response.image_generation_call.partial_image",
    item_id: "ig_media_stream",
    output_index: 0,
    partial_image_index: 0,
    partial_image_b64: STREAM_MEDIA,
  })}\n\n`,
  `data: ${JSON.stringify({
    type: "response.output_text.delta",
    item_id: "msg_media_stream",
    output_index: 1,
    content_index: 0,
    delta: STREAM_VISIBLE,
  })}\n\n`,
  `data: ${JSON.stringify({
    type: "response.completed",
    response: {
      id: "resp_media_stream",
      object: "response",
      status: "completed",
      output: [
        { type: "image_generation_call", id: "ig_media_stream", result: STREAM_MEDIA },
        { type: "message", content: [{ type: "output_text", text: STREAM_VISIBLE }] },
      ],
    },
  })}\n\n`,
  "data: [DONE]\n\n",
];

describe("Responses passthrough keeps generated media out of output guardrails", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let bufferedUpstream: OpenAiUpstream | undefined;
  let streamUpstream: OpenAiUpstream | undefined;
  let moderation: ModerationSink | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    moderation = await startModerationSink();
    bufferedUpstream = await startOpenAiUpstream({ nonStreamBody: bufferedResponse });
    streamUpstream = await startOpenAiUpstream({ rawStreamFrames: streamedResponse });
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const providerKey = await seed.createProviderKey({
      display_name: "passthrough-responses-media-pk",
      secret: "sk-mock",
      api_base: bufferedUpstream.baseUrl,
    });
    await seed.createPassthroughRoute({
      name: "passthrough-responses-media-buffered",
      path_prefix: "/responses-media-buffered",
      target_url: bufferedUpstream.baseUrl,
      provider_key_id: providerKey.id,
    });
    await seed.createPassthroughRoute({
      name: "passthrough-responses-media-stream",
      path_prefix: "/responses-media-stream",
      target_url: streamUpstream.baseUrl,
      provider_key_id: providerKey.id,
    });
    await seed.createGuardrail({
      name: "passthrough-responses-media-output",
      enabled: true,
      hook_point: "output",
      kind: "openai_moderation",
      api_key: "sk-local-moderation",
      endpoint: moderation.baseUrl,
      output_fail_open: false,
    });
    // Seeded last: the successful auth gate proves every resource above has
    // crossed this app's etcd watch before either privacy assertion runs.
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
    await moderation?.close();
  });

  const request = (route: string, stream: boolean) =>
    fetch(`${app!.proxyUrl}/${route}/v1/responses`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({ model: "gpt-4o-mini", input: "go", stream }),
    });

  const expectExternalGuardrailText = (inputs: string[], visible: string, media: string) => {
    expect(inputs.length, "the external output guardrail was invoked").toBeGreaterThan(0);
    expect(inputs.every((input) => !input.includes(media)), inputs.join("\n")).toBe(true);
    expect(inputs.some((input) => input.includes(visible)), inputs.join("\n")).toBe(true);
    const visibleOccurrences = inputs.reduce(
      (count, input) => count + input.split(visible).length - 1,
      0,
    );
    expect(visibleOccurrences, `external guardrail input: ${inputs.join("\n")}`).toBe(1);
  };

  test("buffered Responses media stays out of the external guardrail while output text is scanned", async (ctx) => {
    if (!etcdReachable || !app || !bufferedUpstream || !moderation) {
      ctx.skip();
      return;
    }
    const upstreamBefore = bufferedUpstream.receivedRequests.length;
    const moderationBefore = moderation.inputs.length;
    const response = await request("responses-media-buffered", false);
    const body = await response.text();
    expect(response.status, body).toBe(200);
    expect(body).toContain(BUFFERED_MEDIA);
    expect(body).toContain(BUFFERED_VISIBLE);
    expect(bufferedUpstream.receivedRequests.length).toBe(upstreamBefore + 1);
    expectExternalGuardrailText(
      moderation.inputs.slice(moderationBefore),
      BUFFERED_VISIBLE,
      BUFFERED_MEDIA,
    );
  });

  test("streamed Responses partial-image media stays out of the external guardrail while text is scanned", async (ctx) => {
    if (!etcdReachable || !app || !streamUpstream || !moderation) {
      ctx.skip();
      return;
    }
    const upstreamBefore = streamUpstream.receivedRequests.length;
    const moderationBefore = moderation.inputs.length;
    const response = await request("responses-media-stream", true);
    const body = await response.text();
    expect(response.status, body).toBe(200);
    expect(body).toContain(STREAM_MEDIA);
    expect(body).toContain(STREAM_VISIBLE);
    expect(streamUpstream.receivedRequests.length).toBe(upstreamBefore + 1);
    expectExternalGuardrailText(
      moderation.inputs.slice(moderationBefore),
      STREAM_VISIBLE,
      STREAM_MEDIA,
    );
  });
});
