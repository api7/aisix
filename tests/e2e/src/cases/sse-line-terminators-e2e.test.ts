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

// E2E: an upstream event stream is read by the event-stream rules, whatever
// line ending it uses.
//
// The spec lets a line end in `\r\n`, `\n` or a bare `\r`, and lets the stream
// open with a UTF-8 BOM. An upstream that frames with bare `\r` is a valid
// SSE server, so the gateway has to read it the way a browser `EventSource`
// would: every frame on the typed chat path, and on the native passthrough
// relays every frame restamped with the caller's alias while its bytes keep
// the upstream's own framing.

const CALLER_PLAINTEXT = "sk-sse-line-terminators-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const HEADERS = {
  authorization: `Bearer ${CALLER_PLAINTEXT}`,
  "content-type": "application/json",
};

/** What the provider answers with — never what the caller should see. */
const UPSTREAM_REPORTED_MODEL = "provider-model-v1-20260101";

describe("sse line terminators e2e: a CR-framed upstream is read like any other", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  /** Seed a provider key + model, then the caller key LAST and gate on it. */
  async function seedAlias(
    alias: string,
    upstream: OpenAiUpstream,
    provider: "openai" | "anthropic" = "openai",
  ): Promise<void> {
    const pk = await seed!.createProviderKey({
      display_name: `pk-${alias}`,
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
      provider,
      adapter: provider,
    });
    await seed!.createModel({
      display_name: alias,
      provider,
      model_name: "provider-model-v1",
      provider_key_id: pk.id,
    });
    await seed!.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const r = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: HEADERS.authorization },
      });
      const ok = r.status === 200;
      const body = ok ? ((await r.json()) as { data?: Array<{ id?: string }> }) : undefined;
      if (!ok) await r.text();
      return !!body?.data?.some((m) => m.id === alias);
    });
  }

  test("/v1/chat/completions: every delta of a BOM-opened, CR-framed stream reaches the caller", async (ctx) => {
    if (!etcdReachable || !app || !seed) return void ctx.skip();

    const chunk = (content: string) =>
      `{"id":"chatcmpl-cr","object":"chat.completion.chunk","created":1,"model":"${UPSTREAM_REPORTED_MODEL}","choices":[{"index":0,"delta":{"content":"${content}"},"finish_reason":null}]}`;
    const upstream = await startOpenAiUpstream({
      rawStreamFrames: [
        `﻿data: ${chunk("Hello")}\r\r`,
        `: keep-alive\r\r`,
        `data: ${chunk(", ")}\r\r`,
        `data: ${chunk("world")}\r\r`,
        `data: [DONE]\r\r`,
      ],
      eventDelayMs: 5,
    });
    upstreams.push(upstream);
    await seedAlias("cr-chat", upstream);

    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({
        model: "cr-chat",
        stream: true,
        messages: [{ role: "user", content: "say hello" }],
      }),
    });
    expect(res.status).toBe(200);
    const text = await res.text();
    const content = text
      .split("\n")
      .filter((l) => l.startsWith("data: ") && l !== "data: [DONE]")
      .map((l) => JSON.parse(l.slice("data: ".length)) as {
        choices?: Array<{ delta?: { content?: string } }>;
      })
      .map((c) => c.choices?.[0]?.delta?.content ?? "")
      .join("");
    expect(content).toBe("Hello, world");
    expect(text.trimEnd().endsWith("data: [DONE]")).toBe(true);
  });

  test("/v1/messages native passthrough: a CR-framed stream is restamped and keeps its framing", async (ctx) => {
    if (!etcdReachable || !app || !seed) return void ctx.skip();

    const frames = [
      `event: message_start\rdata: {"type":"message_start","message":{"id":"msg_cr","type":"message","role":"assistant","model":"${UPSTREAM_REPORTED_MODEL}","content":[],"usage":{"input_tokens":4,"output_tokens":0}}}\r\r`,
      `event: content_block_start\rdata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\r\r`,
      `event: content_block_delta\rdata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}\r\r`,
      `event: content_block_stop\rdata: {"type":"content_block_stop","index":0}\r\r`,
      `event: message_delta\rdata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}\r\r`,
      `event: message_stop\rdata: {"type":"message_stop"}\r\r`,
    ];
    const upstream = await startOpenAiUpstream({ rawStreamFrames: frames, eventDelayMs: 5 });
    upstreams.push(upstream);
    await seedAlias("cr-messages", upstream, "anthropic");

    const res = await fetch(`${app.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({
        model: "cr-messages",
        max_tokens: 16,
        stream: true,
        messages: [{ role: "user", content: "say ok" }],
      }),
    });
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).toContain(`"model":"cr-messages"`);
    expect(text).not.toContain(UPSTREAM_REPORTED_MODEL);
    // Byte-for-byte the upstream's stream apart from the one spliced value.
    expect(text).toBe(frames.join("").replace(UPSTREAM_REPORTED_MODEL, "cr-messages"));
  });

  test("/v1/responses native passthrough: every snapshot frame of a CR-framed stream is restamped", async (ctx) => {
    if (!etcdReachable || !app || !seed) return void ctx.skip();

    const snapshot = (status: string) =>
      `"id":"resp_cr","object":"response","status":"${status}","model":"${UPSTREAM_REPORTED_MODEL}"`;
    const upstream = await startOpenAiUpstream({
      rawStreamFrames: [
        `event: response.created\rdata: {"type":"response.created","response":{${snapshot("in_progress")}}}\r\r`,
        `event: response.in_progress\rdata: {"type":"response.in_progress","response":{${snapshot("in_progress")}}}\r\r`,
        `event: response.output_text.delta\rdata: {"type":"response.output_text.delta","delta":"ok"}\r\r`,
        `event: response.completed\rdata: {"type":"response.completed","response":{${snapshot("completed")},"usage":{"input_tokens":4,"output_tokens":2,"total_tokens":6}}}\r\r`,
        `data: [DONE]\r\r`,
      ],
      eventDelayMs: 5,
    });
    upstreams.push(upstream);
    await seedAlias("cr-responses", upstream);

    const res = await fetch(`${app.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({ model: "cr-responses", input: "say ok", stream: true }),
    });
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).not.toContain(UPSTREAM_REPORTED_MODEL);
    expect(text.split(`"model":"cr-responses"`).length - 1).toBe(3);
    expect(text).toContain('"delta":"ok"');
    expect(text.endsWith("data: [DONE]\r\r")).toBe(true);
  });
});
