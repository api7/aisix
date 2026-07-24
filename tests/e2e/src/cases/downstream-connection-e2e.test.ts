import { createHash } from "node:crypto";
import { connect, type Socket } from "node:net";
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

// E2E for AISIX-Cloud#1126: the inbound half of the connection layer.
//
//   - `downstream.idle_timeout_secs` closes a client connection that sits
//     idle *between* requests. The whole point of putting it on hyper's
//     header-read timer is that it must never touch a request in flight,
//     however long the model takes, so both halves are asserted here.
//   - `downstream.sse_keepalive_interval_secs` keeps bytes moving on a
//     streaming response while the model produces nothing, so a proxy in
//     front doesn't call the connection abandoned. Asserted on all three
//     SSE shapes, which reach the client through three different code
//     paths: axum's `Sse` keep-alive (chat), the bridged Anthropic
//     encoder (messages), and the raw byte passthrough (responses).

const CALLER_PLAINTEXT = "sk-downstream-conn-1126";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const IDLE_TIMEOUT_S = 2;
// Comfortably longer than the idle timeout: a request in flight for this
// long must survive, which is exactly what the naive "close anything quiet"
// implementation would get wrong.
const UPSTREAM_THINK_MS = IDLE_TIMEOUT_S * 1000 + 1500;

const HEARTBEAT_INTERVAL_S = 1;
const FIRST_EVENT_DELAY_MS = 2500;

function chatChunk(content: string): string {
  return JSON.stringify({
    id: "evt",
    object: "chat.completion.chunk",
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta: { content }, finish_reason: null }],
  });
}

function chatFinish(): string {
  return JSON.stringify({
    id: "evt",
    object: "chat.completion.chunk",
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
  });
}

const RESPONSES_EVENTS = [
  JSON.stringify({ type: "response.created", response: { id: "resp_1126" } }),
  JSON.stringify({ type: "response.output_text.delta", delta: "late" }),
  JSON.stringify({
    type: "response.completed",
    response: {
      id: "resp_1126",
      status: "completed",
      usage: { input_tokens: 1, output_tokens: 1 },
    },
  }),
  "[DONE]",
];

/**
 * Send one keep-alive request on a raw socket, then report how long the
 * server left the connection open afterwards. Resolves `null` if the
 * connection was still open when `waitMs` elapsed.
 */
function idleUntilClose(port: number, waitMs: number): Promise<number | null> {
  return new Promise((resolve, reject) => {
    const socket: Socket = connect(port, "127.0.0.1");
    let responded = false;
    let idleStartedAt = 0;
    const timer = setTimeout(() => {
      socket.destroy();
      resolve(null);
    }, waitMs);

    socket.on("connect", () => {
      socket.write("GET /livez HTTP/1.1\r\nHost: e2e\r\nConnection: keep-alive\r\n\r\n");
    });
    socket.on("data", (buf) => {
      if (responded) return;
      responded = true;
      expect(buf.toString("utf8")).toContain("HTTP/1.1 200");
      idleStartedAt = Date.now();
    });
    socket.on("close", () => {
      clearTimeout(timer);
      if (!responded) {
        reject(new Error("connection closed before the response arrived"));
        return;
      }
      resolve(Date.now() - idleStartedAt);
    });
    socket.on("error", (err) => {
      clearTimeout(timer);
      reject(err);
    });
  });
}

/** Read a streaming response body to completion as text. */
async function readStream(res: Response): Promise<string> {
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let text = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    text += decoder.decode(value, { stream: true });
  }
  return text;
}

/** SSE comment lines — the heartbeat frames, `:` with no field name. */
function heartbeatCount(sse: string): number {
  return sse.split("\n").filter((line) => line === ":").length;
}

describe("downstream idle timeout (AISIX-Cloud#1126)", () => {
  let app: SpawnedApp | undefined;
  let slow: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    slow = await startOpenAiUpstream({
      // Headers and body both withheld: from the gateway's inbound side
      // the connection is silent for the whole delay.
      responseDelayMs: UPSTREAM_THINK_MS,
      nonStreamBody: {
        id: "cmpl-slow",
        object: "chat.completion",
        created: 0,
        model: "gpt-4o-mini",
        choices: [
          { index: 0, message: { role: "assistant", content: "worth the wait" }, finish_reason: "stop" },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    });

    app = await spawnApp({
      extra: { downstream: { idle_timeout_secs: IDLE_TIMEOUT_S } },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = (
      await seed.createProviderKey({
        display_name: "dc-slow-pk",
        secret: "sk-mock",
        api_base: `${slow.baseUrl}/v1`,
      })
    ).id;
    await seed.createModel({
      display_name: "dc-slow-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk,
      timeout: 30_000,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["dc-slow-model"],
    });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      if (!res.ok) return false;
      const body = (await res.json()) as { data?: Array<{ id: string }> };
      return !!body.data?.some((m) => m.id === "dc-slow-model");
    });
  });

  afterAll(async () => {
    await app?.exit();
    await slow?.close();
  });

  test("an idle keep-alive connection is closed at the configured deadline", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const port = Number(new URL(app.proxyUrl).port);
    const idleMs = await idleUntilClose(port, IDLE_TIMEOUT_S * 1000 + 4000);
    expect(idleMs).not.toBeNull();
    // Not early (the response had just been written) and not never.
    expect(idleMs!).toBeGreaterThanOrEqual(IDLE_TIMEOUT_S * 1000 - 200);
    expect(idleMs!).toBeLessThan(IDLE_TIMEOUT_S * 1000 + 3000);
  });

  // The regression that matters: an LLM call is quiet for far longer than
  // any sane idle timeout, and cutting it would be a self-inflicted 502.
  test("a request in flight past the idle timeout still completes", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const started = Date.now();
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "dc-slow-model",
        messages: [{ role: "user", content: "take your time" }],
      }),
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as { choices: Array<{ message: { content: string } }> };
    expect(body.choices[0]?.message.content).toBe("worth the wait");
    expect(Date.now() - started).toBeGreaterThan(IDLE_TIMEOUT_S * 1000);
  }, 30_000);
});

describe("downstream SSE heartbeat (AISIX-Cloud#1126)", () => {
  let app: SpawnedApp | undefined;
  let chatUpstream: OpenAiUpstream | undefined;
  let responsesUpstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // Headers flush immediately, the first token arrives much later: the
    // window a proxy in front would otherwise see as a dead connection.
    chatUpstream = await startOpenAiUpstream({
      firstEventDelayMs: FIRST_EVENT_DELAY_MS,
      streamEvents: [chatChunk("late reply"), chatFinish(), "[DONE]"],
    });
    responsesUpstream = await startOpenAiUpstream({
      firstEventDelayMs: FIRST_EVENT_DELAY_MS,
      streamEvents: RESPONSES_EVENTS,
    });

    app = await spawnApp({
      extra: { downstream: { sse_keepalive_interval_secs: HEARTBEAT_INTERVAL_S } },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const chatPk = (
      await seed.createProviderKey({
        display_name: "hb-chat-pk",
        secret: "sk-mock",
        api_base: `${chatUpstream.baseUrl}/v1`,
      })
    ).id;
    const responsesPk = (
      await seed.createProviderKey({
        display_name: "hb-responses-pk",
        secret: "sk-mock",
        api_base: `${responsesUpstream.baseUrl}/v1`,
      })
    ).id;
    await seed.createModel({
      display_name: "hb-chat-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: chatPk,
    });
    await seed.createModel({
      display_name: "hb-responses-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: responsesPk,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["hb-chat-model", "hb-responses-model"],
    });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      if (!res.ok) return false;
      const body = (await res.json()) as { data?: Array<{ id: string }> };
      return (
        !!body.data?.some((m) => m.id === "hb-chat-model") &&
        !!body.data?.some((m) => m.id === "hb-responses-model")
      );
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all([chatUpstream?.close(), responsesUpstream?.close()]);
  });

  test("chat completions heartbeat while the model is silent", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "hb-chat-model",
        messages: [{ role: "user", content: "hi" }],
        stream: true,
      }),
    });
    expect(res.status).toBe(200);
    const sse = await readStream(res);
    expect(heartbeatCount(sse)).toBeGreaterThanOrEqual(1);
    // The heartbeat must not disturb the payload it interleaves with.
    expect(sse).toContain("late reply");
    expect(sse).toContain("data: [DONE]");
  }, 30_000);

  test("bridged /v1/messages heartbeats while the model is silent", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const res = await fetch(`${app.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: {
        "x-api-key": CALLER_PLAINTEXT,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "hb-chat-model",
        max_tokens: 16,
        messages: [{ role: "user", content: "hi" }],
        stream: true,
      }),
    });
    expect(res.status).toBe(200);
    const sse = await readStream(res);
    expect(heartbeatCount(sse)).toBeGreaterThanOrEqual(1);
    expect(sse).toContain("late reply");
    expect(sse).toContain("message_stop");
  }, 30_000);

  test("passthrough /v1/responses heartbeats while the model is silent", async (ctx) => {
    if (!etcdReachable || !app) {
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
        model: "hb-responses-model",
        input: "hi",
        stream: true,
      }),
    });
    expect(res.status).toBe(200);
    const sse = await readStream(res);
    expect(heartbeatCount(sse)).toBeGreaterThanOrEqual(1);
    expect(sse).toContain("response.completed");
  }, 30_000);
});
