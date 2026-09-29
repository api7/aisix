import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForLogLine,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: `observability.access_log` is the switch for the access log.
// Off, a gateway logging at `info` writes no access-log line for any
// request — buffered, streamed, or refused before a handler — while its
// other `info` lines, including the ones those same requests produce,
// still appear. Left at its default, the same requests each write one.

const ACCESS_LINE = "proxy request completed";
const CALLER_PLAINTEXT = "sk-access-log-switch-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const MODEL = "access-log-switch-e2e";
const STREAM_MODEL = "access-log-switch-stream-e2e";

const CHAT_COMPLETION = {
  id: "chatcmpl-access-log-switch",
  object: "chat.completion",
  model: "gpt-4o-mini",
  choices: [{ index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "stop" }],
  usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
};
const STREAM_EVENTS = [
  JSON.stringify({
    id: "chatcmpl-access-log-switch",
    object: "chat.completion.chunk",
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta: { content: "hi" }, finish_reason: null }],
  }),
  JSON.stringify({
    id: "chatcmpl-access-log-switch",
    object: "chat.completion.chunk",
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
    usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
  }),
  "[DONE]",
];

interface Sent {
  buffered: string;
  streamed: string;
  refused: string;
}

async function chat(app: SpawnedApp, body: string): Promise<{ status: number; requestId: string }> {
  const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: { authorization: `Bearer ${CALLER_PLAINTEXT}`, "content-type": "application/json" },
    body,
  });
  await res.text();
  return { status: res.status, requestId: res.headers.get("x-aisix-request-id") ?? "" };
}

/** Drive one request of each kind and return their request ids. */
async function drive(app: SpawnedApp): Promise<Sent> {
  const messages = [{ role: "user", content: "hello" }];
  const buffered = await chat(app, JSON.stringify({ model: MODEL, messages }));
  expect(buffered.status).toBe(200);
  const streamed = await chat(app, JSON.stringify({ model: STREAM_MODEL, stream: true, messages }));
  expect(streamed.status).toBe(200);
  // Refused before dispatch: its line is written by a different path.
  const refused = await chat(app, "{not json");
  expect(refused.status).toBe(400);
  for (const id of [buffered.requestId, streamed.requestId, refused.requestId]) {
    expect(id, "every response carries x-aisix-request-id").toBeTruthy();
  }
  return { buffered: buffered.requestId, streamed: streamed.requestId, refused: refused.requestId };
}

describe("observability.access_log switches the access log", () => {
  const upstreams: OpenAiUpstream[] = [];
  let off: SpawnedApp | undefined;
  let on: SpawnedApp | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    const buffered = await startOpenAiUpstream({ nonStreamBody: CHAT_COMPLETION });
    const streamed = await startOpenAiUpstream({ streamEvents: STREAM_EVENTS });
    upstreams.push(buffered, streamed);
    off = await spawnApp({ logLevel: "info", accessLog: false });
    on = await spawnApp({ logLevel: "info" });

    for (const app of [off, on]) {
      const seed = new SeedClient(etcd, app.etcdPrefix);
      for (const [name, upstream] of [
        [MODEL, buffered],
        [STREAM_MODEL, streamed],
      ] as const) {
        const pk = await seed.createProviderKey({
          display_name: `${name}-pk`,
          secret: "sk-mock",
          api_base: `${upstream.baseUrl}/v1`,
        });
        await seed.createModel({
          display_name: name,
          provider: "openai",
          model_name: "gpt-4o-mini",
          provider_key_id: pk.id,
        });
      }
      await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: [MODEL, STREAM_MODEL] });
      const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
      await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
    }
  });

  afterAll(async () => {
    await off?.exit();
    await on?.exit();
    for (const u of upstreams) await u.close();
  });

  test("access_log: false writes no access-log line, and every other line still appears", async (ctx) => {
    if (!etcdReachable || !off) {
      ctx.skip();
      return;
    }
    const sent = await drive(off);

    // Stopping the gateway is the barrier: it drains its log queue before
    // exiting, so every line these requests could have written is ahead of
    // the shutdown line.
    await off.stop();
    await waitForLogLine(off, (l) => l.includes("aisix shut down cleanly"), "the shutdown line");
    const lines = off.output().split("\n");

    expect(lines.filter((l) => l.includes(ACCESS_LINE))).toEqual([]);
    // Application `info` lines are unaffected — the boot line, and the
    // per-attempt lines the same requests wrote.
    expect(lines.some((l) => l.includes("tracing initialised"))).toBe(true);
    for (const id of [sent.buffered, sent.streamed]) {
      expect(
        lines.some((l) => l.includes("provider call completed") && l.includes(`request_id="${id}"`)),
        `the provider-call line for ${id}`,
      ).toBe(true);
    }
  });

  test("by default each request writes its access-log line", async (ctx) => {
    if (!etcdReachable || !on) {
      ctx.skip();
      return;
    }
    const sent = await drive(on);
    const app = on;
    for (const [id, status] of [
      [sent.buffered, 200],
      [sent.streamed, 200],
      [sent.refused, 400],
    ] as const) {
      const line = await waitForLogLine(
        app,
        (l) => l.includes(ACCESS_LINE) && l.includes(`request_id="${id}"`),
        `the access-log line for ${id}`,
      );
      expect(line).toContain(`status=${status}`);
    }
  });
});
