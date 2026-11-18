import { createHash, randomUUID } from "node:crypto";
import { connect } from "node:net";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  awaitWindowHeadroom,
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E for api7/AISIX-Cloud#2028: the environment's `usage_accounting`
// document keeps usage the gateway estimated out of token rate limits.
//
// An upstream that reports no usage leaves the gateway to count the tokens
// itself. With `exclude_estimated_usage: true` such a request adds nothing to
// the caller's TPM, so a key whose TPM is far below one request's estimate is
// still served request after request. Without the document the same traffic
// counts and the key is throttled; with it, a request whose usage the
// upstream DID report still counts. The same contract holds whether the
// counters are per process or shared in Redis.

const TPM = 5;
const hash = (s: string) => createHash("sha256").update(s).digest("hex");
const REDIS_URL = process.env.AISIX_E2E_REDIS ?? "redis://127.0.0.1:6379";

// Long enough on both sides that one request's estimate alone exceeds TPM.
const PROMPT = "Please summarise the quarterly report for the regional sales team in detail.";
const ANSWER = "The regional sales team grew revenue in every month of the quarter, led by new accounts.";

const chatChunk = (json: Record<string, unknown>) =>
  JSON.stringify({ id: "chatcmpl-2028", object: "chat.completion.chunk", created: 0, model: "relay-2028", ...json });
const CHAT_STREAM_NO_USAGE = [
  chatChunk({ choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }] }),
  chatChunk({ choices: [{ index: 0, delta: { content: ANSWER }, finish_reason: null }] }),
  chatChunk({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }),
  "[DONE]",
];
const CHAT_STREAM_WITH_USAGE = [
  ...CHAT_STREAM_NO_USAGE.slice(0, -1),
  chatChunk({ choices: [], usage: { prompt_tokens: 6, completion_tokens: 6, total_tokens: 12 } }),
  "[DONE]",
];
const CHAT_BODY_NO_USAGE = {
  id: "chatcmpl-2028",
  object: "chat.completion",
  created: 0,
  model: "relay-2028",
  choices: [{ index: 0, message: { role: "assistant", content: ANSWER }, finish_reason: "stop" }],
};
const RESPONSES_BODY_NO_USAGE = {
  id: "resp_2028",
  object: "response",
  status: "completed",
  model: "relay-2028",
  output: [
    { type: "message", id: "msg_2028", role: "assistant", content: [{ type: "output_text", text: ANSWER }] },
  ],
};
const COMPLETIONS_BODY_NO_USAGE = {
  id: "cmpl-2028",
  object: "text_completion",
  created: 0,
  model: "relay-2028",
  choices: [{ index: 0, text: ANSWER, finish_reason: "stop" }],
};
const EMBEDDINGS_BODY_NO_USAGE = {
  object: "list",
  model: "relay-2028",
  data: [{ index: 0, object: "embedding", embedding: [0.1, 0.2, 0.3] }],
};
const RESPONSES_STREAM_NO_USAGE = [
  JSON.stringify({ type: "response.created", response: { id: "resp_2028" } }),
  JSON.stringify({ type: "response.output_text.delta", delta: ANSWER }),
  JSON.stringify({ type: "response.completed", response: { id: "resp_2028", status: "completed" } }),
  "[DONE]",
];

async function redisPing(url: string): Promise<boolean> {
  const m = /^redis:\/\/(?:[^@/]*@)?([^:/]+)(?::(\d+))?/.exec(url);
  if (!m) return false;
  return new Promise((resolve) => {
    const sock = connect({ host: m[1], port: m[2] ? Number(m[2]) : 6379 }, () => sock.write("PING\r\n"));
    const done = (ok: boolean) => {
      sock.destroy();
      resolve(ok);
    };
    sock.once("data", (buf) => done(buf.toString().startsWith("+PONG")));
    sock.once("error", () => done(false));
    sock.setTimeout(1000, () => done(false));
  });
}

type Call = (app: SpawnedApp, key: string) => Promise<Response>;

const post = (app: SpawnedApp, key: string, path: string, body: Record<string, unknown>) =>
  fetch(`${app.proxyUrl}${path}`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...(path === "/v1/messages" ? { "x-api-key": key } : { authorization: `Bearer ${key}` }),
    },
    body: JSON.stringify(body),
  });

const chatStream: Call = (app, key) =>
  post(app, key, "/v1/chat/completions", {
    model: "est-chat",
    stream: true,
    messages: [{ role: "user", content: PROMPT }],
  });
const chatBuffered: Call = (app, key) =>
  post(app, key, "/v1/chat/completions", {
    model: "est-chat-buffered",
    messages: [{ role: "user", content: PROMPT }],
  });
const messagesStream: Call = (app, key) =>
  post(app, key, "/v1/messages", {
    model: "est-chat",
    max_tokens: 200,
    stream: true,
    messages: [{ role: "user", content: PROMPT }],
  });
const responsesStream: Call = (app, key) =>
  post(app, key, "/v1/responses", { model: "est-responses", stream: true, input: PROMPT });
// Every sub-call answers without usage: the panel from the buffered
// upstream, the judge buffered or streamed.
const ensembleBuffered: Call = (app, key) =>
  post(app, key, "/v1/chat/completions", { model: "est-ensemble", messages: [{ role: "user", content: PROMPT }] });
const ensembleStream: Call = (app, key) =>
  post(app, key, "/v1/chat/completions", {
    model: "est-ensemble-stream",
    stream: true,
    messages: [{ role: "user", content: PROMPT }],
  });
const messagesBuffered: Call = (app, key) =>
  post(app, key, "/v1/messages", {
    model: "est-chat-buffered",
    max_tokens: 200,
    messages: [{ role: "user", content: PROMPT }],
  });
const responsesBuffered: Call = (app, key) =>
  post(app, key, "/v1/responses", { model: "est-responses-buffered", input: PROMPT });
const completions: Call = (app, key) => post(app, key, "/v1/completions", { model: "est-completions", prompt: PROMPT });
const embeddings: Call = (app, key) => post(app, key, "/v1/embeddings", { model: "est-embeddings", input: PROMPT });
const reportedChatStream: Call = (app, key) =>
  post(app, key, "/v1/chat/completions", {
    model: "reported-chat",
    stream: true,
    messages: [{ role: "user", content: PROMPT }],
  });

/** Status of each of `n` calls, made one after another with the body read
 *  in full. The post-stream commit lands a tick after the client finishes
 *  reading, so the pause between calls lets each one count before the next
 *  is checked. */
async function statuses(n: number, make: () => Promise<Response>): Promise<number[]> {
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const res = await make();
    await res.text();
    out.push(res.status);
    await new Promise((r) => setTimeout(r, 300));
  }
  return out;
}

const BACKENDS = [
  { name: "memory", ratelimit: undefined },
  { name: "redis", ratelimit: { backend: "redis", redis: { url: REDIS_URL } } },
] as const;

describe.each(BACKENDS)("estimated usage and token rate limits ($name counters)", ({ name, ratelimit }) => {
  const apps: { on?: SpawnedApp; off?: SpawnedApp } = {};
  const upstreams: OpenAiUpstream[] = [];
  const keys = new Map<string, string>();
  let ready = false;

  // One key per case and gateway, so every case starts from an empty window.
  const keyFor = (gateway: "on" | "off", label: string) => {
    const id = `${gateway}:${label}`;
    if (!keys.has(id)) keys.set(id, `sk-2028-${name}-${gateway}-${label}-${randomUUID()}`);
    return keys.get(id)!;
  };

  async function seedGateway(app: SpawnedApp, gateway: "on" | "off", labels: string[]) {
    const seed = new SeedClient(new EtcdClient(), app.etcdPrefix);
    // Written first: the watch delivers in revision order, so the keys
    // becoming live below means this document has been applied as well.
    if (gateway === "on") {
      await seed.update("usage_accounting", randomUUID(), { exclude_estimated_usage: true });
    }
    const [chatUp, bufferedUp, responsesUp, reportedUp, responsesBufferedUp, completionsUp, embeddingsUp] =
      upstreams;
    const model = async (display_name: string, upstream: OpenAiUpstream) => {
      const pk = await seed.createProviderKey({
        display_name: `${display_name}-pk`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      await seed.createModel({ display_name, provider: "openai", model_name: "relay-2028", provider_key_id: pk.id });
    };
    await model("est-chat", chatUp);
    await model("est-chat-buffered", bufferedUp);
    await model("est-chat-buffered-2", bufferedUp);
    const panel = [{ model: "est-chat-buffered" }, { model: "est-chat-buffered-2" }];
    await seed.createModel({
      display_name: "est-ensemble",
      ensemble: { panel, judge: { model: "est-chat-buffered" }, min_responses: 2 },
    });
    await seed.createModel({
      display_name: "est-ensemble-stream",
      ensemble: { panel, judge: { model: "est-chat" }, min_responses: 2 },
    });
    await model("est-responses", responsesUp);
    await model("est-responses-buffered", responsesBufferedUp);
    await model("est-completions", completionsUp);
    await model("est-embeddings", embeddingsUp);
    await model("reported-chat", reportedUp);
    for (const label of labels) {
      await seed.createApiKey({
        key_hash: hash(keyFor(gateway, label)),
        allowed_models: ["*"],
        rate_limit: { tpm: TPM },
      });
    }
    for (const label of labels) {
      const probe = new ProxyClient(app.proxyUrl, keyFor(gateway, label));
      await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
    }
  }

  beforeAll(async () => {
    ready = (await new EtcdClient().ping()) && (name === "memory" || (await redisPing(REDIS_URL)));
    if (!ready) return;
    upstreams.push(
      await startOpenAiUpstream({ streamEvents: CHAT_STREAM_NO_USAGE }),
      await startOpenAiUpstream({ nonStreamBody: CHAT_BODY_NO_USAGE }),
      await startOpenAiUpstream({ streamEvents: RESPONSES_STREAM_NO_USAGE }),
      await startOpenAiUpstream({ streamEvents: CHAT_STREAM_WITH_USAGE }),
      await startOpenAiUpstream({ nonStreamBody: RESPONSES_BODY_NO_USAGE }),
      await startOpenAiUpstream({ nonStreamBody: COMPLETIONS_BODY_NO_USAGE }),
      await startOpenAiUpstream({ nonStreamBody: EMBEDDINGS_BODY_NO_USAGE }),
    );
    const extra = ratelimit ? { ratelimit } : undefined;
    apps.on = await spawnApp(extra ? { extra } : {});
    apps.off = await spawnApp(extra ? { extra } : {});
    await seedGateway(apps.on, "on", [
      "chat-stream",
      "chat-buffered",
      "messages",
      "messages-buffered",
      "responses",
      "responses-buffered",
      "completions",
      "embeddings",
      "ensemble-buffered",
      "ensemble-stream",
      "reported",
    ]);
    await seedGateway(apps.off, "off", ["chat-stream"]);
  });

  afterAll(async () => {
    await apps.on?.exit();
    await apps.off?.exit();
    for (const u of upstreams) await u.close();
  });

  const estimatedCalls: Array<[string, Call]> = [
    ["chat-stream", chatStream],
    ["chat-buffered", chatBuffered],
    ["messages", messagesStream],
    ["messages-buffered", messagesBuffered],
    ["responses", responsesStream],
    ["responses-buffered", responsesBuffered],
    ["completions", completions],
    ["embeddings", embeddings],
    ["ensemble-buffered", ensembleBuffered],
    ["ensemble-stream", ensembleStream],
  ];
  for (const [label, call] of estimatedCalls) {
    test(`excluded: estimated ${label} traffic is never throttled by TPM`, async (ctx) => {
      if (!ready || !apps.on) return ctx.skip();
      await awaitWindowHeadroom(20);
      const key = keyFor("on", label);
      expect(await statuses(4, () => call(apps.on!, key))).toEqual([200, 200, 200, 200]);
    });
  }

  test("excluded: the estimate is real — one buffered reply alone exceeds TPM", async (ctx) => {
    if (!ready || !apps.on) return ctx.skip();
    const res = await chatBuffered(apps.on, keyFor("on", "chat-buffered"));
    expect(res.status).toBe(200);
    const body = (await res.json()) as { usage?: { total_tokens?: number } };
    expect(body.usage?.total_tokens ?? 0).toBeGreaterThan(TPM);
  });

  test("excluded: usage the upstream reported still counts", async (ctx) => {
    if (!ready || !apps.on) return ctx.skip();
    await awaitWindowHeadroom(20);
    const got = await statuses(3, () => reportedChatStream(apps.on!, keyFor("on", "reported")));
    expect(got[0]).toBe(200);
    expect(got.slice(1)).toContain(429);
  });

  test("not excluded: the same estimated traffic is throttled by TPM", async (ctx) => {
    if (!ready || !apps.off) return ctx.skip();
    await awaitWindowHeadroom(20);
    const got = await statuses(3, () => chatStream(apps.off!, keyFor("off", "chat-stream")));
    expect(got[0]).toBe(200);
    expect(got.slice(1)).toContain(429);
  });
});
