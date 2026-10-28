import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForLogLine,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: the access-log line of a request whose upstream reported usage
// carries the token counts, on every endpoint — not only on
// /v1/chat/completions. A non-streamed /v1/messages or /v1/embeddings line
// used to have no token fields at all while the same request's usage event
// carried the counts, so an operator reading the log saw a billed request as
// a token-less one.
//
// It also pins the line's field names: the ones that report the same figure
// as the usage event are written under the event's names (`status_code`,
// `error_class`, `requested_model`, `downstream_latency_ms`), and the old
// names are gone rather than written beside them.

const CALLER_PLAINTEXT = "sk-access-log-token-columns";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const CHAT_BODY = {
  id: "chatcmpl-tok",
  object: "chat.completion",
  created: 1,
  model: "tok-upstream",
  choices: [{ index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "stop" }],
  usage: { prompt_tokens: 7, completion_tokens: 3, total_tokens: 10 },
};

const RESPONSES_BODY = {
  id: "resp_tok",
  object: "response",
  model: "tok-upstream",
  status: "completed",
  output: [
    {
      type: "message",
      id: "msg_tok",
      role: "assistant",
      content: [{ type: "output_text", text: "hi" }],
    },
  ],
  usage: { input_tokens: 6, output_tokens: 2, total_tokens: 8 },
};

// One body the single-shot endpoints' parsers all accept: 3 tokens in, 2 out.
const SINGLE_SHOT_BODY = {
  id: "cmpl-tok",
  object: "text_completion",
  created: 1,
  model: "tok-upstream",
  text: "hi",
  choices: [{ index: 0, text: "hi", finish_reason: "stop" }],
  data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2], url: "https://example.com/i.png" }],
  results: [{ index: 0, relevance_score: 0.9 }],
  usage: {
    prompt_tokens: 3,
    completion_tokens: 2,
    total_tokens: 5,
    input_tokens: 3,
    output_tokens: 2,
  },
};

/** One `key=value` field of an access-log line. */
function field(line: string, name: string): string | undefined {
  const m = line.match(new RegExp(`\\b${name}=(?:"([^"]*)"|([^\\s]+))`));
  if (!m) return undefined;
  return m[1] ?? m[2];
}

function tokens(line: string): [string | undefined, string | undefined, string | undefined] {
  return [field(line, "prompt_tokens"), field(line, "completion_tokens"), field(line, "total_tokens")];
}

/** The pre-rename keys, none of which may appear on a line any more. */
function expectNoOldNames(line: string): void {
  for (const old of [" status=", " latency_ms=", " model=", " error_kind="]) {
    expect(line.includes(old), `${old.trim()} is still on the line: ${line}`).toBe(false);
  }
}

describe("access-log token columns and field names", () => {
  let app: SpawnedApp | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];
  const auth = { authorization: `Bearer ${CALLER_PLAINTEXT}` };

  const post = async (path: string, body: unknown, headers: Record<string, string> = auth) => {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: { ...headers, "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    await res.text();
    return { status: res.status, requestId: res.headers.get("x-aisix-request-id") ?? "" };
  };

  const multipart = async (path: string, fields: Record<string, string>) => {
    const form = new FormData();
    for (const [k, v] of Object.entries(fields)) form.set(k, v);
    form.set("file", new Blob([new Uint8Array([0x49, 0x44, 0x33])], { type: "audio/mpeg" }), "f.mp3");
    const res = await fetch(`${app!.proxyUrl}${path}`, { method: "POST", headers: auth, body: form });
    await res.text();
    return { status: res.status, requestId: res.headers.get("x-aisix-request-id") ?? "" };
  };

  const lineOf = (requestId: string) =>
    waitForLogLine(
      app!,
      (l) => l.includes("proxy request completed") && field(l, "request_id") === requestId,
      `the access-log line of ${requestId}`,
    );

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // The access log is an `info` event; the harness defaults to `warn`.
    app = await spawnApp({ logLevel: "info" });
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const chat = await startOpenAiUpstream({ nonStreamBody: CHAT_BODY });
    const responses = await startOpenAiUpstream({ nonStreamBody: RESPONSES_BODY });
    const singleShot = await startOpenAiUpstream({ nonStreamBody: SINGLE_SHOT_BODY });
    const bad = await startOpenAiUpstream({
      status: 500,
      errorBody: { error: { message: "first target is down" } },
    });
    upstreams.push(chat, responses, singleShot, bad);

    const models: Array<[string, OpenAiUpstream]> = [
      ["tok-chat", chat],
      ["tok-responses", responses],
      ["tok-single-shot", singleShot],
      ["tok-bad", bad],
    ];
    for (const [name, upstream] of models) {
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      await seed.createModel({
        display_name: name,
        provider: "openai",
        model_name: "tok-upstream",
        provider_key_id: pk.id,
      });
    }
    await seed.createModel({
      display_name: "tok-group",
      routing: {
        strategy: "failover",
        targets: [{ model: "tok-bad" }, { model: "tok-chat" }],
        retries: 0,
        max_fallbacks: 1,
      },
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [...models.map(([name]) => name), "tok-group"],
    });

    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, { headers: auth });
      await res.text();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    for (const u of upstreams) await u.close();
  });

  test("a non-streamed /v1/messages line carries the upstream's token counts", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await post("/v1/messages", {
      model: "tok-chat",
      max_tokens: 16,
      messages: [{ role: "user", content: "hello" }],
    });
    expect(res.status).toBe(200);
    const line = await lineOf(res.requestId);
    expect(tokens(line), line).toEqual(["7", "3", "10"]);
  });

  test("a /v1/messages line that failed over carries the serving target's counts only", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await post("/v1/messages", {
      model: "tok-group",
      max_tokens: 16,
      messages: [{ role: "user", content: "hello" }],
    });
    expect(res.status).toBe(200);
    const line = await lineOf(res.requestId);
    expect(field(line, "routing_fallback_count"), line).toBe("1");
    expect(tokens(line), line).toEqual(["7", "3", "10"]);
  });

  test("a /v1/embeddings line carries the prompt count", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await post("/v1/embeddings", { model: "tok-single-shot", input: "hello" });
    expect(res.status).toBe(200);
    const line = await lineOf(res.requestId);
    // An embedding generates no output, so all of the total is the input.
    expect(tokens(line), line).toEqual(["3", "0", "3"]);
  });

  test("a non-streamed /v1/responses line carries the upstream's token counts", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await post("/v1/responses", { model: "tok-responses", input: "hello" });
    expect(res.status).toBe(200);
    const line = await lineOf(res.requestId);
    expect(tokens(line), line).toEqual(["6", "2", "8"]);
  });

  test("a /v1/completions line carries the upstream's token counts", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await post("/v1/completions", { model: "tok-single-shot", prompt: "hello" });
    expect(res.status).toBe(200);
    const line = await lineOf(res.requestId);
    expect(tokens(line), line).toEqual(["3", "2", "5"]);
  });

  test("a /v1/rerank line carries the prompt count", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await post("/v1/rerank", { model: "tok-single-shot", query: "q", documents: ["a", "b"] });
    expect(res.status).toBe(200);
    const line = await lineOf(res.requestId);
    expect(tokens(line), line).toEqual(["3", "0", "3"]);
  });

  test("a /v1/images/generations line carries the upstream's token counts", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await post("/v1/images/generations", { model: "tok-single-shot", prompt: "a cat" });
    expect(res.status).toBe(200);
    const line = await lineOf(res.requestId);
    expect(tokens(line), line).toEqual(["3", "2", "5"]);
  });

  test("a non-streamed /v1/audio/transcriptions line carries the upstream's token counts", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await multipart("/v1/audio/transcriptions", { model: "tok-single-shot" });
    expect(res.status).toBe(200);
    const line = await lineOf(res.requestId);
    expect(tokens(line), line).toEqual(["3", "2", "5"]);
  });

  test("a success line is keyed by the usage event's names", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await post("/v1/chat/completions", {
      model: "tok-chat",
      messages: [{ role: "user", content: "hello" }],
    });
    expect(res.status).toBe(200);
    const line = await lineOf(res.requestId);
    expect(field(line, "status_code"), line).toBe("200");
    expect(field(line, "requested_model"), line).toBe("tok-chat");
    expect(field(line, "downstream_latency_ms"), line).toMatch(/^\d+$/);
    expect(line).not.toContain("error_class");
    expectNoOldNames(line);
  });

  test("an authentication failure line is keyed by the usage event's names", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await post(
      "/v1/chat/completions",
      { model: "tok-chat", messages: [{ role: "user", content: "hello" }] },
      { authorization: "Bearer sk-not-a-key" },
    );
    expect(res.status).toBe(401);
    const line = await lineOf(res.requestId);
    expect(field(line, "status_code"), line).toBe("401");
    expect(field(line, "error_class"), line).toBeTruthy();
    expect(field(line, "downstream_latency_ms"), line).toMatch(/^\d+$/);
    expectNoOldNames(line);
  });

  test("a fallback line is keyed by the usage event's names", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await post("/v1/chat/completions", {
      model: "tok-group",
      messages: [{ role: "user", content: "hello" }],
    });
    expect(res.status).toBe(200);
    const line = await lineOf(res.requestId);
    expect(field(line, "status_code"), line).toBe("200");
    expect(field(line, "requested_model"), line).toBe("tok-group");
    expect(field(line, "served_by_model"), line).toBe("tok-chat");
    expect(field(line, "routing_fallback_count"), line).toBe("1");
    expect(field(line, "downstream_latency_ms"), line).toMatch(/^\d+$/);
    expectNoOldNames(line);
  });
});
