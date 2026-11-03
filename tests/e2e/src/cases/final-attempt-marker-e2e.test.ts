import { createHash, randomUUID } from "node:crypto";
import { createServer, type Server as HttpServer } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  slsLogsFor,
  spawnApp,
  startMcpUpstream,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  type McpUpstream,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// `is_final_attempt`: of all the usage rows one client request leaves, exactly
// one says `true`, and it is the last. Observed through an exporter at
// `metadata_only`, on the wire shape the control plane receives too.

const CALLER = "sk-final-attempt-caller";
const CREDENTIAL_REF = "mock";
const LOGSTORE = "final-attempt";
const INPUT_WORD = "fmforbiddeninput";
const OUTPUT_WORD = "fmforbiddenoutput";
const USAGE = { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 };

const chatBody = (content: string) => ({
  id: "chatcmpl-fm",
  object: "chat.completion",
  created: 1,
  model: "gpt-4o-mini",
  choices: [{ index: 0, message: { role: "assistant", content }, finish_reason: "stop" }],
  usage: USAGE,
});
const chatChunk = (delta: Record<string, unknown>, finish: string | null = null) =>
  JSON.stringify({
    id: "chatcmpl-fm",
    object: "chat.completion.chunk",
    created: 1,
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
const CHAT_STREAM = [
  chatChunk({ role: "assistant", content: "ok" }),
  chatChunk({}, "stop"),
  JSON.stringify({
    id: "chatcmpl-fm",
    object: "chat.completion.chunk",
    created: 1,
    model: "gpt-4o-mini",
    choices: [],
    usage: USAGE,
  }),
  "[DONE]",
];
const RESPONSE_OBJECT = {
  id: "resp_fm",
  object: "response",
  status: "completed",
  model: "gpt-4o-mini",
  output: [
    {
      type: "message",
      id: "msg_fm",
      role: "assistant",
      status: "completed",
      content: [{ type: "output_text", text: "ok", annotations: [] }],
    },
  ],
  usage: { input_tokens: 3, output_tokens: 2, total_tokens: 5 },
};

// Answers whichever API the gateway calls it on, in the shape that API has.
function okUpstreamHandler(path: string, body: Record<string, unknown>) {
  if (path.endsWith("/embeddings")) {
    return {
      object: "list",
      model: "text-embedding-3-small",
      data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2] }],
      usage: { prompt_tokens: 3, total_tokens: 3 },
    };
  }
  if (path.endsWith("/completions") && !path.endsWith("/chat/completions")) {
    return {
      id: "cmpl-fm",
      object: "text_completion",
      created: 1,
      model: "gpt-4o-mini",
      choices: [{ index: 0, text: "ok", finish_reason: "stop" }],
      usage: USAGE,
    };
  }
  if (path.endsWith("/responses")) return RESPONSE_OBJECT;
  if (body.messages) return chatBody("ok");
  return { ok: true };
}

const chat = (model: string, content: string, stream = false) => ({
  path: "/v1/chat/completions",
  payload: {
    model,
    messages: [{ role: "user", content }],
    ...(stream ? { stream: true, stream_options: { include_usage: true } } : {}),
  },
});
const messages = (model: string, content: string, stream = false) => ({
  path: "/v1/messages",
  payload: { model, max_tokens: 5, messages: [{ role: "user", content }], ...(stream ? { stream: true } : {}) },
});
const responses = (model: string, content: string, stream = false) => ({
  path: "/v1/responses",
  payload: { model, input: content, ...(stream ? { stream: true } : {}) },
});
const embeddings = (model: string, content: string) => ({
  path: "/v1/embeddings",
  payload: { model, input: content },
});
const completions = (model: string, content: string) => ({
  path: "/v1/completions",
  payload: { model, prompt: content },
});

type Row = Map<string, string>;

describe("is_final_attempt marks the one last row of a request", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let mcp: McpUpstream | undefined;
  let slow: OpenAiUpstream | undefined;
  let trickle: OpenAiUpstream | undefined;
  const upstreams: OpenAiUpstream[] = [];
  const servers: HttpServer[] = [];
  const auth = { authorization: `Bearer ${CALLER}`, "content-type": "application/json" };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "final-attempt-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "aisix-e2e-obs",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });

    const okServer = createServer((req, res) => {
      let raw = "";
      req.on("data", (c) => (raw += c));
      req.on("end", () => {
        const body = (raw ? JSON.parse(raw) : {}) as Record<string, unknown>;
        if (body.stream) {
          res.writeHead(200, { "content-type": "text/event-stream" });
          if (req.url?.endsWith("/responses")) {
            res.write(
              `event: response.completed\ndata: ${JSON.stringify({ type: "response.completed", response: RESPONSE_OBJECT })}\n\n`,
            );
          } else {
            for (const e of CHAT_STREAM) res.write(`data: ${e}\n\n`);
          }
          res.end();
          return;
        }
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify(okUpstreamHandler(req.url ?? "", body)));
      });
    });
    servers.push(okServer);
    await new Promise<void>((r) => okServer.listen(0, "127.0.0.1", r));
    const okBase = `http://127.0.0.1:${(okServer.address() as { port: number }).port}`;

    const modelAt = async (name: string, base: string) => {
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        secret: "sk-mock",
        api_base: `${base}/v1`,
      });
      await seed.createModel({
        display_name: name,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
        retries: 0,
        cooldown: { enabled: false },
      });
      return pk;
    };
    const mocked = async (name: string, opts: Parameters<typeof startOpenAiUpstream>[0]) => {
      const upstream = await startOpenAiUpstream(opts);
      upstreams.push(upstream);
      await modelAt(name, upstream.baseUrl);
      return upstream;
    };

    const okPk = await modelAt("fm-ok", okBase);
    const failing = (message: string) => ({
      status: 503,
      errorBody: { error: { message, type: "server_error" } },
    });
    await mocked("fm-fail-a", failing("first target failed"));
    await mocked("fm-fail-b", failing("second target failed"));
    await mocked("fm-leaky", { nonStreamBody: chatBody(`this says ${OUTPUT_WORD}`) });
    await mocked("fm-cut", {
      streamEvents: [chatChunk({ role: "assistant", content: "one" }), chatChunk({ content: "two" })],
      disconnectAfterEvents: 1,
      eventDelayMs: 20,
    });
    slow = await mocked("fm-slow", { nonStreamBody: chatBody("late"), responseDelayMs: 30_000 });
    trickle = await mocked("fm-trickle", {
      streamEvents: [
        chatChunk({ role: "assistant", content: "one" }),
        chatChunk({ content: "two" }),
        chatChunk({ content: "three" }),
        chatChunk({}, "stop"),
        "[DONE]",
      ],
      eventDelayMs: 500,
    });

    const group = (name: string, ...targets: string[]) =>
      seed.createModel({
        display_name: name,
        routing: { strategy: "failover", targets: targets.map((model) => ({ model })) },
      });
    await group("fm-recovers", "fm-fail-a", "fm-ok");
    await group("fm-all-fail", "fm-fail-a", "fm-fail-b");
    const thenBlocked = await group("fm-then-blocked", "fm-fail-a", "fm-leaky");
    await group("fm-then-cut", "fm-fail-a", "fm-cut");
    await group("fm-then-cancel-head", "fm-fail-a", "fm-slow");
    await group("fm-then-cancel-stream", "fm-fail-a", "fm-trickle");

    const panel = [{ model: "fm-ok" }, { model: "fm-ok" }];
    await seed.createModel({
      display_name: "fm-council",
      ensemble: { panel, judge: { model: "fm-ok" }, min_responses: 2 },
    });
    await seed.createModel({
      display_name: "fm-council-bad-judge",
      ensemble: { panel, judge: { model: "fm-fail-a" }, min_responses: 2 },
    });

    await seed.createCachePolicy({ name: "fm-cache", enabled: true, applies_to: "all" });
    await seed.createGuardrail({
      name: "fm-input-block",
      enabled: true,
      hook_point: "input",
      kind: "keyword",
      patterns: [{ kind: "literal", value: INPUT_WORD }],
    });
    // Attached to the one group, so no other stream here is held back for it.
    const outputGuard = await seed.createGuardrail(
      {
        name: "fm-output-block",
        enabled: true,
        hook_point: "output",
        kind: "keyword",
        patterns: [{ kind: "literal", value: OUTPUT_WORD }],
      },
      { attach: false },
    );
    await seed.attachGuardrailToModel(outputGuard.id, thenBlocked.id);

    await seed.createPassthroughRoute({
      name: "fm-route",
      path_prefix: "/fm-route",
      target_url: okBase,
      provider_key_id: okPk.id,
    });
    await seed.update("a2a_agents", randomUUID(), {
      name: "fm-agent",
      url: okBase,
      protocol_version: "1.0",
      auth_type: "none",
      enabled: true,
    });
    mcp = await startMcpUpstream("fmmcp");
    await seed.update("mcp_servers", randomUUID(), {
      display_name: "fmmcp",
      url: mcp.url,
      enabled: true,
    });

    // Seeded last: the key authenticating implies the whole set is live.
    await seed.createApiKey({
      key_hash: createHash("sha256").update(CALLER).digest("hex"),
      allowed_models: ["*"],
      allowed_routes: ["fm-route"],
      allowed_agents: ["*"],
      allowed_mcp_servers: ["*"],
    });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, { headers: auth });
      await res.arrayBuffer();
      return res.status === 200;
    });
  }, 120_000);

  afterAll(async () => {
    await app?.exit();
    await sls?.close();
    await mcp?.close();
    for (const u of upstreams) await u.close();
    await Promise.all(
      servers.map((s) => new Promise((r) => {
        s.closeAllConnections();
        s.close(r);
      })),
    );
  });

  async function settledRows(match: (r: Row) => boolean, count: number): Promise<Row[]> {
    const matching = () => slsLogsFor(sls!, LOGSTORE).filter(match);
    const deadline = Date.now() + 15_000;
    while (Date.now() < deadline && matching().length < count) {
      await new Promise((r) => setTimeout(r, 100));
    }
    // Past the count: a surplus row would ride the same export.
    await new Promise((r) => setTimeout(r, 500));
    return matching();
  }

  /** The contract itself, on rows in the order the exporter received them. */
  function expectOneFinalRowLast(rows: Row[], expectedStatuses: string[], label: string): void {
    const summary = rows.map(
      (r) => `${r.get("attempt_index")}/${r.get("status_code")}/${r.get("is_final_attempt")}`,
    );
    expect(rows.map((r) => r.get("status_code")), `${label}: ${summary}`).toEqual(expectedStatuses);
    const flags = rows.map((r) => r.get("is_final_attempt"));
    expect(flags, `${label}: ${summary}`).toEqual(
      rows.map((_, i) => (i === rows.length - 1 ? "true" : "false")),
    );
  }

  const post = (path: string, payload: unknown, headers: Record<string, string> = {}) =>
    fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        ...auth,
        "anthropic-version": "2023-06-01",
        accept: "application/json, text/event-stream",
        ...headers,
      },
      body: JSON.stringify(payload),
    });

  async function drive(
    request: { path: string; payload: unknown },
    expectedStatuses: string[],
    label: string,
  ): Promise<Row[]> {
    const res = await post(request.path, request.payload);
    await res.text();
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    expect(requestId, label).not.toBe("");
    const rows = await settledRows((r) => r.get("request_id") === requestId, expectedStatuses.length);
    expectOneFinalRowLast(rows, expectedStatuses, label);
    return rows;
  }

  test.for([
    ["chat", (m: string, c: string) => chat(m, c)],
    ["chat stream", (m: string, c: string) => chat(m, c, true)],
    ["messages", (m: string, c: string) => messages(m, c)],
    ["messages stream", (m: string, c: string) => messages(m, c, true)],
    ["responses", (m: string, c: string) => responses(m, c)],
    ["responses stream", (m: string, c: string) => responses(m, c, true)],
    ["embeddings", embeddings],
    ["completions", completions],
  ] as const)("%s: direct, fallback and all-failed", { timeout: 60_000 }, async ([label, make], ctx) => {
    if (!etcdReachable) return ctx.skip();
    const direct = () => `${label} ${randomUUID()}`;
    await drive(make("fm-ok", direct()), ["200"], `${label} direct`);
    await drive(make("fm-recovers", direct()), ["502", "200"], `${label} fallback`);
    await drive(make("fm-all-fail", direct()), ["502", "502"], `${label} all failed`);
  });

  test.for([
    ["images", "/v1/images/generations", { model: "fm-all-fail", prompt: "a cat" }],
    ["decisions", "/v1/decisions", { model: "fm-all-fail", input: "x", questions: [{ type: "predicate", name: "q", instructions: "i" }] }],
    ["speech", "/v1/audio/speech", { model: "fm-all-fail", input: "x", voice: "alloy" }],
  ] as const)("%s: all-failed ends on the last failure", { timeout: 30_000 }, async ([label, path, payload], ctx) => {
    if (!etcdReachable) return ctx.skip();
    await drive({ path, payload }, ["502", "502"], label);
  });

  test("chat: a cache hit is its own final row", { timeout: 30_000 }, async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const request = chat("fm-ok", `cache ${randomUUID()}`);
    await drive(request, ["200"], "cache miss");
    const [hit] = await drive(request, ["200"], "cache hit");
    expect(hit.get("cache_status")).toBe("hit");
  });

  test("a refusal before dispatch is its own final row", { timeout: 30_000 }, async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    await drive(chat("fm-ok", `say ${INPUT_WORD}`), ["422"], "chat input block");
    await drive(messages("fm-ok", `say ${INPUT_WORD}`), ["422"], "messages input block");
    await drive(chat("fm-no-such-model", "hi"), ["404"], "unknown model");
  });

  test("billed then blocked: the refusal is the final row", { timeout: 30_000 }, async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const rows = await drive(chat("fm-then-blocked", randomUUID()), ["502", "422"], "billed then blocked");
    expect(rows[1].get("guardrail_blocked")).toBe("true");
    expect(Number(rows[1].get("prompt_tokens"))).toBeGreaterThan(0);
  });

  test("a stream that fails part-way ends on its failure row", { timeout: 30_000 }, async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const rows = await drive(chat("fm-then-cut", randomUUID(), true), ["502", "502"], "fallback then cut");
    expect(rows[1].get("error_class")).not.toBe("");
    await drive(chat("fm-cut", randomUUID(), true), ["502"], "direct cut");
  });

  test("a caller that leaves before the head: the 499 is the final row", { timeout: 40_000 }, async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const already = slow!.receivedRequests.length;
    const controller = new AbortController();
    const inflight = fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: auth,
      body: JSON.stringify(chat("fm-then-cancel-head", randomUUID()).payload),
      signal: controller.signal,
    });
    for (let i = 0; i < 200 && slow!.receivedRequests.length === already; i++) {
      await new Promise((r) => setTimeout(r, 25));
    }
    controller.abort();
    await expect(inflight).rejects.toThrow();
    const rows = await settledRows((r) => r.get("requested_model") === "fm-then-cancel-head", 2);
    expectOneFinalRowLast(rows, ["502", "499"], "cancel before head");
  });

  test("a caller that leaves mid-stream: the 499 is the final row", { timeout: 40_000 }, async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const controller = new AbortController();
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: auth,
      body: JSON.stringify(chat("fm-then-cancel-stream", randomUUID(), true).payload),
      signal: controller.signal,
    });
    expect(res.status).toBe(200);
    const reader = res.body!.getReader();
    expect((await reader.read()).done).toBe(false);
    controller.abort();
    // Races the abort: it resolves when cancellation wins and rejects with the
    // body's `AbortError` when the abort does.
    await reader.cancel().catch((err: unknown) => {
      if (!(err instanceof Error) || err.name !== "AbortError") throw err;
    });
    const rows = await settledRows((r) => r.get("requested_model") === "fm-then-cancel-stream", 2);
    expectOneFinalRowLast(rows, ["502", "499"], "cancel mid-stream");
  });

  test("ensemble: panel rows are not final; the judge or the failure row is", { timeout: 30_000 }, async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const ok = await drive(chat("fm-council", randomUUID()), ["200", "200", "200"], "council");
    expect(ok.map((r) => r.get("attempt_kind"))).toEqual(["panel", "panel", "judge"]);
    await drive(
      chat("fm-council-bad-judge", randomUUID()),
      ["200", "200", "502", "502"],
      "council whose judge fails",
    );
  });

  test("the one-row families are final", { timeout: 30_000 }, async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    await drive({ path: "/fm-route/anything", payload: { hello: "world" } }, ["200"], "passthrough");
    await drive(
      {
        path: "/a2a/fm-agent",
        payload: {
          jsonrpc: "2.0",
          id: "1",
          method: "message/send",
          params: { message: { role: "user", parts: [{ kind: "text", text: "hi" }], messageId: "m-1" } },
        },
      },
      ["200"],
      "a2a",
    );
    await drive(
      {
        path: "/mcp",
        payload: {
          jsonrpc: "2.0",
          id: 1,
          method: "tools/call",
          params: { name: "fmmcp__echo", arguments: { text: "hi" } },
        },
      },
      ["200"],
      "mcp",
    );
  });
});
