import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  awaitWindowHeadroom,
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
import { pickFreePort } from "../harness/ports.js";

// E2E: a semantic router on every endpoint that honours one —
// `/v1/chat/completions`, `/v1/messages`, `/v1/messages/count_tokens`,
// `/v1/responses` — and model groups as a router's targets.
//
// What the router classifies is the newest user turn that carries human
// text: tool results, images and `<system-reminder>` blocks are not human
// text, so a turn holding only those is skipped. Agent clients (Claude
// Code on `/v1/messages`, Codex on `/v1/responses`) send exactly such
// turns.
//
// A route target that is a model group dispatches the way the group does
// when addressed directly: its strategy picks members and fails over
// between them. A group that has no member to dispatch to for the caller
// is an unavailable target, and the router serves its `default`.
//
// The embedding upstream maps text to a one-hot vector by keyword, so every
// decision is deterministic: "contract" → the legal route, anything else →
// orthogonal to every example → `default`. It records each batch, whose
// first input is the classified text.

const CALLER_PLAINTEXT = "sk-semantic-protocols-e2e";
const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");
const CALLER_KEY_HASH = sha256(CALLER_PLAINTEXT);
// Callers with their own counters, for the rate-limit cases.
const TEAM_CALLER = "sk-semantic-protocols-e2e-team";
const CAPPED_CALLER = "sk-semantic-protocols-e2e-capped";
const ALL_REFUSED_CALLER = "sk-semantic-protocols-e2e-all-refused";

const LEGAL_ASK = "please review this contract";
const OTHER_ASK = "what time is the meeting tomorrow";
const REMINDER = (text: string) => `<system-reminder>\n${text}\n</system-reminder>`;

function keywordVector(text: string): number[] {
  return text.toLowerCase().includes("contract") ? [1, 0] : [0, 1];
}

interface EmbeddingMock {
  baseUrl: string;
  batches: string[][];
  close(): Promise<void>;
}

async function startEmbeddingMock(): Promise<EmbeddingMock> {
  const batches: string[][] = [];
  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      const body = JSON.parse(raw || "{}") as { input?: string | string[] };
      const inputs = Array.isArray(body.input) ? body.input : [body.input ?? ""];
      batches.push([...inputs]);
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          object: "list",
          model: "embed-mock",
          data: inputs.map((text, index) => ({
            object: "embedding",
            index,
            embedding: keywordVector(text),
          })),
          usage: { prompt_tokens: inputs.length, total_tokens: inputs.length },
        }),
      );
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    batches,
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
}

interface ChatMock {
  baseUrl: string;
  hits: number;
  close(): Promise<void>;
}

/**
 * An OpenAI-compatible chat upstream answering `content`, streamed or not as
 * the request asks; `failing` answers every request with a 500.
 */
async function startChatMock(content: string, failing = false): Promise<ChatMock> {
  const mock = { hits: 0 } as ChatMock;
  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      mock.hits += 1;
      if (failing) {
        res.statusCode = 500;
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify({ error: { message: "member down", type: "server_error" } }));
        return;
      }
      const stream = (JSON.parse(raw || "{}") as { stream?: boolean }).stream === true;
      const created = Math.floor(Date.now() / 1000);
      if (!stream) {
        res.setHeader("content-type", "application/json");
        res.end(
          JSON.stringify({
            id: `chatcmpl-${content}`,
            object: "chat.completion",
            created,
            model: "mock-chat",
            choices: [{ index: 0, message: { role: "assistant", content }, finish_reason: "stop" }],
            usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
          }),
        );
        return;
      }
      res.setHeader("content-type", "text/event-stream");
      const chunk = (delta: object, finish: string | null, usage?: object) =>
        `data: ${JSON.stringify({
          id: `chatcmpl-${content}`,
          object: "chat.completion.chunk",
          created,
          model: "mock-chat",
          choices: [{ index: 0, delta, finish_reason: finish }],
          ...(usage ? { usage } : {}),
        })}\n\n`;
      res.write(chunk({ role: "assistant", content }, null));
      res.write(chunk({}, "stop", { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 }));
      res.end("data: [DONE]\n\n");
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  mock.baseUrl = `http://127.0.0.1:${port}`;
  mock.close = () => new Promise<void>((resolve) => server.close(() => resolve()));
  return mock;
}

function field(line: string, name: string): string | undefined {
  const m = new RegExp(`\\b${name}=("(?:[^"\\\\]|\\\\.)*"|\\S+)`).exec(line);
  if (!m) return undefined;
  return m[1].startsWith('"') ? (JSON.parse(m[1]) as string) : m[1];
}

interface Result {
  status: number;
  /** The whole response body as text (an SSE stream included). */
  text: string;
  json: unknown;
  route: string | null;
  requestId: string;
}

describe("semantic routing on every protocol, with model-group targets", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  let embed: EmbeddingMock | undefined;
  const chatMocks: Record<string, ChatMock> = {};
  const anthropicMocks: Record<string, OpenAiUpstream> = {};

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp({ exposeDebugHeaders: true });
    seed = new SeedClient(etcd, app.etcdPrefix);
    embed = await startEmbeddingMock();

    const embedPk = await seed.createProviderKey({
      display_name: "sp-embed-pk",
      secret: "sk-mock",
      api_base: `${embed.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "sp-embed",
      provider: "openai",
      model_name: "embed-mock",
      provider_key_id: embedPk.id,
      embedding: { dimensions: 2, normalize: true },
    });

    // Chat-family targets, reached from every chat-family endpoint (the
    // OpenAI-compatible bridge serves /v1/messages and /v1/responses too).
    const chatTarget = async (
      name: string,
      content: string,
      opts: { failing?: boolean; allowed_cidrs?: string[]; rate_limit?: object } = {},
    ) => {
      const mock = await startChatMock(content, opts.failing);
      chatMocks[name] = mock;
      const pk = await seed!.createProviderKey({
        display_name: `${name}-pk`,
        provider: "deepseek",
        adapter: "openai",
        secret: "sk-mock",
        api_base: `${mock.baseUrl}/v1`,
      });
      await seed!.createModel({
        display_name: name,
        provider: "deepseek",
        model_name: "deepseek-chat",
        provider_key_id: pk.id,
        ...(opts.allowed_cidrs ? { allowed_cidrs: opts.allowed_cidrs } : {}),
        ...(opts.rate_limit ? { rate_limit: opts.rate_limit } : {}),
      });
    };
    await chatTarget("sp-legal", "served-by-legal");
    await chatTarget("sp-default", "served-by-default");
    await chatTarget("sp-member-down", "never-served", { failing: true });
    await chatTarget("sp-member-up", "served-by-group-member");
    await chatTarget("sp-limited-member", "served-by-limited-member");
    // A member this caller (127.0.0.1) may not reach.
    await chatTarget("sp-locked-member", "never-served-locked", {
      allowed_cidrs: ["10.0.0.0/8"],
    });

    // Anthropic targets for count_tokens, which only Anthropic upstreams serve.
    const anthropicTarget = async (name: string, inputTokens: number | "fail") => {
      const mock = await startOpenAiUpstream(
        inputTokens === "fail"
          ? { status: 500, errorBody: { type: "error", error: { type: "api_error", message: "down" } } }
          : { nonStreamBody: { input_tokens: inputTokens } },
      );
      anthropicMocks[name] = mock;
      const pk = await seed!.createProviderKey({
        display_name: `${name}-pk`,
        secret: "sk-ant-mock",
        api_base: mock.baseUrl,
      });
      await seed!.createModel({
        display_name: name,
        provider: "anthropic",
        model_name: "claude-haiku-4-5-20251001",
        provider_key_id: pk.id,
      });
    };
    await anthropicTarget("ct-legal", 11);
    await anthropicTarget("ct-default", 22);
    await anthropicTarget("ct-member-down", "fail");
    await anthropicTarget("ct-member-up", 33);

    const group = (name: string, members: string[], extra: object = {}) =>
      seed!.createModel({
        display_name: name,
        routing: { strategy: "failover", targets: members.map((model) => ({ model })) },
        ...extra,
      });
    const legalGroup = await group("sp-legal-group", ["sp-member-down", "sp-member-up"]);
    await chatTarget("sp-full-a", "served-by-full-a", { rate_limit: { rpm: 1 } });
    await chatTarget("sp-full-b", "served-by-full-b", { rate_limit: { rpm: 1 } });
    await chatTarget("sp-free", "served-by-free");
    await group("sp-locked-group", ["sp-locked-member"]);
    await group("sp-limited-group", ["sp-limited-member"], { rate_limit: { rpm: 2 } });
    await group("ct-legal-group", ["ct-member-down", "ct-member-up"]);
    await group("sp-capped-group", ["sp-member-up"], { rate_limit: { rpm: 1 } });
    await group("sp-full-group", ["sp-full-a", "sp-full-b"]);
    await seed.createRateLimitPolicy({
      name: "sp-group-pinned",
      conditions: [
        { dimension: "team", operator: "in", value: ["sp-team"] },
        { dimension: "model", operator: "in", value: [legalGroup.id] },
      ],
      limits: { rpm: 2 },
    });

    const router = (name: string, legalTarget: string, defaultTarget: string) =>
      seed!.createModel({
        display_name: name,
        semantic: {
          embedding_model: "sp-embed",
          routes: [
            {
              name: "legal",
              target: legalTarget,
              examples: ["analyze this contract for legal risk"],
              threshold: 0.5,
            },
          ],
          default: defaultTarget,
          match: { threshold: 0.5 },
        },
      });
    await router("sp-router", "sp-legal", "sp-default");
    await router("sp-group-router", "sp-legal-group", "sp-default");
    await router("sp-locked-router", "sp-locked-group", "sp-default");
    await router("sp-limited-router", "sp-limited-group", "sp-default");
    await router("ct-router", "ct-legal", "ct-default");
    await router("ct-group-router", "ct-legal-group", "ct-default");
    await router("sp-capped-router", "sp-capped-group", "sp-default");
    await router("sp-full-router", "sp-full-group", "sp-default");
    // An embedding model that does not exist: every decision fails, and
    // the failure target serves.
    await seed.createModel({
      display_name: "sp-failing-router",
      semantic: {
        embedding_model: "sp-no-such-embed",
        routes: [
          { name: "legal", target: "sp-legal", examples: ["analyze this contract"], threshold: 0.5 },
        ],
        default: "sp-default",
        match: { threshold: 0.5 },
        on_embedding_failure: { target: "sp-legal-group" },
      },
    });

    // Seeded last: authenticating it proves every resource above landed.
    await seed.createApiKey({ key_hash: sha256(TEAM_CALLER), allowed_models: ["*"], team_id: "sp-team" });
    await seed.createApiKey({ key_hash: sha256(CAPPED_CALLER), allowed_models: ["*"], rate_limit: { rpm: 1 } });
    await seed.createApiKey({ key_hash: sha256(ALL_REFUSED_CALLER), allowed_models: ["*"], rate_limit: { rpm: 1 } });
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await embed?.close();
    await Promise.all(Object.values(chatMocks).map((m) => m.close()));
    await Promise.all(Object.values(anthropicMocks).map((m) => m.close()));
  });

  const ready = () => etcdReachable && !!app && !!embed;

  async function post(path: string, body: object, anthropic = false, key = CALLER_PLAINTEXT): Promise<Result> {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        ...(anthropic
          ? { "x-api-key": key, "anthropic-version": "2023-06-01" }
          : { authorization: `Bearer ${key}` }),
      },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    let json: unknown;
    try {
      json = JSON.parse(text);
    } catch {
      json = undefined;
    }
    return {
      status: res.status,
      text,
      json,
      route: res.headers.get("x-aisix-route"),
      requestId: res.headers.get("x-aisix-request-id") ?? "",
    };
  }

  const chat = (model: string, messages: object[], stream = false) =>
    post("/v1/chat/completions", { model, messages, stream });
  const messages = (model: string, msgs: object[], stream = false) =>
    post("/v1/messages", { model, max_tokens: 64, messages: msgs, stream }, true);
  const countTokens = (model: string, msgs: object[]) =>
    post("/v1/messages/count_tokens", { model, messages: msgs }, true);
  const responses = (model: string, input: unknown, stream = false) =>
    post("/v1/responses", { model, input, stream });

  /** The text the router classified on the request just made. */
  const classified = () => embed!.batches.at(-1)?.[0];

  async function accessLine(r: Result): Promise<string> {
    expect(r.requestId).not.toBe("");
    return waitForLogLine(
      app!,
      (l) => l.includes("proxy request completed") && l.includes(`"${r.requestId}"`),
      `access-log line for ${r.requestId}`,
    );
  }

  describe("a semantic router on each protocol", () => {
    test("/v1/messages, non-streaming", async (ctx) => {
      if (!ready()) return ctx.skip();
      const r = await messages("sp-router", [{ role: "user", content: LEGAL_ASK }]);
      expect(r.status, r.text).toBe(200);
      expect(r.text).toContain("served-by-legal");
      expect(r.route).toBe("legal");
      const line = await accessLine(r);
      expect(field(line, "semantic_route"), line).toBe("legal");

      const d = await messages("sp-router", [{ role: "user", content: OTHER_ASK }]);
      expect(d.status, d.text).toBe(200);
      expect(d.text).toContain("served-by-default");
      expect(d.route).toBeNull();
      expect(field(await accessLine(d), "semantic_fallback")).toBe("no_match");
    });

    test("/v1/messages, streaming", async (ctx) => {
      if (!ready()) return ctx.skip();
      const r = await messages("sp-router", [{ role: "user", content: LEGAL_ASK }], true);
      expect(r.status, r.text).toBe(200);
      expect(r.text).toContain("event: message_start");
      expect(r.text).toContain("served-by-legal");
      expect(r.route).toBe("legal");
      expect(field(await accessLine(r), "semantic_route")).toBe("legal");
    });

    test("/v1/messages/count_tokens decides the same way", async (ctx) => {
      if (!ready()) return ctx.skip();
      const r = await countTokens("ct-router", [{ role: "user", content: LEGAL_ASK }]);
      expect(r.status, r.text).toBe(200);
      expect(r.json).toEqual({ input_tokens: 11 });
      expect(r.route).toBe("legal");
      expect(classified()).toBe(LEGAL_ASK);

      const d = await countTokens("ct-router", [{ role: "user", content: OTHER_ASK }]);
      expect(d.status, d.text).toBe(200);
      expect(d.json).toEqual({ input_tokens: 22 });
      expect(d.route).toBeNull();
    });

    test("/v1/responses, non-streaming", async (ctx) => {
      if (!ready()) return ctx.skip();
      const r = await responses("sp-router", LEGAL_ASK);
      expect(r.status, r.text).toBe(200);
      expect((r.json as { object?: string }).object).toBe("response");
      expect(r.text).toContain("served-by-legal");
      expect(r.route).toBe("legal");
      expect(field(await accessLine(r), "semantic_route")).toBe("legal");

      const d = await responses("sp-router", OTHER_ASK);
      expect(d.status, d.text).toBe(200);
      expect(d.text).toContain("served-by-default");
      expect(d.route).toBeNull();
    });

    test("/v1/responses, streaming", async (ctx) => {
      if (!ready()) return ctx.skip();
      const r = await responses(
        "sp-router",
        [{ type: "message", role: "user", content: [{ type: "input_text", text: LEGAL_ASK }] }],
        true,
      );
      expect(r.status, r.text).toBe(200);
      expect(r.text).toContain("response.completed");
      expect(r.text).toContain("served-by-legal");
      expect(r.route).toBe("legal");
    });
  });

  describe("what the router classifies", () => {
    test("a newest turn of tool results only is skipped for the ask before it", async (ctx) => {
      if (!ready()) return ctx.skip();
      const r = await messages("sp-router", [
        { role: "user", content: LEGAL_ASK },
        {
          role: "assistant",
          content: [{ type: "tool_use", id: "toolu_1", name: "read_file", input: { path: "a" } }],
        },
        {
          role: "user",
          content: [{ type: "tool_result", tool_use_id: "toolu_1", content: OTHER_ASK }],
        },
      ]);
      expect(r.status, r.text).toBe(200);
      expect(classified()).toBe(LEGAL_ASK);
      expect(r.text).toContain("served-by-legal");

      const resp = await responses("sp-router", [
        { type: "message", role: "user", content: [{ type: "input_text", text: LEGAL_ASK }] },
        { type: "function_call", call_id: "call_1", name: "read_file", arguments: "{}" },
        { type: "function_call_output", call_id: "call_1", output: OTHER_ASK },
      ]);
      expect(resp.status, resp.text).toBe(200);
      expect(classified()).toBe(LEGAL_ASK);
      expect(resp.text).toContain("served-by-legal");
    });

    test("a <system-reminder> block beside the ask is not classified", async (ctx) => {
      if (!ready()) return ctx.skip();
      // The reminder alone would match the legal route; the ask does not.
      const r = await messages("sp-router", [
        {
          role: "user",
          content: [
            { type: "text", text: REMINDER("the user has a contract open in the editor") },
            { type: "text", text: OTHER_ASK },
          ],
        },
      ]);
      expect(r.status, r.text).toBe(200);
      expect(classified()).toBe(OTHER_ASK);
      expect(r.text).toContain("served-by-default");
    });

    test("a newest turn holding only a reminder is skipped", async (ctx) => {
      if (!ready()) return ctx.skip();
      const r = await messages("sp-router", [
        { role: "user", content: LEGAL_ASK },
        { role: "assistant", content: "Sure." },
        { role: "user", content: [{ type: "text", text: REMINDER(OTHER_ASK) }] },
      ]);
      expect(r.status, r.text).toBe(200);
      expect(classified()).toBe(LEGAL_ASK);
      expect(r.text).toContain("served-by-legal");
    });

    test("an image-only newest turn on chat is skipped for the ask before it", async (ctx) => {
      if (!ready()) return ctx.skip();
      const r = await chat("sp-router", [
        { role: "user", content: LEGAL_ASK },
        { role: "assistant", content: "Send me the scan." },
        {
          role: "user",
          content: [{ type: "image_url", image_url: { url: "data:image/png;base64,iVBORw0KGgo=" } }],
        },
      ]);
      expect(r.status, r.text).toBe(200);
      expect(classified()).toBe(LEGAL_ASK);
      expect(r.text).toContain("served-by-legal");
      expect(r.route).toBe("legal");
    });
  });

  describe("a model group as a route target", () => {
    // The group's failover strategy tries `*-member-down` first; it answers
    // 500 and the group fails over to `*-member-up`.
    test("/v1/chat/completions fails over inside the group", async (ctx) => {
      if (!ready()) return ctx.skip();
      const down = chatMocks["sp-member-down"]!.hits;
      const r = await chat("sp-group-router", [{ role: "user", content: LEGAL_ASK }]);
      expect(r.status, r.text).toBe(200);
      expect(r.text).toContain("served-by-group-member");
      expect(r.route).toBe("legal");
      expect(chatMocks["sp-member-down"]!.hits).toBe(down + 1);
    });

    test("/v1/chat/completions streaming fails over inside the group", async (ctx) => {
      if (!ready()) return ctx.skip();
      const down = chatMocks["sp-member-down"]!.hits;
      const r = await chat("sp-group-router", [{ role: "user", content: LEGAL_ASK }], true);
      expect(r.status, r.text).toBe(200);
      expect(r.text).toContain("served-by-group-member");
      expect(chatMocks["sp-member-down"]!.hits).toBe(down + 1);
    });

    test("/v1/messages fails over inside the group", async (ctx) => {
      if (!ready()) return ctx.skip();
      const down = chatMocks["sp-member-down"]!.hits;
      const r = await messages("sp-group-router", [{ role: "user", content: LEGAL_ASK }]);
      expect(r.status, r.text).toBe(200);
      expect(r.text).toContain("served-by-group-member");
      expect(r.route).toBe("legal");
      expect(chatMocks["sp-member-down"]!.hits).toBe(down + 1);
    });

    test("/v1/responses fails over inside the group", async (ctx) => {
      if (!ready()) return ctx.skip();
      const down = chatMocks["sp-member-down"]!.hits;
      const r = await responses("sp-group-router", LEGAL_ASK, true);
      expect(r.status, r.text).toBe(200);
      expect(r.text).toContain("served-by-group-member");
      expect(r.route).toBe("legal");
      expect(chatMocks["sp-member-down"]!.hits).toBe(down + 1);
    });

    test("/v1/messages/count_tokens fails over inside the group", async (ctx) => {
      if (!ready()) return ctx.skip();
      const down = anthropicMocks["ct-member-down"]!.receivedRequests.length;
      const r = await countTokens("ct-group-router", [{ role: "user", content: LEGAL_ASK }]);
      expect(r.status, r.text).toBe(200);
      expect(r.json).toEqual({ input_tokens: 33 });
      expect(r.route).toBe("legal");
      expect(anthropicMocks["ct-member-down"]!.receivedRequests.length).toBe(down + 1);
    });

    test("a failure target that is a group fails over inside the group", async (ctx) => {
      if (!ready()) return ctx.skip();
      const down = chatMocks["sp-member-down"]!.hits;
      const r = await chat("sp-failing-router", [{ role: "user", content: LEGAL_ASK }]);
      expect(r.status, r.text).toBe(200);
      expect(r.text).toContain("served-by-group-member");
      expect(r.route).toBeNull();
      expect(chatMocks["sp-member-down"]!.hits).toBe(down + 1);
      expect(field(await accessLine(r), "semantic_fallback")).toBe("decision_failed");
    });

    test("the group's own rate limit binds traffic the router sends it", async (ctx) => {
      if (!ready()) return ctx.skip();
      await awaitWindowHeadroom(5);
      const ask = [{ role: "user", content: LEGAL_ASK }];
      const first = await chat("sp-limited-router", ask);
      expect(first.status, first.text).toBe(200);
      expect(first.text).toContain("served-by-limited-member");
      const second = await messages("sp-limited-router", ask);
      expect(second.status, second.text).toBe(200);
      const third = await responses("sp-limited-router", LEGAL_ASK);
      expect(third.status, third.text).toBe(429);
      expect(chatMocks["sp-limited-member"]!.hits).toBe(2);
    });

    test("a policy pinned to the group counts once per request, failovers included", async (ctx) => {
      if (!ready()) return ctx.skip();
      await awaitWindowHeadroom(10);
      const ask = { model: "sp-group-router", messages: [{ role: "user", content: LEGAL_ASK }] };
      const down = chatMocks["sp-member-down"]!.hits;
      // Each request fails over from the down member: two attempts per request.
      for (let i = 0; i < 2; i++) {
        const r = await post("/v1/chat/completions", ask, false, TEAM_CALLER);
        expect(r.status, `request ${i + 1}: ${r.text}`).toBe(200);
      }
      expect(chatMocks["sp-member-down"]!.hits - down).toBe(2);
      const third = await post("/v1/chat/completions", ask, false, TEAM_CALLER);
      expect(third.status, third.text).toBe(429);
    });

    test("a request the group's own limit refuses leaves the router's gate unconsumed", async (ctx) => {
      if (!ready()) return ctx.skip();
      await awaitWindowHeadroom(10);
      const ask = (model: string, text: string) => ({ model, messages: [{ role: "user", content: text }] });
      // Fill the group's rpm 1 with another key.
      expect((await chat("sp-capped-router", [{ role: "user", content: LEGAL_ASK }])).status).toBe(200);
      const refused = await post("/v1/chat/completions", ask("sp-capped-router", LEGAL_ASK), false, CAPPED_CALLER);
      expect(refused.status, refused.text).toBe(429);
      // The key's only slot (rpm 1) is still free.
      const ok = await post("/v1/chat/completions", ask("sp-router", OTHER_ASK), false, CAPPED_CALLER);
      expect(ok.status, ok.text).toBe(200);
      const over = await post("/v1/chat/completions", ask("sp-router", OTHER_ASK), false, CAPPED_CALLER);
      expect(over.status).toBe(429);
    });

    test("a request whose every group member refuses leaves the router's gate unconsumed", async (ctx) => {
      if (!ready()) return ctx.skip();
      await awaitWindowHeadroom(10);
      const ask = (model: string, text: string) => ({ model, messages: [{ role: "user", content: text }] });
      expect((await chat("sp-full-a", [{ role: "user", content: OTHER_ASK }])).status).toBe(200);
      expect((await chat("sp-full-b", [{ role: "user", content: OTHER_ASK }])).status).toBe(200);
      const refused = await post("/v1/chat/completions", ask("sp-full-router", LEGAL_ASK), false, ALL_REFUSED_CALLER);
      expect(refused.status, refused.text).toBe(429);
      const ok = await post("/v1/chat/completions", ask("sp-free", OTHER_ASK), false, ALL_REFUSED_CALLER);
      expect(ok.status, ok.text).toBe(200);
      const over = await post("/v1/chat/completions", ask("sp-free", OTHER_ASK), false, ALL_REFUSED_CALLER);
      expect(over.status).toBe(429);
    });

    test("a group with no member this caller may reach is unavailable: default serves", async (ctx) => {
      if (!ready()) return ctx.skip();
      const locked = chatMocks["sp-locked-member"]!.hits;
      for (const r of [
        await chat("sp-locked-router", [{ role: "user", content: LEGAL_ASK }]),
        await messages("sp-locked-router", [{ role: "user", content: LEGAL_ASK }]),
      ]) {
        expect(r.status, r.text).toBe(200);
        expect(r.text).toContain("served-by-default");
        expect(r.route).toBeNull();
        expect(field(await accessLine(r), "semantic_fallback")).toBe("target_unavailable");
      }
      expect(chatMocks["sp-locked-member"]!.hits).toBe(locked);
    });
  });
});
