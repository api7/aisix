import { createHash, randomUUID } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startMockControlPlane,
  waitConfigPropagation,
  waitForLogLine,
  type MockControlPlane,
  type SpawnedApp,
} from "../harness/index.js";
import { pickFreePort } from "../harness/ports.js";

// E2E: semantic routing with session affinity. A request that names a
// session and matches a route pins that session to the route; later
// requests of the session, with the same API key, go to the pinned route
// without being classified, whatever their text says.
//
// The embedding upstream maps text to a one-hot vector by keyword:
// "contract" → the legal route, "python" → the code route, anything else
// matches nothing and goes to `default`. It records every batch, so a
// request that was not classified leaves no batch behind.

const CALLER = "sk-semantic-affinity-e2e";
const OTHER_CALLER = "sk-semantic-affinity-e2e-other";
const hash = (key: string) => createHash("sha256").update(key).digest("hex");

const LEGAL = "please review this contract";
const CODE = "write a python script that sorts a list";
const OTHER = "what time is the meeting tomorrow";

function keywordVector(text: string): number[] {
  const t = text.toLowerCase();
  if (t.includes("contract")) return [1, 0, 0];
  if (t.includes("python")) return [0, 1, 0];
  return [0, 0, 1];
}

interface Mock {
  baseUrl: string;
  hits: number;
  batches: string[][];
  close(): Promise<void>;
}

async function serve(handler: (path: string, body: Record<string, unknown>, mock: Mock) => [number, unknown, string?]): Promise<Mock> {
  const mock = { hits: 0, batches: [] } as unknown as Mock;
  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      mock.hits += 1;
      const [status, body, contentType] = handler(req.url ?? "", JSON.parse(raw || "{}") as Record<string, unknown>, mock);
      res.statusCode = status;
      res.setHeader("content-type", contentType ?? "application/json");
      res.end(typeof body === "string" ? body : JSON.stringify(body));
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  mock.baseUrl = `http://127.0.0.1:${port}`;
  mock.close = () => new Promise<void>((resolve) => server.close(() => resolve()));
  return mock;
}

const startEmbeddingMock = () =>
  serve((_, body, mock) => {
    const inputs = Array.isArray(body.input) ? (body.input as string[]) : [String(body.input ?? "")];
    mock.batches.push(inputs);
    return [
      200,
      {
        object: "list",
        model: "embed-mock",
        data: inputs.map((text, index) => ({ object: "embedding", index, embedding: keywordVector(text) })),
        usage: { prompt_tokens: inputs.length, total_tokens: inputs.length },
      },
    ];
  });

/** An OpenAI-compatible chat upstream answering `content`, streamed or not. */
const startChatMock = (content: string) =>
  serve((_, body) => {
    const created = Math.floor(Date.now() / 1000);
    if (body.stream !== true) {
      return [
        200,
        {
          id: `chatcmpl-${content}`,
          object: "chat.completion",
          created,
          model: "mock-chat",
          choices: [{ index: 0, message: { role: "assistant", content }, finish_reason: "stop" }],
          usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
        },
      ];
    }
    const chunk = (delta: object, finish: string | null, usage?: object) =>
      `data: ${JSON.stringify({
        id: `chatcmpl-${content}`,
        object: "chat.completion.chunk",
        created,
        model: "mock-chat",
        choices: [{ index: 0, delta, finish_reason: finish }],
        ...(usage ? { usage } : {}),
      })}\n\n`;
    return [
      200,
      chunk({ role: "assistant", content }, null) +
        chunk({}, "stop", { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 }) +
        "data: [DONE]\n\n",
      "text/event-stream",
    ];
  });

/** An Anthropic upstream: `count_tokens` answers `tokens`, messages answer `content`. */
const startAnthropicMock = (content: string, tokens: number) =>
  serve((path) =>
    path.includes("count_tokens")
      ? [200, { input_tokens: tokens }]
      : [
          200,
          {
            id: `msg_${content}`,
            type: "message",
            role: "assistant",
            model: "claude-haiku-4-5-20251001",
            content: [{ type: "text", text: content }],
            stop_reason: "end_turn",
            usage: { input_tokens: 3, output_tokens: 2 },
          },
        ],
  );

function field(line: string, name: string): string | undefined {
  const m = new RegExp(`\\b${name}=("(?:[^"\\\\]|\\\\.)*"|\\S+)`).exec(line);
  if (!m) return undefined;
  return m[1].startsWith('"') ? (JSON.parse(m[1]) as string) : m[1];
}

interface Result {
  status: number;
  text: string;
  json: unknown;
  route: string | null;
  requestId: string;
}

type Headers = Record<string, string>;

async function call(baseUrl: string, path: string, body: object, headers: Headers, key = CALLER): Promise<Result> {
  const anthropic = path.startsWith("/v1/messages");
  const res = await fetch(`${baseUrl}${path}`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...(anthropic ? { "x-api-key": key, "anthropic-version": "2023-06-01" } : { authorization: `Bearer ${key}` }),
      ...headers,
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

async function accessLine(app: SpawnedApp, r: Result): Promise<string> {
  expect(r.requestId).not.toBe("");
  return waitForLogLine(
    app,
    (l) => l.includes("proxy request completed") && l.includes(`"${r.requestId}"`),
    `access-log line for ${r.requestId}`,
  );
}

const user = (text: string) => [{ role: "user", content: text }];
const session = () => `sess-${randomUUID()}`;

type Doc = Record<string, unknown>;
type Put = (kind: string, doc: Doc) => Promise<{ id: string; value: Doc }>;

/**
 * A writer that gives each document the same id under every prefix it seeds:
 * gateways of one environment read one set of resources.
 */
function stableWriter(seed: SeedClient, ids: Map<string, string>): Put {
  return async (kind, doc) => {
    const name = `${kind}:${String(doc.display_name ?? doc.key_hash)}`;
    const id = ids.get(name) ?? randomUUID();
    ids.set(name, id);
    await seed.update(kind, id, doc);
    return { id, value: doc };
  };
}

async function startUpstreams(): Promise<Record<string, Mock>> {
  const mocks: Record<string, Mock> = { embed: await startEmbeddingMock() };
  for (const name of ["sa-legal", "sa-code", "sa-default", "sa-flip"]) {
    mocks[name] = await startChatMock(`served-by-${name}`);
  }
  for (const [name, tokens] of [["ct-legal", 11], ["ct-code", 22], ["ct-default", 33]] as const) {
    mocks[name] = await startAnthropicMock(`served-by-${name}`, tokens);
  }
  return mocks;
}

const routerDoc = (name: string, legal: string, code: string, dflt: string, affinity: object | null): Doc => ({
  display_name: name,
  semantic: {
    embedding_model: "sa-embed",
    routes: [
      { name: "legal", target: legal, examples: ["analyze this contract"], threshold: 0.5 },
      { name: "code", target: code, examples: ["python code"], threshold: 0.5 },
    ],
    default: dflt,
    match: { threshold: 0.5 },
    ...(affinity ? { session_affinity: affinity } : {}),
  },
});

/** Seed the targets and routers both suites use. */
async function seedRouters(put: Put, mocks: Record<string, Mock>) {
  const embedPk = await put("provider_keys", {
    display_name: "sa-embed-pk",
    provider: "openai",
    adapter: "openai",
    secret: "sk-mock",
    api_base: `${mocks.embed!.baseUrl}/v1`,
  });
  await put("models", {
    display_name: "sa-embed",
    provider: "openai",
    model_name: "embed-mock",
    provider_key_id: embedPk.id,
    embedding: { dimensions: 3, normalize: true },
  });
  const models: Record<string, { id: string; value: Doc }> = {};
  for (const name of ["sa-legal", "sa-code", "sa-default", "sa-flip"]) {
    const pk = await put("provider_keys", {
      display_name: `${name}-pk`,
      provider: "deepseek",
      adapter: "openai",
      secret: "sk-mock",
      api_base: `${mocks[name]!.baseUrl}/v1`,
    });
    models[name] = await put("models", {
      display_name: name,
      provider: "deepseek",
      model_name: "deepseek-chat",
      provider_key_id: pk.id,
    });
  }
  for (const name of ["ct-legal", "ct-code", "ct-default"]) {
    const pk = await put("provider_keys", {
      display_name: `${name}-pk`,
      provider: "openai",
      adapter: "openai",
      secret: "sk-ant-mock",
      api_base: mocks[name]!.baseUrl,
    });
    await put("models", {
      display_name: name,
      provider: "anthropic",
      model_name: "claude-haiku-4-5-20251001",
      provider_key_id: pk.id,
    });
  }
  const routers: Record<string, { id: string; value: Doc }> = {};
  const on = { enabled: true, ttl_seconds: 600 };
  routers["sa-router"] = await put("models", routerDoc("sa-router", "sa-legal", "sa-code", "sa-default", on));
  routers["sa-off"] = await put("models", routerDoc("sa-off", "sa-legal", "sa-code", "sa-default", { enabled: false, ttl_seconds: 600 }));
  routers["sa-drop"] = await put("models", routerDoc("sa-drop", "sa-legal", "sa-code", "sa-default", on));
  routers["sa-flip-router"] = await put("models", routerDoc("sa-flip-router", "sa-legal", "sa-flip", "sa-default", on));
  routers["ct-router"] = await put("models", routerDoc("ct-router", "ct-legal", "ct-code", "ct-default", on));
  await put("api_keys", { key_hash: hash(OTHER_CALLER), allowed_models: ["*"] });
  // Seeded last: authenticating it proves every resource above landed.
  await put("api_keys", { key_hash: hash(CALLER), allowed_models: ["*"] });
  return { models, routers };
}

describe("semantic session affinity on one gateway", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let embed: Mock | undefined;
  let seeded: Awaited<ReturnType<typeof seedRouters>> | undefined;
  let mocks: Record<string, Mock> = {};
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    app = await spawnApp({ exposeDebugHeaders: true });
    seed = new SeedClient(etcd, app.etcdPrefix);
    mocks = await startUpstreams();
    embed = mocks.embed;
    seeded = await seedRouters(stableWriter(seed, new Map()), mocks);
    const probe = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(Object.values(mocks).map((m) => m.close()));
  });

  const ready = () => etcdReachable && !!app && !!embed;
  const post = (path: string, body: object, headers: Headers = {}, key = CALLER) =>
    call(app!.proxyUrl, path, body, headers, key);
  const chat = (model: string, text: string, headers: Headers = {}, extra: object = {}, key = CALLER) =>
    post("/v1/chat/completions", { model, messages: user(text), ...extra }, headers, key);
  const messages = (model: string, text: string, headers: Headers = {}, extra: object = {}) =>
    post("/v1/messages", { model, max_tokens: 64, messages: user(text), ...extra }, headers);
  const countTokens = (model: string, text: string, headers: Headers = {}) =>
    post("/v1/messages/count_tokens", { model, messages: user(text) }, headers);
  const responses = (model: string, text: string, headers: Headers = {}, stream = false) =>
    post("/v1/responses", { model, input: text, stream }, headers);
  const batches = () => embed!.batches.length;

  /** Wait until every etcd write so far is applied, by seeding a marker model after them. */
  async function settle() {
    const marker = `sa-marker-${randomUUID()}`;
    await seed!.createModel({ ...seeded!.models["sa-default"]!.value, display_name: marker });
    const probe = new ProxyClient(app!.proxyUrl, CALLER);
    await waitConfigPropagation(async () => {
      const r = await probe.listModels();
      return r.status === 200 && JSON.stringify(r.body).includes(marker);
    });
  }

  async function expectPinned(r: Result, target: string, route: string) {
    expect(r.status, r.text).toBe(200);
    expect(r.text).toContain(`served-by-${target}`);
    expect(r.route).toBe(route);
    const line = await accessLine(app!, r);
    expect(field(line, "semantic_pinned"), line).toBe("true");
    expect(field(line, "semantic_route"), line).toBe(route);
    expect(field(line, "semantic_score"), line).toBeUndefined();
  }

  async function expectClassified(r: Result, target: string, route: string | null) {
    expect(r.status, r.text).toBe(200);
    expect(r.text).toContain(`served-by-${target}`);
    expect(r.route).toBe(route);
    const line = await accessLine(app!, r);
    expect(field(line, "semantic_pinned"), line).toBeUndefined();
    expect(field(line, "semantic_score"), line).toBeDefined();
  }

  describe("a matched route pins the session on every protocol", () => {
    test("chat, x-aisix-session-id", async (ctx) => {
      if (!ready()) return ctx.skip();
      const h = { "x-aisix-session-id": session() };
      await expectClassified(await chat("sa-router", LEGAL, h), "sa-legal", "legal");
      const before = batches();
      await expectPinned(await chat("sa-router", CODE, h), "sa-legal", "legal");
      await expectPinned(await chat("sa-router", CODE, h, { stream: true }), "sa-legal", "legal");
      expect(batches(), "a pinned request is not classified").toBe(before);
    });

    test("messages, x-claude-code-session-id", async (ctx) => {
      if (!ready()) return ctx.skip();
      const h = { "x-claude-code-session-id": randomUUID() };
      await expectClassified(await messages("sa-router", CODE, h), "sa-code", "code");
      const before = batches();
      await expectPinned(await messages("sa-router", LEGAL, h), "sa-code", "code");
      expect(batches()).toBe(before);
    });

    test("messages, metadata.user_id as a JSON string", async (ctx) => {
      if (!ready()) return ctx.skip();
      const userId = JSON.stringify({ device_id: "d1", account_uuid: "", session_id: randomUUID() });
      const md = { metadata: { user_id: userId } };
      await expectClassified(await messages("sa-router", LEGAL, {}, md), "sa-legal", "legal");
      await expectPinned(await messages("sa-router", CODE, {}, md), "sa-legal", "legal");
    });

    test("responses, Codex session-id from a Codex caller", async (ctx) => {
      if (!ready()) return ctx.skip();
      const h = { "user-agent": "codex_cli_rs/0.80.0 (Linux; x86_64)", "session-id": randomUUID() };
      await expectClassified(await responses("sa-router", CODE, h), "sa-code", "code");
      const before = batches();
      await expectPinned(await responses("sa-router", LEGAL, h, true), "sa-code", "code");
      expect(batches()).toBe(before);
      // The same header from a caller that is not Codex names no session.
      const plain = { "session-id": h["session-id"] };
      await expectClassified(await responses("sa-router", LEGAL, plain), "sa-legal", "legal");
    });

    test("chat, body metadata.session_id", async (ctx) => {
      if (!ready()) return ctx.skip();
      const md = { metadata: { session_id: session() } };
      await expectClassified(await chat("sa-router", CODE, {}, md), "sa-code", "code");
      await expectPinned(await chat("sa-router", LEGAL, {}, md), "sa-code", "code");
    });
  });

  test("count_tokens follows a pin but never creates one", async (ctx) => {
    if (!ready()) return ctx.skip();
    const h = { "x-aisix-session-id": session() };
    const first = await countTokens("ct-router", LEGAL, h);
    expect(first.json).toEqual({ input_tokens: 11 });
    const second = await countTokens("ct-router", CODE, h);
    expect(second.json, "count_tokens pinned nothing").toEqual({ input_tokens: 22 });

    await expectClassified(await messages("ct-router", LEGAL, h), "ct-legal", "legal");
    const pinned = await countTokens("ct-router", CODE, h);
    expect(pinned.json).toEqual({ input_tokens: 11 });
    expect(pinned.route).toBe("legal");
    expect(field(await accessLine(app!, pinned), "semantic_pinned")).toBe("true");
  });

  describe("a client's side request follows a pin but never sets one", () => {
    const CLAUDE_TITLE = (ask: string) =>
      `<session>\n${ask}\n</session>\n\nWrite the title in the predominant language of the session — a stray word or code token in another language doesn't change it, and neither does the English of these instructions.`;
    const CODEX_UA = "codex_cli_rs/0.167.0 (Linux; x86_64)";

    for (const [protocol, send] of [
      ["messages", messages],
      ["chat", chat],
    ] as const) {
      test(`${protocol}: a title request is classified and pins nothing`, async (ctx) => {
        if (!ready()) return ctx.skip();
        const h = { "x-claude-code-session-id": randomUUID() };
        await expectClassified(await send("sa-router", CLAUDE_TITLE(LEGAL), h), "sa-legal", "legal");
        await expectClassified(await send("sa-router", CODE, h), "sa-code", "code");
        await expectPinned(await send("sa-router", LEGAL, h), "sa-code", "code");
      });
    }

    test("a title request in a pinned session goes to the pin", async (ctx) => {
      if (!ready()) return ctx.skip();
      const h = { "x-claude-code-session-id": randomUUID() };
      await expectClassified(await messages("sa-router", LEGAL, h), "sa-legal", "legal");
      await expectPinned(await messages("sa-router", CLAUDE_TITLE(CODE), h), "sa-legal", "legal");
    });

    test("x-claude-code-request-class: auxiliary pins nothing", async (ctx) => {
      if (!ready()) return ctx.skip();
      const session = { "x-claude-code-session-id": randomUUID() };
      const aux = { ...session, "x-claude-code-request-class": "auxiliary" };
      await expectClassified(await messages("sa-router", LEGAL, aux), "sa-legal", "legal");
      await expectClassified(await messages("sa-router", CODE, session), "sa-code", "code");
      await expectPinned(await messages("sa-router", LEGAL, aux), "sa-code", "code");
    });

    test("responses: a Codex title request pins nothing", async (ctx) => {
      if (!ready()) return ctx.skip();
      const h = { "user-agent": CODEX_UA, "session-id": randomUUID() };
      const title = `Generate a concise, single-line task title of at most 36 characters and under five words where possible.\n\n${LEGAL}`;
      await expectClassified(await responses("sa-router", title, h), "sa-legal", "legal");
      await expectClassified(await responses("sa-router", CODE, h), "sa-code", "code");
      await expectPinned(await responses("sa-router", LEGAL, h), "sa-code", "code");
    });

    test("responses: what Codex injects into a user turn is not classified", async (ctx) => {
      if (!ready()) return ctx.skip();
      const input = [
        {
          type: "message",
          role: "user",
          content: [
            { type: "input_text", text: `# AGENTS.md instructions for /repo\n\n<INSTRUCTIONS>\n${LEGAL}\n</INSTRUCTIONS>` },
            { type: "input_text", text: "<environment_context>\n  <cwd>/repo</cwd>\n  <note>contract</note>\n</environment_context>" },
          ],
        },
        {
          type: "message",
          role: "user",
          content: [{ type: "input_image", image_url: "data:image/png;base64,iVBORw0KGgo=" }],
        },
      ];
      const codex = { "user-agent": CODEX_UA, "session-id": randomUUID() };
      const r = await post("/v1/responses", { model: "sa-router", input }, codex);
      expect(r.status, r.text).toBe(200);
      expect(r.text).toContain("served-by-sa-default");
      expect(r.route).toBeNull();
      expect(field(await accessLine(app!, r), "semantic_fallback")).toBe("empty_prompt");
      await expectClassified(await responses("sa-router", CODE, codex), "sa-code", "code");

      // From any other client the same turn is text, and is classified.
      const other = { "x-aisix-session-id": session() };
      await expectClassified(await post("/v1/responses", { model: "sa-router", input }, other), "sa-legal", "legal");
    });
  });

  test("a request without a session id is classified every time", async (ctx) => {
    if (!ready()) return ctx.skip();
    await expectClassified(await chat("sa-router", LEGAL), "sa-legal", "legal");
    await expectClassified(await chat("sa-router", CODE), "sa-code", "code");
  });

  test("a request that falls back pins nothing; a later match does", async (ctx) => {
    if (!ready()) return ctx.skip();
    const h = { "x-aisix-session-id": session() };
    const d = await chat("sa-router", OTHER, h);
    expect(d.text).toContain("served-by-sa-default");
    expect(field(await accessLine(app!, d), "semantic_fallback")).toBe("no_match");
    await expectClassified(await chat("sa-router", CODE, h), "sa-code", "code");
    await expectPinned(await chat("sa-router", LEGAL, h), "sa-code", "code");
  });

  test("two API keys with one session id are pinned separately", async (ctx) => {
    if (!ready()) return ctx.skip();
    const h = { "x-aisix-session-id": session() };
    await expectClassified(await chat("sa-router", LEGAL, h), "sa-legal", "legal");
    await expectClassified(await chat("sa-router", CODE, h, {}, OTHER_CALLER), "sa-code", "code");
    await expectPinned(await chat("sa-router", CODE, h), "sa-legal", "legal");
    await expectPinned(await chat("sa-router", LEGAL, h, {}, OTHER_CALLER), "sa-code", "code");
  });

  test("with enabled: false nothing is pinned", async (ctx) => {
    if (!ready()) return ctx.skip();
    const h = { "x-aisix-session-id": session() };
    await expectClassified(await chat("sa-off", LEGAL, h), "sa-legal", "legal");
    await expectClassified(await chat("sa-off", CODE, h), "sa-code", "code");
  });

  test("turning affinity off ignores existing pins", async (ctx) => {
    if (!ready()) return ctx.skip();
    const h = { "x-aisix-session-id": session() };
    const { routers } = seeded!;
    const id = routers["sa-drop"]!.id;
    await expectClassified(await chat("sa-drop", LEGAL, h), "sa-legal", "legal");
    await seed!.update("models", id, routerDoc("sa-drop", "sa-legal", "sa-code", "sa-default", { enabled: false }));
    await settle();
    await expectClassified(await chat("sa-drop", CODE, h), "sa-code", "code");
    await seed!.update("models", id, routers["sa-drop"]!.value);
    await settle();
    await expectPinned(await chat("sa-drop", CODE, h), "sa-legal", "legal");
  });

  test("a pin is dropped once its target is no longer a route target", async (ctx) => {
    if (!ready()) return ctx.skip();
    const h = { "x-aisix-session-id": session() };
    const { routers } = seeded!;
    const id = routers["sa-drop"]!.id;
    await expectClassified(await chat("sa-drop", LEGAL, h), "sa-legal", "legal");
    // The legal route now sends to sa-flip: sa-legal is no route's target.
    await seed!.update("models", id, routerDoc("sa-drop", "sa-flip", "sa-code", "sa-default", { enabled: true, ttl_seconds: 600 }));
    await settle();
    try {
      await expectClassified(await chat("sa-drop", CODE, h), "sa-code", "code");
      // The match re-pinned the session, to the code route.
      await expectPinned(await chat("sa-drop", LEGAL, h), "sa-code", "code");
    } finally {
      await seed!.update("models", id, routers["sa-drop"]!.value);
      await settle();
    }
  });

  test("an unavailable pinned target serves default and keeps the pin", async (ctx) => {
    if (!ready()) return ctx.skip();
    const h = { "x-aisix-session-id": session() };
    const { models } = seeded!;
    const flip = models["sa-flip"]!;
    await expectClassified(await chat("sa-flip-router", CODE, h), "sa-flip", "code");
    await seed!.update("models", flip.id, { ...flip.value, allowed_cidrs: ["10.0.0.0/8"] });
    await settle();
    try {
      const before = batches();
      const r = await chat("sa-flip-router", LEGAL, h);
      expect(r.status, r.text).toBe(200);
      expect(r.text).toContain("served-by-sa-default");
      expect(r.route).toBeNull();
      const line = await accessLine(app!, r);
      expect(field(line, "semantic_fallback"), line).toBe("target_unavailable");
      expect(field(line, "semantic_pinned"), line).toBe("true");
      expect(batches()).toBe(before);
    } finally {
      await seed!.update("models", flip.id, flip.value);
      await settle();
    }
    await expectPinned(await chat("sa-flip-router", LEGAL, h), "sa-flip", "code");
  });
});

describe("semantic session affinity across managed gateways", () => {
  let cp: MockControlPlane | undefined;
  const apps: SpawnedApp[] = [];
  let mocks: Record<string, Mock> = {};
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    cp = await startMockControlPlane();
    mocks = await startUpstreams();
    for (let i = 0; i < 2; i++) {
      const app = await spawnApp({
        exposeDebugHeaders: true,
        extra: { managed: cp.managed },
        awaitProxyListener: false,
      });
      apps.push(app);
    }
    // Each gateway reads its own prefix; the same documents under the same
    // ids make them two gateways of one environment.
    const ids = new Map<string, string>();
    for (const app of apps) {
      await seedRouters(stableWriter(new SeedClient(etcd, `${app.etcdPrefix}/${cp.envId}`), ids), mocks);
    }
    for (const app of apps) {
      const probe = new ProxyClient(app.proxyUrl, CALLER);
      await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
    }
  });

  afterAll(async () => {
    await Promise.all(apps.map((a) => a.exit()));
    await cp?.close();
    await Promise.all(Object.values(mocks).map((m) => m.close()));
  });

  const ready = () => etcdReachable && apps.length === 2;
  const chat = (app: SpawnedApp, text: string, h: Headers) =>
    call(app.proxyUrl, "/v1/chat/completions", { model: "sa-router", messages: user(text) }, h);
  const servedBy = (r: Result) => /served-by-(sa-[a-z]+)/.exec(r.text)?.[1];

  test("a pin made on one gateway holds on the other", async (ctx) => {
    if (!ready()) return ctx.skip();
    const [a, b] = apps as [SpawnedApp, SpawnedApp];
    const h = { "x-aisix-session-id": session() };
    expect(servedBy(await chat(a, LEGAL, h))).toBe("sa-legal");
    const r = await chat(b, CODE, h);
    expect(servedBy(r), r.text).toBe("sa-legal");
    expect(field(await accessLine(b, r), "semantic_pinned")).toBe("true");
    const keys = [...cp!.store.keys()].filter((k) => k.startsWith("session_affinity/"));
    expect(keys.length).toBeGreaterThan(0);
    expect(keys.every((k) => /^session_affinity\/[0-9a-f]{64}$/.test(k)), keys.join()).toBe(true);
  });

  test("a side request writes no pin to the store", async (ctx) => {
    if (!ready()) return ctx.skip();
    const [a, b] = apps as [SpawnedApp, SpawnedApp];
    const h = { "x-aisix-session-id": session(), "x-claude-code-request-class": "auxiliary" };
    const stored = () => [...cp!.store.keys()].filter((k) => k.startsWith("session_affinity/")).length;
    const before = stored();
    expect(servedBy(await chat(a, LEGAL, h))).toBe("sa-legal");
    expect(stored()).toBe(before);
    const r = await chat(b, CODE, h);
    expect(servedBy(r), r.text).toBe("sa-code");
    expect(field(await accessLine(b, r), "semantic_pinned")).toBeUndefined();
  });

  test("a gateway that lost the race to pin goes where the first pin went", async (ctx) => {
    if (!ready()) return ctx.skip();
    const [a, b] = apps as [SpawnedApp, SpawnedApp];
    const h = { "x-aisix-session-id": session() };
    // b learns the session holds no pin, and will not read the store for it again for a while.
    expect(servedBy(await chat(b, OTHER, h))).toBe("sa-default");
    expect(servedBy(await chat(a, LEGAL, h))).toBe("sa-legal");
    // b's own match loses the create to a's pin, and follows it.
    const r = await chat(b, CODE, h);
    expect(servedBy(r), r.text).toBe("sa-legal");
    expect(field(await accessLine(b, r), "semantic_pinned")).toBe("true");
  });

  test("simultaneous first requests converge on one pin", async (ctx) => {
    if (!ready()) return ctx.skip();
    const [a, b] = apps as [SpawnedApp, SpawnedApp];
    const h = { "x-aisix-session-id": session() };
    const [ra, rb] = await Promise.all([chat(a, LEGAL, h), chat(b, CODE, h)]);
    expect(servedBy(ra), ra.text).toBe(servedBy(rb));
    const pinned = servedBy(ra);
    expect(servedBy(await chat(a, OTHER, h))).toBe(pinned);
    expect(servedBy(await chat(b, OTHER, h))).toBe(pinned);
  });

  test("with the store down, requests are served and pinned on the gateway", async (ctx) => {
    if (!ready()) return ctx.skip();
    const [a] = apps as [SpawnedApp, SpawnedApp];
    cp!.storeStatusOverride = 503;
    try {
      const h = { "x-aisix-session-id": session() };
      const first = await chat(a, LEGAL, h);
      expect(first.status, first.text).toBe(200);
      expect(servedBy(first)).toBe("sa-legal");
      const started = Date.now();
      const second = await chat(a, CODE, h);
      expect(servedBy(second), second.text).toBe("sa-legal");
      expect(Date.now() - started, "a request does not wait on a store that is down").toBeLessThan(900);
    } finally {
      cp!.storeStatusOverride = undefined;
    }
  });
});
