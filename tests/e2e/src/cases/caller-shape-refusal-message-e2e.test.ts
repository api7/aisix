import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  waitConfigPropagation,
  waitForLogLine,
  type SpawnedApp,
} from "../harness/index.js";

// A request whose shape a provider's bridge cannot translate — the legacy
// `functions` fields, a non-function tool, a tool result with no call
// before it — is refused with 400 `invalid_request_error` before anything
// is sent upstream. The message has to point at the request: one prefixed
// "invalid upstream configuration" sends the caller to the ProviderKey and
// Model, where there is nothing to fix.
//
// The refusals are raised while the upstream body is being built, so the
// upstream here only counts requests; every case asserts it saw none.
//
// The records name the refusal the same way the caller's envelope does: the
// access log's `error_kind` (which the usage event's `error_class` mirrors)
// is `invalid_request_error`, the class every other refusal of the caller's
// request records — not `invalid_config`, which points an operator at the
// target's configuration.

function field(line: string, name: string): string | undefined {
  const m = new RegExp(`(?:^|\\s)${name}=("(?:[^"\\\\]|\\\\.)*"|\\S+)`).exec(line);
  if (!m) return undefined;
  return m[1].startsWith('"') ? (JSON.parse(m[1]) as string) : m[1];
}

const CALLER_PLAINTEXT = "sk-caller-shape-refusal";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const WEATHER_FUNCTION = {
  name: "get_weather",
  parameters: { type: "object", properties: { city: { type: "string" } } },
};

function post(app: SpawnedApp, path: string, body: unknown): Promise<Response> {
  return fetch(`${app.proxyUrl}${path}`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "x-api-key": CALLER_PLAINTEXT,
      "content-type": "application/json",
    },
    body: JSON.stringify(body),
  });
}

describe("a refused request shape is described as a request error", () => {
  let app: SpawnedApp | undefined;
  let upstream: Server | undefined;
  let upstreamHits = 0;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = createServer((req, res) => {
      upstreamHits += 1;
      req.resume();
      res.writeHead(500, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: { message: "upstream must not be reached" } }));
    });
    await new Promise<void>((resolve) => upstream!.listen(0, "127.0.0.1", resolve));
    const addr = upstream.address();
    if (addr === null || typeof addr === "string") throw new Error("no port");
    const baseUrl = `http://127.0.0.1:${addr.port}`;

    // The access log is an INFO line.
    app = await spawnApp({ logLevel: "info" });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const vertexPk = await seed.createProviderKey({
      display_name: "refusal-vertex-pk",
      provider: "google",
      adapter: "vertex",
      secret: JSON.stringify({
        access_token: "ya29.caller-shape-refusal",
        project: "proj-e2e",
        region: "us-central1",
      }),
      api_base: baseUrl,
    });
    await seed.createModel({
      display_name: "refusal-gemini",
      provider: "google",
      model_name: "gemini-2.5-flash",
      provider_key_id: vertexPk.id,
    });
    const bedrockPk = await seed.createProviderKey({
      display_name: "refusal-bedrock-pk",
      provider: "bedrock",
      adapter: "bedrock",
      secret: JSON.stringify({
        access_key_id: "AKIA-refusal",
        secret_access_key: "sk-refusal",
        region: "us-east-1",
      }),
      api_base: baseUrl,
    });
    await seed.createModel({
      display_name: "refusal-nova",
      provider: "bedrock",
      model_name: "us.amazon.nova-lite-v1:0",
      provider_key_id: bedrockPk.id,
    });
    await seed.createModel({
      display_name: "refusal-bedrock-claude",
      provider: "bedrock",
      model_name: "us.anthropic.claude-haiku-4-5-20251001-v1:0",
      provider_key_id: bedrockPk.id,
    });
    await seed.createModel({
      display_name: "refusal-vertex-claude",
      provider: "google",
      model_name: "claude-sonnet-4-5@20250929",
      provider_key_id: vertexPk.id,
    });
    const anthropicPk = await seed.createProviderKey({
      display_name: "refusal-anthropic-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-refusal",
      api_base: baseUrl,
    });
    await seed.createModel({
      display_name: "refusal-claude",
      provider: "anthropic",
      model_name: "claude-sonnet-4-5",
      provider_key_id: anthropicPk.id,
    });
    // Seeded last, so this key authenticating implies the whole seed set
    // has reached the gateway's snapshot.
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
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
    await new Promise<void>((resolve) => (upstream ? upstream.close(() => resolve()) : resolve()));
  });

  async function expectRequestRefusal(res: Response, mentions: RegExp) {
    expect(res.status).toBe(400);
    const body = (await res.json()) as any;
    expect(body.error.type).toBe("invalid_request_error");
    expect(body.error.message).not.toMatch(/invalid upstream configuration/i);
    expect(body.error.message).toMatch(mentions);
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    expect(requestId).not.toBe("");
    const line = await waitForLogLine(
      app!,
      (l) => l.includes(requestId) && l.includes("proxy request completed"),
      `the access log of ${requestId}`,
    );
    expect(field(line, "error_kind"), line).toBe("invalid_request_error");
  }

  const USER = { role: "user", content: "Weather in Paris?" };
  const ORPHAN_TOOL_RESULT = { role: "tool", tool_call_id: "call_unknown", content: "sunny" };

  test.for<[string, string, Record<string, unknown>, RegExp]>([
    ["gemini: legacy functions", "refusal-gemini", { functions: [WEATHER_FUNCTION] }, /^`functions` is not supported/],
    [
      "gemini: legacy function_call",
      "refusal-gemini",
      { tools: [{ type: "function", function: WEATHER_FUNCTION }], function_call: "auto" },
      /^`function_call` is not supported/,
    ],
    ["gemini: a non-function tool", "refusal-gemini", { tools: [{ type: "web_search" }] }, /^tool type "web_search" is not supported/],
    [
      "gemini: an orphan tool result",
      "refusal-gemini",
      { messages: [USER, ORPHAN_TOOL_RESULT] },
      /^tool message has no corresponding tool call/,
    ],
    ["bedrock: legacy functions", "refusal-nova", { functions: [WEATHER_FUNCTION] }, /^`functions` is not supported/],
  ])("chat: %s", async ([, model, extra, mentions], ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    for (const stream of [false, true]) {
      const before = upstreamHits;
      const res = await post(app, "/v1/chat/completions", {
        model,
        messages: [USER],
        stream,
        ...extra,
      });
      await expectRequestRefusal(res, mentions);
      expect(upstreamHits, `stream=${stream}`).toBe(before);
    }
  });

  // A tool result has to name the call it answers; the Anthropic-wire
  // bridges cannot pair one that does not. Bedrock streams Claude through
  // ConverseStream instead of `/invoke`, so only its non-streaming call is
  // translated here.
  const UNNAMED_TOOL_RESULT = [
    USER,
    {
      role: "assistant",
      content: null,
      tool_calls: [
        { id: "call_1", type: "function", function: { name: "get_weather", arguments: "{}" } },
      ],
    },
    { role: "tool", content: "sunny" },
  ];
  test.for<[string, boolean[]]>([
    ["refusal-claude", [false, true]],
    ["refusal-vertex-claude", [false, true]],
    ["refusal-bedrock-claude", [false]],
  ])("chat: a tool message without tool_call_id on %s", async ([model, streams], ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    for (const stream of streams) {
      const before = upstreamHits;
      const res = await post(app, "/v1/chat/completions", {
        model,
        stream,
        tools: [{ type: "function", function: WEATHER_FUNCTION }],
        messages: UNNAMED_TOOL_RESULT,
      });
      await expectRequestRefusal(res, /^tool message missing tool_call_id/);
      expect(upstreamHits, `stream=${stream}`).toBe(before);
    }
  });

  // A request with nothing but instructions gives the model no turn to
  // answer, and Bedrock rejects it on every rail. It is the request that has
  // to change, so the gateway refuses it the same way whichever rail serves
  // it: Converse for Nova, `/invoke` for a non-streaming Claude call, and
  // ConverseStream for a streaming one. Each entry point reaches the same
  // bridge.
  const SYSTEM_ONLY: Record<string, (model: string, stream: boolean) => unknown> = {
    "/v1/chat/completions": (model, stream) => ({
      model,
      stream,
      messages: [{ role: "system", content: "You are terse." }],
    }),
    "/v1/messages": (model, stream) => ({
      model,
      stream,
      max_tokens: 64,
      system: "You are terse.",
      messages: [],
    }),
    "/v1/responses": (model, stream) => ({
      model,
      stream,
      instructions: "You are terse.",
      input: [],
    }),
  };
  test.for<[string, string]>(
    ["refusal-nova", "refusal-bedrock-claude"].flatMap((model) =>
      Object.keys(SYSTEM_ONLY).map((path): [string, string] => [path, model]),
    ),
  )("%s: a system-only request on %s", async ([path, model], ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    for (const stream of [false, true]) {
      const before = upstreamHits;
      const res = await post(app, path, SYSTEM_ONLY[path]!(model, stream));
      await expectRequestRefusal(res, /at least one user or assistant turn/);
      expect(upstreamHits, `stream=${stream}`).toBe(before);
    }
  });

  test("messages: a tool_result with no tool_use before it", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    for (const stream of [false, true]) {
      const before = upstreamHits;
      const res = await post(app, "/v1/messages", {
        model: "refusal-gemini",
        max_tokens: 64,
        stream,
        messages: [
          { role: "user", content: "Weather in Paris?" },
          {
            role: "user",
            content: [{ type: "tool_result", tool_use_id: "toolu_unknown", content: "sunny" }],
          },
        ],
      });
      await expectRequestRefusal(res, /^tool message has no corresponding tool call/);
      expect(upstreamHits, `stream=${stream}`).toBe(before);
    }
  });

  test("responses: a function_call_output with no function_call before it", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    for (const stream of [false, true]) {
      const before = upstreamHits;
      const res = await post(app, "/v1/responses", {
        model: "refusal-gemini",
        stream,
        input: [
          { role: "user", content: "Weather in Paris?" },
          { type: "function_call_output", call_id: "call_unknown", output: "sunny" },
        ],
      });
      await expectRequestRefusal(res, /^tool message has no corresponding tool call/);
      expect(upstreamHits, `stream=${stream}`).toBe(before);
    }
  });
});
