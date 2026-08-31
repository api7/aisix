import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  pickFreePort,
  scrapeMetrics,
  spawnApp,
  sumMetric,
  waitConfigPropagation,
  type SpawnedApp,
} from "../harness/index.js";

// E2E for `provider_keys[].apis` (AISIX-Cloud#1388 and the two-protocol
// upstream). One credential, one host, two protocol entry points — the
// shape DeepSeek/Zhipu/Kimi ship: an OpenAI-compatible path and an
// Anthropic-compatible one, plus no `/v1/responses` route at all.
//
// The mock below is deliberately path-strict, because that is the whole
// point of the field: a request that lands on the wrong path must fail
// visibly here rather than be absorbed by a mock that answers everything.

const CALLER_PLAINTEXT = "sk-pk-apis-e2e";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const CHAT_REPLY = {
  id: "chatcmpl-apis-1",
  object: "chat.completion",
  created: 1730000000,
  model: "upstream-model",
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: "from chat completions" },
      finish_reason: "stop",
    },
  ],
  usage: { prompt_tokens: 11, completion_tokens: 5, total_tokens: 16 },
};

const MESSAGES_REPLY = {
  id: "msg_apis_1",
  type: "message",
  role: "assistant",
  model: "upstream-model",
  content: [{ type: "text", text: "from anthropic messages" }],
  stop_reason: "end_turn",
  usage: { input_tokens: 12, output_tokens: 6 },
};

const COUNT_TOKENS_REPLY = { input_tokens: 42 };

// An OpenAI Responses-API reply, for the upstream that does implement
// the route — on a path of its own, so the test can tell the declared
// entry's base from `api_base`.
const RESPONSES_REPLY = {
  id: "resp_apis_1",
  object: "response",
  created_at: 1730000000,
  model: "upstream-model",
  status: "completed",
  output: [
    {
      type: "message",
      id: "msg_apis_1",
      role: "assistant",
      status: "completed",
      content: [{ type: "output_text", text: "from the responses route", annotations: [] }],
    },
  ],
  usage: { input_tokens: 9, output_tokens: 4, total_tokens: 13 },
};

interface DualProtocolUpstream {
  baseUrl: string;
  /** Every request the mock saw, in order. */
  seen: { method: string; path: string; body: string }[];
  close(): Promise<void>;
}

/**
 * An upstream that serves the OpenAI wire under `/v1` and the Anthropic
 * wire under `/anthropic/v1`, and has **no** `/v1/responses` route — the
 * exact combination `api_base` alone cannot describe.
 */
async function startDualProtocolUpstream(): Promise<DualProtocolUpstream> {
  const seen: DualProtocolUpstream["seen"] = [];
  const server: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c) => chunks.push(c as Buffer));
    req.on("end", () => {
      const body = Buffer.concat(chunks).toString("utf8");
      const path = (req.url ?? "").split("?")[0].replace(/\/+$/, "");
      seen.push({ method: req.method ?? "", path, body });

      const json = (status: number, payload: unknown) => {
        const data = JSON.stringify(payload);
        res.writeHead(status, {
          "content-type": "application/json",
          "content-length": Buffer.byteLength(data),
        });
        res.end(data);
      };

      switch (path) {
        case "/v1/chat/completions":
          return json(200, CHAT_REPLY);
        case "/anthropic/v1/messages":
          return json(200, MESSAGES_REPLY);
        case "/anthropic/v1/messages/count_tokens":
          return json(200, COUNT_TOKENS_REPLY);
        case "/responses-host/v1/responses":
          return json(200, RESPONSES_REPLY);
        default:
          // Everything else — `/v1/responses` and `/v1/messages` included
          // — is absent, the way a chat-completions-only deployment and a
          // single-path vendor really behave.
          return json(404, { detail: "Not Found" });
      }
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    seen,
    close: () => new Promise((resolve) => server.close(() => resolve())),
  };
}

function callGateway(
  app: SpawnedApp,
  path: string,
  body: unknown,
): Promise<Response> {
  return fetch(`${app.proxyUrl}${path}`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    },
    body: JSON.stringify(body),
  });
}

describe("provider_keys[].apis — native surfaces and their entry points", () => {
  let app: SpawnedApp | undefined;
  let upstream: DualProtocolUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startDualProtocolUpstream();
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    // 1. The #1388 shape: the `openai` catalog vendor pointed at an
    //    OpenAI-compatible endpoint that has no `/v1/responses`. An empty
    //    `apis` map is the operator saying exactly that.
    const declaredPk = await seed.createProviderKey({
      display_name: "apis-declared",
      provider: "openai",
      adapter: "openai",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
      apis: {},
    });
    await seed.createModel({
      display_name: "declared-chat-only",
      provider: "openai",
      model_name: "upstream-model",
      provider_key_id: declaredPk.id,
    });

    // 2. The same endpoint with no `apis` map at all — the control that
    //    pins the pre-field behavior (vendor id ⇒ verbatim passthrough).
    const legacyPk = await seed.createProviderKey({
      display_name: "apis-legacy",
      provider: "openai",
      adapter: "openai",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "legacy-chat-only",
      provider: "openai",
      model_name: "upstream-model",
      provider_key_id: legacyPk.id,
    });

    // 3. Two protocols, one credential: OpenAI wire at `/v1`, Anthropic
    //    wire at `/anthropic`. One Model, addressable by both.
    const dualPk = await seed.createProviderKey({
      display_name: "apis-dual",
      provider: "deepseek",
      adapter: "openai",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
      apis: { messages: { base: `${upstream.baseUrl}/anthropic` } },
    });
    await seed.createModel({
      display_name: "dual-protocol",
      provider: "deepseek",
      model_name: "upstream-model",
      provider_key_id: dualPk.id,
    });

    // 4. An anthropic-adapter key whose Anthropic wire lives on the
    //    declared path. Its OpenAI-wire traffic is translated INTO that
    //    wire by the bridge, which has to reach the same host the
    //    verbatim passthrough does.
    const anthropicPk = await seed.createProviderKey({
      display_name: "apis-anthropic",
      provider: "byo",
      adapter: "anthropic",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
      apis: { messages: { base: `${upstream.baseUrl}/anthropic` } },
    });
    await seed.createModel({
      display_name: "anthropic-declared",
      provider: "byo",
      model_name: "upstream-model",
      provider_key_id: anthropicPk.id,
    });

    // 5. A non-OpenAI vendor whose endpoint DOES serve `/v1/responses`,
    //    on a path of its own. Nothing but the declaration can reach
    //    this: the vendor id says "not OpenAI", so without `apis` the
    //    request would be translated.
    const responsesPk = await seed.createProviderKey({
      display_name: "apis-responses",
      provider: "deepseek",
      adapter: "openai",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
      // A version-bearing root, the shape an OpenAI-compatible endpoint
      // is configured with: the endpoint path is appended to it as-is.
      apis: { responses: { base: `${upstream.baseUrl}/responses-host/v1` } },
    });
    await seed.createModel({
      display_name: "responses-declared",
      provider: "deepseek",
      model_name: "upstream-model",
      provider_key_id: responsesPk.id,
    });

    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [
        "declared-chat-only",
        "legacy-chat-only",
        "dual-protocol",
        "anthropic-declared",
        "responses-declared",
      ],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("an apis map without `responses` translates /v1/responses instead of 404ing upstream", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    await waitConfigPropagation(async () => {
      try {
        const r = await callGateway(app!, "/v1/chat/completions", {
          model: "declared-chat-only",
          messages: [{ role: "user", content: "ready" }],
        });
        return r.status === 200;
      } catch {
        return false;
      }
    });

    const before = upstream.seen.length;
    const res = await callGateway(app, "/v1/responses", {
      model: "declared-chat-only",
      input: "hello",
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as Record<string, unknown>;
    expect(body.object).toBe("response");
    expect(JSON.stringify(body.output)).toContain("from chat completions");

    const dispatched = upstream.seen.slice(before);
    expect(dispatched.map((r) => r.path)).toEqual(["/v1/chat/completions"]);
  });

  test("without an apis map the same endpoint still takes the verbatim /v1/responses passthrough", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    const before = upstream.seen.length;
    const res = await callGateway(app, "/v1/responses", {
      model: "legacy-chat-only",
      input: "hello",
    });
    // The upstream has no such route, so this is the failure #1388
    // reported — reproduced here to pin that the `apis` map, and nothing
    // else, is what changes the dispatch.
    expect(res.status).toBe(404);
    expect(upstream.seen.slice(before).map((r) => r.path)).toEqual([
      "/v1/responses",
    ]);
  });

  test("a declared `messages` entry forwards the Anthropic body verbatim to its own path", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    const before = upstream.seen.length;
    const res = await callGateway(app, "/v1/messages", {
      model: "dual-protocol",
      max_tokens: 64,
      system: [
        {
          type: "text",
          text: "you are a helpful assistant",
          cache_control: { type: "ephemeral" },
        },
      ],
      messages: [{ role: "user", content: "hello" }],
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as Record<string, unknown>;
    expect(body.type).toBe("message");

    const dispatched = upstream.seen.slice(before);
    expect(dispatched.map((r) => r.path)).toEqual(["/anthropic/v1/messages"]);
    // Verbatim, not translated: the prompt-cache breakpoint survives.
    // Going through the chat bridge would have dropped it, which is the
    // cost this field exists to avoid.
    const forwarded = JSON.parse(dispatched[0].body) as Record<string, unknown>;
    expect(JSON.stringify(forwarded.system)).toContain("cache_control");
    // Still the caller's own Anthropic body: a bridge round-trip would
    // have rebuilt `messages` into the canonical chat shape and dropped
    // `max_tokens`' Anthropic spelling along the way.
    expect(forwarded.max_tokens).toBe(64);
    expect(forwarded.messages).toEqual([{ role: "user", content: "hello" }]);
  });

  test("count_tokens rides the same declared entry", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    const before = upstream.seen.length;
    const res = await callGateway(app, "/v1/messages/count_tokens", {
      model: "dual-protocol",
      messages: [{ role: "user", content: "hello" }],
    });
    expect(res.status).toBe(200);
    expect(await res.json()).toMatchObject({ input_tokens: 42 });
    expect(upstream.seen.slice(before).map((r) => r.path)).toEqual([
      "/anthropic/v1/messages/count_tokens",
    ]);
  });

  test("the declared entry does not move chat completions off api_base", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    const before = upstream.seen.length;
    const res = await callGateway(app, "/v1/chat/completions", {
      model: "dual-protocol",
      messages: [{ role: "user", content: "hello" }],
    });
    expect(res.status).toBe(200);
    expect(upstream.seen.slice(before).map((r) => r.path)).toEqual([
      "/v1/chat/completions",
    ]);
  });

  test("the bridge that translates INTO the Anthropic wire uses the declared entry too", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    const before = upstream.seen.length;
    const res = await callGateway(app, "/v1/chat/completions", {
      model: "anthropic-declared",
      messages: [{ role: "user", content: "hello" }],
    });
    expect(res.status).toBe(200);
    // Not `/v1/messages` at api_base: one upstream route must not resolve
    // to two hosts depending on which inbound protocol asked for it.
    expect(upstream.seen.slice(before).map((r) => r.path)).toEqual([
      "/anthropic/v1/messages",
    ]);
  });

  test("a declared `responses` entry forwards verbatim, to its own base", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    const before = upstream.seen.length;
    const res = await callGateway(app, "/v1/responses", {
      model: "responses-declared",
      input: "hello",
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as Record<string, unknown>;
    expect(body.object).toBe("response");
    expect(JSON.stringify(body.output)).toContain("from the responses route");

    const dispatched = upstream.seen.slice(before);
    // The declared base, not `api_base`, and not the chat-completions
    // route a translated request would have taken.
    expect(dispatched.map((r) => r.path)).toEqual(["/responses-host/v1/responses"]);
    // Verbatim: `input` reaches the upstream unconverted.
    expect(JSON.parse(dispatched[0].body)).toMatchObject({ input: "hello" });
  });

  test("usage is attributed to the target model's own vendor, not the wire it took", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    // Both requests run on `deepseek` models. The verbatim Anthropic
    // passthrough and the translated Responses path used to hard-code
    // "anthropic" / "openai" here, which split one model's series in two
    // and misattributed the priced row.
    await callGateway(app, "/v1/messages", {
      model: "dual-protocol",
      max_tokens: 16,
      messages: [{ role: "user", content: "hello" }],
    });
    await callGateway(app, "/v1/responses", {
      model: "declared-chat-only",
      input: "hello",
    });
    // The verbatim Responses path on a non-OpenAI vendor — the case that
    // used to report a hard-coded "openai".
    await callGateway(app, "/v1/responses", {
      model: "responses-declared",
      input: "hello",
    });

    const samples = await scrapeMetrics(app.metricsUrl);
    const deepseekMessages = sumMetric(samples, "aisix_llm_requests_total", {
      provider: "deepseek",
      endpoint: "/v1/messages",
    });
    expect(deepseekMessages).toBeGreaterThan(0);
    // Scoped to the two endpoints this file drives through a verbatim
    // path, so adding an actually-Anthropic model here later does not
    // make the assertion fire for the wrong reason.
    for (const endpoint of ["/v1/messages", "/v1/messages/count_tokens"]) {
      expect(
        sumMetric(samples, "aisix_llm_requests_total", {
          provider: "anthropic",
          endpoint,
        }),
        `${endpoint} must be labelled with the vendor, not the wire`,
      ).toBe(0);
    }
    expect(
      sumMetric(samples, "aisix_llm_requests_total", {
        provider: "deepseek",
        endpoint: "/v1/responses",
      }),
      "the verbatim Responses path must report the vendor too",
    ).toBeGreaterThan(0);
    // The bridged Responses request rides the `openai` catalog vendor, so
    // it legitimately reports openai — the assertion that matters is that
    // it reports the MODEL's vendor rather than a constant.
    expect(
      sumMetric(samples, "aisix_llm_requests_total", {
        provider: "openai",
        endpoint: "/v1/responses",
      }),
    ).toBeGreaterThan(0);
  });
});
