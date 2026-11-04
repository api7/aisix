import { createHash, randomUUID } from "node:crypto";
import { createServer, type Server as HttpServer } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { WebSocket as WsClient } from "ws";
import {
  awaitWindowHeadroom,
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// AISIX-Cloud#1848: `error_source` on the request counters tells the gateway's
// own refusals from an upstream's failures, which a status code cannot: an
// AISIX rate limit and an upstream's answer are both a 429.

const CALLER = "sk-error-source-caller";
const NARROW = "sk-error-source-narrow";
const BROKE = "sk-error-source-broke";
const ROUTE_LIMITED = "sk-error-source-route-limited";
const USER = "error-source-user";
const BROKE_KEY_ID = randomUUID();
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

type Labels = Record<string, string>;

const COUNTERS = [
  "aisix_llm_requests_total",
  "aisix_proxy_requests_total",
] as const;

/** Every sample of `family` carrying all of `labels`, with its value. */
function samples(text: string, family: string, labels: Labels): { line: string; value: number }[] {
  return text
    .split("\n")
    .filter((l) => l.startsWith(`${family}{`))
    .filter((l) => Object.entries(labels).every(([k, v]) => l.includes(`${k}="${v}"`)))
    .map((line) => ({ line, value: Number(line.slice(line.lastIndexOf(" ") + 1)) }));
}

const chat = (model: string, extra: Record<string, unknown> = {}) => ({
  model,
  messages: [{ role: "user", content: "hi" }],
  ...extra,
});

describe("error_source on the request counters", () => {
  let app: SpawnedApp | undefined;
  const upstreams: OpenAiUpstream[] = [];
  const httpServers: HttpServer[] = [];
  let etcdReachable = false;
  let seed: SeedClient;

  async function fixedUpstream(status: number): Promise<string> {
    const server = createServer((req, res) => {
      req.resume();
      req.on("end", () => {
        res.writeHead(status, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: { message: "sentinel-upstream-failure" } }));
      });
    });
    httpServers.push(server);
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    return `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  }

  async function pair(
    name: string,
    upstream: Parameters<typeof startOpenAiUpstream>[0],
    model: Record<string, unknown> = {},
  ): Promise<void> {
    const mock = await startOpenAiUpstream(upstream);
    upstreams.push(mock);
    const pk = await seed.createProviderKey({
      display_name: `${name}-pk`,
      secret: "sk-mock",
      api_base: `${mock.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: name,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
      ...model,
    });
  }

  const failing = (status: number, message = "sentinel-upstream-failure") => ({
    status,
    errorBody: { error: { message } },
  });
  const limited = { rate_limit: { rpm: 1 } };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    await pair("es-ok", {});
    await pair("es-limited", {}, limited);
    await pair("es-upstream-429", failing(429));

    // Each group has its own members: a failed dispatch feeds its target's
    // cooldown, which would take a shared one out of rotation.
    await pair("es-fb-limited", {}, limited);
    await pair("es-fb-ok", {});
    await pair("es-a-upstream", failing(429));
    await pair("es-a-limited", {}, limited);
    await pair("es-u-limited", {}, limited);
    await pair("es-u-upstream", failing(429));
    const group = (name: string, a: string, b: string) =>
      seed.createModel({
        display_name: name,
        routing: {
          strategy: "failover",
          targets: [{ model: a }, { model: b }],
          retry_on_429: true,
        },
      });
    await group("es-group-recovers", "es-fb-limited", "es-fb-ok");
    await group("es-group-ends-local", "es-a-upstream", "es-a-limited");
    await group("es-group-ends-upstream", "es-u-limited", "es-u-upstream");

    await pair("es-stream-in-band", {
      streamEvents: [
        JSON.stringify({ error: { message: "sentinel-in-band", type: "server_error" } }),
      ],
    });
    await pair("es-stream-slow", {
      streamEvents: Array.from({ length: 6 }, () =>
        JSON.stringify({
          id: "chatcmpl-es",
          object: "chat.completion.chunk",
          created: 1,
          model: "gpt-4o-mini",
          choices: [{ index: 0, delta: { content: "x" }, finish_reason: null }],
        }),
      ),
      eventDelayMs: 500,
    });

    await pair("es-embed-500", failing(500));
    await pair("es-embed-limited", {}, limited);
    await pair("es-messages-500", failing(500));
    await pair("es-messages-limited", {}, limited);

    const unmappedPk = await seed.createProviderKey({
      display_name: "es-video-unmapped-pk",
      secret: "sk-mock",
      provider: "anthropic",
      api_base: "http://127.0.0.1:9",
    });
    await seed.createModel({
      display_name: "es-video-unmapped",
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: unmappedPk.id,
    });

    const guard = await seed.createGuardrail(
      {
        name: "es-output-block",
        enabled: true,
        hook_point: "output",
        kind: "keyword",
        patterns: [{ kind: "literal", value: "esforbiddenword" }],
      },
      { attach: false },
    );
    const leakyUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "chatcmpl-leaky",
        object: "chat.completion",
        created: 1,
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "esforbiddenword" },
            finish_reason: "stop",
          },
        ],
      },
    });
    upstreams.push(leakyUpstream);
    const leaky = await seed.createProviderKey({
      display_name: "es-leaky-pk",
      secret: "sk-mock",
      api_base: `${leakyUpstream.baseUrl}/v1`,
    });
    const leakyModel = await seed.createModel({
      display_name: "es-leaky-guarded",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: leaky.id,
    });
    await seed.attachGuardrailToModel(guard.id, leakyModel.id);

    const routeUpstream = await startOpenAiUpstream(failing(500));
    upstreams.push(routeUpstream);
    const routePk = await seed.createProviderKey({
      display_name: "es-route-pk",
      secret: "sk-mock",
      api_base: routeUpstream.baseUrl,
    });
    await seed.createPassthroughRoute({
      name: "es-route-500",
      path_prefix: "/es-route-500",
      target_url: routeUpstream.baseUrl,
      provider_key_id: routePk.id,
    });

    await seed.update("a2a_agents", randomUUID(), {
      name: "es-agent",
      url: await fixedUpstream(500),
      protocol_version: "1.0",
      auth_type: "none",
      enabled: true,
    });
    const refusing = await fixedUpstream(400);
    for (const name of ["es-jobs-400", "es-realtime-400"]) {
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        secret: "sk-mock",
        api_base: `${refusing}/v1`,
      });
      await seed.createModel({
        display_name: name,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
    }

    await seed.update("budgets", randomUUID(), {
      scope: "api_key",
      scope_ref: BROKE_KEY_ID,
      period: "month",
      exhausted: {
        reason: {
          type: "billing_error",
          code: "budget_exceeded",
          message: "api key budget exceeded",
          scope: "api_key",
          scope_id: BROKE_KEY_ID,
        },
      },
    });
    await seed.update("api_keys", BROKE_KEY_ID, {
      key_hash: hash(BROKE),
      allowed_models: ["es-ok"],
    });
    await seed.createApiKey({ key_hash: hash(NARROW), allowed_models: ["es-ok"] });
    await seed.createApiKey({
      key_hash: hash(ROUTE_LIMITED),
      allowed_models: [],
      allowed_routes: ["es-route-500"],
      rate_limit: { rpm: 1 },
    });
    // Written last: its authenticating implies every row above has landed.
    await seed.createApiKey({
      key_hash: hash(CALLER),
      allowed_models: ["*"],
      allowed_routes: ["es-route-500"],
      allowed_agents: ["*"],
      user_id: USER,
    });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER}` },
      });
      await res.text();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
    await Promise.all(httpServers.map((s) => new Promise<void>((r) => s.close(() => r()))));
  });

  async function post(path: string, body: unknown, key = CALLER): Promise<Response> {
    return fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: { authorization: `Bearer ${key}`, "content-type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  async function send(path: string, body: unknown, key = CALLER): Promise<number> {
    const res = await post(path, body, key);
    await res.text();
    return res.status;
  }

  async function scrape(): Promise<string> {
    const res = await fetch(`${app!.metricsUrl}/metrics`);
    expect(res.status).toBe(200);
    return res.text();
  }

  /**
   * The request ended on exactly this `error_source` on every counter that
   * carries it. Polled, because a stream's samples are written when it ends.
   */
  async function expectSource(labels: Labels, source: string, families: readonly string[] = COUNTERS) {
    await expect
      .poll(
        async () => {
          const text = await scrape();
          return families.map((family) => samples(text, family, labels).map((s) => s.line));
        },
        { timeout: 5_000 },
      )
      .toSatisfy((rows: string[][]) => rows.every((r) => r.length > 0));
    const text = await scrape();
    for (const family of families) {
      for (const { line } of samples(text, family, labels)) {
        expect(line, family).toContain(`error_source="${source}"`);
      }
    }
  }

  const ready = () => etcdReachable && !!app;

  test("the gateway's 429 and the upstream's 429 differ, and the first can be excluded", async (ctx) => {
    if (!ready()) return ctx.skip();
    await awaitWindowHeadroom();
    expect(await send("/v1/chat/completions", chat("es-ok"))).toBe(200);
    expect(await send("/v1/chat/completions", chat("es-limited"))).toBe(200);
    expect(await send("/v1/chat/completions", chat("es-limited"))).toBe(429);
    expect(await send("/v1/chat/completions", chat("es-upstream-429"))).toBe(429);

    const text = await scrape();
    const allFamilies = [...COUNTERS, "aisix_proxy_failed_requests_total"];
    for (const family of allFamilies) {
      expect(
        samples(text, family, { model: "es-limited", status: "429", error_source: "aisix" }),
        family,
      ).toHaveLength(1);
      expect(
        samples(text, family, { model: "es-upstream-429", status: "429", error_source: "upstream" }),
        family,
      ).toHaveLength(1);
    }
    // The customer's query: what this user called, without what AISIX refused.
    const called = samples(text, "aisix_llm_requests_total", { user_id: USER })
      .filter(({ line }) => !line.includes('error_source="aisix"'))
      .filter(({ line }) => /model="es-(ok|limited|upstream-429)"/.test(line))
      .reduce((sum, { value }) => sum + value, 0);
    const refused = samples(text, "aisix_llm_requests_total", {
      user_id: USER,
      model: "es-limited",
      error_source: "aisix",
    }).reduce((sum, { value }) => sum + value, 0);
    expect(called).toBe(3);
    expect(refused).toBe(1);
    // A label of the counters only.
    expect(text).not.toMatch(/^aisix_(llm|proxy)_request_duration_seconds_[a-z]+\{[^}]*error_source=/m);
    expect(text).not.toMatch(/^aisix_requests_total\{[^}]*error_source=/m);
  });

  test("a success has an empty source", async (ctx) => {
    if (!ready()) return ctx.skip();
    expect(await send("/v1/chat/completions", chat("es-ok"))).toBe(200);
    await expectSource({ model: "es-ok", status: "200", user_id: USER }, "");
  });

  test("a budget refusal is the gateway's", async (ctx) => {
    if (!ready()) return ctx.skip();
    expect(await send("/v1/chat/completions", chat("es-ok"), BROKE)).toBe(429);
    await expectSource({ endpoint: "/v1/chat/completions", api_key_id: BROKE_KEY_ID, status: "429" }, "aisix");
  });

  test("a model the key may not use and an invalid body are the gateway's", async (ctx) => {
    if (!ready()) return ctx.skip();
    expect(await send("/v1/chat/completions", chat("es-limited"), NARROW)).toBe(403);
    await expectSource({ endpoint: "/v1/chat/completions", status: "403" }, "aisix");
    expect(await send("/v1/chat/completions", { model: "es-ok" })).toBe(400);
    await expectSource({ endpoint: "/v1/chat/completions", status: "400" }, "aisix");
    expect(await send("/v1/chat/completions", chat("es-no-such-model"))).toBe(404);
    await expectSource({ endpoint: "/v1/chat/completions", status: "404" }, "aisix");
  });

  test("the request's last failure decides a routing group", async (ctx) => {
    if (!ready()) return ctx.skip();
    await awaitWindowHeadroom();
    // Burn each group member's own one-per-minute window.
    for (const member of ["es-fb-limited", "es-a-limited", "es-u-limited"]) {
      expect(await send("/v1/chat/completions", chat(member)), member).toBe(200);
    }

    expect(await send("/v1/chat/completions", chat("es-group-recovers"))).toBe(200);
    await expectSource({ model: "es-group-recovers", status: "200", is_fallback: "true" }, "");

    expect(await send("/v1/chat/completions", chat("es-group-ends-local"))).toBe(429);
    await expectSource({ model: "es-group-ends-local", status: "429" }, "aisix");

    expect(await send("/v1/chat/completions", chat("es-group-ends-upstream"))).toBe(429);
    await expectSource({ model: "es-group-ends-upstream", status: "429" }, "upstream");
  });

  test("an upstream error inside a stream is the upstream's", async (ctx) => {
    if (!ready()) return ctx.skip();
    const res = await post("/v1/chat/completions", chat("es-stream-in-band", { stream: true }));
    await res.text();
    await expectSource({ model: "es-stream-in-band", stream: "true", user_id: USER }, "upstream");
    const text = await scrape();
    const [sample] = samples(text, "aisix_proxy_requests_total", { model: "es-stream-in-band" });
    expect(sample.line).not.toContain('status="200"');
  });

  test("a caller that leaves mid-stream has an empty source", async (ctx) => {
    if (!ready()) return ctx.skip();
    const abort = new AbortController();
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify(chat("es-stream-slow", { stream: true })),
      signal: abort.signal,
    });
    expect(res.status).toBe(200);
    const reader = res.body!.getReader();
    await reader.read();
    abort.abort();
    await reader.cancel().catch(() => undefined);
    await expectSource({ model: "es-stream-slow", status: "499" }, "");
  });

  test("a guardrail refusing the upstream's answer is the gateway's", async (ctx) => {
    if (!ready()) return ctx.skip();
    expect(await send("/v1/chat/completions", chat("es-leaky-guarded"))).toBe(422);
    await expectSource({ model: "es-leaky-guarded", status: "422" }, "aisix");
  });

  test("passthrough: the upstream's 500 and the route's rate limit", async (ctx) => {
    if (!ready()) return ctx.skip();
    await awaitWindowHeadroom();
    expect(await send("/es-route-500/anything", { x: 1 })).toBe(500);
    await expectSource(
      { endpoint: "/passthrough_route", status: "500" },
      "upstream",
      ["aisix_proxy_requests_total", "aisix_proxy_failed_requests_total"],
    );
    await send("/es-route-500/anything", { x: 1 }, ROUTE_LIMITED);
    expect(await send("/es-route-500/anything", { x: 1 }, ROUTE_LIMITED)).toBe(429);
    await expectSource(
      { endpoint: "/passthrough_route", status: "429" },
      "aisix",
      ["aisix_proxy_requests_total", "aisix_proxy_failed_requests_total"],
    );
  });

  test("embeddings and messages tell the two apart as chat does", async (ctx) => {
    if (!ready()) return ctx.skip();
    await awaitWindowHeadroom();
    expect(await send("/v1/embeddings", { model: "es-embed-500", input: "hi" })).toBeGreaterThanOrEqual(500);
    await expectSource({ endpoint: "/v1/embeddings", model: "es-embed-500" }, "upstream");
    await send("/v1/embeddings", { model: "es-embed-limited", input: "hi" });
    expect(await send("/v1/embeddings", { model: "es-embed-limited", input: "hi" })).toBe(429);
    await expectSource({ endpoint: "/v1/embeddings", model: "es-embed-limited", status: "429" }, "aisix");

    const messages = (model: string) => ({
      model,
      max_tokens: 8,
      messages: [{ role: "user", content: "hi" }],
    });
    expect(await send("/v1/messages", messages("es-messages-500"))).toBeGreaterThanOrEqual(500);
    await expectSource({ endpoint: "/v1/messages", model: "es-messages-500" }, "upstream");
    await send("/v1/messages", messages("es-messages-limited"));
    expect(await send("/v1/messages", messages("es-messages-limited"))).toBe(429);
    await expectSource({ endpoint: "/v1/messages", model: "es-messages-limited", status: "429" }, "aisix");
  });

  test("a provider /v1/videos does not map is the gateway's 501", async (ctx) => {
    if (!ready()) return ctx.skip();
    expect(await send("/v1/videos", { model: "es-video-unmapped", prompt: "a cat" })).toBe(501);
    await expectSource({ model: "es-video-unmapped", status: "501" }, "aisix", [
      "aisix_proxy_requests_total",
    ]);
  });

  test("routes that hold no typed error still name the source", async (ctx) => {
    if (!ready()) return ctx.skip();
    const proxyOnly = ["aisix_proxy_requests_total"] as const;
    const call = {
      jsonrpc: "2.0",
      id: "1",
      method: "message/send",
      params: { message: { role: "user", parts: [{ kind: "text", text: "hi" }], messageId: "m-1" } },
    };
    expect(await send("/a2a/es-agent", call)).toBe(502);
    await expectSource({ endpoint: "/a2a", status: "502" }, "upstream", proxyOnly);
    expect(await send("/a2a/es-no-such-agent", call)).toBe(404);
    await expectSource({ endpoint: "/a2a", status: "404" }, "aisix", proxyOnly);

    expect(await send("/v1/batches?model=es-jobs-400", { input_file_id: "f" })).toBe(400);
    await expectSource({ endpoint: "/v1/batches", status: "400" }, "upstream", proxyOnly);

    await new Promise<void>((resolve, reject) => {
      const ws = new WsClient(
        `${app!.proxyUrl.replace("http://", "ws://")}/v1/realtime?model=es-realtime-400`,
        { headers: { authorization: `Bearer ${CALLER}` } },
      );
      ws.on("close", () => resolve());
      ws.on("error", reject);
    });
    await expectSource({ endpoint: "/v1/realtime", status: "502" }, "upstream");

    const mcp = await fetch(`${app!.proxyUrl}/mcp`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: "not json",
    });
    await mcp.text();
    expect(mcp.status).toBeGreaterThanOrEqual(400);
    await expectSource({ endpoint: "/mcp", status: String(mcp.status) }, "aisix", proxyOnly);
  });
});
