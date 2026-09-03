import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
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
import { pickFreePort } from "../harness/ports.js";

// E2E: the semantic router honors its MEMBERS' own gates (model-kind
// audit — the semantic selection path used to bypass every one of
// them, while routing groups enforce all of them per target):
//
//   1. A route target excluded by its `allowed_cidrs` is a hard miss:
//      the request falls through to `default` and the served-route
//      header is cleared.
//   2. Route target AND default both excluded → the same 403 the entry
//      gate produces, leaking no member names.
//   3. A route target in request-path cooldown (tripped by a 500) is
//      skipped on the NEXT request — health is consumed at selection,
//      not just recorded.
//   4. The embed sub-call inherits the embedding model's own `timeout`
//      when the router sets no `embedding_timeout_ms` — a hung
//      embedding upstream degrades via `on_embedding_failure` instead
//      of stalling the request unbounded.
//   5. The router's own top-level `retries` is the group slot of the
//      member → group → deployment-default retry chain (a semantic
//      parent has no `routing` block to carry it).

const CALLER_PLAINTEXT = "sk-semantic-gates-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

function keywordVector(text: string): number[] {
  const t = text.toLowerCase();
  if (t.includes("code") || t.includes("python")) return [0, 1, 0, 0];
  return [0, 0, 0, 1];
}

interface EmbeddingMock {
  baseUrl: string;
  close(): Promise<void>;
}

/** Deterministic keyword-vector `/v1/embeddings` mock; `delayMs` holds
 * every response open to model a hung embedding upstream. */
async function startEmbeddingMock(opts: { delayMs?: number } = {}): Promise<EmbeddingMock> {
  const server: Server = createServer((req, res) => {
    res.on("error", () => {});
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      if (!req.url?.includes("/embeddings")) {
        res.statusCode = 404;
        res.end("{}");
        return;
      }
      const respond = () => {
        let body: { input?: string | string[] };
        try {
          body = JSON.parse(raw || "{}") as { input?: string | string[] };
        } catch {
          res.statusCode = 400;
          res.end("{}");
          return;
        }
        const inputs = Array.isArray(body.input) ? body.input : [body.input ?? ""];
        res.statusCode = 200;
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
      };
      if (opts.delayMs) setTimeout(respond, opts.delayMs);
      else respond();
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    async close() {
      server.closeAllConnections?.();
      await new Promise<void>((resolve) => {
        server.close(() => resolve());
      });
    },
  };
}

function chatBody(content: string) {
  return {
    id: `cmpl-${content}`,
    object: "chat.completion",
    created: 0,
    model: "gpt-4o-mini",
    choices: [
      { index: 0, message: { role: "assistant", content }, finish_reason: "stop" },
    ],
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
  };
}

describe("semantic router member gates e2e", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];
  const embedMocks: EmbeddingMock[] = [];

  async function chatUpstream(
    content: string,
    extra: Parameters<typeof startOpenAiUpstream>[0] = {},
  ): Promise<OpenAiUpstream> {
    const u = await startOpenAiUpstream({ nonStreamBody: chatBody(content), ...extra });
    upstreams.push(u);
    return u;
  }

  async function directModel(
    displayName: string,
    upstream: OpenAiUpstream,
    extra: Record<string, unknown> = {},
  ): Promise<void> {
    if (!seed) throw new Error("seed not ready");
    const pk = await seed.createProviderKey({
      display_name: `${displayName}-pk`,
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: displayName,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
      ...extra,
    });
  }

  async function embeddingModel(
    displayName: string,
    mock: EmbeddingMock,
    extra: Record<string, unknown> = {},
  ): Promise<void> {
    if (!seed) throw new Error("seed not ready");
    const pk = await seed.createProviderKey({
      display_name: `${displayName}-pk`,
      secret: "sk-mock",
      api_base: `${mock.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: displayName,
      provider: "openai",
      model_name: "embed-mock",
      provider_key_id: pk.id,
      embedding: { dimensions: 4 },
      ...extra,
    });
  }

  async function chat(
    model: string,
    prompt: string,
  ): Promise<{ status: number; content: string; route: string | null; body: string }> {
    if (!app) throw new Error("app not ready");
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({ model, messages: [{ role: "user", content: prompt }] }),
    });
    const body = await res.text();
    let content = "";
    try {
      const parsed = JSON.parse(body) as {
        choices?: Array<{ message?: { content?: string } }>;
      };
      content = parsed.choices?.[0]?.message?.content ?? "";
    } catch {
      // non-JSON body (never expected) — surface via `body` in asserts
    }
    return { status: res.status, content, route: res.headers.get("x-aisix-route"), body };
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const embed = await startEmbeddingMock();
    const slowEmbed = await startEmbeddingMock({ delayMs: 3000 });
    embedMocks.push(embed, slowEmbed);
    await embeddingModel("smg-bge", embed);
    // The slow embedder's own timeout is the knob under test in case 4.
    await embeddingModel("smg-bge-slow", slowEmbed, { timeout: 500 });

    // Members. The loopback caller is NOT in 10.0.0.0/8, so "restricted"
    // members are excluded for every test request.
    await directModel("smg-blocked", await chatUpstream("served-blocked"), {
      allowed_cidrs: ["10.0.0.0/8"],
    });
    await directModel("smg-blocked-2", await chatUpstream("served-blocked-2"), {
      allowed_cidrs: ["10.0.0.0/8"],
    });
    await directModel("smg-open", await chatUpstream("served-open"));
    await directModel("smg-flaky", await chatUpstream("unused", { status: 500 }), {
      cooldown: { enabled: true, default_seconds: 120 },
    });
    await directModel("smg-t4-code", await chatUpstream("served-t4-code"));
    await directModel("smg-t4-default", await chatUpstream("served-t4-default"));
    // 4 scripted failures, then success: recoverable only with the
    // router-level retry budget of 4 (deployment default is 2).
    await directModel(
      "smg-retry-target",
      await chatUpstream("served-after-retries", {
        scriptedResponses: [
          { status: 500, errorBody: { error: { message: "boom", type: "server_error" } } },
          { status: 500, errorBody: { error: { message: "boom", type: "server_error" } } },
          { status: 500, errorBody: { error: { message: "boom", type: "server_error" } } },
          { status: 500, errorBody: { error: { message: "boom", type: "server_error" } } },
        ],
      }),
    );

    // Streaming-loop twin of the retries case: 4 scripted 500s, then the
    // static SSE fixture serves the success — recoverable only with the
    // router-level budget of 4 through the STREAMING dispatch loop.
    await directModel(
      "smg-retry-stream-target",
      await chatUpstream("unused-static", {
        scriptedResponses: [
          { status: 500, errorBody: { error: { message: "boom", type: "server_error" } } },
          { status: 500, errorBody: { error: { message: "boom", type: "server_error" } } },
          { status: 500, errorBody: { error: { message: "boom", type: "server_error" } } },
          { status: 500, errorBody: { error: { message: "boom", type: "server_error" } } },
        ],
        streamEvents: [
          JSON.stringify({
            id: "smg-sse",
            object: "chat.completion.chunk",
            model: "gpt-4o-mini",
            choices: [
              { index: 0, delta: { content: "served-stream" }, finish_reason: "stop" },
            ],
            usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
          }),
          "[DONE]",
        ],
      }),
    );
    // Background-unhealthy displacement: this member's upstream always
    // 500s, request-path cooldown is DISABLED (so only the background
    // prober's Unhealthy verdict can displace it), and the 5s probe
    // interval marks it within the test's 30s poll budget.
    await directModel("smg-unhealthy", await chatUpstream("unused-500", { status: 500 }), {
      cooldown: { enabled: false },
      background_model_check: {
        enabled: true,
        // Schema minimum — the prober's first verdict lands within ~5s.
        interval_seconds: 5,
        stale_after_seconds: 300,
        prompt: "ping",
        max_tokens: 1,
        timeout_seconds: 2,
      },
    });

    const router = (name: string, target: string, def: string, extra: Record<string, unknown> = {}) =>
      seed!.createModel({
        display_name: name,
        semantic: {
          embedding_model: "smg-bge",
          routes: [
            { name: "code", target, examples: ["write python code"], threshold: 0.5 },
          ],
          default: def,
          match: { threshold: 0.5 },
        },
        ...extra,
      });

    await router("smg-router-ip", "smg-blocked", "smg-open");
    await router("smg-router-all-blocked", "smg-blocked", "smg-blocked-2");
    await router("smg-router-cooldown", "smg-flaky", "smg-open");
    await router("smg-router-retries", "smg-retry-target", "smg-open", { retries: 4 });
    await router("smg-router-retries-stream", "smg-retry-stream-target", "smg-open", {
      retries: 4,
    });
    await router("smg-router-unhealthy", "smg-unhealthy", "smg-open");
    await seed.createModel({
      display_name: "smg-router-slow-embed",
      semantic: {
        embedding_model: "smg-bge-slow",
        routes: [
          { name: "code", target: "smg-t4-code", examples: ["write python code"], threshold: 0.5 },
        ],
        default: "smg-t4-default",
        match: { threshold: 0.5 },
      },
    });

    // The caller key is seeded LAST: once it authenticates, revision
    // order implies every resource above is in the snapshot
    // (tests/e2e/AGENTS.md). The gate exercises none of the member-gate
    // behavior under test, so a defect there fails its own case by name
    // instead of surfacing as a propagation timeout here.
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      await res.arrayBuffer(); // release the socket between polls
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
    await Promise.all(embedMocks.map((m) => m.close()));
  });

  test("route target excluded by allowed_cidrs falls through to default and clears the route header", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const r = await chat("smg-router-ip", "write python code please");
    expect(r.status).toBe(200);
    expect(r.content).toBe("served-open");
    // The winning route did not serve; the header must not claim it did.
    expect(r.route).toBeNull();
  });

  test("route target and default both excluded yields the entry-gate 403 without leaking members", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const r = await chat("smg-router-all-blocked", "write python code please");
    expect(r.status).toBe(403);
    expect(r.body).not.toContain("smg-blocked");
  });

  test("a cooled-down route target is skipped at selection on the next request", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    // First request dispatches to the 500-ing target and trips its
    // cooldown (selection had no health signal yet).
    const first = await chat("smg-router-cooldown", "write python code please");
    expect(first.status).toBeGreaterThanOrEqual(500);
    // Second request: the winner is in cooldown → selection prefers the
    // default instead of grinding the broken target again.
    const second = await chat("smg-router-cooldown", "write python code please");
    expect(second.status).toBe(200);
    expect(second.content).toBe("served-open");
    expect(second.route).toBeNull();
  });

  test("the embed sub-call inherits the embedding model's own timeout", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const started = Date.now();
    const r = await chat("smg-router-slow-embed", "write python code please");
    const elapsed = Date.now() - started;
    expect(r.status).toBe(200);
    // Pre-fix the 3s hang completed and routed by keyword to the code
    // target; with the member timeout honored the embed call dies at
    // ~500ms and on_embedding_failure serves the default.
    expect(r.content).toBe("served-t4-default");
    expect(elapsed).toBeLessThan(2500);
  });

  test("the streaming dispatch loop honors the router-level retry budget too", async (ctx) => {
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
        model: "smg-router-retries-stream",
        messages: [{ role: "user", content: "write python code please" }],
        stream: true,
      }),
    });
    const body = await res.text();
    // 4 scripted 500s exhaust only with the router's budget of 4; the
    // 5th attempt opens the real SSE stream.
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("text/event-stream");
    expect(body).toContain("served-stream");
  });

  test("a background-unhealthy route target is displaced by the healthy default", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    // Cooldown is disabled on the member, so only the background
    // prober's Unhealthy verdict can displace it. Poll until the 5s
    // probe interval has marked it and selection serves the default.
    const deadline = Date.now() + 30_000;
    let displaced: { status: number; content: string; route: string | null } | undefined;
    while (Date.now() < deadline) {
      const r = await chat("smg-router-unhealthy", "write python code please");
      if (r.status === 200 && r.content === "served-open") {
        displaced = r;
        break;
      }
      // Until the mark lands, the winner dispatches and its upstream
      // 500s — cooldown being disabled keeps this the only mechanism.
      expect(r.status).toBeGreaterThanOrEqual(500);
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
    expect(displaced).toBeDefined();
    expect(displaced?.route).toBeNull();
  });

  test("the router's top-level retries is the group slot of the retry chain", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    // 4 scripted 500s then success: only a budget of 4 retries (router
    // level) survives; the deployment default of 2 fails pre-fix.
    const r = await chat("smg-router-retries", "write python code please");
    expect(r.status).toBe(200);
    expect(r.content).toBe("served-after-retries");
    expect(r.route).toBe("code");
  });
});
