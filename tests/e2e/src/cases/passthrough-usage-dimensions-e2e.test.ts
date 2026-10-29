import { createHash } from "node:crypto";
import { createServer, type IncomingMessage, type Server } from "node:http";
import { gunzipSync } from "node:zlib";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  pickFreePort,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForLogLine,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";
import {
  metricDelta,
  scrapeMetrics,
  type MetricSample,
} from "../harness/metrics.js";

// E2E: a passthrough route reports EVERY token dimension the exchange
// carried, not just the canonical prompt/completion pair.
//
// A route that auto-detects a first-class envelope must meter what the
// typed endpoint serving that envelope would: `UsageEvent` has seven token
// fields and cp-api prices the cache ones at their own rates (cache-write
// ~1.25x prompt, cache-read ~0.10x), so dropping them makes a cached
// workload — the dominant shape for IDE agent traffic, where the cache read
// can be 98% of the prompt — unpriceable and invisible.
//
// Three journeys, one per usage shape a route actually meets:
//
//   1. OPAQUE stream, server-labelled usage frame. An IDE agent backend
//      reached through a forward-proxy route has no recognisable envelope
//      and reports its counts as a FLAT token object on its own
//      `event: token_usage` frame — no `usage` wrapper. Read there, and
//      only there.
//   2. OPAQUE stream, unlabelled frame. The same flat shape on a frame the
//      server did not name a usage report mints nothing: an opaque stream
//      offers no envelope to authenticate token-shaped fields against.
//   3. Detected chat envelope, buffered. The nested OpenAI details
//      (`prompt_tokens_details.cached_tokens`,
//      `completion_tokens_details.reasoning_tokens`) reach the event, and
//      the caller's `model` alias attributes the row.
//
// A second spec drives every LLM protocol a route identifies — Anthropic
// Messages (buffered and streamed), OpenAI Responses, a streamed OpenAI
// Chat, embeddings, image generation — and requires each to be metered the
// way the typed endpoint for that protocol meters it: the cache counters in
// that protocol's own accounting (Anthropic's BESIDE the prompt, OpenAI's
// INSIDE it), a cache-inclusive access-log total, and one request in the
// LLM request tier beside its `aisix_llm_*` token families. The opaque agent
// stream stays out of the LLM tier.
//
// Asserted through a real `datadog` exporter, because that sink serialises
// the WHOLE `UsageEvent` — the OTLP span builder is an explicit allowlist
// that carries only the input/output pair, so it cannot see the dimensions
// under test.

const CALLER_PLAINTEXT = "sk-ptr-usage-dims-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const CREDENTIAL_REF = "ptrdims";
const DD_API_KEY = "dd-ptr-usage-dims-key";
const INTAKE_PATH = "/api/v2/logs";

interface MockDatadog {
  site: string;
  logs: Record<string, unknown>[];
  close(): Promise<void>;
}

/** Mock Datadog Logs intake: collects every delivered log object. */
async function startMockDatadog(): Promise<MockDatadog> {
  const logs: Record<string, unknown>[] = [];
  const server: Server = createServer(
    (req: IncomingMessage, res: import("node:http").ServerResponse) => {
      const chunks: Buffer[] = [];
      req.on("data", (c: Buffer) => chunks.push(c));
      req.on("end", () => {
        const path = (req.url ?? "").split("?")[0];
        if (req.method === "POST" && path === INTAKE_PATH) {
          try {
            const parsed: unknown = JSON.parse(
              gunzipSync(Buffer.concat(chunks)).toString("utf8"),
            );
            if (Array.isArray(parsed)) {
              for (const entry of parsed) {
                if (entry && typeof entry === "object") {
                  logs.push(entry as Record<string, unknown>);
                }
              }
            }
          } catch {
            // A body we cannot decode leaves `logs` short, so the poll
            // below times out loudly instead of passing on nothing.
          }
        }
        res.statusCode = 202;
        res.end();
      });
    },
  );
  const port = await pickFreePort();
  await new Promise<void>((resolve) =>
    server.listen(port, "127.0.0.1", resolve),
  );
  return {
    site: `127.0.0.1:${port}`,
    logs,
    close: () =>
      new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      }),
  };
}

describe("passthrough-route usage event: token dimensions and the caller's wait", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let dd: MockDatadog | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];

  // The agent-backend report captured from a real forward-proxied IDE: a
  // flat token object on the server's own `token_usage` event. Its total is
  // prompt + completion, so the prompt is the WHOLE input and the cache read
  // (96% of it) is a part of it, not an addition to it.
  // The upstream holds the first frame this long, then spaces the
  // remaining four out — so the whole stream runs far longer than the wait
  // the caller actually experienced.
  const FIRST_FRAME_MS = 120;
  const TAIL_STEP_MS = 400;

  const AGENT_USAGE = {
    prompt_tokens: 14603,
    completion_tokens: 8,
    total_tokens: 14611,
    cache_creation_input_tokens: 0,
    cache_read_input_tokens: 14272,
    reasoning_tokens: 8,
  };

  // Anthropic Messages usage: the cache counters sit BESIDE `input_tokens`.
  const ANTHROPIC_USAGE = {
    input_tokens: 12,
    output_tokens: 7,
    cache_creation_input_tokens: 300,
    cache_read_input_tokens: 1200,
    cache_creation: { ephemeral_5m_input_tokens: 100, ephemeral_1h_input_tokens: 200 },
  };
  const ANTHROPIC_TOTAL_INPUT =
    ANTHROPIC_USAGE.input_tokens +
    ANTHROPIC_USAGE.cache_creation_input_tokens +
    ANTHROPIC_USAGE.cache_read_input_tokens;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    dd = await startMockDatadog();
    app = await spawnApp({
      // The access-log line is written at `info`.
      logLevel: "info",
      extraEnv: {
        [`DD_CRED_${CREDENTIAL_REF.toUpperCase()}_API_KEY`]: DD_API_KEY,
      },
    });
    seed = new SeedClient(etcd, app.etcdPrefix);

    await seed.createObservabilityExporter({
      name: "ptr-dims-datadog",
      enabled: true,
      kind: "datadog",
      site: dd.site,
      credential_ref: CREDENTIAL_REF,
      service: "aisix-ptr-dims",
      content_mode: "metadata_only",
    });

    // 1. Opaque stream that labels its usage frame.
    const labelled = await startOpenAiUpstream({
      rawStreamFrames: [
        `event:thought\ndata:${JSON.stringify({ thought: "hm" })}\n\n`,
        `event:output\ndata:${JSON.stringify({ text: "hi" })}\n\n`,
        `event:token_usage\ndata:${JSON.stringify({ name: "", ...AGENT_USAGE })}\n\n`,
      ],
    });
    // 2. The same flat shape on a frame the server never named usage.
    const unlabelled = await startOpenAiUpstream({
      rawStreamFrames: [
        `event:history\ndata:${JSON.stringify({ ...AGENT_USAGE })}\n\n`,
        `event:done\ndata:${JSON.stringify({ ok: true })}\n\n`,
      ],
    });
    // A chat-shaped exchange whose server reports usage only on its own
    // labelled frame: the typed parser reads nothing, so the label decides.
    const detectedLabelled = await startOpenAiUpstream({
      rawStreamFrames: [
        `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: "hi" } }] })}\n\n`,
        `event:token_usage\ndata:${JSON.stringify({ name: "", ...AGENT_USAGE })}\n\n`,
        "data: [DONE]\n\n",
      ],
    });
    // The same, plus a typed usage chunk: the typed report wins alone.
    const detectedBoth = await startOpenAiUpstream({
      rawStreamFrames: [
        `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: "hi" } }] })}\n\n`,
        `data: ${JSON.stringify({ choices: [], usage: { prompt_tokens: 3, completion_tokens: 4, total_tokens: 7 } })}\n\n`,
        `event:token_usage\ndata:${JSON.stringify({ name: "", ...AGENT_USAGE })}\n\n`,
        "data: [DONE]\n\n",
      ],
    });
    // 3. Detected chat envelope, buffered, nested OpenAI usage details.
    const chat = await startOpenAiUpstream({
      nonStreamBody: {
        id: "chatcmpl-dims",
        object: "chat.completion",
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "ok" },
            finish_reason: "stop",
          },
        ],
        usage: {
          prompt_tokens: 900,
          completion_tokens: 25,
          total_tokens: 925,
          prompt_tokens_details: { cached_tokens: 768, cache_write_tokens: 113 },
          completion_tokens_details: { reasoning_tokens: 17 },
        },
      },
    });
    // 4. A stream whose frames are far apart: the caller's wait ends at the
    //    FIRST frame, but the exchange runs for the whole stream.
    const slow = await startOpenAiUpstream({
      streamEvents: [
        JSON.stringify({ choices: [{ delta: { content: "a" } }] }),
        JSON.stringify({ choices: [{ delta: { content: "b" } }] }),
        JSON.stringify({ choices: [{ delta: { content: "c" } }] }),
        JSON.stringify({ choices: [{ delta: { content: "d" } }] }),
        JSON.stringify({
          choices: [],
          usage: { prompt_tokens: 3, completion_tokens: 4 },
        }),
      ],
      // A real (measurable) wait before the first frame, then a long tail.
      firstEventDelayMs: FIRST_FRAME_MS,
      eventDelayMs: TAIL_STEP_MS,
    });
    // 5. Anthropic Messages, buffered.
    const anthropic = await startOpenAiUpstream({
      nonStreamBody: {
        id: "msg_dims",
        type: "message",
        role: "assistant",
        model: "claude-sonnet-4-5",
        content: [{ type: "text", text: "ok" }],
        stop_reason: "end_turn",
        usage: ANTHROPIC_USAGE,
      },
    });
    // 6. Anthropic Messages, streamed: the input and cache counters ride
    //    `message_start`, the output count the terminal `message_delta`.
    const anthropicStream = await startOpenAiUpstream({
      rawStreamFrames: [
        `event: message_start\ndata: ${JSON.stringify({
          type: "message_start",
          message: {
            id: "msg_dims_stream",
            type: "message",
            role: "assistant",
            model: "claude-sonnet-4-5",
            content: [],
            usage: { ...ANTHROPIC_USAGE, output_tokens: 1 },
          },
        })}\n\n`,
        `event: content_block_delta\ndata: ${JSON.stringify({
          type: "content_block_delta",
          index: 0,
          delta: { type: "text_delta", text: "ok" },
        })}\n\n`,
        `event: message_delta\ndata: ${JSON.stringify({
          type: "message_delta",
          delta: { stop_reason: "end_turn" },
          usage: { output_tokens: ANTHROPIC_USAGE.output_tokens },
        })}\n\n`,
        `event: message_stop\ndata: ${JSON.stringify({ type: "message_stop" })}\n\n`,
      ],
    });
    // 7. OpenAI Responses, buffered.
    const responses = await startOpenAiUpstream({
      nonStreamBody: {
        id: "resp_dims",
        object: "response",
        status: "completed",
        output: [
          {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "ok" }],
          },
        ],
        usage: {
          input_tokens: 30,
          output_tokens: 9,
          total_tokens: 39,
          input_tokens_details: { cached_tokens: 25 },
          output_tokens_details: { reasoning_tokens: 4 },
        },
      },
    });
    // 8. OpenAI Chat, streamed, cache hit inside the prompt.
    const chatStream = await startOpenAiUpstream({
      streamEvents: [
        JSON.stringify({ choices: [{ index: 0, delta: { content: "ok" } }] }),
        JSON.stringify({
          choices: [],
          usage: {
            prompt_tokens: 900,
            completion_tokens: 25,
            total_tokens: 925,
            prompt_tokens_details: { cached_tokens: 768 },
          },
        }),
        "[DONE]",
      ],
    });
    // 9. OpenAI embeddings: the body carries the Responses `input` key.
    const embeddings = await startOpenAiUpstream({
      nonStreamBody: {
        object: "list",
        model: "text-embedding-3-small",
        data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2] }],
        usage: { prompt_tokens: 8, total_tokens: 8 },
      },
    });
    // 10. OpenAI image generation: the body carries the Completions
    //     `prompt` key, the usage the Responses-like input/output spelling.
    const images = await startOpenAiUpstream({
      nonStreamBody: {
        created: 1,
        data: [{ b64_json: "aGk=" }],
        usage: { input_tokens: 50, output_tokens: 4160, total_tokens: 4210 },
      },
    });
    upstreams.push(
      detectedLabelled,
      detectedBoth,
      embeddings,
      images,
      labelled,
      unlabelled,
      chat,
      slow,
      anthropic,
      anthropicStream,
      responses,
      chatStream,
    );

    const pk = await seed.createProviderKey({
      display_name: "ptr-dims-pk",
      secret: "sk-mock",
      api_base: "http://unused-on-routes",
    });
    for (const [name, prefix, upstream] of [
      ["ptr-dims-labelled", "/dims-labelled", labelled],
      ["ptr-dims-unlabelled", "/dims-unlabelled", unlabelled],
      ["ptr-dims-detected-labelled", "/dims-detected-labelled", detectedLabelled],
      ["ptr-dims-detected-both", "/dims-detected-both", detectedBoth],
      ["ptr-dims-chat", "/dims-chat", chat],
      ["ptr-dims-slow", "/dims-slow", slow],
      ["ptr-dims-anthropic", "/dims-anthropic", anthropic],
      ["ptr-dims-anthropic-stream", "/dims-anthropic-stream", anthropicStream],
      ["ptr-dims-responses", "/dims-responses", responses],
      ["ptr-dims-chat-stream", "/dims-chat-stream", chatStream],
      ["ptr-dims-embeddings", "/dims-embeddings", embeddings],
      ["ptr-dims-images", "/dims-images", images],
    ] as const) {
      await seed.createPassthroughRoute({
        name,
        path_prefix: prefix,
        target_url: upstream.baseUrl,
        provider_key_id: pk.id,
      });
    }

    // The caller key is seeded LAST, so gating on it authenticating implies
    // every resource above is already in the snapshot.
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
      allowed_routes: ["*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
    await dd?.close();
  });

  const post = (
    path: string,
    body: unknown,
    headers: Record<string, string> = {},
  ) =>
    fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
        ...headers,
      },
      body: JSON.stringify(body),
    });

  /** Poll the intake for the usage log of one route. */
  async function logFor(
    route: string,
    timeoutMs = 15_000,
  ): Promise<Record<string, unknown>> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      const hit = dd!.logs.find(
        (l) => l["aisix.passthrough_route_name"] === route,
      );
      if (hit) return hit;
      await new Promise((r) => setTimeout(r, 50));
    }
    throw new Error(`no usage log delivered for route ${route}`);
  }

  test("every token dimension a passthrough exchange reports reaches the usage event", async (ctx) => {
    if (!etcdReachable || !app || !seed || !dd) {
      ctx.skip();
      return;
    }

    await waitConfigPropagation(async () => {
      const r = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      await r.text();
      return r.status === 200;
    });

    // 1. Opaque agent stream: private request schema (no `messages` /
    //    `input` / `prompt`), usage on the server's own labelled frame.
    const labelledRes = await post(
      "/dims-labelled/api/agent/v3/create_agent_task",
      {
        config_name: "agent",
        payload: "opaque",
      },
    );
    expect(labelledRes.status).toBe(200);
    await labelledRes.text();

    const labelled = await logFor("ptr-dims-labelled");
    // The agent's `prompt_tokens` is the whole input, so it is the input
    // count as reported — the cache read is a part of it, not added to it.
    expect(labelled["gen_ai.usage.input_tokens"]).toBe(
      AGENT_USAGE.prompt_tokens,
    );
    expect(labelled["gen_ai.usage.output_tokens"]).toBe(
      AGENT_USAGE.completion_tokens,
    );
    expect(labelled["aisix.cached_prompt_tokens"]).toBe(
      AGENT_USAGE.cache_read_input_tokens,
    );
    expect(labelled["aisix.cache_read_tokens"] ?? 0).toBe(0);
    expect(labelled["aisix.cache_creation_tokens"] ?? 0).toBe(0);
    expect(labelled["aisix.reasoning_tokens"]).toBe(
      AGENT_USAGE.reasoning_tokens,
    );
    expect(labelled["aisix.total_tokens"]).toBe(AGENT_USAGE.total_tokens);
    // An opaque body's `config_name` is not a model alias, so the row
    // claims no model identity.
    expect(labelled["aisix.requested_model"] ?? "").toBe("");

    // 2. The identical token object on an unlabelled frame mints nothing —
    //    the no-phantom-tokens guarantee, now covering opaque STREAMS too.
    const unlabelledRes = await post("/dims-unlabelled/api/agent/v3/history", {
      payload: "opaque",
    });
    expect(unlabelledRes.status).toBe(200);
    await unlabelledRes.text();

    const unlabelled = await logFor("ptr-dims-unlabelled");
    expect(unlabelled["gen_ai.usage.input_tokens"]).toBe(0);
    expect(unlabelled["gen_ai.usage.output_tokens"]).toBe(0);
    expect(unlabelled["aisix.cache_read_tokens"] ?? 0).toBe(0);
    expect(unlabelled["aisix.cache_creation_tokens"] ?? 0).toBe(0);
    expect(unlabelled["aisix.reasoning_tokens"] ?? 0).toBe(0);

    // 3. Detected chat envelope: OpenAI's NESTED cache/reasoning details
    //    reach the event, and the caller's alias attributes the row.
    const chatRes = await post("/dims-chat/v1/chat/completions", {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(chatRes.status).toBe(200);
    await chatRes.text();

    const chat = await logFor("ptr-dims-chat");
    expect(chat["gen_ai.usage.input_tokens"]).toBe(900);
    expect(chat["gen_ai.usage.output_tokens"]).toBe(25);
    expect(chat["aisix.cached_prompt_tokens"]).toBe(768);
    expect(chat["aisix.cache_write_tokens"]).toBe(113);
    expect(chat["aisix.reasoning_tokens"]).toBe(17);
    expect(chat["aisix.requested_model"]).toBe("gpt-4o-mini");
  });

  test("a detected protocol falls back to the server-labelled usage frame only when it read none itself", async (ctx) => {
    if (!etcdReachable || !app || !seed || !dd) {
      ctx.skip();
      return;
    }
    await waitConfigPropagation(async () => {
      const r = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      await r.text();
      return r.status === 200;
    });
    const body = {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "hi" }],
      stream: true,
    };

    const labelledRes = await post("/dims-detected-labelled/v1/chat/completions", body);
    expect(labelledRes.status).toBe(200);
    await labelledRes.text();
    const labelled = await logFor("ptr-dims-detected-labelled");
    expect(labelled["gen_ai.usage.input_tokens"]).toBe(AGENT_USAGE.prompt_tokens);
    expect(labelled["gen_ai.usage.output_tokens"]).toBe(AGENT_USAGE.completion_tokens);
    expect(labelled["aisix.cached_prompt_tokens"]).toBe(AGENT_USAGE.cache_read_input_tokens);
    expect(labelled["aisix.requested_model"]).toBe("gpt-4o-mini");

    const bothRes = await post("/dims-detected-both/v1/chat/completions", body);
    expect(bothRes.status).toBe(200);
    await bothRes.text();
    const both = await logFor("ptr-dims-detected-both");
    expect(both["gen_ai.usage.input_tokens"]).toBe(3);
    expect(both["gen_ai.usage.output_tokens"]).toBe(4);
    expect(both["aisix.cached_prompt_tokens"] ?? 0).toBe(0);
  });

  test("a streamed relay reports the caller's wait to the first frame, not the whole stream", async (ctx) => {
    if (!etcdReachable || !app || !seed || !dd) {
      ctx.skip();
      return;
    }

    // `downstream_latency_ms` means "what the caller waited for before it
    // got something it can use": the complete body when buffered, the first
    // token when streamed. `/a2a` is the ONE documented exception, because
    // an agent's stream of task updates is the call's product — a relay is
    // a delivery mechanism for a response, so it follows the normal rule.
    // Pre-fix a route stamped BOTH `upstream_latency_ms` and
    // `downstream_latency_ms` with the whole-request elapsed at
    // end-of-stream, so a long stream reported the caller as having waited
    // for all of it.
    const started = Date.now();
    const res = await post("/dims-slow/v1/chat/completions", {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "hi" }],
      stream: true,
    });
    expect(res.status).toBe(200);
    await res.text();
    const wallMs = Date.now() - started;

    // The exchange really did run for the whole spaced-out stream, so a
    // whole-stream figure and a first-frame figure are far apart here.
    const tailMs = TAIL_STEP_MS * 4;
    expect(wallMs).toBeGreaterThan(tailMs);

    const slow = await logFor("ptr-dims-slow");
    const downstream = Number(slow["aisix.downstream_latency_ms"] ?? 0);
    const upstreamTotal = Number(slow["aisix.upstream_latency_ms"] ?? 0);
    const ttft = Number(slow["aisix.upstream_ttft_ms"] ?? 0);

    // The caller's wait is the first frame's arrival — bounded well below
    // the tail the stream went on to spend.
    expect(downstream).toBeGreaterThanOrEqual(FIRST_FRAME_MS / 2);
    expect(downstream).toBeLessThan(tailMs);
    // ...while the attempt itself ran for the whole stream.
    expect(upstreamTotal).toBeGreaterThan(tailMs);

    // TTFT is attempt-scoped and the caller's wait is request-scoped over
    // the same first frame, so gateway-side work is exactly their
    // difference — never negative.
    expect(ttft).toBeGreaterThanOrEqual(FIRST_FRAME_MS / 2);
    expect(ttft).toBeLessThanOrEqual(downstream);
  });

  /** Scrape the gateway's Prometheus counters. */
  const scrape = (): Promise<MetricSample[]> => scrapeMetrics(app!.metricsUrl);

  const ROUTE_TOKENS = { endpoint: "/passthrough_route" };

  /** The `aisix_llm_*` token counters one action added on route traffic. */
  function tokenDeltas(before: MetricSample[], after: MetricSample[]) {
    const d = (name: string) => metricDelta(before, after, name, ROUTE_TOKENS);
    return {
      // The LLM request tier: the denominator of every tokens-per-request
      // query, so an identified exchange counts here exactly once.
      requests: d("aisix_llm_requests_total"),
      input: d("aisix_llm_input_tokens_total"),
      output: d("aisix_llm_output_tokens_total"),
      total: d("aisix_llm_total_tokens_total"),
      cached: d("aisix_llm_cached_input_tokens_total"),
      cacheRead: d("aisix_llm_cache_read_input_tokens_total"),
      cacheCreation: d("aisix_llm_cache_creation_input_tokens_total"),
    };
  }

  /**
   * The latency families one action added on route traffic: time to first
   * token (upstream side) and end-to-end latency, each as an observation
   * count.
   */
  function latencyDeltas(before: MetricSample[], after: MetricSample[]) {
    const d = (name: string, extra: Record<string, string> = {}) =>
      metricDelta(before, after, name, { ...ROUTE_TOKENS, ...extra });
    return {
      ttft: d("aisix_llm_time_to_first_token_seconds_count", { side: "upstream" }),
      requestTtft: d("aisix_request_ttft_seconds_count", { side: "upstream" }),
      e2e: d("aisix_request_e2e_latency_seconds_count", { side: "downstream" }),
    };
  }

  /**
   * The token deltas once the route's terminal emit has landed. A streamed
   * relay emits from its drop, which can trail the client's last read, so
   * poll until the input count moves rather than reading once.
   */
  async function settledTokenDeltas(before: MetricSample[]) {
    const deadline = Date.now() + 5_000;
    for (;;) {
      const deltas = tokenDeltas(before, await scrape());
      if (deltas.requests > 0 || Date.now() >= deadline) return deltas;
      await new Promise((r) => setTimeout(r, 50));
    }
  }

  /** The access-log line of one response, by its request id. */
  async function accessLineFor(res: Response): Promise<string> {
    const id = res.headers.get("x-aisix-request-id") ?? "";
    expect(id, "every response carries x-aisix-request-id").toBeTruthy();
    return waitForLogLine(
      app!,
      (l) =>
        l.includes("proxy request completed") && l.includes(`request_id="${id}"`),
      `the access-log line for ${id}`,
    );
  }

  test("each identified LLM envelope is metered as its typed endpoint meters it", async (ctx) => {
    if (!etcdReachable || !app || !seed || !dd) {
      ctx.skip();
      return;
    }

    await waitConfigPropagation(async () => {
      const r = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      await r.text();
      return r.status === 200;
    });

    const anthropicBody = {
      model: "claude-sonnet-4-5",
      max_tokens: 64,
      messages: [{ role: "user", content: "hi" }],
    };
    const anthropicHeaders = { "anthropic-version": "2023-06-01" };

    // Anthropic Messages, buffered then streamed: cache counters BESIDE the
    // prompt, so the input the request processed is all three, and the
    // access-log total includes the cache like `/v1/messages`' does.
    for (const [route, prefix, stream] of [
      ["ptr-dims-anthropic", "/dims-anthropic", false],
      ["ptr-dims-anthropic-stream", "/dims-anthropic-stream", true],
    ] as const) {
      const before = await scrape();
      const res = await post(
        `${prefix}/v1/messages`,
        { ...anthropicBody, stream },
        anthropicHeaders,
      );
      expect(res.status).toBe(200);
      await res.text();

      const row = await logFor(route);
      expect(row["gen_ai.usage.input_tokens"], route).toBe(ANTHROPIC_TOTAL_INPUT);
      expect(row["gen_ai.usage.output_tokens"], route).toBe(
        ANTHROPIC_USAGE.output_tokens,
      );
      expect(row["aisix.cache_creation_tokens"], route).toBe(
        ANTHROPIC_USAGE.cache_creation_input_tokens,
      );
      expect(row["aisix.cache_read_tokens"], route).toBe(
        ANTHROPIC_USAGE.cache_read_input_tokens,
      );
      // The write's split by cache lifetime, as the upstream reported it.
      expect(row["aisix.cache_creation_5m_tokens"], route).toBe(100);
      expect(row["aisix.cache_creation_1h_tokens"], route).toBe(200);
      expect(row["aisix.cached_prompt_tokens"] ?? 0, route).toBe(0);
      // Anthropic reports no total of its own, and the event records only an
      // upstream-stated one — exactly as `/v1/messages` records it.
      expect(row["aisix.total_tokens"] ?? 0, route).toBe(0);
      expect(row["aisix.requested_model"], route).toBe("claude-sonnet-4-5");

      const line = await accessLineFor(res);
      expect(line, route).toContain(
        `total_tokens=${ANTHROPIC_TOTAL_INPUT + ANTHROPIC_USAGE.output_tokens}`,
      );

      expect(await settledTokenDeltas(before), route).toEqual({
        requests: 1,
        input: ANTHROPIC_USAGE.input_tokens,
        output: ANTHROPIC_USAGE.output_tokens,
        total: ANTHROPIC_TOTAL_INPUT + ANTHROPIC_USAGE.output_tokens,
        cached: 0,
        cacheRead: ANTHROPIC_USAGE.cache_read_input_tokens,
        cacheCreation: ANTHROPIC_USAGE.cache_creation_input_tokens,
      });
      // `/v1/messages` times every request end to end, and a streamed one
      // to its first token as well.
      expect(latencyDeltas(before, await scrape()), route).toEqual({
        ttft: stream ? 1 : 0,
        requestTtft: stream ? 1 : 0,
        e2e: 1,
      });
    }

    // OpenAI Responses: the `input`/`output` spelling, cache hit inside.
    {
      const before = await scrape();
      const res = await post("/dims-responses/v1/responses", {
        model: "gpt-5",
        input: "hi",
      });
      expect(res.status).toBe(200);
      await res.text();
      const row = await logFor("ptr-dims-responses");
      expect(row["gen_ai.usage.input_tokens"]).toBe(30);
      expect(row["gen_ai.usage.output_tokens"]).toBe(9);
      expect(row["aisix.cached_prompt_tokens"]).toBe(25);
      expect(row["aisix.reasoning_tokens"]).toBe(4);
      expect(row["aisix.total_tokens"]).toBe(39);
      expect(row["aisix.cache_read_tokens"] ?? 0).toBe(0);
      expect(await accessLineFor(res)).toContain("total_tokens=39");
      expect(await settledTokenDeltas(before)).toEqual({
        requests: 1,
        input: 30,
        output: 9,
        total: 39,
        cached: 25,
        cacheRead: 0,
        cacheCreation: 0,
      });
    }

    // OpenAI Chat, streamed: the cache hit is a part of the prompt.
    {
      const before = await scrape();
      const res = await post("/dims-chat-stream/v1/chat/completions", {
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "hi" }],
        stream: true,
      });
      expect(res.status).toBe(200);
      await res.text();
      const row = await logFor("ptr-dims-chat-stream");
      expect(row["gen_ai.usage.input_tokens"]).toBe(900);
      expect(row["aisix.cached_prompt_tokens"]).toBe(768);
      expect(row["aisix.cache_read_tokens"] ?? 0).toBe(0);
      expect(await accessLineFor(res)).toContain("total_tokens=925");
      expect(await settledTokenDeltas(before)).toEqual({
        requests: 1,
        input: 900,
        output: 25,
        total: 925,
        cached: 768,
        cacheRead: 0,
        cacheCreation: 0,
      });
      expect(latencyDeltas(before, await scrape())).toEqual({
        ttft: 1,
        requestTtft: 1,
        e2e: 1,
      });
    }

    // The opaque agent stream reports usage on its event, but it is not an
    // LLM envelope the gateway identified, so it stays out of the LLM token
    // families.
    {
      const before = await scrape();
      const res = await post("/dims-labelled/api/agent/v3/create_agent_task", {
        config_name: "agent",
        payload: "opaque",
      });
      expect(res.status).toBe(200);
      await res.text();
      // The access-log line is written with the usage event, so once it is
      // visible the request's metrics are too.
      expect(await accessLineFor(res)).toContain(
        `total_tokens=${AGENT_USAGE.total_tokens}`,
      );
      const after = await scrape();
      const deltas = tokenDeltas(before, after);
      expect(deltas.requests).toBe(0);
      expect(deltas.input).toBe(0);
      // A streamed reply is timed whatever its protocol: an opaque relay
      // reports its first frame and its end-to-end latency like any route.
      expect(latencyDeltas(before, after)).toEqual({ ttft: 1, requestTtft: 1, e2e: 1 });
    }

    // OpenAI embeddings and image generation are identified from their
    // paths and read as `/v1/embeddings` and `/v1/images/generations` read
    // them — although their bodies carry another envelope's key.
    {
      const before = await scrape();
      const res = await post("/dims-embeddings/v1/embeddings", {
        model: "text-embedding-3-small",
        input: "hi",
      });
      expect(res.status).toBe(200);
      await res.text();
      const row = await logFor("ptr-dims-embeddings");
      expect(row["gen_ai.usage.input_tokens"]).toBe(8);
      expect(row["gen_ai.usage.output_tokens"] ?? 0).toBe(0);
      expect(await settledTokenDeltas(before)).toEqual({
        requests: 1,
        input: 8,
        output: 0,
        total: 8,
        cached: 0,
        cacheRead: 0,
        cacheCreation: 0,
      });
    }
    {
      const before = await scrape();
      const res = await post("/dims-images/v1/images/generations", {
        model: "gpt-image-1",
        prompt: "a cat",
      });
      expect(res.status).toBe(200);
      await res.text();
      const row = await logFor("ptr-dims-images");
      expect(row["gen_ai.usage.input_tokens"]).toBe(50);
      expect(row["gen_ai.usage.output_tokens"]).toBe(4160);
      expect(await settledTokenDeltas(before)).toEqual({
        requests: 1,
        input: 50,
        output: 4160,
        total: 4210,
        cached: 0,
        cacheRead: 0,
        cacheCreation: 0,
      });
    }
  });
});
