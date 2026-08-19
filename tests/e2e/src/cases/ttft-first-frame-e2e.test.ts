import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  pickFreePort,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: `upstream_ttft_ms` stops on the FIRST streamed frame of any type
// (AISIX-Cloud#1225, both rounds).
//
// The clock deliberately uses the industry convention — LiteLLM's
// `completion_start_time` and caller-side gateways (istio/Higress-class
// AI gateways) all stamp the first response chunk, whatever it carries.
// A "generated output only" predicate looked more truthful but made the
// figure structurally incomparable: a hidden-reasoning upstream that
// streams nothing it considers output while thinking (Azure GPT-5.x on
// /v1/responses without reasoning summaries) pushed our TTFT to the END
// of the thinking phase (23,595 ms on a 23,903 ms attempt) while the
// gateway in front measured the same request's first frame at 2,822 ms —
// an inversion customers keep filing as a bug. Frame types no whitelist
// anticipated (each provider adds its own) reopen that gap forever; the
// first-frame convention closes it by construction.
//
// Asserted per inbound protocol, since the stamp lives at five sites
// across the handler family (chat, /v1/messages native + bridge,
// /v1/responses native + bridge).

const CALLER_PLAINTEXT = "sk-ttft-first-frame-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

/** Inter-chunk gap. */
const GAP_MS = 300;
/** Delay before the first SSE event, so a correct TTFT is measurably > 0. */
const LEAD_MS = 250;
/** Frames streamed between the stream opener and the first visible output. */
const QUIET_FRAMES = 6;
/**
 * When the first visible-output frame lands — what a generated-output
 * predicate reports. A first-frame TTFT sits at LEAD_MS instead, so the
 * midpoint separates the two by a wide margin either way.
 */
const OUTPUT_STARTS_MS = LEAD_MS + (QUIET_FRAMES + 1) * GAP_MS;

interface OtlpReceiver {
  url: string;
  spans: Array<Record<string, string>>;
  close(): Promise<void>;
}

async function startOtlpReceiver(): Promise<OtlpReceiver> {
  const spans: Array<Record<string, string>> = [];
  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      try {
        const body = JSON.parse(raw);
        for (const rs of body.resourceSpans ?? []) {
          for (const ss of rs.scopeSpans ?? []) {
            for (const span of ss.spans ?? []) {
              const attrs: Record<string, string> = {};
              for (const a of span.attributes ?? []) {
                const v = a.value ?? {};
                attrs[a.key] =
                  v.stringValue ?? String(v.intValue ?? v.boolValue ?? "");
              }
              spans.push(attrs);
            }
          }
        }
      } catch {
        // ignore malformed bodies — assertions fail on missing spans
      }
      res.statusCode = 200;
      res.end("{}");
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    url: `http://127.0.0.1:${port}/v1/traces`,
    spans,
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

async function waitForSpan(
  recv: OtlpReceiver,
  requestId: string,
  timeoutMs = 10_000,
): Promise<Record<string, string>> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    // Attempt carriers only: since AISIX-Cloud#1279 the export also ships
    // the trace's structural SERVER / logical spans, which share
    // `aisix.request_id` but carry no per-attempt usage attributes.
    const hit = recv.spans.find(
      (a) =>
        a["aisix.request_id"] === requestId &&
        a["aisix.attempt_index"] !== undefined,
    );
    if (hit) return hit;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(`no usage span for request_id=${requestId}`);
}

function chatChunk(delta: unknown): string {
  return JSON.stringify({
    id: "chatcmpl-ttft-first-frame",
    object: "chat.completion.chunk",
    created: 1,
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta, finish_reason: null }],
  });
}

const CHAT_TERMINAL = JSON.stringify({
  id: "chatcmpl-ttft-first-frame",
  object: "chat.completion.chunk",
  created: 1,
  model: "gpt-4o-mini",
  choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
  usage: { prompt_tokens: 5, completion_tokens: 8, total_tokens: 13 },
});

/**
 * OpenAI's opener (a role-only frame carrying no output) followed by a
 * long quiet phase, then the visible answer. The opener is the frame a
 * front-side gateway stamps, so the clock must stop there — waiting for
 * the first visible output re-opens the customer's inversion.
 */
function openerThenQuietThenContent(quietFrames = QUIET_FRAMES): string[] {
  return [
    chatChunk({ role: "assistant", content: "" }),
    ...Array.from({ length: quietFrames }, (_, i) =>
      chatChunk({ content: null, reasoning_content: `think${i} ` }),
    ),
    chatChunk({ content: "answer " }),
    chatChunk({ content: "here" }),
    CHAT_TERMINAL,
    "[DONE]",
  ];
}

/**
 * Gap for the chat-opener case, where the discriminating distance is a
 * single frame: the opener at LEAD_MS vs the first reasoning delta one
 * gap later. Wide, so the assertion `ttft < LEAD_MS + CHAT_GAP_MS`
 * separates the two stamps by a full 900 ms rather than scheduler
 * noise.
 */
const CHAT_GAP_MS = 900;

/**
 * The Azure GPT-5.x /v1/responses shape from the second #1225 report:
 * `response.created` arrives with the headers, the whole thinking phase
 * emits only bookkeeping events (no reasoning summaries on that
 * deployment), and the visible output lands as a late burst just before
 * `response.completed`.
 */
function responsesSilentThinkingStream(): string[] {
  return [
    JSON.stringify({ type: "response.created", response: { id: "resp_ttft" } }),
    JSON.stringify({
      type: "response.output_item.added",
      output_index: 0,
      item: { type: "reasoning", id: "rs_ttft", summary: [] },
    }),
    ...Array.from({ length: QUIET_FRAMES - 1 }, () =>
      JSON.stringify({ type: "response.in_progress", response: { id: "resp_ttft" } }),
    ),
    JSON.stringify({ type: "response.output_text.delta", delta: "answer " }),
    JSON.stringify({ type: "response.output_text.delta", delta: "here" }),
    JSON.stringify({
      type: "response.completed",
      response: {
        id: "resp_ttft",
        status: "completed",
        usage: {
          input_tokens: 6,
          output_tokens: 9,
          output_tokens_details: { reasoning_tokens: 4 },
        },
      },
    }),
    "[DONE]",
  ];
}

/**
 * Anthropic-native stream whose thinking phase is only pings:
 * `message_start` opens immediately, then nothing content-shaped until a
 * late `content_block_start`/`delta` burst. `message_start` is what the
 * caller side observes first, so it stops the clock.
 */
function anthropicQuietThinkingStream(): string[] {
  return [
    JSON.stringify({
      type: "message_start",
      message: {
        id: "msg_ttft",
        model: "mco-5",
        usage: { input_tokens: 7, output_tokens: 1 },
      },
    }),
    ...Array.from({ length: QUIET_FRAMES }, () =>
      JSON.stringify({ type: "ping" }),
    ),
    JSON.stringify({
      type: "content_block_start",
      index: 0,
      content_block: { type: "text", text: "" },
    }),
    JSON.stringify({
      type: "content_block_delta",
      index: 0,
      delta: { type: "text_delta", text: "answer here" },
    }),
    JSON.stringify({ type: "content_block_stop", index: 0 }),
    JSON.stringify({
      type: "message_delta",
      delta: { stop_reason: "end_turn" },
      usage: { output_tokens: 9 },
    }),
    JSON.stringify({ type: "message_stop" }),
  ];
}

describe("upstream TTFT stops on the first streamed frame", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let otlp: OtlpReceiver | undefined;
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);
    otlp = await startOtlpReceiver();
    await seed.createObservabilityExporter({
      name: "ttft-first-frame-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
    await otlp?.close();
  });

  /**
   * `provider` selects the dispatch path under test. `openai` takes the
   * native routes; `anthropic` the /v1/messages byte passthrough; any
   * other OpenAI-compatible provider (here `deepseek`) routes
   * `/v1/messages` through the cross-provider translation and
   * `/v1/responses` through the chat-completions bridge.
   */
  async function createModel(
    displayName: string,
    upstream: OpenAiUpstream,
    provider: "openai" | "deepseek" | "anthropic" = "openai",
  ): Promise<void> {
    if (!seed) throw new Error("seed client not initialized");
    const pk = await seed.createProviderKey({
      display_name: `${displayName}-pk`,
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    const modelName = {
      openai: "gpt-4o-mini",
      deepseek: "deepseek-reasoner",
      anthropic: "mco-5",
    }[provider];
    await seed.createModel({
      display_name: displayName,
      provider,
      model_name: modelName,
      provider_key_id: pk.id,
    });
  }

  /** Seed a throwaway key AFTER the config under test, then poll until it authenticates. */
  async function awaitPropagation(tag: string): Promise<void> {
    const canary = `sk-canary-${tag}-${Date.now()}`;
    await seed!.createApiKey({
      key_hash: createHash("sha256").update(canary).digest("hex"),
      allowed_models: ["*"],
    });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${canary}` },
      });
      return res.status === 200;
    });
  }

  test("the role-only opener stops the clock on /v1/chat/completions", async (ctx) => {
    if (!etcdReachable || !app || !seed || !otlp) {
      ctx.skip();
      return;
    }

    const upstream = await startOpenAiUpstream({
      streamEvents: openerThenQuietThenContent(2),
      firstEventDelayMs: LEAD_MS,
      eventDelayMs: CHAT_GAP_MS,
    });
    upstreams.push(upstream);
    await createModel("ttft-ff-chat", upstream);
    await awaitPropagation("ff-chat");

    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "ttft-ff-chat",
        messages: [{ role: "user", content: "think then answer" }],
        stream: true,
      }),
    });
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-call-id");
    expect(requestId).toBeTruthy();
    await res.text();

    const span = await waitForSpan(otlp, requestId!);
    const ttft = Number(span["aisix.upstream_ttft_ms"]);
    const downstream = Number(span["aisix.downstream_latency_ms"]);

    // The opener is the first frame, so the clock stops near LEAD_MS. A
    // predicate that skips role-only frames waits at least one full gap
    // for the first reasoning delta, so this bound separates the two
    // stamps deterministically: the mock cannot deliver frame two before
    // LEAD_MS + CHAT_GAP_MS.
    expect(Number.isFinite(ttft)).toBe(true);
    expect(ttft).toBeGreaterThan(0);
    expect(ttft).toBeLessThan(LEAD_MS + CHAT_GAP_MS);

    // The caller-facing figure can never precede the upstream frame it
    // waited on. With both clocks stamping the same first frame (at
    // request scope vs attempt scope) this holds by construction; an
    // output-only TTFT inverted it — the customer's 2,726 ms caller
    // latency against a 23,595 ms "first token".
    expect(downstream).toBeGreaterThanOrEqual(ttft);
  });

  test("silent thinking on native /v1/responses: created stops the clock (#1225 round 2)", async (ctx) => {
    if (!etcdReachable || !app || !seed || !otlp) {
      ctx.skip();
      return;
    }

    const upstream = await startOpenAiUpstream({
      streamEvents: responsesSilentThinkingStream(),
      firstEventDelayMs: LEAD_MS,
      eventDelayMs: GAP_MS,
    });
    upstreams.push(upstream);
    await createModel("ttft-ff-responses", upstream);
    await awaitPropagation("ff-responses");

    const res = await fetch(`${app.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "ttft-ff-responses",
        input: "think silently then answer",
        stream: true,
      }),
    });
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id");
    expect(requestId).toBeTruthy();
    await res.text();

    const span = await waitForSpan(otlp, requestId!);
    const ttft = Number(span["aisix.upstream_ttft_ms"]);
    const downstream = Number(span["aisix.downstream_latency_ms"]);

    // `response.created` is the first frame the caller side observes, so
    // it stops the clock. A generated-output whitelist ran the clock
    // through the whole quiet phase and landed at the late output burst.
    expect(Number.isFinite(ttft)).toBe(true);
    expect(ttft).toBeGreaterThan(0);
    expect(ttft).toBeLessThan(OUTPUT_STARTS_MS / 2);
    expect(downstream).toBeGreaterThanOrEqual(ttft);
  });

  test("quiet thinking on native /v1/messages: message_start stops the clock", async (ctx) => {
    if (!etcdReachable || !app || !seed || !otlp) {
      ctx.skip();
      return;
    }

    const upstream = await startOpenAiUpstream({
      streamEvents: anthropicQuietThinkingStream(),
      firstEventDelayMs: LEAD_MS,
      eventDelayMs: GAP_MS,
    });
    upstreams.push(upstream);
    await createModel("ttft-ff-messages-native", upstream, "anthropic");
    await awaitPropagation("ff-messages-native");

    const res = await fetch(`${app.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "ttft-ff-messages-native",
        max_tokens: 128,
        messages: [{ role: "user", content: "think then answer" }],
        stream: true,
      }),
    });
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id");
    expect(requestId).toBeTruthy();
    await res.text();

    const span = await waitForSpan(otlp, requestId!);
    const ttft = Number(span["aisix.upstream_ttft_ms"]);

    expect(Number.isFinite(ttft)).toBe(true);
    expect(ttft).toBeGreaterThan(0);
    expect(ttft).toBeLessThan(OUTPUT_STARTS_MS / 2);
  });

  test("the first bridged chunk stops the clock on the /v1/messages bridge", async (ctx) => {
    if (!etcdReachable || !app || !seed || !otlp) {
      ctx.skip();
      return;
    }

    // Anthropic-shaped request, OpenAI-compatible upstream: the
    // cross-provider translation, not the byte-for-byte passthrough.
    // The wide gap makes the assertion discriminate this call site's
    // stamp: a generated-output predicate skips the opener and waits a
    // full CHAT_GAP_MS for the first reasoning delta.
    const upstream = await startOpenAiUpstream({
      streamEvents: openerThenQuietThenContent(2),
      firstEventDelayMs: LEAD_MS,
      eventDelayMs: CHAT_GAP_MS,
    });
    upstreams.push(upstream);
    await createModel("ttft-ff-messages-bridge", upstream, "deepseek");
    await awaitPropagation("ff-messages-bridge");

    const res = await fetch(`${app.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "ttft-ff-messages-bridge",
        max_tokens: 128,
        messages: [{ role: "user", content: "think then answer" }],
        stream: true,
      }),
    });
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id");
    expect(requestId).toBeTruthy();
    await res.text();

    const span = await waitForSpan(otlp, requestId!);
    const ttft = Number(span["aisix.upstream_ttft_ms"]);

    // Same discriminating bound as the chat case: the mock cannot
    // deliver frame two before LEAD_MS + CHAT_GAP_MS, so a predicate
    // that skips the opener always lands past this bound.
    expect(Number.isFinite(ttft)).toBe(true);
    expect(ttft).toBeGreaterThan(0);
    expect(ttft).toBeLessThan(LEAD_MS + CHAT_GAP_MS);
  });

  test("the first bridged chunk stops the clock on the /v1/responses bridge", async (ctx) => {
    if (!etcdReachable || !app || !seed || !otlp) {
      ctx.skip();
      return;
    }

    // A non-OpenAI provider serves /v1/responses through the
    // chat-completions bridge, so the upstream speaks chat SSE. Wide gap
    // for the same reason as the messages bridge above: the bound must
    // fail if this call site regains a generated-output predicate.
    const upstream = await startOpenAiUpstream({
      streamEvents: openerThenQuietThenContent(2),
      firstEventDelayMs: LEAD_MS,
      eventDelayMs: CHAT_GAP_MS,
    });
    upstreams.push(upstream);
    await createModel("ttft-ff-resp-bridge", upstream, "deepseek");
    await awaitPropagation("ff-resp-bridge");

    const res = await fetch(`${app.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "ttft-ff-resp-bridge",
        input: "think then answer",
        stream: true,
      }),
    });
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id");
    expect(requestId).toBeTruthy();
    await res.text();

    const span = await waitForSpan(otlp, requestId!);
    const ttft = Number(span["aisix.upstream_ttft_ms"]);

    // Same discriminating bound as the chat case — see the messages
    // bridge above.
    expect(Number.isFinite(ttft)).toBe(true);
    expect(ttft).toBeGreaterThan(0);
    expect(ttft).toBeLessThan(LEAD_MS + CHAT_GAP_MS);
  });
});
