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

// E2E: what `upstream_ttft_ms` is allowed to stop on (AISIX-Cloud#1225).
//
// A reasoning model streams its chain of thought first and only emits
// `content` once it has stopped thinking. The reported request spent
// ~182s reasoning, so a TTFT that only recognises `content` frames read
// 183,581 ms while the gateway in front of it measured the first token at
// 1,175 ms. The clock must stop on the first frame carrying generated
// output of ANY kind — reasoning text included.
//
// The mirror-image defect: OpenAI opens every stream with a role-only
// `{"role":"assistant","content":""}` frame. Testing `content.is_some()`
// accepted that empty string, so the clock stopped when the stream
// opened rather than when a token arrived.
//
// Both are asserted per inbound protocol, since the predicate lives at
// four sites across the handler family.

const CALLER_PLAINTEXT = "sk-ttft-reasoning-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

/** Inter-chunk gap. */
const GAP_MS = 250;
/** Delay before the first SSE event, so a correct TTFT is measurably > 0. */
const LEAD_MS = 250;
/** Reasoning frames streamed before the first content frame. */
const REASONING_FRAMES = 6;
/**
 * When the first content frame lands — what a `content`-only predicate
 * reports. A correct TTFT sits at the first reasoning frame instead, so
 * the midpoint separates the two by a wide margin either way.
 */
const CONTENT_STARTS_MS = LEAD_MS + REASONING_FRAMES * GAP_MS;

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
    const hit = recv.spans.find((a) => a["aisix.request_id"] === requestId);
    if (hit) return hit;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(`no usage span for request_id=${requestId}`);
}

function chatChunk(delta: unknown): string {
  return JSON.stringify({
    id: "chatcmpl-ttft-reasoning",
    object: "chat.completion.chunk",
    created: 1,
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta, finish_reason: null }],
  });
}

const CHAT_TERMINAL = JSON.stringify({
  id: "chatcmpl-ttft-reasoning",
  object: "chat.completion.chunk",
  created: 1,
  model: "gpt-4o-mini",
  choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
  usage: { prompt_tokens: 5, completion_tokens: 8, total_tokens: 13 },
});

/**
 * The customer's shape: DeepSeek-class reasoning frames carry the text at
 * `reasoning_content` and leave `content` null, then real content follows.
 */
function reasoningThenContent(): string[] {
  return [
    ...Array.from({ length: REASONING_FRAMES }, (_, i) =>
      chatChunk({ content: null, reasoning_content: `think${i} ` }),
    ),
    chatChunk({ content: "answer " }),
    chatChunk({ content: "here" }),
    CHAT_TERMINAL,
    "[DONE]",
  ];
}

describe("upstream TTFT stops on the first generated-output frame", () => {
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
      name: "ttft-reasoning-otlp",
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

  async function createModel(
    displayName: string,
    upstream: OpenAiUpstream,
  ): Promise<void> {
    if (!seed) throw new Error("seed client not initialized");
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

  test("reasoning text stops the clock on /v1/chat/completions", async (ctx) => {
    if (!etcdReachable || !app || !seed || !otlp) {
      ctx.skip();
      return;
    }

    const upstream = await startOpenAiUpstream({
      streamEvents: reasoningThenContent(),
      firstEventDelayMs: LEAD_MS,
      eventDelayMs: GAP_MS,
    });
    upstreams.push(upstream);
    await createModel("ttft-reasoning-chat", upstream);
    await awaitPropagation("reasoning-chat");

    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "ttft-reasoning-chat",
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

    // The first reasoning frame is a token arriving, so the clock stops
    // near LEAD_MS — not at CONTENT_STARTS_MS, which is what a
    // content-only predicate reported.
    expect(Number.isFinite(ttft)).toBe(true);
    expect(ttft).toBeGreaterThan(0);
    expect(ttft).toBeLessThan(CONTENT_STARTS_MS / 2);

    // The caller-facing figure can never precede the upstream TTFT it
    // waited on. Skipping reasoning frames broke exactly that: the
    // gateway forwarded them (so downstream marked the first one) while
    // TTFT held out for content a whole thinking phase later.
    expect(downstream).toBeGreaterThanOrEqual(ttft);
  });

  test("an empty role-only frame does not stop the clock", async (ctx) => {
    if (!etcdReachable || !app || !seed || !otlp) {
      ctx.skip();
      return;
    }

    // OpenAI's stream opener, then a long wait for the first real token.
    const upstream = await startOpenAiUpstream({
      streamEvents: [
        chatChunk({ role: "assistant", content: "" }),
        chatChunk({ content: "answer " }),
        chatChunk({ content: "here" }),
        CHAT_TERMINAL,
        "[DONE]",
      ],
      eventDelayMs: GAP_MS * 2,
    });
    upstreams.push(upstream);
    await createModel("ttft-role-only-chat", upstream);
    await awaitPropagation("role-only-chat");

    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "ttft-role-only-chat",
        messages: [{ role: "user", content: "answer" }],
        stream: true,
      }),
    });
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-call-id");
    expect(requestId).toBeTruthy();
    await res.text();

    const span = await waitForSpan(otlp, requestId!);
    const ttft = Number(span["aisix.upstream_ttft_ms"]);

    // `content: ""` is the stream opening, not a token; the clock has to
    // wait for the frame one gap later. Reading the opener put this at ~0.
    expect(Number.isFinite(ttft)).toBe(true);
    expect(ttft).toBeGreaterThan(GAP_MS);
  });

  test("reasoning summaries stop the clock on /v1/responses", async (ctx) => {
    if (!etcdReachable || !app || !seed || !otlp) {
      ctx.skip();
      return;
    }

    const upstream = await startOpenAiUpstream({
      streamEvents: [
        JSON.stringify({ type: "response.created", response: { id: "resp_ttft" } }),
        ...Array.from({ length: REASONING_FRAMES }, (_, i) =>
          JSON.stringify({
            type: "response.reasoning_summary_text.delta",
            delta: `think${i} `,
          }),
        ),
        JSON.stringify({ type: "response.output_text.delta", delta: "answer" }),
        JSON.stringify({
          type: "response.completed",
          response: {
            id: "resp_ttft",
            status: "completed",
            usage: { input_tokens: 6, output_tokens: 9 },
          },
        }),
        "[DONE]",
      ],
      firstEventDelayMs: LEAD_MS,
      eventDelayMs: GAP_MS,
    });
    upstreams.push(upstream);
    await createModel("ttft-reasoning-responses", upstream);
    await awaitPropagation("reasoning-responses");

    const res = await fetch(`${app.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "ttft-reasoning-responses",
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

    // `response.created` is not generated output, so the clock runs past
    // it; the first reasoning-summary delta stops it — one frame later,
    // not REASONING_FRAMES later.
    expect(Number.isFinite(ttft)).toBe(true);
    expect(ttft).toBeGreaterThan(0);
    expect(ttft).toBeLessThan(CONTENT_STARTS_MS / 2 + GAP_MS);
  });
});
