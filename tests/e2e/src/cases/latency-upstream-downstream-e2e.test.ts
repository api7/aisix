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

// E2E: the UsageEvent carries two latency families measured against
// different clocks (see `UsageEvent` in aisix-obs/src/usage.rs):
//
//   upstream_ttft_ms      attempt-scoped — when the UPSTREAM produced its
//                         first generated chunk.
//   downstream_latency_ms request-scoped — when the CALLER got its first
//                         usable bytes. Includes everything the gateway
//                         did in between.
//
// Two properties pin the split:
//
//  1. With no output guardrail the gateway forwards chunks straight
//     through, so the two figures nearly coincide.
// The hold-back counterpart — where a masking output guardrail makes the
// two diverge — needs an env-scoped guardrail, which would leak into
// every later request here, so it lives in
// `latency-guardrail-holdback-e2e.test.ts` with its own gateway.
//
// It also covers a bug fixed alongside: `/v1/responses` never recorded a
// TTFT at all, so codex-class clients showed a blank figure.

const CALLER_PLAINTEXT = "sk-latency-split-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

/** Inter-chunk gap; the hold-back case pays the whole stream before releasing. */
const CHUNK_GAP_MS = 250;
const CHUNK_COUNT = 4;
/** Total time the upstream spends streaming after its first chunk. */
const STREAM_TAIL_MS = CHUNK_GAP_MS * (CHUNK_COUNT - 1);

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

/** OpenAI-shape SSE chunks whose text is benign (nothing for the guardrail to mask). */
function chatChunks(): string[] {
  const events = Array.from({ length: CHUNK_COUNT }, (_, i) =>
    JSON.stringify({
      id: "chatcmpl-latency-split",
      object: "chat.completion.chunk",
      created: 1,
      model: "gpt-4o-mini",
      choices: [{ index: 0, delta: { content: `part${i} ` }, finish_reason: null }],
    }),
  );
  events.push(
    JSON.stringify({
      id: "chatcmpl-latency-split",
      object: "chat.completion.chunk",
      created: 1,
      model: "gpt-4o-mini",
      choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
      usage: { prompt_tokens: 5, completion_tokens: 8, total_tokens: 13 },
    }),
    "[DONE]",
  );
  return events;
}

describe("upstream vs downstream latency split", () => {
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
      name: "latency-split-otlp",
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

  test("without an output guardrail the two figures nearly coincide", async (ctx) => {
    if (!etcdReachable || !app || !seed || !otlp) {
      ctx.skip();
      return;
    }

    const upstream = await startOpenAiUpstream({
      streamEvents: chatChunks(),
      eventDelayMs: CHUNK_GAP_MS,
    });
    upstreams.push(upstream);
    await createModel("latency-plain", upstream);
    await awaitPropagation("plain");

    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "latency-plain",
        messages: [{ role: "user", content: "stream please" }],
        stream: true,
      }),
    });
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-call-id");
    expect(requestId).toBeTruthy();
    await res.text();

    const span = await waitForSpan(otlp, requestId!);
    const upstreamTtft = Number(span["aisix.upstream_ttft_ms"]);
    const downstream = Number(span["aisix.downstream_latency_ms"]);

    expect(Number.isFinite(upstreamTtft)).toBe(true);
    expect(Number.isFinite(downstream)).toBe(true);
    // Live-forward: the client gets the first chunk as it arrives, so the
    // caller-facing figure sits just above the upstream's TTFT — and well
    // below the point where the whole stream has finished.
    expect(downstream).toBeGreaterThanOrEqual(upstreamTtft);
    expect(downstream - upstreamTtft).toBeLessThan(STREAM_TAIL_MS);
  });

  test("/v1/responses streaming records a TTFT (it previously reported none)", async (ctx) => {
    if (!etcdReachable || !app || !seed || !otlp) {
      ctx.skip();
      return;
    }

    const upstream = await startOpenAiUpstream({
      streamEvents: [
        JSON.stringify({ type: "response.created", response: { id: "resp_lat" } }),
        JSON.stringify({ type: "response.output_text.delta", delta: "hello " }),
        JSON.stringify({ type: "response.output_text.delta", delta: "there" }),
        JSON.stringify({
          type: "response.completed",
          response: {
            id: "resp_lat",
            status: "completed",
            usage: { input_tokens: 6, output_tokens: 9 },
          },
        }),
        "[DONE]",
      ],
      eventDelayMs: CHUNK_GAP_MS,
    });
    upstreams.push(upstream);
    await createModel("latency-responses", upstream);
    await awaitPropagation("responses");

    const res = await fetch(`${app.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "latency-responses",
        input: "measure ttft",
        stream: true,
      }),
    });
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id");
    expect(requestId).toBeTruthy();
    await res.text();

    const span = await waitForSpan(otlp, requestId!);
    // The regression: this attribute was absent entirely on /v1/responses.
    expect(span["aisix.upstream_ttft_ms"]).toBeDefined();
    expect(Number(span["aisix.upstream_ttft_ms"])).toBeGreaterThan(0);
    expect(Number(span["aisix.downstream_latency_ms"])).toBeGreaterThan(0);
  });
});
