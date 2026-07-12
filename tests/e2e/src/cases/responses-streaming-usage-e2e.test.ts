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

// E2E regression for AISIX-Cloud#808: a STREAMING /v1/responses 200 must
// emit a UsageEvent. Codex (the OpenAI CLI) talks to /v1/responses and
// always streams, so pre-#808 every successful Codex call was invisible to
// the dashboard Logs and the budget ledger — while a 4xx/5xx on the same
// endpoint still produced a (zero-token) log row. The verbatim streaming
// path returned `usage: None` and never parsed the SSE bytes, so no event
// was emitted at all.
//
// Post-fix: the response stream's Drop guard parses the terminal
// `response.completed` event's `usage` block and emits a UsageEvent at
// end-of-stream — the same end-of-stream emission the Anthropic
// /v1/messages streaming path already does (#245/#790).
//
// Usage telemetry has no cp-api receiver in DP e2e, so — like the
// per-attempt telemetry test (#655) — we observe the emitted field values
// through the per-env OTLP/HTTP fan-out: register a mock OTLP receiver as
// an observability_exporter, drive one streamed request, and assert a span
// carrying the request_id reports the terminal-event token counts.

const CALLER_PLAINTEXT = "sk-issue-808-streaming";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const INPUT_TOKENS = 17;
const OUTPUT_TOKENS = 23;

// Real Responses-API streaming wire shape: created → output_text deltas →
// terminal `response.completed` carrying the authoritative `usage` block
// (nested under `response`), then `[DONE]`.
const STREAM_EVENTS = [
  JSON.stringify({ type: "response.created", response: { id: "resp_808" } }),
  JSON.stringify({ type: "response.output_text.delta", delta: "hello " }),
  JSON.stringify({ type: "response.output_text.delta", delta: "from codex" }),
  JSON.stringify({
    type: "response.completed",
    response: {
      id: "resp_808",
      status: "completed",
      usage: {
        input_tokens: INPUT_TOKENS,
        output_tokens: OUTPUT_TOKENS,
        output_tokens_details: { reasoning_tokens: 4 },
        input_tokens_details: { cached_tokens: 5 },
      },
    },
  }),
  "[DONE]",
];

interface OtlpReceiver {
  url: string;
  spanAttrs: Array<Record<string, string>>;
  close(): Promise<void>;
}

async function startOtlpReceiver(): Promise<OtlpReceiver> {
  const spanAttrs: Array<Record<string, string>> = [];
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
              spanAttrs.push(attrs);
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
    spanAttrs,
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

/**
 * Collect every usage span emitted for `requestId`. Waits for the first to
 * arrive, then settles briefly so a (buggy) duplicate emit has time to show
 * up — the caller asserts cardinality, which is the point: the fix must emit
 * exactly one usage event per streamed request (the `usage_handled_by_stream`
 * guard prevents the handler from double-emitting alongside the Drop guard).
 */
async function collectUsageSpans(
  recv: OtlpReceiver,
  requestId: string,
  timeoutMs = 10_000,
): Promise<Array<Record<string, string>>> {
  const matches = () =>
    recv.spanAttrs.filter(
      (a) =>
        a["aisix.request_id"] === requestId &&
        a["gen_ai.usage.input_tokens"] !== undefined,
    );
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (matches().length > 0) break;
    await new Promise((r) => setTimeout(r, 50));
  }
  if (matches().length === 0) {
    throw new Error(`no usage span for request_id=${requestId}`);
  }
  // Settle: catch a duplicate emitted shortly after the first.
  await new Promise((r) => setTimeout(r, 300));
  return matches();
}

describe("responses streaming usage emission (#808)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let otlp: OtlpReceiver | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      streamEvents: STREAM_EVENTS,
      eventDelayMs: 2,
    });
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);
    otlp = await startOtlpReceiver();
    await seed.createObservabilityExporter({
      name: "issue808-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });

    const pk = await seed.createProviderKey({
      display_name: "issue808-pk",
      secret: "sk-openai-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "gpt-5-codex",
      provider: "openai",
      model_name: "gpt-5-codex",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["gpt-5-codex"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await otlp?.close();
  });

  test("a streamed 200 emits a UsageEvent with the terminal-event token counts", async (ctx) => {
    if (!etcdReachable || !app || !upstream || !otlp) {
      ctx.skip();
      return;
    }

    // Streaming readiness probe (the mock always streams) — drained and
    // kept separate from the measured request, which is matched by its own
    // request_id, so the readiness spans don't interfere.
    await waitConfigPropagation(async () => {
      try {
        const r = await fetch(`${app!.proxyUrl}/v1/responses`, {
          method: "POST",
          headers: {
            authorization: `Bearer ${CALLER_PLAINTEXT}`,
            "content-type": "application/json",
          },
          body: JSON.stringify({
            model: "gpt-5-codex",
            input: "ready",
            stream: true,
          }),
        });
        const text = await r.text();
        return r.status === 200 && text.includes("response.completed");
      } catch {
        return false;
      }
    });

    const res = await fetch(`${app.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
        "user-agent": "codex_cli_rs/0.5.0",
      },
      body: JSON.stringify({
        model: "gpt-5-codex",
        input: "issue 808 streamed",
        stream: true,
      }),
    });
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id");
    expect(requestId).toBeTruthy();

    // Bytes pass through verbatim — the terminal event + [DONE] reach the
    // client unchanged.
    const body = await res.text();
    expect(body).toContain("response.completed");
    expect(body).toContain("[DONE]");

    // The outbound upstream request streamed against /v1/responses.
    const lastReq = upstream.receivedRequests.at(-1);
    expect(lastReq?.path).toBe("/v1/responses");
    expect((JSON.parse(lastReq!.body) as { stream?: unknown }).stream).toBe(true);

    // Exactly one UsageEvent for the streamed request (not zero, not a
    // double-emit), carrying the terminal-event token counts.
    const spans = await collectUsageSpans(otlp, requestId!);
    expect(spans).toHaveLength(1);
    const [span] = spans;
    expect(span["gen_ai.usage.input_tokens"]).toBe(String(INPUT_TOKENS));
    expect(span["gen_ai.usage.output_tokens"]).toBe(String(OUTPUT_TOKENS));
    expect(span["http.response.status_code"]).toBe("200");
  });
});
