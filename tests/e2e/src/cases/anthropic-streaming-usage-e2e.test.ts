import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: Anthropic /v1/messages STREAMING usage telemetry (#245,
// dp-blocker). Parity with the OpenAI streaming fix (#225).
//
// Pre-fix the Anthropic passthrough streaming branch forwarded the
// upstream SSE bytes verbatim WITHOUT parsing them, so the DP recorded
// prompt_tokens=0 / completion_tokens=0 for every streaming
// /v1/messages request — silently zeroing budget enforcement and
// under-counting billed tokens across the whole Anthropic-SDK
// streaming surface.
//
// This test drives a real streaming /v1/messages request through the
// DP binary against a mock Anthropic streaming upstream whose SSE
// carries non-zero token counts (input_tokens in message_start,
// output_tokens in message_delta), then scrapes the DP's own
// /metrics endpoint and asserts the per-request LLM token counters
// are NON-ZERO. With the pre-#245 bug these counters stay at 0, so
// this test fails red on a regression.
//
// Why metrics rather than cp-api: the DP e2e harness runs the gateway
// standalone (no cp-api in the loop). `aisix_llm_input_tokens_total` /
// `aisix_llm_output_tokens_total` are recorded by the same
// emit_anthropic_usage_event call that ships the UsageEvent, so a
// non-zero token counter proves the streaming usage parser ran.
//
// References:
// - Issue: api7/AISIX-Cloud#245
// - Anthropic streaming wire shape:
//   https://docs.anthropic.com/en/api/messages-streaming

const CALLER_PLAINTEXT = "sk-anth-stream-usage-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

// Anthropic streaming SSE frames. The harness wraps each as
// `data: <frame>\n\n`; the DP parser keys off the JSON `type` field
// (it does not require the textual `event:` line). input_tokens lands
// in message_start; the running output_tokens lands in message_delta.
const INPUT_TOKENS = 37;
const OUTPUT_TOKENS = 52;
const STREAM_EVENTS = [
  JSON.stringify({
    type: "message_start",
    message: {
      id: "msg_e2e_245",
      role: "assistant",
      content: [],
      model: "claude-3-5-haiku-20241022",
      stop_reason: null,
      usage: { input_tokens: INPUT_TOKENS, output_tokens: 1 },
    },
  }),
  JSON.stringify({
    type: "content_block_start",
    index: 0,
    content_block: { type: "text", text: "" },
  }),
  JSON.stringify({
    type: "content_block_delta",
    index: 0,
    delta: { type: "text_delta", text: "hello there" },
  }),
  JSON.stringify({ type: "content_block_stop", index: 0 }),
  JSON.stringify({
    type: "message_delta",
    delta: { stop_reason: "end_turn" },
    usage: { output_tokens: OUTPUT_TOKENS },
  }),
  JSON.stringify({ type: "message_stop" }),
];

describe("anthropic /v1/messages streaming usage telemetry (#245)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    // Path-agnostic mock: the DP forwards /v1/messages to
    // {api_base}/v1/messages; the mock streams STREAM_EVENTS back.
    upstream = await startOpenAiUpstream({
      streamEvents: STREAM_EVENTS,
      // Small per-event delay so the request takes measurable time
      // (TTFT > 0) and the stream is genuinely incremental.
      eventDelayMs: 2,
    });
    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    // Anthropic-provider model → exercises the byte-passthrough
    // streaming branch (the one #245 fixes), not the translated path.
    const pk = await admin.createProviderKey({
      display_name: "anth-stream-usage-pk",
      secret: "sk-anth-mock",
      api_base: upstream.baseUrl,
    });
    await admin.createModel({
      display_name: "anth-stream-usage",
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["anth-stream-usage"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("streaming /v1/messages records non-zero token metrics (#245)", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    await waitConfigPropagation(async () => {
      try {
        const res = await fetch(`${app!.proxyUrl}/v1/messages`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-api-key": CALLER_PLAINTEXT,
          },
          body: JSON.stringify({
            model: "anth-stream-usage",
            max_tokens: 100,
            stream: true,
            messages: [{ role: "user", content: "probe" }],
          }),
        });
        return res.ok;
      } catch {
        return false;
      }
    });

    const res = await fetch(`${app.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": CALLER_PLAINTEXT,
      },
      body: JSON.stringify({
        model: "anth-stream-usage",
        max_tokens: 200,
        stream: true,
        messages: [{ role: "user", content: "What is the capital of France?" }],
      }),
    });
    expect(res.status).toBe(200);

    // Bytes pass through verbatim — the client still sees the exact
    // Anthropic SSE wire shape (message_start / message_delta / stop).
    const body = await res.text();
    expect(body).toContain("message_start");
    expect(body).toContain("hello there");
    expect(body).toContain("message_stop");

    // The DP emits usage on stream completion asynchronously (Drop
    // guard at end-of-stream). Poll /metrics until the token counters
    // for /v1/messages appear non-zero, with a bounded timeout so a
    // regression (counters stuck at 0) fails rather than hangs.
    const deadline = Date.now() + 5_000;
    let inTokens = 0;
    let outTokens = 0;
    while (Date.now() < deadline) {
      const scrape = await fetch(`${app.metricsUrl}/metrics`).then((r) =>
        r.text(),
      );
      inTokens = sumMetric(scrape, "aisix_llm_input_tokens_total", "/v1/messages");
      outTokens = sumMetric(
        scrape,
        "aisix_llm_output_tokens_total",
        "/v1/messages",
      );
      if (inTokens > 0 && outTokens > 0) break;
      await new Promise((r) => setTimeout(r, 100));
    }

    expect(
      inTokens,
      "input_tokens metric must be non-zero — #245 (pre-fix it was 0)",
    ).toBeGreaterThanOrEqual(INPUT_TOKENS);
    expect(
      outTokens,
      "output_tokens metric must be non-zero — #245 (pre-fix it was 0)",
    ).toBeGreaterThanOrEqual(OUTPUT_TOKENS);
  });
});

/**
 * Sum the values of all `<metric>{...endpoint="<endpoint>"...}` counter
 * lines in a prometheus scrape. Counters accumulate across requests in
 * the same process (the readiness probe + the asserted call), so we sum
 * matching label sets rather than expecting a single line.
 */
function sumMetric(scrape: string, metric: string, endpoint: string): number {
  let total = 0;
  for (const line of scrape.split("\n")) {
    if (!line.startsWith(`${metric}{`)) continue;
    if (!line.includes(`endpoint="${endpoint}"`)) continue;
    const valueStr = line.split("}").at(-1)?.trim() ?? "";
    const v = Number.parseFloat(valueStr);
    if (!Number.isNaN(v)) total += v;
  }
  return total;
}
