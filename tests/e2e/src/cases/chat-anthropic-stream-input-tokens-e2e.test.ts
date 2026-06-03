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

// E2E: /v1/chat/completions STREAMING against an Anthropic-provider model
// records prompt (input) tokens (#450, finding #1).
//
// This is the bridge `chat_stream` path (OpenAI-compatible chat → Anthropic
// upstream), distinct from the /v1/messages passthrough fixed in #245.
// Pre-fix, `AnthropicStreamStartMessage` dropped `usage.input_tokens` from
// the `message_start` event, so prompt tokens were recorded as 0 for the
// whole stream — silently under-counting TPM/budget/telemetry on every
// Anthropic (and Vertex Claude) streaming chat request.
//
// We drive a real streaming request through the DP binary against a mock
// Anthropic streaming upstream, then scrape /metrics and assert the per-
// request input-token counter is non-zero.

const CALLER = "sk-chat-anth-stream-input";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const INPUT_TOKENS = 41;
const OUTPUT_TOKENS = 58;
const STREAM_EVENTS = [
  JSON.stringify({
    type: "message_start",
    message: {
      id: "msg_chat_450",
      role: "assistant",
      content: [],
      model: "claude-3-5-haiku-20241022",
      stop_reason: null,
      usage: { input_tokens: INPUT_TOKENS, output_tokens: 1 },
    },
  }),
  JSON.stringify({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
  JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "hi there" } }),
  JSON.stringify({ type: "content_block_stop", index: 0 }),
  JSON.stringify({ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: OUTPUT_TOKENS } }),
  JSON.stringify({ type: "message_stop" }),
];

describe("/v1/chat/completions anthropic streaming input tokens (#450)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    upstream = await startOpenAiUpstream({ streamEvents: STREAM_EVENTS, eventDelayMs: 2 });
    app = await spawnApp();
    const admin = new AdminClient(app.adminUrl, app.adminKey);
    const pk = await admin.createProviderKey({
      display_name: "chat-anth-stream-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-anth-mock",
      api_base: upstream.baseUrl,
    });
    await admin.createModel({
      display_name: "chat-anth-stream",
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({ key_hash: CALLER_HASH, allowed_models: ["chat-anth-stream"] });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("records non-zero input_tokens on streaming chat completions (#450)", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    await waitConfigPropagation(async () => {
      try {
        const r = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
          method: "POST",
          headers: { "content-type": "application/json", authorization: `Bearer ${CALLER}` },
          body: JSON.stringify({ model: "chat-anth-stream", stream: true, messages: [{ role: "user", content: "probe" }] }),
        });
        return r.ok;
      } catch {
        return false;
      }
    });

    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${CALLER}` },
      body: JSON.stringify({
        model: "chat-anth-stream",
        stream: true,
        messages: [{ role: "user", content: "What is the capital of France?" }],
      }),
    });
    expect(res.status).toBe(200);
    await res.text();

    const deadline = Date.now() + 5_000;
    let inTok = 0;
    let outTok = 0;
    while (Date.now() < deadline) {
      const scrape = await fetch(`${app.adminUrl}/metrics`).then((r) => r.text());
      inTok = sumMetric(scrape, "aisix_llm_input_tokens_total", "/v1/chat/completions");
      outTok = sumMetric(scrape, "aisix_llm_output_tokens_total", "/v1/chat/completions");
      if (inTok > 0 && outTok > 0) break;
      await new Promise((r) => setTimeout(r, 100));
    }

    expect(
      inTok,
      "input_tokens must reflect message_start usage — #450 (pre-fix it was 0)",
    ).toBeGreaterThanOrEqual(INPUT_TOKENS);
    expect(outTok).toBeGreaterThanOrEqual(OUTPUT_TOKENS);
  });
});

function sumMetric(scrape: string, metric: string, endpoint: string): number {
  let total = 0;
  for (const line of scrape.split("\n")) {
    if (!line.startsWith(`${metric}{`)) continue;
    if (!line.includes(`endpoint="${endpoint}"`)) continue;
    const v = Number.parseFloat(line.split("}").at(-1)?.trim() ?? "");
    if (!Number.isNaN(v)) total += v;
  }
  return total;
}
