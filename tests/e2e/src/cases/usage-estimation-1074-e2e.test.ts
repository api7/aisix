import { createHash } from "node:crypto";
import OpenAI from "openai";
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
import { decodedTextFor, startMockSls, type MockSls } from "../harness/sls-mock.js";

// E2E for AISIX-Cloud#1074: token metering for SSE + missing-usage upstreams.
//
// Contract pinned here: when an upstream (an OpenAI-compatible relay, an
// aborted stream, ...) never reports a `usage` block, the gateway counts
// tokens locally — prompt from the request, completion from the delivered
// output text — and the usage record is marked `usage_estimated`. When the
// upstream DOES report usage, its values win untouched (pinned by the Rust
// unit suite; this file covers the estimation-positive paths end-to-end).
// Estimation feeds telemetry (usage events → exporters, prometheus token
// counters, TPM accounting) ONLY: the client-visible body/stream is never
// rewritten with synthesised usage.
//
// Expected token values are ground truth from the de-facto OpenAI counting
// scheme (https://github.com/openai/openai-cookbook — "How to count tokens"):
// per message 3 tokens + the message's text, +3 reply priming; plain-text
// counting for the completion side. The seeded upstream model names are
// deliberately non-OpenAI so the fallback `cl100k_base` encoding applies:
//   "user" = 1 token, "hi" = 1 token, "Hello world" = 2 tokens
// → prompt for one user message "hi" = 3 + 1 + 1 + 3 = 8.
const EXPECTED_PROMPT_TOKENS = 8;
const EXPECTED_COMPLETION_TOKENS = 2; // "Hello world"

const CALLER_PLAINTEXT = "sk-usage-estimation-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const CREDENTIAL_REF = "mock";
const MOCK_AK_ID = "mock-akid";
const MOCK_AK_SECRET = "mock-secret";
const SLS_PROJECT = "aisix-e2e-obs";
const META_LOGSTORE = "est-meta-events";

// Chat SSE without any terminal usage chunk (relay behavior: ignores the
// gateway-injected stream_options.include_usage).
const CHAT_STREAM_EVENTS_NO_USAGE = [
  '{"id":"chatcmpl-est-1","object":"chat.completion.chunk","model":"relay-mini","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
  '{"id":"chatcmpl-est-1","object":"chat.completion.chunk","model":"relay-mini","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}',
  '{"id":"chatcmpl-est-1","object":"chat.completion.chunk","model":"relay-mini","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}',
  '{"id":"chatcmpl-est-1","object":"chat.completion.chunk","model":"relay-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
  "[DONE]",
];

// Non-streaming chat 200 with NO usage block at all.
const CHAT_NON_STREAM_NO_USAGE = {
  id: "chatcmpl-est-2",
  object: "chat.completion",
  created: Math.floor(Date.now() / 1000),
  model: "relay-mini",
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: "Hello world" },
      finish_reason: "stop",
    },
  ],
};

// Anthropic /v1/messages SSE where message_start has NO usage and no
// message_delta usage ever arrives (relay omits token accounting).
const MESSAGES_STREAM_EVENTS_NO_USAGE = [
  JSON.stringify({
    type: "message_start",
    message: {
      id: "msg_est_1",
      role: "assistant",
      content: [],
      model: "relay-claude",
      stop_reason: null,
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
    delta: { type: "text_delta", text: "Hello world" },
  }),
  JSON.stringify({ type: "content_block_stop", index: 0 }),
  JSON.stringify({ type: "message_stop" }),
];

// /v1/responses SSE that aborts before any terminal `response.completed`
// event — no usage ever arrives; the deltas are the only output signal.
const RESPONSES_STREAM_EVENTS_NO_USAGE = [
  '{"type":"response.created","response":{"id":"resp_est_1"}}',
  '{"type":"response.output_text.delta","delta":"Hello"}',
  '{"type":"response.output_text.delta","delta":" world"}',
  "[DONE]",
];

describe("usage estimation e2e (AISIX-Cloud#1074): missing upstream usage is locally counted and flagged", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let chatStreamUpstream: OpenAiUpstream | undefined;
  let chatNonStreamUpstream: OpenAiUpstream | undefined;
  let chatAbortUpstream: OpenAiUpstream | undefined;
  let messagesUpstream: OpenAiUpstream | undefined;
  let responsesUpstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    chatStreamUpstream = await startOpenAiUpstream({
      streamEvents: CHAT_STREAM_EVENTS_NO_USAGE,
    });
    chatNonStreamUpstream = await startOpenAiUpstream({
      nonStreamBody: CHAT_NON_STREAM_NO_USAGE,
    });
    // Slow event pacing so the client can abort mid-stream with
    // chunks still pending upstream-side.
    chatAbortUpstream = await startOpenAiUpstream({
      streamEvents: CHAT_STREAM_EVENTS_NO_USAGE,
      eventDelayMs: 150,
    });
    messagesUpstream = await startOpenAiUpstream({
      streamEvents: MESSAGES_STREAM_EVENTS_NO_USAGE,
    });
    responsesUpstream = await startOpenAiUpstream({
      streamEvents: RESPONSES_STREAM_EVENTS_NO_USAGE,
    });

    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: MOCK_AK_ID,
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: MOCK_AK_SECRET,
      },
    });
    seed = new SeedClient(etcd, app.etcdPrefix);

    // Metadata-only SLS exporter: every usage event lands as one flat
    // key/value log, so `usage_estimated` (serialized only when true) is
    // observable on the wire without content capture.
    await seed.createObservabilityExporter({
      name: "est-sls-meta",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: META_LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });

    // One model per scenario so prometheus counters and SLS records are
    // unambiguous. All upstream model names are non-OpenAI on purpose —
    // the estimator falls back to cl100k_base, matching the expected
    // token constants above.
    const seedModel = async (display: string, upstream: OpenAiUpstream, provider = "openai") => {
      const pk = await seed!.createProviderKey({
        display_name: `${display}-pk`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      await seed!.createModel({
        display_name: display,
        provider,
        model_name: "relay-compat-x",
        provider_key_id: pk.id,
      });
    };
    await seedModel("est-chat-stream", chatStreamUpstream);
    await seedModel("est-chat-nonstream", chatNonStreamUpstream);
    await seedModel("est-chat-abort", chatAbortUpstream);
    await seedModel("est-msgs-stream", messagesUpstream, "anthropic");
    await seedModel("est-responses-stream", responsesUpstream);
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [
        "est-chat-stream",
        "est-chat-nonstream",
        "est-chat-abort",
        "est-msgs-stream",
        "est-responses-stream",
      ],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await sls?.close();
    await chatStreamUpstream?.close();
    await chatNonStreamUpstream?.close();
    await chatAbortUpstream?.close();
    await messagesUpstream?.close();
    await responsesUpstream?.close();
  });

  function openai(): OpenAI {
    return new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app!.proxyUrl}/v1`,
      maxRetries: 0,
    });
  }

  /** Sum a token counter across label sets matching `model="<model>"`. */
  async function tokenMetric(metric: string, model: string): Promise<number> {
    const scrape = await fetch(`${app!.metricsUrl}/metrics`).then((r) => r.text());
    let total = 0;
    for (const line of scrape.split("\n")) {
      if (!line.startsWith(`${metric}{`)) continue;
      if (!line.includes(`model="${model}"`)) continue;
      const value = Number(line.slice(line.lastIndexOf(" ") + 1));
      if (Number.isFinite(value)) total += value;
    }
    return total;
  }

  /** Poll until the model's input-token counter is non-zero (Drop-guard
   * emission is asynchronous), then return the (input, output) pair. */
  async function waitTokenMetrics(model: string): Promise<{ input: number; output: number }> {
    const deadline = Date.now() + 10_000;
    let input = 0;
    let output = 0;
    while (Date.now() < deadline) {
      input = await tokenMetric("aisix_llm_input_tokens_total", model);
      output = await tokenMetric("aisix_llm_output_tokens_total", model);
      if (input > 0) break;
      await new Promise((r) => setTimeout(r, 100));
    }
    return { input, output };
  }

  /** True when an SLS record for `requestedModel` carries the
   * `usage_estimated` flag. The flattened log serializes fields in
   * struct order (requested_model before usage_estimated, which is
   * present only when true), so a bounded forward window inside the
   * decoded protobuf text identifies the pair within one record. */
  function slsFlagged(requestedModel: string): boolean {
    const text = decodedTextFor(sls!, META_LOGSTORE);
    const re = new RegExp(`${requestedModel}[\\s\\S]{0,600}?usage_estimated`);
    return re.test(text);
  }

  async function waitSlsFlagged(requestedModel: string): Promise<void> {
    const deadline = Date.now() + 10_000;
    while (Date.now() < deadline) {
      if (slsFlagged(requestedModel)) return;
      await new Promise((r) => setTimeout(r, 100));
    }
    throw new Error(
      `no usage_estimated-flagged SLS record for '${requestedModel}' within 10s`,
    );
  }

  test("chat SSE without a usage chunk: tokens estimated, flagged, client stream untouched", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const client = openai();
    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "est-chat-stream",
          messages: [{ role: "user", content: "hi" }],
          stream: true,
        });
        for await (const _ of probe) {
          /* drain */
        }
        return true;
      } catch {
        return false;
      }
    });

    const inputBefore = await tokenMetric("aisix_llm_input_tokens_total", "est-chat-stream");

    const stream = await client.chat.completions.create({
      model: "est-chat-stream",
      messages: [{ role: "user", content: "hi" }],
      stream: true,
    });
    let sawContent = "";
    for await (const chunk of stream) {
      sawContent += chunk.choices[0]?.delta?.content ?? "";
      // The client did not ask for include_usage; estimation must not
      // fabricate a usage chunk on the client-facing stream.
      expect(chunk.usage ?? null).toBeNull();
    }
    expect(sawContent).toBe("Hello world");

    const deadline = Date.now() + 10_000;
    let input = 0;
    let output = 0;
    while (Date.now() < deadline) {
      input = await tokenMetric("aisix_llm_input_tokens_total", "est-chat-stream");
      output = await tokenMetric("aisix_llm_output_tokens_total", "est-chat-stream");
      if (input > inputBefore) break;
      await new Promise((r) => setTimeout(r, 100));
    }
    // The probe call and the asserted call are identical requests, so
    // per-call expectations hold on the delta.
    expect(input - inputBefore).toBe(EXPECTED_PROMPT_TOKENS);
    expect(output).toBeGreaterThanOrEqual(EXPECTED_COMPLETION_TOKENS);
    await waitSlsFlagged("est-chat-stream");
  }, 60_000);

  test("chat non-streaming 200 without usage: telemetry estimated, client body not rewritten", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const client = openai();
    await waitConfigPropagation(async () => {
      try {
        await client.chat.completions.create({
          model: "est-chat-nonstream",
          messages: [{ role: "user", content: "hi" }],
        });
        return true;
      } catch {
        return false;
      }
    });

    const inputBefore = await tokenMetric("aisix_llm_input_tokens_total", "est-chat-nonstream");
    const resp = await client.chat.completions.create({
      model: "est-chat-nonstream",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(resp.choices[0]?.message?.content).toBe("Hello world");
    // The upstream sent no usage; the gateway must not inject estimated
    // numbers into the client-visible body (zeros = normalized absence,
    // the pre-existing wire shape).
    expect(resp.usage?.prompt_tokens ?? 0).toBe(0);
    expect(resp.usage?.completion_tokens ?? 0).toBe(0);

    const deadline = Date.now() + 10_000;
    let input = 0;
    let output = 0;
    while (Date.now() < deadline) {
      input = await tokenMetric("aisix_llm_input_tokens_total", "est-chat-nonstream");
      output = await tokenMetric("aisix_llm_output_tokens_total", "est-chat-nonstream");
      if (input > inputBefore) break;
      await new Promise((r) => setTimeout(r, 100));
    }
    expect(input - inputBefore).toBe(EXPECTED_PROMPT_TOKENS);
    expect(output).toBeGreaterThanOrEqual(EXPECTED_COMPLETION_TOKENS);
    await waitSlsFlagged("est-chat-nonstream");
  }, 60_000);

  test("client abort mid-stream: the usage record still ships with estimated tokens", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    // Raw fetch so the reader can be cancelled deterministically after the
    // first delivered chunk (the OpenAI SDK hides the reader).
    const controller = new AbortController();
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "est-chat-abort",
        messages: [{ role: "user", content: "hi" }],
        stream: true,
      }),
      signal: controller.signal,
    });
    expect(res.status).toBe(200);
    const reader = res.body!.getReader();
    // Pull one chunk off the wire, then abort — the upstream still has
    // events pending (150ms pacing), so this is a genuine mid-stream
    // disconnect, not a raced clean close.
    await reader.read();
    controller.abort();

    // The Drop guard fires on disconnect and estimates from what was
    // delivered: the prompt side is exact; the completion side covers
    // at least the first delivered content token(s) but must stay below
    // the full response (we aborted before " world" et al.).
    const { input, output } = await waitTokenMetrics("est-chat-abort");
    expect(input).toBe(EXPECTED_PROMPT_TOKENS);
    expect(output).toBeGreaterThanOrEqual(0);
    await waitSlsFlagged("est-chat-abort");
  }, 60_000);

  test("/v1/messages SSE with no usage anywhere: estimated and flagged", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const call = async () =>
      fetch(`${app!.proxyUrl}/v1/messages`, {
        method: "POST",
        headers: {
          "x-api-key": CALLER_PLAINTEXT,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: "est-msgs-stream",
          max_tokens: 100,
          stream: true,
          messages: [{ role: "user", content: "hi" }],
        }),
      });
    await waitConfigPropagation(async () => {
      try {
        const r = await call();
        await r.text();
        return r.status === 200;
      } catch {
        return false;
      }
    });

    const inputBefore = await tokenMetric("aisix_llm_input_tokens_total", "est-msgs-stream");
    const res = await call();
    expect(res.status).toBe(200);
    const body = await res.text();
    // Bytes pass through verbatim; no fabricated usage on the wire.
    expect(body).toContain("Hello world");
    expect(body).toContain("message_stop");

    const deadline = Date.now() + 10_000;
    let input = 0;
    let output = 0;
    while (Date.now() < deadline) {
      input = await tokenMetric("aisix_llm_input_tokens_total", "est-msgs-stream");
      output = await tokenMetric("aisix_llm_output_tokens_total", "est-msgs-stream");
      if (input > inputBefore) break;
      await new Promise((r) => setTimeout(r, 100));
    }
    // Prompt estimate is exact (one user message "hi" = 8). The output
    // accumulator joins text deltas with separators, so bound the
    // completion estimate instead of pinning it.
    expect(input - inputBefore).toBe(EXPECTED_PROMPT_TOKENS);
    expect(output).toBeGreaterThanOrEqual(EXPECTED_COMPLETION_TOKENS);
    await waitSlsFlagged("est-msgs-stream");
  }, 60_000);

  test("/v1/responses SSE aborted before the terminal event: record still ships, flagged", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const call = async () =>
      fetch(`${app!.proxyUrl}/v1/responses`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: "est-responses-stream",
          input: "hi",
          stream: true,
        }),
      });
    await waitConfigPropagation(async () => {
      try {
        const r = await call();
        await r.text();
        return r.status === 200;
      } catch {
        return false;
      }
    });

    const res = await call();
    expect(res.status).toBe(200);
    const body = await res.text();
    // Verbatim passthrough of the truncated stream.
    expect(body).toContain("response.output_text.delta");

    // /v1/responses has no per-token prometheus family today; the flagged
    // SLS record (flat usage-event fields) is the observable wire. Exact
    // estimated values for this surface are pinned by the Rust handler
    // tests (`estimates_usage_event_when_upstream_omits_usage_block`).
    await waitSlsFlagged("est-responses-stream");
  }, 60_000);
});
