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

// E2E regression for AISIX-Cloud#790: Anthropic /v1/messages STREAMING
// through an OpenAI-protocol provider recorded prompt_tokens=0 /
// completion_tokens=0 — the translated upstream request carried
// `stream: true` without `stream_options: {include_usage: true}`, so
// OpenAI-protocol upstreams never attached usage to the stream.
//
// Post-fix contracts exercised here against a real `aisix` binary:
//  1. The outbound (translated) request carries the injected
//     stream_options.
//  2. The terminal usage-only frame (OpenAI's real include_usage shape:
//     empty `choices` + `usage`, AFTER the stop chunk) feeds the token
//     metrics — pre-fix these stayed 0 because the SSE pump stopped at
//     the stop chunk.
//  3. The client-visible Anthropic SSE carries the real counts in the
//     closing `message_delta.usage` (pre-fix: output_tokens 0).
//  4. An upstream that IGNORES stream_options still produces a
//     well-formed Anthropic close (message_delta + message_stop) at
//     stream end.

const CALLER_PLAINTEXT = "sk-issue-790-regression";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const PROMPT_TOKENS = 17;
const COMPLETION_TOKENS = 23;

function chunk(json: Record<string, unknown>): string {
  return JSON.stringify({
    id: "chatcmpl-issue790",
    object: "chat.completion.chunk",
    created: 1765000000,
    model: "mog-6",
    ...json,
  });
}

// OpenAI's real include_usage stream shape: content → stop chunk
// (no usage) → usage-only frame with empty choices → [DONE].
const STREAM_WITH_TRAILING_USAGE = [
  chunk({ choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }] }),
  chunk({ choices: [{ index: 0, delta: { content: "hello from mog-6" }, finish_reason: null }] }),
  chunk({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }),
  chunk({
    choices: [],
    usage: {
      prompt_tokens: PROMPT_TOKENS,
      completion_tokens: COMPLETION_TOKENS,
      total_tokens: PROMPT_TOKENS + COMPLETION_TOKENS,
    },
  }),
  "[DONE]",
];

// An upstream that ignores stream_options: stop chunk, no usage, ever.
const STREAM_NO_USAGE = [
  chunk({ choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }] }),
  chunk({ choices: [{ index: 0, delta: { content: "hello from mog-6" }, finish_reason: null }] }),
  chunk({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }),
  "[DONE]",
];

async function postMessages(proxyUrl: string, model: string): Promise<Response> {
  return fetch(`${proxyUrl}/v1/messages`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": CALLER_PLAINTEXT,
      "user-agent": "claude-cli/2.1.118 (external, cli)",
    },
    body: JSON.stringify({
      model,
      max_tokens: 200,
      stream: true,
      messages: [{ role: "user", content: "issue 790 regression" }],
    }),
  });
}

/** Sum a token counter across label sets matching the given model label. */
function sumMetricByModel(scrape: string, metric: string, model: string): number {
  let total = 0;
  for (const line of scrape.split("\n")) {
    if (!line.startsWith(`${metric}{`)) continue;
    if (!line.includes(`model="${model}"`)) continue;
    const v = Number(line.slice(line.lastIndexOf(" ") + 1));
    if (Number.isFinite(v)) total += v;
  }
  return total;
}

describe("anthropic→openai streaming usage (#790)", () => {
  let app: SpawnedApp | undefined;
  let upstreamUsage: OpenAiUpstream | undefined;
  let upstreamNoUsage: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    upstreamUsage = await startOpenAiUpstream({
      streamEvents: STREAM_WITH_TRAILING_USAGE,
      eventDelayMs: 2,
    });
    upstreamNoUsage = await startOpenAiUpstream({
      streamEvents: STREAM_NO_USAGE,
      eventDelayMs: 2,
    });
    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pkUsage = await admin.createProviderKey({
      display_name: "issue790-pk",
      secret: "sk-openai-mock",
      api_base: `${upstreamUsage.baseUrl}/v1`,
    });
    const pkNoUsage = await admin.createProviderKey({
      display_name: "issue790-pk-nousage",
      secret: "sk-openai-mock",
      api_base: `${upstreamNoUsage.baseUrl}/v1`,
    });
    // Mirrors the customer config from #790: alias gpt-5.5 → upstream
    // model mog-6 on an OpenAI-protocol provider.
    await admin.createModel({
      display_name: "gpt-5.5",
      provider: "openai",
      model_name: "mog-6",
      provider_key_id: pkUsage.id,
    });
    await admin.createModel({
      display_name: "gpt-5.5-nousage",
      provider: "openai",
      model_name: "mog-6",
      provider_key_id: pkNoUsage.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["gpt-5.5", "gpt-5.5-nousage"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstreamUsage?.close();
    await upstreamNoUsage?.close();
  });

  test("injects stream_options, records tokens, surfaces usage in message_delta", async (ctx) => {
    if (!etcdReachable || !app || !upstreamUsage) {
      ctx.skip();
      return;
    }

    await waitConfigPropagation(async () => {
      try {
        return (await postMessages(app!.proxyUrl, "gpt-5.5")).ok;
      } catch {
        return false;
      }
    });

    const res = await postMessages(app.proxyUrl, "gpt-5.5");
    expect(res.status).toBe(200);
    const body = await res.text();

    // Anthropic SSE wire shape intact, content delivered.
    expect(body).toContain("message_start");
    expect(body).toContain("hello from mog-6");
    expect(body).toContain("message_stop");

    // (1) The translated upstream request asked for the usage frame.
    const lastReq = upstreamUsage.receivedRequests.at(-1);
    expect(lastReq).toBeDefined();
    const outbound = JSON.parse(lastReq!.body) as Record<string, unknown>;
    expect(outbound.model).toBe("mog-6");
    expect(outbound.stream).toBe(true);
    expect(outbound.stream_options).toEqual({ include_usage: true });

    // (3) The closing message_delta carries the real counts — the
    // pre-fix encoder closed at the stop chunk with output_tokens 0.
    const messageDelta = body
      .split("\n")
      .filter((l) => l.startsWith("data: "))
      .map((l) => JSON.parse(l.slice("data: ".length)))
      .find((d) => d.type === "message_delta");
    expect(messageDelta?.usage?.output_tokens).toBe(COMPLETION_TOKENS);
    expect(messageDelta?.usage?.input_tokens).toBe(PROMPT_TOKENS);

    // (2) Token metrics record the usage (pre-fix: stuck at 0). The
    // readiness probes above also hit gpt-5.5 so use >= not ==.
    const deadline = Date.now() + 5_000;
    let inTok = 0;
    let outTok = 0;
    while (Date.now() < deadline) {
      const scrape = await fetch(`${app.adminUrl}/metrics`).then((r) => r.text());
      inTok = sumMetricByModel(scrape, "aisix_llm_input_tokens_total", "gpt-5.5");
      outTok = sumMetricByModel(scrape, "aisix_llm_output_tokens_total", "gpt-5.5");
      if (inTok >= PROMPT_TOKENS && outTok >= COMPLETION_TOKENS) break;
      await new Promise((r) => setTimeout(r, 100));
    }
    expect(inTok).toBeGreaterThanOrEqual(PROMPT_TOKENS);
    expect(outTok).toBeGreaterThanOrEqual(COMPLETION_TOKENS);
  });

  test("upstream ignoring stream_options still gets a well-formed close", async (ctx) => {
    if (!etcdReachable || !app || !upstreamNoUsage) {
      ctx.skip();
      return;
    }

    const res = await postMessages(app.proxyUrl, "gpt-5.5-nousage");
    expect(res.status).toBe(200);
    const body = await res.text();

    // The closing pair is withheld waiting for a usage frame that never
    // comes — stream end must flush it (force_finish), keeping the
    // Anthropic wire shape valid for the client.
    expect(body).toContain("hello from mog-6");
    const events = body
      .split("\n")
      .filter((l) => l.startsWith("data: "))
      .map((l) => JSON.parse(l.slice("data: ".length)));
    const messageDelta = events.find((d) => d.type === "message_delta");
    expect(messageDelta?.delta?.stop_reason).toBe("end_turn");
    expect(events.some((d) => d.type === "message_stop")).toBe(true);
  });
});
