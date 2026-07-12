import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  ProxyClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E for #688: streamed /v1/messages and /v1/responses skipped post-stream
// token accounting — their reservation dropped at handler return without ever
// committing the terminal token cost, so the TPM/TPD counters never moved and a
// caller could bypass token-rate limits (and hold more than N concurrent
// streams) simply by streaming. The fix carries the reservation into the
// end-of-stream guard and commits the terminal usage via
// `add_tokens_post_stream`, matching the chat streaming path.
//
// Both endpoints use a TPM cap of 10 against an upstream that reports 16 tokens
// per stream: the first streamed call must succeed (committing 16) and the next
// call must be rejected 429. Pre-fix the counter stayed 0 and the next call
// also succeeded. Covers the two dispatch families — /v1/messages via the
// cross-provider bridge and /v1/responses via the verbatim passthrough.

const TPM = 10;
const PROMPT_TOKENS = 8;
const COMPLETION_TOKENS = 8; // 8 + 8 = 16 > TPM → one stream exhausts it

const MSG_CALLER = "sk-688-msg-caller";
const RESP_CALLER = "sk-688-resp-caller";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

// /v1/messages cross-provider: the request is bridged to the OpenAI-protocol
// upstream, so the mock speaks chat.completion.chunk with a trailing usage
// frame (OpenAI's real include_usage shape).
function chunk(json: Record<string, unknown>): string {
  return JSON.stringify({
    id: "chatcmpl-688",
    object: "chat.completion.chunk",
    created: 0,
    model: "up-688",
    ...json,
  });
}
const MSG_STREAM = [
  chunk({ choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }] }),
  chunk({ choices: [{ index: 0, delta: { content: "hi from 688" }, finish_reason: null }] }),
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

// /v1/responses verbatim passthrough: the upstream speaks the Responses-API SSE
// wire shape, terminal `response.completed` carrying the authoritative usage.
const RESP_STREAM = [
  JSON.stringify({ type: "response.created", response: { id: "resp_688" } }),
  JSON.stringify({ type: "response.output_text.delta", delta: "hi from 688" }),
  JSON.stringify({
    type: "response.completed",
    response: {
      id: "resp_688",
      status: "completed",
      usage: { input_tokens: PROMPT_TOKENS, output_tokens: COMPLETION_TOKENS },
    },
  }),
  "[DONE]",
];

/**
 * Send `make()` repeatedly until it returns 429 or the deadline passes. The
 * post-stream token commit fires when the streamed response body is dropped on
 * the server, a tick after the client finishes reading, so the first follow-up
 * call can race ahead of it — retrying absorbs that tick. Pre-fix the counter
 * never moves, so this exhausts the deadline and returns the last (200) status,
 * failing the caller's `toBe(429)`.
 */
async function nextCallEventually429(make: () => Promise<Response>): Promise<number> {
  const deadline = Date.now() + 5_000;
  let last = 0;
  while (Date.now() < deadline) {
    const res = await make();
    last = res.status;
    await res.text();
    if (last === 429) return 429;
    await new Promise((r) => setTimeout(r, 100));
  }
  return last;
}

describe("streaming TPM commit (#688)", () => {
  let app: SpawnedApp | undefined;
  let upstreamMsg: OpenAiUpstream | undefined;
  let upstreamResp: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstreamMsg = await startOpenAiUpstream({ streamEvents: MSG_STREAM, eventDelayMs: 2 });
    upstreamResp = await startOpenAiUpstream({ streamEvents: RESP_STREAM, eventDelayMs: 2 });
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pkMsg = await seed.createProviderKey({
      display_name: "688-msg-pk",
      secret: "sk-mock",
      api_base: `${upstreamMsg.baseUrl}/v1`,
    });
    const pkResp = await seed.createProviderKey({
      display_name: "688-resp-pk",
      secret: "sk-mock",
      api_base: `${upstreamResp.baseUrl}/v1`,
    });
    // Both models are OpenAI-protocol: /v1/messages bridges to it
    // (cross_provider_dispatch), /v1/responses forwards verbatim
    // (responses_to_target).
    await seed.createModel({
      display_name: "msg-tpm",
      provider: "openai",
      model_name: "up-688",
      provider_key_id: pkMsg.id,
    });
    await seed.createModel({
      display_name: "resp-tpm",
      provider: "openai",
      model_name: "up-688",
      provider_key_id: pkResp.id,
    });
    // Separate caller keys so each endpoint's TPM counter is independent.
    await seed.createApiKey({
      key_hash: hash(MSG_CALLER),
      allowed_models: ["msg-tpm"],
      rate_limit: { tpm: TPM },
    });
    await seed.createApiKey({
      key_hash: hash(RESP_CALLER),
      allowed_models: ["resp-tpm"],
      rate_limit: { tpm: TPM },
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstreamMsg?.close();
    await upstreamResp?.close();
  });

  function postMessages(): Promise<Response> {
    return fetch(`${app!.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": MSG_CALLER,
      },
      body: JSON.stringify({
        model: "msg-tpm",
        max_tokens: 200,
        stream: true,
        messages: [{ role: "user", content: "trip the tpm" }],
      }),
    });
  }

  function postResponses(): Promise<Response> {
    return fetch(`${app!.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${RESP_CALLER}`,
      },
      body: JSON.stringify({ model: "resp-tpm", input: "trip the tpm", stream: true }),
    });
  }

  test("streamed /v1/messages commits tokens post-stream and the next call is 429", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    // listModels doesn't consume the token budget, so it's a safe readiness
    // probe that leaves the TPM quota intact for the measured stream.
    const probe = new ProxyClient(app.proxyUrl, MSG_CALLER);
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      if (res.status !== 200) return false;
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return data.some((m) => m.id === "msg-tpm");
    });

    // First streamed call succeeds and commits its 16 tokens against TPM=10.
    const first = await postMessages();
    expect(first.status).toBe(200);
    expect(await first.text()).toContain("message_stop");

    // The next call is rejected once those streamed tokens are committed.
    // Pre-fix (no post-stream commit) the counter stayed 0 and this stayed 200.
    expect(await nextCallEventually429(postMessages)).toBe(429);
  });

  test("streamed /v1/responses commits tokens post-stream and the next call is 429", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const probe = new ProxyClient(app.proxyUrl, RESP_CALLER);
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      if (res.status !== 200) return false;
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return data.some((m) => m.id === "resp-tpm");
    });

    const first = await postResponses();
    expect(first.status).toBe(200);
    expect(await first.text()).toContain("response.completed");

    expect(await nextCallEventually429(postResponses)).toBe(429);
  });
});
