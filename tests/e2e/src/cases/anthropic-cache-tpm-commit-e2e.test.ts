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

// E2E for AISIX-Cloud#995: the native Anthropic paths committed only
// prompt + completion tokens against TPM/TPD, silently excluding the
// separate cache_creation_input_tokens / cache_read_input_tokens counters.
// The OpenAI bridge already folds cache tokens into total_tokens (#679) and
// the CP display total includes them (#906), so a caller hitting
// /v1/messages natively (or /v1/responses bridged to Anthropic) was
// rate-limited on a smaller total than the one displayed — and than what the
// upstream actually billed.
//
// Each scenario uses a TPM cap the first call only exhausts when cache
// tokens are counted; the follow-up call must then 429. Pre-fix the
// cache-heavy usage kept the counter under (or at exactly) the cap and the
// follow-up stayed 200.

const TPM = 10;

const hash = (s: string) => createHash("sha256").update(s).digest("hex");

// Non-streaming usage: base total (2+2=4) stays under TPM=10, with-cache
// total (2+2+5+3=12) exceeds it. The non-streaming commit happens before the
// response is written, so a single immediate follow-up call is
// deterministic — no retry loop needed.
const NONSTREAM_USAGE = {
  input_tokens: 2,
  output_tokens: 2,
  cache_creation_input_tokens: 5,
  cache_read_input_tokens: 3,
};

function anthropicMessageBody(usage: Record<string, number>) {
  return {
    id: "msg_995",
    type: "message",
    role: "assistant",
    content: [{ type: "text", text: "hello from cache" }],
    model: "claude-3-5-haiku-20241022",
    stop_reason: "end_turn",
    usage,
  };
}

// Streaming usage: ONLY cache tokens are non-zero. The streamed commit fires
// asynchronously after the response body is dropped, so the follow-up must
// retry until 429 — and a retry loop only discriminates the fix if the
// pre-fix commit is exactly 0 per call (base tokens would accumulate across
// retries and eventually trip the cap even without the fix). A zero
// input_tokens with a large cache_read is the fully-cached-prompt shape
// (input_tokens excludes cached input on the Anthropic wire).
const STREAM_EVENTS = [
  JSON.stringify({
    type: "message_start",
    message: {
      id: "msg_995s",
      model: "claude-3-5-haiku-20241022",
      usage: {
        input_tokens: 0,
        output_tokens: 0,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 16,
      },
    },
  }),
  JSON.stringify({
    type: "content_block_delta",
    index: 0,
    delta: { type: "text_delta", text: "hello from cache" },
  }),
  JSON.stringify({
    type: "message_delta",
    delta: { stop_reason: "end_turn" },
    usage: { output_tokens: 0 },
  }),
  JSON.stringify({ type: "message_stop" }),
];

const MSG_CALLER = "sk-995-msg-caller";
const MSG_STREAM_CALLER = "sk-995-msg-stream-caller";
const RESP_CALLER = "sk-995-resp-caller";

/** Retry `make()` until 429 or deadline; returns the last status. */
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

describe("anthropic cache tokens count toward TPM (AISIX-Cloud#995)", () => {
  let app: SpawnedApp | undefined;
  let upstreamMsg: OpenAiUpstream | undefined;
  let upstreamMsgStream: OpenAiUpstream | undefined;
  let upstreamResp: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstreamMsg = await startOpenAiUpstream({
      nonStreamBody: anthropicMessageBody(NONSTREAM_USAGE),
    });
    upstreamMsgStream = await startOpenAiUpstream({
      streamEvents: STREAM_EVENTS,
      eventDelayMs: 2,
    });
    upstreamResp = await startOpenAiUpstream({
      nonStreamBody: anthropicMessageBody(NONSTREAM_USAGE),
    });
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    // The Anthropic adapter appends `/v1/messages` to the api_base, so the
    // provider key points at the bare mock host.
    const mkAnthropicModel = async (
      name: string,
      upstream: OpenAiUpstream,
      caller: string,
    ) => {
      const pk = await seed!.createProviderKey({
        display_name: `${name}-pk`,
        provider: "anthropic",
        adapter: "anthropic",
        secret: "sk-ant-mock",
        api_base: upstream.baseUrl,
      });
      await seed!.createModel({
        display_name: name,
        provider: "anthropic",
        model_name: "claude-3-5-haiku-20241022",
        provider_key_id: pk.id,
      });
      // Separate caller keys so each scenario's TPM counter is independent.
      await seed!.createApiKey({
        key_hash: hash(caller),
        allowed_models: [name],
        rate_limit: { tpm: TPM },
      });
    };
    await mkAnthropicModel("msg-cache-tpm", upstreamMsg!, MSG_CALLER);
    await mkAnthropicModel("msg-cache-tpm-stream", upstreamMsgStream!, MSG_STREAM_CALLER);
    await mkAnthropicModel("resp-cache-tpm", upstreamResp!, RESP_CALLER);
  });

  afterAll(async () => {
    await app?.exit();
    await upstreamMsg?.close();
    await upstreamMsgStream?.close();
    await upstreamResp?.close();
  });

  async function waitModelVisible(caller: string, model: string) {
    // listModels doesn't consume the token budget, so it's a safe readiness
    // probe that leaves the TPM quota intact for the measured call.
    const probe = new ProxyClient(app!.proxyUrl, caller);
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      if (res.status !== 200) return false;
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return data.some((m) => m.id === model);
    });
  }

  function postMessages(caller: string, model: string, stream: boolean): Promise<Response> {
    return fetch(`${app!.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: { "content-type": "application/json", "x-api-key": caller },
      body: JSON.stringify({
        model,
        max_tokens: 200,
        stream,
        messages: [{ role: "user", content: "count my cache tokens" }],
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
      body: JSON.stringify({ model: "resp-cache-tpm", input: "count my cache tokens" }),
    });
  }

  test("non-streaming /v1/messages: cache tokens exhaust TPM and the next call is 429", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    await waitModelVisible(MSG_CALLER, "msg-cache-tpm");

    // First call succeeds and commits 2+2+5+3 = 12 tokens against TPM=10.
    const first = await postMessages(MSG_CALLER, "msg-cache-tpm", false);
    expect(first.status).toBe(200);
    const body = (await first.json()) as { usage?: Record<string, number> };
    expect(body.usage?.cache_creation_input_tokens).toBe(5);

    // The non-streaming commit is synchronous, so the very next call must be
    // rejected. Pre-fix only 4 tokens were committed and this stayed 200.
    const second = await postMessages(MSG_CALLER, "msg-cache-tpm", false);
    expect(second.status).toBe(429);
    await second.text();
  });

  test("streaming /v1/messages: cache tokens commit post-stream and the next call is 429", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    await waitModelVisible(MSG_STREAM_CALLER, "msg-cache-tpm-stream");

    // First streamed call succeeds; its terminal usage is 16 cache-read
    // tokens (prompt/completion both 0) against TPM=10.
    const first = await postMessages(MSG_STREAM_CALLER, "msg-cache-tpm-stream", true);
    expect(first.status).toBe(200);
    expect(await first.text()).toContain("message_stop");

    // Pre-fix the post-stream commit was 0+0 and the counter never moved, so
    // this exhausted the deadline still 200.
    expect(
      await nextCallEventually429(() =>
        postMessages(MSG_STREAM_CALLER, "msg-cache-tpm-stream", true),
      ),
    ).toBe(429);
  });

  test("/v1/responses bridged to Anthropic: cache tokens exhaust TPM and the next call is 429", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    await waitModelVisible(RESP_CALLER, "resp-cache-tpm");

    const first = await postResponses();
    expect(first.status).toBe(200);
    await first.text();

    const second = await postResponses();
    expect(second.status).toBe(429);
    await second.text();
  });
});
