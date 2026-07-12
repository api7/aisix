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

// E2E: streaming + usage accumulation accuracy.
//
// Per OpenAI's chat completions wire shape — when the caller sets
// `stream_options.include_usage = true`, the upstream emits a final
// SSE chunk with `choices: []` and a `usage` block summarising
// prompt / completion / total tokens
// (https://platform.openai.com/docs/api-reference/chat/streaming).
//
// One contract pinned here:
//
//   - Token usage is preserved end-to-end across the streaming path
//     and equals the non-streaming response's usage for the same
//     prompt against the same upstream.
//
// Why this matters: token counts feed billing. A regression that
// dropped the trailing usage-only chunk, or rewrote any of the
// three numeric fields between upstream and caller, would
// silently undercount or overcount real usage.
//
// Reference: OpenAI streaming API + the OpenAI Node SDK chunk type
// `ChatCompletionChunk` (which exposes `usage` directly).

const CALLER_PLAINTEXT = "sk-stream-usage-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

// Canonical usage values. Both the non-streaming response body and
// the streaming usage-only chunk emit identical numbers — equality
// across the two paths is then the cross-validation.
const PROMPT_TOKENS = 17;
const COMPLETION_TOKENS = 23;
const TOTAL_TOKENS = PROMPT_TOKENS + COMPLETION_TOKENS;

const NON_STREAM_BODY = {
  id: "chatcmpl-non-stream-1",
  object: "chat.completion",
  created: Math.floor(Date.now() / 1000),
  model: "gpt-4o-mini",
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: "hello" },
      finish_reason: "stop",
    },
  ],
  usage: {
    prompt_tokens: PROMPT_TOKENS,
    completion_tokens: COMPLETION_TOKENS,
    total_tokens: TOTAL_TOKENS,
  },
};

const STREAM_EVENTS = [
  '{"id":"chatcmpl-stream-1","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}],"usage":null}',
  '{"id":"chatcmpl-stream-1","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}],"usage":null}',
  '{"id":"chatcmpl-stream-1","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":null}',
  // Terminal usage-only chunk — choices array empty, usage populated.
  // This is the OpenAI shape when stream_options.include_usage=true.
  `{"id":"chatcmpl-stream-1","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":${PROMPT_TOKENS},"completion_tokens":${COMPLETION_TOKENS},"total_tokens":${TOTAL_TOKENS}}}`,
  "[DONE]",
];

describe("streaming usage accumulation e2e: stream final usage == non-stream usage", () => {
  let app: SpawnedApp | undefined;
  let nonStreamUpstream: OpenAiUpstream | undefined;
  let streamUpstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    nonStreamUpstream = await startOpenAiUpstream({
      nonStreamBody: NON_STREAM_BODY,
    });
    streamUpstream = await startOpenAiUpstream({
      streamEvents: STREAM_EVENTS,
    });

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    // Two ProviderKeys → two Models so receivedRequests counts on
    // each mock are unambiguous, and so the streaming and non-
    // streaming calls cannot accidentally cross-pollinate request
    // bodies in the assertion below.
    const pkNon = await seed.createProviderKey({
      display_name: "stream-usage-non-pk",
      secret: "sk-mock",
      api_base: `${nonStreamUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "stream-usage-non",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pkNon.id,
    });
    const pkStream = await seed.createProviderKey({
      display_name: "stream-usage-stream-pk",
      secret: "sk-mock",
      api_base: `${streamUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "stream-usage-stream",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pkStream.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["stream-usage-non", "stream-usage-stream"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await nonStreamUpstream?.close();
    await streamUpstream?.close();
  });

  test("non-stream and stream paths surface identical usage triplets", async (ctx) => {
    if (
      !etcdReachable ||
      !app ||
      !nonStreamUpstream ||
      !streamUpstream
    ) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // Probe each Model on its own dispatcher path so registration
    // races don't show up as flaky usage assertions.
    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "stream-usage-non",
          messages: [{ role: "user", content: "ready-probe-non" }],
        });
        return probe.usage?.total_tokens === TOTAL_TOKENS;
      } catch {
        return false;
      }
    });
    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "stream-usage-stream",
          messages: [{ role: "user", content: "ready-probe-stream" }],
          stream: true,
          stream_options: { include_usage: true },
        });
        for await (const _chunk of probe) {
          break;
        }
        return true;
      } catch {
        return false;
      }
    });

    // Baseline-isolate request counts so the upstream wire-shape
    // assertions below measure only the actual test calls.
    const nonBaseline = nonStreamUpstream.receivedRequests.length;
    const streamBaseline = streamUpstream.receivedRequests.length;

    // Non-streaming reference call.
    const nonStreamResp = await client.chat.completions.create({
      model: "stream-usage-non",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(nonStreamResp.usage?.prompt_tokens).toBe(PROMPT_TOKENS);
    expect(nonStreamResp.usage?.completion_tokens).toBe(
      COMPLETION_TOKENS,
    );
    expect(nonStreamResp.usage?.total_tokens).toBe(TOTAL_TOKENS);

    // Streaming call with include_usage=true. Capture the usage from
    // whichever chunk carries it (OpenAI emits it on a usage-only
    // terminal chunk, but the contract is "exactly one chunk in the
    // stream carries the usage block, and its values are these").
    const stream = await client.chat.completions.create({
      model: "stream-usage-stream",
      messages: [{ role: "user", content: "hi" }],
      stream: true,
      stream_options: { include_usage: true },
    });
    type UsageView = {
      prompt: number;
      completion: number;
      total: number;
    };
    const usageChunks: UsageView[] = [];
    for await (const chunk of stream) {
      if (chunk.usage) {
        usageChunks.push({
          prompt: chunk.usage.prompt_tokens,
          completion: chunk.usage.completion_tokens,
          total: chunk.usage.total_tokens,
        });
      }
    }

    // (1) Exactly one chunk carries the usage block — the terminal
    // usage-only chunk. A regression that emitted usage repeatedly,
    // or never, fails here.
    expect(usageChunks).toHaveLength(1);
    const streamUsage = usageChunks[0]!;

    // (2) Stream usage triplet is byte-equal to the non-stream
    // response's triplet. Both paths run against the same canonical
    // upstream output; if the gateway mutated either side, the
    // numbers would diverge here.
    expect(streamUsage.prompt).toBe(nonStreamResp.usage?.prompt_tokens);
    expect(streamUsage.completion).toBe(
      nonStreamResp.usage?.completion_tokens,
    );
    expect(streamUsage.total).toBe(nonStreamResp.usage?.total_tokens);

    // (3) Stream usage triplet matches the canonical upstream-emitted
    // values. Belt-and-suspenders against (2) — if the non-stream
    // path were also broken in the same way, (2) would still pass.
    expect(streamUsage.prompt).toBe(PROMPT_TOKENS);
    expect(streamUsage.completion).toBe(COMPLETION_TOKENS);
    expect(streamUsage.total).toBe(TOTAL_TOKENS);

    // (4) Upstream wire-shape assertions. The gateway must have
    // forwarded `stream_options.include_usage = true` to the
    // upstream when the caller asked for it; otherwise OpenAI never
    // emits the trailing usage chunk and the test would silently
    // pass against a mock that emits it anyway. Closes the same
    // blind spot CLAUDE.md §7 highlights.
    const sentNon = nonStreamUpstream.receivedRequests
      .slice(nonBaseline)
      .filter((r) => r.path === "/v1/chat/completions");
    expect(sentNon).toHaveLength(1);
    const nonBody = JSON.parse(sentNon[0]!.body);
    expect(nonBody.stream ?? false).toBe(false);

    const sentStream = streamUpstream.receivedRequests
      .slice(streamBaseline)
      .filter((r) => r.path === "/v1/chat/completions");
    expect(sentStream).toHaveLength(1);
    const streamBody = JSON.parse(sentStream[0]!.body);
    expect(streamBody.stream).toBe(true);
    expect(streamBody.stream_options?.include_usage).toBe(true);
  }, 60_000);
});
