import { createHash } from "node:crypto";
import OpenAI from "openai";
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

// E2E: streaming-edge cases derived from `docs/api-proxy.md` §5:
//
//   - Client abort mid-stream — caller starts a streaming chat
//     completion, then aborts the request via AbortSignal. After
//     the abort the gateway must remain HEALTHY: subsequent
//     streaming calls from the same caller run to completion.
//     Pins the "no resource leak / no parser corruption" property
//     that's externally observable from the client.
//
//   - Upstream disconnects mid-stream — mock upstream emits N
//     SSE chunks then closes the TCP connection without `[DONE]`.
//     Per docs §5 ("If the upstream stream terminates abnormally,
//     aisix sends a final error chunk and closes the response
//     without `[DONE]`"), the caller MUST see partial chunks +
//     iterator-time error + no synthetic `finish_reason:"stop"`.
//
// Note on scope (client-abort case): verifies gateway LIVENESS
// post-abort, not upstream-side disconnect propagation. The
// harness has no signal for "upstream observed the client
// disconnect", so a regression where the gateway holds the
// upstream connection open (silently consuming chunks after the
// client aborted) would pass this test. Filing that as a separate
// harness-extension task; today's coverage is "gateway stays
// alive", which is the load-bearing user-visible contract.
//
// References:
// - Gateway's own streaming contract: `docs/api-proxy.md` §5
// - OpenAI Chat Completions streaming spec
//   <https://platform.openai.com/docs/api-reference/chat/streaming>
// - OpenAI Node SDK stream-cancel pattern via AbortSignal
//   <https://github.com/openai/openai-node#canceling-a-request>

const CALLER_PLAINTEXT = "sk-stream-edge-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("streaming edges e2e: client abort mid-stream", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  test("client aborts mid-stream: gateway stays healthy, subsequent request succeeds", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    // Mock upstream emits 5 SSE chunks with 200ms between each →
    // the full response takes >1s. The caller will abort after
    // receiving the first chunk (well before the upstream finishes).
    const upstream = await startOpenAiUpstream({
      streamEvents: [
        '{"id":"abrt","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
        '{"id":"abrt","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"chunk-1 "},"finish_reason":null}]}',
        '{"id":"abrt","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"chunk-2 "},"finish_reason":null}]}',
        '{"id":"abrt","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"chunk-3 "},"finish_reason":null}]}',
        '{"id":"abrt","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
        "[DONE]",
      ],
      eventDelayMs: 200,
    });
    upstreams.push(upstream);

    const pk = await seed.createProviderKey({
      display_name: "stream-abort-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "stream-abort",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // Use ProxyClient.listModels for snapshot-readiness gating
    // (matches concurrency-e2e / streaming-disconnect convention).
    // A streaming chat probe would burn ≥200ms per attempt due to
    // eventDelayMs; listModels is faster and consistent with how
    // other tests gate readiness against a streaming-only mock.
    const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const r = await probe.listModels();
      if (r.status !== 200) return false;
      const data = (r.body as { data?: Array<{ id?: string }> }).data ?? [];
      return data.some((m) => m.id === "stream-abort");
    });

    // Start streaming, abort after the first chunk via
    // AbortSignal. The OpenAI Node SDK forwards the abort to
    // the underlying fetch; the gateway sees the client
    // connection close.
    const controller = new AbortController();
    const stream = await client.chat.completions.create(
      {
        model: "stream-abort",
        messages: [{ role: "user", content: "abort me early" }],
        stream: true,
      },
      { signal: controller.signal },
    );

    // Iterate until we've received the first chunk, then abort
    // and break. Whether the iterator subsequently throws or
    // silently ends is SDK-internal and not part of the
    // gateway's externally-observable contract — what IS
    // observable is "the gateway is still healthy after the
    // abort", which we verify with a follow-up request below.
    let firstChunkSeen = false;
    try {
      for await (const _chunk of stream) {
        firstChunkSeen = true;
        controller.abort();
        break;
      }
    } catch {
      // Iterator may throw on abort or may not, depending on
      // SDK timing. Both are acceptable — the load-bearing
      // assertion is the followup call below.
    }
    expect(firstChunkSeen).toBe(true);

    // Load-bearing: gateway must remain healthy after the
    // mid-stream abort. A regression that left a dangling
    // upstream connection, leaked a per-caller resource, or
    // corrupted shared parser state would surface as the next
    // call hanging or 5xx-ing. Use a streaming followup since
    // the mock upstream is configured for streaming responses.
    const followupStream = await client.chat.completions.create({
      model: "stream-abort",
      messages: [{ role: "user", content: "still alive?" }],
      stream: true,
    });
    let followupChunkCount = 0;
    let followupFinishReason: string | null | undefined;
    for await (const chunk of followupStream) {
      followupChunkCount++;
      followupFinishReason ??= chunk.choices[0]?.finish_reason ?? undefined;
    }
    // The followup ran to completion (saw all chunks AND the
    // finish_reason), proving the gateway is fully functional
    // post-abort.
    expect(followupChunkCount).toBeGreaterThan(0);
    expect(followupFinishReason).toBe("stop");
  });

  test("upstream disconnects mid-stream: partial chunks reach caller, iterator throws, no [DONE]", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    // Mock upstream emits 2 SSE chunks then drops the connection
    // (`disconnectAfterEvents: 2`). No `finish_reason: "stop"`,
    // no `[DONE]` from the upstream — premature close per docs §5.
    //
    // `eventDelayMs: 200` between writes ensures both chunks flush
    // to the gateway BEFORE `res.destroy()` aborts the connection.
    // Without the delay, write buffering on the mock side can
    // cause the destroy to land before the second chunk reaches
    // the wire, making the test flaky on which chunks the gateway
    // actually receives. Matches the peer client-abort case for
    // consistency.
    const upstream = await startOpenAiUpstream({
      streamEvents: [
        '{"id":"disc","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
        '{"id":"disc","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"partial "},"finish_reason":null}]}',
        // Never emitted (disconnect happens at i >= 2):
        '{"id":"disc","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"never "},"finish_reason":null}]}',
        '{"id":"disc","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
        "[DONE]",
      ],
      eventDelayMs: 200,
      disconnectAfterEvents: 2,
    });
    upstreams.push(upstream);

    const pk = await seed.createProviderKey({
      display_name: "stream-disc-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "stream-disc",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // Use ProxyClient.listModels for snapshot-readiness gating —
    // chat-completions probes don't work cleanly here because the
    // mock upstream is configured for streaming with disconnects,
    // so any chat-completions probe (with or without `stream:true`)
    // would match the streaming-cutoff failure mode the test is
    // meant to verify.
    const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const r = await probe.listModels();
      if (r.status !== 200) return false;
      const data = (r.body as { data?: Array<{ id?: string }> }).data ?? [];
      return data.some((m) => m.id === "stream-disc");
    });

    // Per docs §5 strict contract:
    //   - 2 partial chunks reach the caller (the bytes the upstream
    //     emitted before the close)
    //   - Iterator THROWS during streaming (gateway emits SSE
    //     `event: error` chunk, OpenAI SDK surfaces it as iteration
    //     error)
    //   - NO synthetic `finish_reason: "stop"` injection (would
    //     turn truncated responses into apparently-complete ones)
    //   - NO synthetic `[DONE]` after error (would make the SDK
    //     treat the truncated stream as a clean completion)
    const collected: string[] = [];
    let sawFinish = false;
    let surfacedError = false;
    let errCtor = "";
    let errMessage = "";

    const stream = await client.chat.completions.create({
      model: "stream-disc",
      messages: [{ role: "user", content: "give me content" }],
      stream: true,
    });
    try {
      for await (const chunk of stream) {
        const delta = chunk.choices[0]?.delta;
        if (delta?.content) collected.push(delta.content);
        if (chunk.choices[0]?.finish_reason) sawFinish = true;
      }
    } catch (e) {
      surfacedError = true;
      errCtor =
        (e as { constructor?: { name?: string } })?.constructor?.name ?? "";
      errMessage = (e as { message?: string })?.message ?? "";
    }

    // Partial chunks reached the caller. Chunk 2 carried the
    // string "partial "; chunk 1 was role-only with no content.
    expect(collected.join("")).toBe("partial ");
    // No fake completion signal. The upstream's mid-stream close
    // happened BEFORE the chunk that carries finish_reason:"stop".
    expect(sawFinish).toBe(false);
    // Iterator surfaced an error. The OpenAI Node SDK throws on
    // an SSE `event: error` chunk during stream iteration.
    expect(surfacedError).toBe(true);

    // Tighten: distinguish typed `APIError` from `SyntaxError`.
    // Without this, a regression that re-introduces a plain-string
    // `data:` payload on the SSE error frame would still pass
    // `surfacedError === true` because the SDK's `JSON.parse(sse.data)`
    // (in `streaming.ts`, called BEFORE the `sse.event === "error"`
    // check) would throw `SyntaxError("Could not parse message into
    // JSON: ...")` instead of the typed `APIError` callers expect.
    // See OpenAI Node SDK <https://github.com/openai/openai-node/blob/main/src/streaming.ts>
    // for the parse-then-classify ordering.
    expect(errCtor).not.toBe("SyntaxError");
    expect(errMessage).not.toContain("Could not parse message into JSON");
  });
});
