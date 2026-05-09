import { createHash } from "node:crypto";
import OpenAI from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  ProxyClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: streaming-edge case. One production failure mode pinned
// here, derived from `docs/api-proxy.md` §5:
//
//   - Client abort mid-stream — caller starts a streaming chat
//     completion, then aborts the request via AbortSignal. After
//     the abort the gateway must remain HEALTHY: subsequent
//     streaming calls from the same caller run to completion.
//     This pins the "no resource leak / no parser corruption"
//     property that's externally observable from the client.
//
// (The "upstream disconnect mid-stream" case — partial chunks
// reach the caller, then iterator surfaces error per docs §5
// "aisix sends a final error chunk and closes without [DONE]" —
// is held back. The gateway today short-circuits to a request-
// time 502 instead of forwarding partial chunks. See follow-up
// issue.)
//
// Note on scope: this case verifies gateway LIVENESS post-abort,
// not upstream-side disconnect propagation. The harness has no
// signal for "upstream observed the client disconnect", so a
// regression where the gateway holds the upstream connection open
// (silently consuming chunks after the client aborted) would
// pass this test. Filing that as a separate harness-extension
// task; today's coverage is "gateway stays alive", which is the
// load-bearing user-visible contract.
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
  let admin: AdminClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  test("client aborts mid-stream: gateway stays healthy, subsequent request succeeds", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
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

    const pk = await admin.createProviderKey({
      display_name: "stream-abort-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
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

});
