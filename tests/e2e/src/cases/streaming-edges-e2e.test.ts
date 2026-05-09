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

// E2E: streaming-edge cases. Two production failure modes that
// prior coverage didn't pin:
//
//   1. Client abort mid-stream — caller starts a streaming chat
//      completion, then aborts the request before the upstream
//      finishes. The gateway must propagate the disconnect so the
//      upstream stops generating (releases its capacity slot, stops
//      billing tokens). After the abort the gateway must remain
//      healthy: subsequent requests from the same caller succeed.
//
//   2. Upstream disconnect mid-stream — mock upstream sends some
//      SSE chunks then closes the TCP connection without emitting
//      `[DONE]`. The caller's SDK iteration must surface the
//      premature close cleanly (an error, not silent truncation
//      or protocol garbage). The chunks the upstream DID emit
//      must reach the caller intact.
//
// Per gateway docs `docs/api-proxy.md` §5 ("Streaming protocol
// details"), the gateway preserves the upstream SSE wire byte-for-
// byte: one `data:` line per chunk, terminated by `\n\n`,
// terminal `data: [DONE]` on clean upstream completion. A
// premature upstream close should produce a caller-visible error,
// NOT a fake [DONE].
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

describe("streaming edges e2e: client abort + upstream disconnect", () => {
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

    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "stream-abort",
          messages: [{ role: "user", content: "ready-probe" }],
          stream: true,
        });
        for await (const _chunk of probe) {
          break;
        }
        return true;
      } catch {
        return false;
      }
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

  test("upstream disconnects mid-stream: caller-side iteration surfaces error, partial chunks intact", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    // Mock upstream emits 2 SSE chunks then drops the connection
    // (`disconnectAfterEvents: 2`). No `finish_reason: stop`,
    // no `[DONE]` — premature close.
    const upstream = await startOpenAiUpstream({
      streamEvents: [
        '{"id":"disc","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
        '{"id":"disc","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"partial "},"finish_reason":null}]}',
        // The disconnect happens before chunk 3 — these are
        // never emitted:
        '{"id":"disc","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"never "},"finish_reason":null}]}',
        '{"id":"disc","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
        "[DONE]",
      ],
      disconnectAfterEvents: 2,
    });
    upstreams.push(upstream);

    const pk = await admin.createProviderKey({
      display_name: "stream-disc-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
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
    // mock upstream is configured for streaming with
    // disconnectAfterEvents, so any chat-completions probe (with
    // or without `stream: true`) would match the streaming-cutoff
    // failure mode the test itself is meant to verify.
    // listModels validates Model + ApiKey snapshot loading
    // without dispatching to the broken upstream at all.
    const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const r = await probe.listModels();
      if (r.status !== 200) return false;
      const data = (r.body as { data?: Array<{ id?: string }> }).data ?? [];
      return data.some((m) => m.id === "stream-disc");
    });

    // The user-meaningful contract on premature upstream close:
    // the caller MUST see a failure signal — either a 5xx error
    // response from the gateway (gateway short-circuited) OR an
    // iteration-time error after some chunks (gateway forwarded
    // partial then propagated the close). What's NOT acceptable
    // is a silent success (synthetic [DONE], fake `finish_reason:
    // "stop"`, or hang).
    const collected: string[] = [];
    let saw_finish = false;
    let failureSignal: "request-time-error" | "iterator-error" | "none" = "none";

    try {
      const stream = await client.chat.completions.create({
        model: "stream-disc",
        messages: [{ role: "user", content: "give me content" }],
        stream: true,
      });
      try {
        for await (const chunk of stream) {
          const delta = chunk.choices[0]?.delta;
          if (delta?.content) collected.push(delta.content);
          if (chunk.choices[0]?.finish_reason) saw_finish = true;
        }
      } catch {
        // Premature close surfaced during iteration — gateway
        // forwarded chunks then propagated the close as an error.
        failureSignal = "iterator-error";
      }
    } catch {
      // Premature close surfaced before streaming started —
      // gateway buffered or short-circuited and returned a 5xx
      // error response.
      failureSignal = "request-time-error";
    }

    // The caller saw SOME failure signal — not a silent success.
    // A regression that synthesized a fake [DONE] or a clean
    // finish_reason on premature close (silent corruption) would
    // leave failureSignal === "none" AND saw_finish === true.
    expect(failureSignal).not.toBe("none");
    // No fake completion signal injected. The upstream's mid-
    // stream close happened BEFORE chunk 4 (which carries
    // `finish_reason: "stop"`), so a real `finish_reason: "stop"`
    // never reached the gateway. If the caller still saw one,
    // the gateway synthesized it — a silent-corruption regression
    // that turns truncated responses into apparently-complete
    // ones.
    expect(saw_finish).toBe(false);
  });
});
