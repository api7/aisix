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

// E2E: streaming responses are NEVER cached, regardless of policy.
//
// Per `docs/api-proxy.md` §3 (PR #191) the `x-aisix-cache` response
// header is "Absent for streaming responses." Per §4.2, only
// non-streaming requests participate in the cache layer. This is
// the published product contract — there is no terminal value to
// store on a stream, and replaying a partial event sequence would
// break SSE consumers downstream.
//
// Closes #151 C4.3 — cache-policy-e2e and cache-scenarios-e2e both
// only cover the non-streaming path; the "streaming bypass" half
// of the contract was never pinned.
//
// Two contracts pinned in one test:
//
//   1. Two identical streaming requests both dispatch upstream
//      (count grows by 2), AND neither response carries an
//      `x-aisix-cache` header — bypass is silent, not a `miss` or
//      `disabled` label.
//
//   2. Sanity: with the SAME CachePolicy in scope, a non-streaming
//      request still hits the cache normally (header=miss on
//      first, hit on second). Confirms the policy is loaded and
//      caching is on — the streaming bypass is not "no policy at
//      all".
//
// Reference: docs/api-proxy.md §3 (header rules) and §4.2
// (fingerprint contents + non-streaming-only).

const CALLER_PLAINTEXT = "sk-stream-nocache-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const STREAM_PROMPT = "stream-nocache-prompt";
const NON_STREAM_PROMPT = "non-stream-prompt";

// Minimal SSE event sequence — three role/content/done frames plus
// the terminal `[DONE]` sentinel. Mirrors the shape used in
// cross-provider-matrix-e2e streaming branch.
const SSE_EVENTS = [
  '{"id":"c1","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
  '{"id":"c1","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}',
  '{"id":"c1","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
  "[DONE]",
];

describe("cache streaming bypass e2e: streaming responses are never cached", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    // Mock dispatches by config: when streamEvents is configured,
    // it always returns SSE. So requests on the stream Model see
    // SSE; the non-stream Model uses a separate mock with only
    // nonStreamBody. Two upstreams, two ProviderKeys, two Models
    // — keeps the receivedRequests counts unambiguous per path.
    const streamMock = await startOpenAiUpstream({
      streamEvents: SSE_EVENTS,
    });
    const nonStreamMock = await startOpenAiUpstream();
    upstream = streamMock; // primary handle for the streaming side
    // Stash the second mock so afterAll closes it.
    (upstream as unknown as { _second?: OpenAiUpstream })._second =
      nonStreamMock;

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pkStream = await admin.createProviderKey({
      display_name: "stream-nocache-stream-pk",
      secret: "sk-mock",
      api_base: `${streamMock.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "stream-nocache-stream-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pkStream.id,
    });

    const pkNon = await admin.createProviderKey({
      display_name: "stream-nocache-non-pk",
      secret: "sk-mock",
      api_base: `${nonStreamMock.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "stream-nocache-non-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pkNon.id,
    });

    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [
        "stream-nocache-stream-model",
        "stream-nocache-non-model",
      ],
    });
    // Single enabled policy applies to all — both Models are in
    // its scope. The streaming Model should bypass it anyway; the
    // non-streaming Model uses it normally.
    await admin.json("POST", "/admin/v1/cache_policies", {
      name: "stream-nocache-policy",
      enabled: true,
      applies_to: "all",
    });
  });

  afterAll(async () => {
    await app?.exit();
    const second = (upstream as unknown as {
      _second?: OpenAiUpstream;
    })?._second;
    await upstream?.close();
    await second?.close();
  });

  test(
    "streaming: no x-aisix-cache header, upstream re-hit on every call",
    async (ctx) => {
      if (!etcdReachable || !app || !upstream) {
        ctx.skip();
        return;
      }
      const second = (upstream as unknown as {
        _second?: OpenAiUpstream;
      })._second!;

      const reqHeaders = {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      };

      // Readiness — non-streaming probe through the non-streaming
      // mock confirms the CachePolicy + Models + ApiKey are loaded
      // (waits for x-aisix-cache: miss like other cache-edge tests).
      await waitConfigPropagation(async () => {
        try {
          const r = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
            method: "POST",
            headers: reqHeaders,
            body: JSON.stringify({
              model: "stream-nocache-non-model",
              messages: [{ role: "user", content: "ready-probe" }],
            }),
          });
          await r.text();
          return (
            r.status === 200 &&
            r.headers.get("x-aisix-cache") === "miss"
          );
        } catch {
          return false;
        }
      });

      // (1) STREAMING side: same fingerprint, two calls — both
      // must dispatch to upstream and neither response carries
      // x-aisix-cache (header-absent per docs §3).
      const streamBaseline = upstream.receivedRequests.length;
      const streamBody = JSON.stringify({
        model: "stream-nocache-stream-model",
        messages: [{ role: "user", content: STREAM_PROMPT }],
        stream: true,
      });
      const s1 = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: reqHeaders,
        body: streamBody,
      });
      expect(s1.status).toBe(200);
      expect(s1.headers.get("content-type")).toContain(
        "text/event-stream",
      );
      // The contract under test: no cache header on streams,
      // regardless of whether the policy was applied or not.
      expect(s1.headers.get("x-aisix-cache")).toBeNull();
      // Drain the body so the connection finalises cleanly before
      // the second call.
      await s1.text();

      const s2 = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: reqHeaders,
        body: streamBody,
      });
      expect(s2.status).toBe(200);
      expect(s2.headers.get("content-type")).toContain(
        "text/event-stream",
      );
      expect(s2.headers.get("x-aisix-cache")).toBeNull();
      await s2.text();

      // Both calls reached upstream — exactly two new requests on
      // the streaming mock since the baseline. A regression that
      // somehow cached and replayed a stream from a prior call
      // (or coalesced concurrent streams) would land at 1, not 2.
      const newStreamHits =
        upstream.receivedRequests.length - streamBaseline;
      expect(newStreamHits).toBe(2);

      // (2) NON-STREAMING side: same CachePolicy is in scope.
      // First call misses, second hits. Sanity: the policy IS
      // active in this test fixture; the streaming bypass above
      // is not "no policy at all".
      const nonBaseline = second.receivedRequests.length;
      const nonBody = JSON.stringify({
        model: "stream-nocache-non-model",
        messages: [{ role: "user", content: NON_STREAM_PROMPT }],
      });
      const n1 = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: reqHeaders,
        body: nonBody,
      });
      expect(n1.status).toBe(200);
      expect(n1.headers.get("x-aisix-cache")).toBe("miss");
      await n1.text();
      expect(second.receivedRequests.length).toBe(nonBaseline + 1);

      const n2 = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: reqHeaders,
        body: nonBody,
      });
      expect(n2.status).toBe(200);
      expect(n2.headers.get("x-aisix-cache")).toBe("hit");
      await n2.text();
      // Cache hit means upstream NOT re-hit.
      expect(second.receivedRequests.length).toBe(nonBaseline + 1);
    },
    60_000,
  );
});
