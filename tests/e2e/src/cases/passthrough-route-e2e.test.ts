import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { harnessRequest } from "../harness/http.js";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";
import { startMockOtlp, type MockOtlp } from "../harness/otlp-mock.js";

// E2E: explicit PassthroughRoute resources — the successor of the removed
// implicit `/passthrough/{provider}/*rest` tunnel. A route binds a gateway
// entry (path prefix and/or inbound Host) to ONE upstream target with its
// own auth mode and credential handling, so there is no implicit
// provider→Model credential borrowing (AISIX-Cloud#1127) and the caller's
// own upstream credential can be forwarded verbatim (AISIX-Cloud#1312).
//
// Journeys pinned here:
//
//   1. Migration shape: an inject-mode route claiming the old
//      `/passthrough/openai` prefix serves the old URL unchanged —
//      including the #164 double-/v1 dedup and Bearer injection.
//   2. Anthropic inject shape: `x-api-key` + `anthropic-version`, never
//      a Bearer alongside (#166).
//   3. An unclaimed path in the namespace is an ordinary 404 miss
//      for any unclaimed `/passthrough/*` path.
//   4. Forward-proxy BYO: a host-matched route with
//      `credential_mode: forward_client` + `auth_mode: header_key`
//      forwards the caller's own Authorization verbatim and strips the
//      gateway's side-channel key header — even when the path collides
//      with a typed gateway route (`/v1/chat/completions`).
//   5. `auth_mode: anonymous` binds traffic to the configured principal,
//      gated by `source_cidrs` (real TCP, so 127.0.0.1 resolves).
//   6. SSE relay: a streaming upstream is forwarded as SSE with the
//      frames intact.
//   7. Envelope auto-detection: a route has NO protocol/streaming
//      configuration — usage extraction follows the request body's own
//      envelope (chat buffered, Responses streamed) and a non-LLM
//      exchange is never probed for phantom usage.

const CALLER_PLAINTEXT = "sk-ptr-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");
const STREAM_LIMITED_PLAINTEXT = "sk-ptr-e2e-stream-limit";
const STREAM_LIMITED_KEY_HASH = createHash("sha256")
  .update(STREAM_LIMITED_PLAINTEXT)
  .digest("hex");
const STREAM_TIMEOUT_PLAINTEXT = "sk-ptr-e2e-stream-timeout";
const STREAM_TIMEOUT_KEY_HASH = createHash("sha256")
  .update(STREAM_TIMEOUT_PLAINTEXT)
  .digest("hex");

describe("passthrough-route e2e: explicit routes, BYO credentials, unclaimed paths", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];
  const otlps: MockOtlp[] = [];

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
      allowed_routes: ["*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
    await Promise.all(otlps.map((o) => o.close()));
  });

  test("inject route on the legacy prefix: /v1 dedup, verbatim body, Bearer injection", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    const upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "file-ptr-openai-01",
        object: "file",
        purpose: "batch",
      },
    });
    upstreams.push(upstream);

    const pk = await seed.createProviderKey({
      display_name: "ptr-openai-pk",
      secret: "sk-mock",
      api_base: "http://unused-on-routes",
    });
    // The migration shape from the removed implicit tunnel: the route
    // claims the old `/passthrough/openai` prefix, the target carries
    // the `/v1` suffix like the old api_base docs example — callers
    // keep their old URLs byte-for-byte.
    await seed.createPassthroughRoute({
      name: "ptr-openai-tunnel",
      path_prefix: "/passthrough/openai",
      target_url: `${upstream.baseUrl}/v1`,
      provider_key_id: pk.id,
    });

    const headers = {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    };

    await waitConfigPropagation(async () => {
      try {
        const r = await fetch(`${app!.proxyUrl}/passthrough/openai/v1/files`, {
          method: "POST",
          headers,
          body: JSON.stringify({ purpose: "batch" }),
        });
        if (r.status !== 200) {
          await r.text();
          return false;
        }
        const j = (await r.json()) as { object?: unknown };
        return j.object === "file";
      } catch {
        return false;
      }
    });

    const baseline = upstream.receivedRequests.length;
    const res = await fetch(`${app.proxyUrl}/passthrough/openai/v1/files?limit=3`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        purpose: "batch",
        arbitrary_unknown_field: "must-pass-through-untouched",
      }),
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as { id?: unknown };
    expect(body.id).toBe("file-ptr-openai-01");

    const calls = upstream.receivedRequests.slice(baseline);
    // #164 dedup carried over: `/v1` target tail + `v1/...` remainder →
    // one `/v1`, and the query string survives.
    expect(calls.filter((r) => r.path.startsWith("/v1/v1/"))).toHaveLength(0);
    const hit = calls.find((r) => r.path === "/v1/files?limit=3");
    expect(hit).toBeDefined();
    // Inject mode: the ProviderKey secret rides upstream; the caller's
    // gateway key does not.
    expect(hit?.headers["authorization"]).toBe("Bearer sk-mock");
    expect(JSON.parse(hit!.body) as Record<string, unknown>).toMatchObject({
      arbitrary_unknown_field: "must-pass-through-untouched",
    });
  });

  test("anthropic inject shape: x-api-key + anthropic-version, no Bearer", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    const upstream = await startOpenAiUpstream({
      nonStreamBody: { id: "msgbatch-1", type: "message_batch" },
    });
    upstreams.push(upstream);

    const pk = await seed.createProviderKey({
      display_name: "ptr-anthropic-pk",
      secret: "sk-ant-mock",
      api_base: "http://unused-on-routes",
      provider: "anthropic",
      adapter: "anthropic",
    });
    await seed.createPassthroughRoute({
      name: "ptr-anthropic-tunnel",
      path_prefix: "/passthrough/anthropic",
      target_url: upstream.baseUrl,
      provider_key_id: pk.id,
    });

    const headers = {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    };
    await waitConfigPropagation(async () => {
      try {
        const r = await fetch(
          `${app!.proxyUrl}/passthrough/anthropic/v1/messages/batches`,
          { method: "POST", headers, body: "{}" },
        );
        if (r.status !== 200) {
          await r.text();
          return false;
        }
        return true;
      } catch {
        return false;
      }
    });

    const hit = upstream.receivedRequests.at(-1)!;
    expect(hit.headers["x-api-key"]).toBe("sk-ant-mock");
    expect(hit.headers["anthropic-version"]).toBe("2023-06-01");
    expect(hit.headers["authorization"]).toBeUndefined();
  });

  test("unclaimed /passthrough/* takes the plain 404 miss path", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const res = await fetch(
      `${app.proxyUrl}/passthrough/some-unclaimed-provider/v1/models`,
      {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      },
    );
    expect(res.status).toBe(404);
    expect(await res.text()).toBe("");
  });

  test("route boundary traversal and query conflicts never reach the configured upstream", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    const upstream = await startOpenAiUpstream({
      nonStreamBody: { object: "safe-route-target" },
    });
    upstreams.push(upstream);

    const pk = await seed.createProviderKey({
      display_name: "ptr-boundary-pk",
      secret: "sk-mock",
      api_base: "http://unused-on-routes",
    });
    await seed.createPassthroughRoute({
      name: "ptr-boundary",
      path_prefix: "/ptr-boundary",
      target_url: `${upstream.baseUrl}/provider/v1?tenant=operator`,
      provider_key_id: pk.id,
    });
    await seed.createPassthroughRoute({
      name: "ptr-form-key-boundary",
      path_prefix: "/ptr-form-key-boundary",
      target_url: `${upstream.baseUrl}/provider/v1?tenant_id=operator`,
      provider_key_id: pk.id,
    });

    const headers = { authorization: `Bearer ${CALLER_PLAINTEXT}` };
    await waitConfigPropagation(async () => {
      try {
        const ready = await fetch(`${app!.proxyUrl}/ptr-form-key-boundary/models`, {
          headers,
        });
        await ready.text();
        return ready.status === 200;
      } catch {
        return false;
      }
    });

    expect(upstream.receivedRequests.at(-1)?.path).toBe(
      "/provider/v1/models?tenant_id=operator",
    );
    const allowedQuery = await harnessRequest(
      `${app.proxyUrl}/ptr-boundary/models?limit=3`,
      { headers },
    );
    expect(allowedQuery.statusCode).toBe(200);
    await allowedQuery.body.text();
    expect(upstream.receivedRequests.at(-1)?.path).toBe(
      "/provider/v1/models?tenant=operator&limit=3",
    );

    const baseline = upstream.receivedRequests.length;
    // Use undici's raw request helper: fetch implementations are allowed to
    // normalize URL escapes before the gateway receives the wire path.
    for (const remainder of [
      "%252e%252e%252fmodels",
      "..;ignored/models",
      "%2e%2e%3bignored/models",
      "%252e%252e%253bignored/models",
      "%2e%2e%3bignored/%2e%2e%3bignored/admin",
    ]) {
      const rejected = await harnessRequest(
        `${app.proxyUrl}/ptr-boundary/${remainder}`,
        { headers },
      );
      expect(rejected.statusCode, remainder).toBe(400);
      await rejected.body.text();
      expect(upstream.receivedRequests, remainder).toHaveLength(baseline);
    }

    const conflictingQuery = await harnessRequest(
      `${app.proxyUrl}/ptr-boundary/models?tenant=caller`,
      { headers },
    );
    expect(conflictingQuery.statusCode).toBe(400);
    await conflictingQuery.body.text();
    expect(upstream.receivedRequests).toHaveLength(baseline);

    const nestedConflictingQuery = await harnessRequest(
      `${app.proxyUrl}/ptr-boundary/models?%2574enant=caller`,
      { headers },
    );
    expect(nestedConflictingQuery.statusCode).toBe(400);
    await nestedConflictingQuery.body.text();
    expect(upstream.receivedRequests).toHaveLength(baseline);

    const encodedQueryDelimiter = await harnessRequest(
      `${app.proxyUrl}/ptr-boundary/models?safe=1%26tenant%3Dcaller`,
      { headers },
    );
    expect(encodedQueryDelimiter.statusCode).toBe(400);
    await encodedQueryDelimiter.body.text();
    expect(upstream.receivedRequests).toHaveLength(baseline);

    const nestedEncodedQueryDelimiter = await harnessRequest(
      `${app.proxyUrl}/ptr-boundary/models?safe=1%2526tenant%253Dcaller`,
      { headers },
    );
    expect(nestedEncodedQueryDelimiter.statusCode).toBe(400);
    await nestedEncodedQueryDelimiter.body.text();
    expect(upstream.receivedRequests).toHaveLength(baseline);

    for (const query of [
      "+tenant=caller",
      "%20tenant=caller",
      "%2520tenant=caller",
      "tenant%00suffix=caller",
      "tenant%2500suffix=caller",
    ]) {
      const phpFormKeyConflict = await harnessRequest(
        `${app.proxyUrl}/ptr-boundary/models?${query}`,
        { headers },
      );
      expect(phpFormKeyConflict.statusCode, query).toBe(400);
      await phpFormKeyConflict.body.text();
      expect(upstream.receivedRequests, query).toHaveLength(baseline);
    }

    for (const query of [
      "safe=1;tenant=caller",
      "safe=1%3Btenant%3Dcaller",
      "safe=1%253Btenant%253Dcaller",
    ]) {
      const semicolonQueryDelimiter = await harnessRequest(
        `${app.proxyUrl}/ptr-boundary/models?${query}`,
        { headers },
      );
      expect(semicolonQueryDelimiter.statusCode, query).toBe(400);
      await semicolonQueryDelimiter.body.text();
      expect(upstream.receivedRequests, query).toHaveLength(baseline);
    }

    for (const query of [
      "tenant.id=caller",
      "tenant%2Eid=caller",
      "tenant%252Eid=caller",
      "tenant+id=caller",
      "tenant%20id=caller",
    ]) {
      const normalizedKeyConflict = await harnessRequest(
        `${app.proxyUrl}/ptr-form-key-boundary/models?${query}`,
        { headers },
      );
      expect(normalizedKeyConflict.statusCode, query).toBe(400);
      await normalizedKeyConflict.body.text();
      expect(upstream.receivedRequests, query).toHaveLength(baseline);
    }

    const bracketedKeyConflict = await harnessRequest(
      `${app.proxyUrl}/ptr-boundary/models?tenant%5Brole%5D=caller`,
      { headers },
    );
    expect(bracketedKeyConflict.statusCode).toBe(400);
    await bracketedKeyConflict.body.text();
    expect(upstream.receivedRequests).toHaveLength(baseline);
  });

  test("forward-proxy BYO: host match beats typed routes; Authorization forwarded verbatim", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    const upstream = await startOpenAiUpstream({
      nonStreamBody: { routed: "byo-upstream" },
    });
    upstreams.push(upstream);

    await seed.createPassthroughRoute({
      name: "ptr-byo-host",
      hosts: ["ai-upstream.example.com"],
      target_url: upstream.baseUrl,
      auth_mode: "header_key",
      auth_header_name: "x-aisix-api-key",
      credential_mode: "forward_client",
      identity_header: "x-aisix-user",
    });

    // The colliding path is the whole point: with the foreign Host the
    // request must reach the route, not the typed chat handler.
    // `fetch` (undici) treats Host as a forbidden header and silently
    // drops it, so the raw undici request helper carries it instead —
    // exactly what a TLS-terminating device on the wire would send.
    const call = async () =>
      harnessRequest(`${app!.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          host: "ai-upstream.example.com",
          authorization: "Bearer employee-official-token",
          "x-aisix-api-key": CALLER_PLAINTEXT,
          "x-aisix-user": "employee-42",
          "content-type": "application/json",
        },
        body: JSON.stringify({ model: "gpt-4o", messages: [] }),
      });

    await waitConfigPropagation(async () => {
      try {
        const r = await call();
        if (r.statusCode !== 200) {
          await r.body.text();
          return false;
        }
        const j = (await r.body.json()) as { routed?: unknown };
        return j.routed === "byo-upstream";
      } catch {
        return false;
      }
    });

    const baseline = upstream.receivedRequests.length;
    const res = await call();
    expect(res.statusCode).toBe(200);
    await res.body.text();
    const hit = upstream.receivedRequests.slice(baseline).at(-1)!;
    // BYO: the employee's own credential reached the upstream verbatim…
    expect(hit.headers["authorization"]).toBe("Bearer employee-official-token");
    // …and the gateway's side-channel headers did not.
    expect(hit.headers["x-aisix-api-key"]).toBeUndefined();
    expect(hit.headers["x-aisix-user"]).toBeUndefined();

    // Missing gateway key: the 401 must point at the route's configured
    // header, not `Authorization` — on this route Authorization carries the
    // caller's own upstream credential and is exactly the wrong thing to fix.
    const noKey = await harnessRequest(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        host: "ai-upstream.example.com",
        authorization: "Bearer employee-official-token",
        "content-type": "application/json",
      },
      body: JSON.stringify({ model: "gpt-4o", messages: [] }),
    });
    expect(noKey.statusCode).toBe(401);
    const noKeyBody = (await noKey.body.json()) as {
      error?: { message?: string };
    };
    expect(noKeyBody.error?.message).toContain("x-aisix-api-key");
    expect(noKeyBody.error?.message).not.toContain("Authorization");
  });

  test("a prefixed host route claims a reserved namespace and mirrors the path", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    // The forward-proxy shape an agent CLI needs: one upstream host serving
    // several protocols under different prefixes (GitHub Copilot's CLI talks
    // to its MCP server at /mcp/readonly on the very host it uses for chat).
    //
    // What this pins end-to-end: a host-matched route MAY claim `/mcp`,
    // which the typed routes own for host-less traffic — host dispatch runs
    // ahead of them — and such a route still mounts normally, stripping its
    // prefix onto the target base.
    //
    // The mirroring half of the pair (a `preserve_host` route forwards the
    // COMPLETE path) cannot run here: `preserve_host` dials
    // `https://<inbound host>`, and this mock listens on 127.0.0.1 with no
    // way to own a public name. It is covered by the `match_route` unit test
    // (`longest_prefix_and_host_specificity_win`) and by the live
    // Copilot-CLI verification behind AISIX-Cloud#1312.
    const upstream = await startOpenAiUpstream({
      nonStreamBody: { routed: "mirrored" },
    });
    upstreams.push(upstream);

    await seed.createPassthroughRoute({
      name: "ptr-mirror-mcp",
      hosts: ["agent-upstream.example.com"],
      path_prefix: "/mcp",
      target_url: upstream.baseUrl,
      auth_mode: "header_key",
      auth_header_name: "x-aisix-api-key",
      credential_mode: "forward_client",
    });

    const call = async () =>
      harnessRequest(`${app!.proxyUrl}/mcp/readonly`, {
        method: "POST",
        headers: {
          host: "agent-upstream.example.com",
          "x-aisix-api-key": CALLER_PLAINTEXT,
          authorization: "Bearer caller-own-upstream-token",
          "content-type": "application/json",
        },
        body: JSON.stringify({ jsonrpc: "2.0", method: "initialize", id: 1 }),
      });

    await waitConfigPropagation(async () => {
      try {
        const r = await call();
        const ok = r.statusCode === 200;
        await r.body.text();
        return ok;
      } catch {
        return false;
      }
    });

    const baseline = upstream.receivedRequests.length;
    const res = await call();
    expect(res.statusCode).toBe(200);
    expect(await res.body.json()).toMatchObject({ routed: "mirrored" });

    const hit = upstream.receivedRequests.slice(baseline).at(-1)!;
    // A target_url route mounts at its prefix, so the upstream sees the
    // remainder. (A preserve_host route on the same prefix would relay
    // "/mcp/readonly" whole — see the note above.)
    expect(hit.path).toBe("/readonly");
    // BYO credential relayed verbatim; the gateway key never leaves.
    expect(hit.headers["authorization"]).toBe("Bearer caller-own-upstream-token");
    expect(hit.headers["x-aisix-api-key"]).toBeUndefined();
  });

  test("anonymous route binds the configured principal behind source_cidrs", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    const upstream = await startOpenAiUpstream({
      nonStreamBody: { ok: true },
    });
    upstreams.push(upstream);

    const anonKey = await seed.createApiKey({
      key_hash: createHash("sha256").update("sk-ptr-anon-principal").digest("hex"),
      allowed_models: [],
      allowed_routes: ["ptr-anon"],
    });
    await seed.createPassthroughRoute({
      name: "ptr-anon",
      path_prefix: "/anon-tunnel",
      target_url: upstream.baseUrl,
      auth_mode: "anonymous",
      anonymous_key_id: anonKey.id,
      source_cidrs: ["127.0.0.0/8", "::1/128"],
      credential_mode: "forward_client",
    });

    // No gateway credential at all — the route's bound principal carries
    // the request.
    await waitConfigPropagation(async () => {
      try {
        const r = await fetch(`${app!.proxyUrl}/anon-tunnel/health`, {});
        if (r.status !== 200) {
          await r.text();
          return false;
        }
        return true;
      } catch {
        return false;
      }
    });

    const res = await fetch(`${app.proxyUrl}/anon-tunnel/health`);
    expect(res.status).toBe(200);
    expect((await res.json()) as Record<string, unknown>).toMatchObject({ ok: true });
  });

  test("SSE upstream is relayed as SSE with frames intact", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    const streamEvents = [
      JSON.stringify({ choices: [{ delta: { content: "hel" } }] }),
      JSON.stringify({ choices: [{ delta: { content: "lo" } }] }),
      JSON.stringify({
        choices: [],
        usage: { prompt_tokens: 5, completion_tokens: 2 },
      }),
      "[DONE]",
    ];
    const upstream = await startOpenAiUpstream({
      // The propagation probe below reaches the same passthrough route but
      // receives this complete unary response, leaving the first stream for
      // the journey asserted by the test.
      scriptedResponses: [
        { nonStreamBody: { ready: true } },
        { streamEvents },
      ],
    });
    upstreams.push(upstream);

    const pk = await seed.createProviderKey({
      display_name: "ptr-sse-pk",
      secret: "sk-mock",
      api_base: "http://unused-on-routes",
    });
    await seed.createPassthroughRoute({
      name: "ptr-sse",
      path_prefix: "/sse-tunnel",
      target_url: upstream.baseUrl,
      provider_key_id: pk.id,
    });

    const headers = {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    };
    const call = async () =>
      fetch(`${app!.proxyUrl}/sse-tunnel/chat/completions`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          model: "gpt-4o",
          messages: [{ role: "user", content: "hi" }],
          stream: true,
        }),
      });

    await waitConfigPropagation(async () => {
      try {
        const probe = await call();
        const ready =
          probe.status === 200 &&
          !(probe.headers.get("content-type") ?? "").includes("text/event-stream");
        await probe.text();
        return ready;
      } catch {
        return false;
      }
    });

    const res = await call();
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type") ?? "").toContain("text/event-stream");
    const text = await res.text();
    // Frames arrive intact and in order; the relay adds nothing.
    expect(text).toContain('"content":"hel"');
    expect(text).toContain('"content":"lo"');
    expect(text).toContain("[DONE]");
  });

  test("an SSE passthrough holds its concurrency slot until the body ends or cancels", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    // Headers arrive immediately but the body remains open long enough to
    // make the second request observe the in-flight stream, rather than a
    // short unary exchange that happened to have already finished.
    const streamEvents = [
      JSON.stringify({ choices: [{ delta: { content: "one" } }] }),
      JSON.stringify({ choices: [{ delta: { content: "two" } }] }),
      "[DONE]",
    ];
    const upstream = await startOpenAiUpstream({
      scriptedResponses: [
        // A complete probe establishes propagation on this route without
        // retaining a streaming concurrency slot. The first real stream
        // then stalls until the client cancels it; the second ends naturally.
        { nonStreamBody: { ready: true } },
        { streamEvents, firstEventDelayMs: 10_000 },
        { streamEvents, firstEventDelayMs: 25, eventDelayMs: 25 },
        { streamEvents },
      ],
    });
    upstreams.push(upstream);

    const pk = await seed.createProviderKey({
      display_name: "ptr-sse-concurrency-pk",
      secret: "sk-mock",
      api_base: "http://unused-on-routes",
    });
    await seed.createPassthroughRoute({
      name: "ptr-sse-concurrency",
      path_prefix: "/sse-concurrency",
      target_url: upstream.baseUrl,
      provider_key_id: pk.id,
    });
    // Write the constrained principal last. A completed unary probe proves
    // the preceding route has reached the same snapshot and released its
    // slot before this test starts the first real stream.
    await seed.createApiKey({
      key_hash: STREAM_LIMITED_KEY_HASH,
      allowed_models: ["*"],
      allowed_routes: ["ptr-sse-concurrency"],
      rate_limit: { concurrency: 1 },
    });

    const headers = {
      authorization: `Bearer ${STREAM_LIMITED_PLAINTEXT}`,
      "content-type": "application/json",
    };
    const call = () =>
      fetch(`${app!.proxyUrl}/sse-concurrency/chat/completions`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          model: "gpt-4o",
          messages: [{ role: "user", content: "hi" }],
          stream: true,
        }),
      });

    await waitConfigPropagation(async () => {
      try {
        const probe = await call();
        const ready =
          probe.status === 200 &&
          !(probe.headers.get("content-type") ?? "").includes("text/event-stream");
        await probe.text();
        return ready;
      } catch {
        return false;
      }
    });

    // Fetch resolves as soon as the upstream headers are relayed. Keep this
    // body unread while issuing the second request: it is the real caller
    // journey that used to release the slot at handler return.
    const first = await call();
    expect(first.status).toBe(200);
    const upstreamCallsWhileStreaming = upstream.receivedRequests.length;
    expect(upstreamCallsWhileStreaming).toBe(2);

    const second = await call();
    expect(second.status).toBe(429);
    expect(second.headers.get("x-ratelimit-scope")).toBe("concurrency");
    await second.text();
    expect(upstream.receivedRequests).toHaveLength(upstreamCallsWhileStreaming);

    // A cancelled client must release the same hold. The admitted follow-up
    // is deliberately a separate, naturally ending stream.
    expect(first.body).not.toBeNull();
    await first.body!.cancel();
    let naturallyEnding: Response | undefined;
    await waitConfigPropagation(async () => {
      const afterCancel = await call();
      const admitted = afterCancel.status === 200;
      if (admitted) naturallyEnding = afterCancel;
      else await afterCancel.text();
      return admitted;
    }, 3_000);
    expect(naturallyEnding).toBeDefined();
    await naturallyEnding!.text();

    const afterEnd = await call();
    expect(afterEnd.status).toBe(200);
    await afterEnd.text();

    expect(upstream.receivedRequests).toHaveLength(upstreamCallsWhileStreaming + 2);
  });

  test("a silent SSE gap terminates and releases the concurrency slot", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    const upstream = await startOpenAiUpstream({
      scriptedResponses: [
        {
          streamEvents: [
            JSON.stringify({ choices: [{ delta: { content: "first" } }] }),
            JSON.stringify({ choices: [{ delta: { content: "late" } }] }),
            "[DONE]",
          ],
          // The first event is immediate; the next one violates the route's
          // 100 ms per-read budget after headers have already been relayed.
          eventDelayMs: 1_500,
        },
        { nonStreamBody: { recovered: true } },
      ],
    });
    upstreams.push(upstream);

    const pk = await seed.createProviderKey({
      display_name: "ptr-sse-timeout-pk",
      secret: "sk-mock",
      api_base: "http://unused-on-routes",
    });
    await seed.createPassthroughRoute({
      name: "ptr-sse-timeout",
      path_prefix: "/sse-timeout",
      target_url: upstream.baseUrl,
      provider_key_id: pk.id,
      timeout_ms: 100,
    });
    await seed.createApiKey({
      key_hash: STREAM_TIMEOUT_KEY_HASH,
      allowed_models: ["*"],
      allowed_routes: ["ptr-sse-timeout"],
      rate_limit: { concurrency: 1 },
    });

    const headers = {
      authorization: `Bearer ${STREAM_TIMEOUT_PLAINTEXT}`,
      "content-type": "application/json",
    };
    const call = () =>
      fetch(`${app!.proxyUrl}/sse-timeout/chat/completions`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          model: "gpt-4o",
          messages: [{ role: "user", content: "hi" }],
          stream: true,
        }),
      });

    await waitConfigPropagation(async () => {
      try {
        const probe = await fetch(`${app!.proxyUrl}/v1/models`, {
          headers: { authorization: `Bearer ${STREAM_TIMEOUT_PLAINTEXT}` },
        });
        await probe.text();
        return probe.status === 200;
      } catch {
        return false;
      }
    });

    const started = Date.now();
    const stalled = await call();
    expect(stalled.status).toBe(200);
    const body = await stalled.text();
    const elapsed = Date.now() - started;
    expect(body).toContain("first");
    expect(body).not.toContain("late");
    expect(elapsed).toBeLessThan(1_000);

    // A new request can enter after the timeout. A leaked concurrency hold
    // would instead be a gateway 429 and would never consume step three.
    const recovered = await call();
    expect(recovered.status).toBe(200);
    expect(await recovered.json()).toEqual({ recovered: true });
  }, 10_000);

  test("envelope auto-detection: usage follows the request body, never the config", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    // A route carries NO protocol or streaming configuration — the
    // envelope is detected per exchange from the request body's top-level
    // keys. Three exchanges through three identical-config routes:
    //
    //   - chat-shaped request, buffered response  → `usage` extracted
    //   - Responses-shaped request, SSE response  → nested usage on the
    //     terminal `response.completed` event extracted (the shape the
    //     GitHub Copilot CLI streams)
    //   - JSON-RPC request (MCP)                  → raw: a usage-looking
    //     object in the response is NOT probed (no phantom tokens)
    const otlp = await startMockOtlp();
    otlps.push(otlp);
    await seed.createObservabilityExporter({
      name: "ptr-auto-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });

    const chatUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "chatcmpl-auto",
        choices: [{ message: { role: "assistant", content: "ok" } }],
        usage: { prompt_tokens: 7, completion_tokens: 3 },
      },
    });
    const responsesUpstream = await startOpenAiUpstream({
      streamEvents: [
        JSON.stringify({ type: "response.output_text.delta", delta: "he" }),
        JSON.stringify({ type: "response.output_text.delta", delta: "y" }),
        JSON.stringify({
          type: "response.completed",
          response: { usage: { input_tokens: 11, output_tokens: 4 } },
        }),
      ],
    });
    const rpcUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        jsonrpc: "2.0",
        id: 1,
        result: { usage: { prompt_tokens: 99, completion_tokens: 99 } },
      },
    });
    upstreams.push(chatUpstream, responsesUpstream, rpcUpstream);

    const pk = await seed.createProviderKey({
      display_name: "ptr-auto-pk",
      secret: "sk-mock",
      api_base: "http://unused-on-routes",
    });
    for (const [name, prefix, upstream] of [
      ["ptr-auto-chat", "/auto-chat", chatUpstream],
      ["ptr-auto-resp", "/auto-resp", responsesUpstream],
      ["ptr-auto-rpc", "/auto-rpc", rpcUpstream],
    ] as const) {
      await seed.createPassthroughRoute({
        name,
        path_prefix: prefix,
        target_url: upstream.baseUrl,
        provider_key_id: pk.id,
      });
    }

    // Readiness sentinel, seeded LAST: the snapshot watch applies etcd
    // revisions in order, so this route serving proves every resource
    // seeded before it (exporter, the three routes under test) is live —
    // without the probe exercising any exchange the test asserts on.
    await seed.createPassthroughRoute({
      name: "ptr-auto-ready",
      path_prefix: "/auto-ready",
      target_url: rpcUpstream.baseUrl,
      provider_key_id: pk.id,
    });

    const headers = {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    };
    const post = (path: string, body: unknown) =>
      fetch(`${app!.proxyUrl}${path}`, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
      });

    await waitConfigPropagation(async () => {
      try {
        const r = await post("/auto-ready/ping", {});
        await r.text();
        return r.status === 200;
      } catch {
        return false;
      }
    });

    const chatRes = await post("/auto-chat/chat/completions", {
      model: "m",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(chatRes.status).toBe(200);
    await chatRes.text();
    const responsesRes = await post("/auto-resp/responses", {
      model: "m",
      input: [{ role: "user", content: "hi" }],
      stream: true,
    });
    expect(responsesRes.status).toBe(200);
    await responsesRes.text();
    const rpcRes = await post("/auto-rpc/mcp", {
      jsonrpc: "2.0",
      id: 1,
      method: "tools/list",
    });
    expect(rpcRes.status).toBe(200);
    await rpcRes.text();

    const spanFor = async (route: string) => {
      const deadline = Date.now() + 10_000;
      while (Date.now() < deadline) {
        const hit = otlp.spans.find(
          (s) => s.attributes["aisix.passthrough.route_name"] === route,
        );
        if (hit) return hit;
        await new Promise((r) => setTimeout(r, 50));
      }
      throw new Error(`no OTLP span for route ${route}`);
    };

    const chatSpan = await spanFor("ptr-auto-chat");
    expect(chatSpan.attributes["gen_ai.usage.input_tokens"]).toBe(7);
    expect(chatSpan.attributes["gen_ai.usage.output_tokens"]).toBe(3);

    const respSpan = await spanFor("ptr-auto-resp");
    expect(respSpan.attributes["gen_ai.usage.input_tokens"]).toBe(11);
    expect(respSpan.attributes["gen_ai.usage.output_tokens"]).toBe(4);

    const rpcSpan = await spanFor("ptr-auto-rpc");
    expect(rpcSpan.attributes["gen_ai.usage.input_tokens"] ?? 0).toBe(0);
    expect(rpcSpan.attributes["gen_ai.usage.output_tokens"] ?? 0).toBe(0);
  });
});
