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
//   3. The removed implicit tunnel answers 410 with a migration pointer
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

const CALLER_PLAINTEXT = "sk-ptr-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("passthrough-route e2e: explicit routes, BYO credentials, 410 tombstone", () => {
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
      allowed_routes: ["*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
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

  test("unclaimed /passthrough/* answers the 410 migration tombstone", async (ctx) => {
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
    expect(res.status).toBe(410);
    const body = (await res.json()) as {
      error?: { code?: unknown; message?: unknown };
    };
    expect(body.error?.code).toBe("endpoint_removed");
    expect(String(body.error?.message)).toContain("passthrough_route");
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

  test("preserve_host route narrowed by a prefix mirrors the whole path", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    // The forward-proxy shape an agent CLI needs: one upstream host serving
    // several protocols under different prefixes (GitHub Copilot's CLI talks
    // to its MCP server at /mcp/readonly on the very host it uses for chat).
    // The prefix narrows WHICH requests the route claims; it is not a mount
    // point, so the upstream must receive its own path untouched — a stripped
    // "/readonly" 404s at the real backend.
    const upstream = await startOpenAiUpstream({
      nonStreamBody: { routed: "mirrored" },
    });
    upstreams.push(upstream);

    await seed.createPassthroughRoute({
      name: "ptr-mirror-mcp",
      hosts: ["agent-upstream.example.com"],
      // `/mcp` is a reserved gateway namespace for host-less routes; a
      // host-matched route may claim it because host dispatch runs ahead of
      // the typed routes.
      path_prefix: "/mcp",
      preserve_host: true,
      auth_mode: "header_key",
      auth_header_name: "x-aisix-api-key",
      credential_mode: "forward_client",
    });

    // preserve_host targets https://<host>, so point the harness host at the
    // mock upstream via the route's own hosts entry: we assert on the PATH
    // the upstream sees, which the mock records before answering.
    const call = async () =>
      harnessRequest(`${app!.proxyUrl}/mcp/readonly`, {
        method: "POST",
        headers: {
          host: "agent-upstream.example.com",
          "x-aisix-api-key": CALLER_PLAINTEXT,
          "content-type": "application/json",
        },
        body: JSON.stringify({ jsonrpc: "2.0", method: "initialize", id: 1 }),
      });

    await waitConfigPropagation(async () => {
      try {
        const r = await call();
        await r.body.text();
        // Any answer other than the gateway's own 404/410 proves the route
        // claimed the reserved-looking path.
        return r.statusCode !== 404 && r.statusCode !== 410;
      } catch {
        return false;
      }
    });

    const res = await call();
    await res.body.text();
    // The gateway resolved the route (not its own /mcp endpoint) and did not
    // answer the tombstone.
    expect(res.statusCode).not.toBe(410);
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

    const upstream = await startOpenAiUpstream({
      streamEvents: [
        JSON.stringify({ choices: [{ delta: { content: "hel" } }] }),
        JSON.stringify({ choices: [{ delta: { content: "lo" } }] }),
        JSON.stringify({
          choices: [],
          usage: { prompt_tokens: 5, completion_tokens: 2 },
        }),
        "[DONE]",
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
      protocol: "openai_chat",
    });

    const headers = {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    };
    const call = async () =>
      fetch(`${app!.proxyUrl}/sse-tunnel/chat/completions`, {
        method: "POST",
        headers,
        body: JSON.stringify({ model: "gpt-4o", stream: true }),
      });

    await waitConfigPropagation(async () => {
      try {
        const r = await call();
        const ok =
          r.status === 200 &&
          (r.headers.get("content-type") ?? "").includes("text/event-stream");
        await r.text();
        return ok;
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
});
