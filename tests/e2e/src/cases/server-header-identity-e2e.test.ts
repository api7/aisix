import { createHash } from "node:crypto";
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

// Header-identity scenarios this test pins. Comment lives at the
// top so the test body stays focused on assertions rather than
// re-narrating intent. Each scenario maps to a numbered block below.
//
//   1. 200 happy-path                       — handler return
//   2. 401 OpenAI envelope                  — auth-fail short-circuit
//   3. 404 OpenAI envelope                  — unknown model
//   4. /livez                               — bare handler, no upstream
//   5. SSE / streaming chat completion      — flushed headers
//   6. 401 Anthropic envelope (/v1/messages)— separate rendering path

// E2E: the data plane must identify itself via the `Server` response
// header on every response, using the `AISIX/<version>` product token.
// Pinned because the header is a customer-visible contract — clients
// (and intermediaries) use `Server` to identify the gateway without
// round-tripping a status endpoint, and the header must survive every
// response path the DP can emit:
//
//   1. Success body                — happy-path 200 from upstream
//   2. Auth-failure envelope       — 401 from the auth layer
//   3. Routing-failure envelope    — 404 for unknown model
//   4. Liveness probe              — bare `/livez` GET
//
// All four must carry an *identical* `Server` value: the gateway's
// identity must not change between request paths, and must never leak
// an upstream provider's `Server` token (provider fingerprinting via
// error envelopes is a known leakage vector).
//
// References:
// - RFC 9110 §10.2.4 (Server header, `product/version` format):
//   <https://www.rfc-editor.org/rfc/rfc9110#section-10.2.4>
// - APISIX `Server` convention (`APISIX/<version>`):
//   <https://apisix.apache.org/docs/apisix/admin-api/>

const VALID_PLAINTEXT = "sk-server-header-e2e-valid";
const VALID_KEY_HASH = createHash("sha256")
  .update(VALID_PLAINTEXT)
  .digest("hex");
const UNKNOWN_PLAINTEXT = "sk-server-header-e2e-unregistered";

// Semver-anchored: `AISIX/` + `<major>.<minor>.<patch>` with optional
// pre-release / build metadata. The version segment comes from
// `CARGO_PKG_VERSION`, which the workspace pins to semver — a regression
// that swaps in e.g. `CARGO_PKG_NAME` (yielding `AISIX/aisix-proxy`)
// would slip past a looser `.+` pattern. Tightening to semver locks the
// documented contract.
const SERVER_HEADER_PATTERN = /^AISIX\/\d+\.\d+\.\d+([-+][\w.-]+)?$/;

describe("data plane identifies itself via Server header on every response", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "server-header-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "server-header-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: VALID_KEY_HASH,
      allowed_models: ["server-header-model"],
    });

    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${VALID_PLAINTEXT}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: "server-header-model",
          messages: [{ role: "user", content: "ready-probe" }],
        }),
      });
      await res.text();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("every response — success, 401, 404, livez — carries the same AISIX/<version> token", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    // 1. Happy-path chat completion.
    const ok = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${VALID_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "server-header-model",
        messages: [{ role: "user", content: "hello" }],
      }),
    });
    await ok.text();
    expect(ok.status).toBe(200);
    const okServer = ok.headers.get("server");
    expect(okServer, "success response missing Server header").not.toBeNull();
    expect(okServer).toMatch(SERVER_HEADER_PATTERN);

    // 2. Auth-failure envelope (401).
    const unauth = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${UNKNOWN_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "server-header-model",
        messages: [{ role: "user", content: "bad auth" }],
      }),
    });
    await unauth.text();
    expect(unauth.status).toBe(401);
    const unauthServer = unauth.headers.get("server");
    expect(unauthServer, "401 envelope missing Server header").not.toBeNull();
    expect(unauthServer).toMatch(SERVER_HEADER_PATTERN);

    // 3. Routing-failure envelope (404 unknown model).
    const notfound = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${VALID_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "does-not-exist",
        messages: [{ role: "user", content: "missing model" }],
      }),
    });
    await notfound.text();
    expect(notfound.status).toBe(404);
    const notfoundServer = notfound.headers.get("server");
    expect(notfoundServer, "404 envelope missing Server header").not.toBeNull();
    expect(notfoundServer).toMatch(SERVER_HEADER_PATTERN);

    // 4. Bare liveness probe.
    const livez = await fetch(`${app.proxyUrl}/livez`);
    await livez.text();
    expect(livez.status).toBe(200);
    const livezServer = livez.headers.get("server");
    expect(livezServer, "livez missing Server header").not.toBeNull();
    expect(livezServer).toMatch(SERVER_HEADER_PATTERN);

    // 5. SSE / streaming chat completion. Response headers are flushed
    // at handler-return-time, before body framing starts — a regression
    // where a streaming route is mounted outside the identity layer
    // (or where the streaming response is constructed via a hyper Body
    // pipeline that bypasses tower-http) would lose the Server header
    // on every streamed call. Drain so the connection closes cleanly.
    const sse = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${VALID_PLAINTEXT}`,
        "content-type": "application/json",
        accept: "text/event-stream",
      },
      body: JSON.stringify({
        model: "server-header-model",
        messages: [{ role: "user", content: "stream me" }],
        stream: true,
      }),
    });
    await sse.text();
    expect(sse.status).toBe(200);
    expect(sse.headers.get("content-type")).toMatch(/text\/event-stream/);
    const sseServer = sse.headers.get("server");
    expect(sseServer, "SSE response missing Server header").not.toBeNull();
    expect(sseServer).toMatch(SERVER_HEADER_PATTERN);

    // 6. Anthropic-shape error envelope (`/v1/messages` rejects auth
    // through a separate `into_anthropic_response()` rendering path
    // from the OpenAI envelope). A regression that builds the
    // Anthropic 401 via a Response constructor bypassing the outer
    // layer would slip past an OpenAI-only test.
    const anth401 = await fetch(`${app.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: {
        "x-api-key": UNKNOWN_PLAINTEXT,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: "server-header-model",
        max_tokens: 10,
        messages: [{ role: "user", content: "hi" }],
      }),
    });
    await anth401.text();
    expect(anth401.status).toBe(401);
    const anth401Server = anth401.headers.get("server");
    expect(anth401Server, "Anthropic-envelope 401 missing Server header").not.toBeNull();
    expect(anth401Server).toMatch(SERVER_HEADER_PATTERN);

    // The gateway's identity must be stable across paths — clients
    // observing different `Server` values on success vs. error would
    // (legitimately) treat that as two different servers in the chain.
    expect(unauthServer).toBe(okServer);
    expect(notfoundServer).toBe(okServer);
    expect(livezServer).toBe(okServer);
    expect(sseServer).toBe(okServer);
    expect(anth401Server).toBe(okServer);
  });
});
