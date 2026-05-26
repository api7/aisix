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

// `AISIX/` followed by a non-empty token. Version shape is intentionally
// loose — the contract is "product/version", not a specific semver
// today. Tightening to semver here would tie this test to the workspace
// versioning policy.
const SERVER_HEADER_PATTERN = /^AISIX\/.+$/;

describe("data plane identifies itself via Server header on every response", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "server-header-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "server-header-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
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

    // The gateway's identity must be stable across paths — clients
    // observing different `Server` values on success vs. error would
    // (legitimately) treat that as two different servers in the chain.
    expect(unauthServer).toBe(okServer);
    expect(notfoundServer).toBe(okServer);
    expect(livezServer).toBe(okServer);
  });
});
