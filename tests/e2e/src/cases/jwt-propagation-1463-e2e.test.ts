import { createHash } from "node:crypto";
import { randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  agentClaims,
  EtcdClient,
  SeedClient,
  spawnApp,
  startMockIdp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type MockIdp,
  type OpenAiUpstream,
  type ReceivedRequest,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: delivering the caller's verified JWT to an internal upstream
// (`forward_jwt_header`, AISIX-Cloud#1463).
//
// The gateway is inserted into a call chain whose hops already carry an
// internal JWT, and the upstreams authorize on the end user's claims. So
// the gateway must keep AUTHENTICATING the token (signature, expiry, claim
// binding — unchanged) and additionally RELAY it, unmodified, to the
// upstream, in a header the operator names per upstream.
//
// One mock upstream serves all three surfaces and records the headers it
// received, which is the whole observable contract:
//
//   1. /v1/chat/completions — the ProviderKey's request overrides.
//   2. /passthrough/*       — the route's own field.
//   3. /mcp                 — an `openapi` server whose generated tools
//                             call the same REST upstream (a system server
//                             registered as tools).
//
// And the three ways it must NOT fire, each of them a real deployment:
// an upstream nobody opted in, an API-key caller with no token to relay,
// and the reserved slot — a token must never displace the gateway's own
// credential unless the operator named that slot.
//
// References:
// - RFC 9110 §11.6.2 (Authorization = <scheme> <credentials>)
//   <https://www.rfc-editor.org/rfc/rfc9110#section-11.6.2>
// - RFC 7519 (JWT) <https://datatracker.ietf.org/doc/html/rfc7519>

const JWT_HEADER = "x-user-jwt";
const FORWARDING_MODEL = "jwt-fwd-model";
const PLAIN_MODEL = "jwt-plain-model";
const ANTHROPIC_MODEL = "jwt-anthropic-model";
const ROUTE_PREFIX = "/passthrough/claims";
const AGENT_KEY = "sk-jwt-propagation-agent";

function claims(): Record<string, unknown> {
  return agentClaims(idpIssuer, { sub: "poc-agent" });
}

/** Set once the mock IdP is up; `claims()` needs its issuer. */
let idpIssuer = "";

/** The OpenAPI document exposing the upstream's `/v1/models` as one tool. */
function systemServerSpec(): Record<string, unknown> {
  return {
    openapi: "3.0.0",
    info: { title: "system-server", version: "1" },
    paths: {
      "/models": {
        get: {
          operationId: "listmodels",
          responses: { "200": { description: "ok" } },
        },
      },
    },
  };
}

describe("jwt propagation e2e: forward_jwt_header across all three upstream surfaces", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let idp: MockIdp | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  /** Requests the upstream received since `mark`. */
  const since = (mark: number): ReceivedRequest[] =>
    upstream!.receivedRequests.slice(mark);

  /** The value of the forwarded header on the single request since `mark`. */
  const forwardedOn = (mark: number): string | undefined => {
    const seen = since(mark);
    expect(seen.length).toBeGreaterThan(0);
    return seen[seen.length - 1]!.headers[JWT_HEADER];
  };

  const chat = async (
    token: string,
    model: string,
    scheme: "Bearer" = "Bearer",
  ): Promise<Response> =>
    fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `${scheme} ${token}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: "probe" }],
      }),
    });

  const mcpPost = async (
    token: string,
    body: unknown,
  ): Promise<Record<string, any> | undefined> => {
    const res = await fetch(`${app!.proxyUrl}/mcp`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${token}`,
        "content-type": "application/json",
        accept: "application/json, text/event-stream",
      },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    try {
      return text ? JSON.parse(text) : undefined;
    } catch {
      return undefined;
    }
  };

  /** Spec-faithful per-operation handshake; the endpoint is stateless. */
  const callSystemServerTool = async (token: string): Promise<void> => {
    await mcpPost(token, {
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: {
        protocolVersion: "2025-11-25",
        capabilities: {},
        clientInfo: { name: "jwt-propagation-e2e", version: "0.1" },
      },
    });
    await mcpPost(token, {
      jsonrpc: "2.0",
      method: "notifications/initialized",
    });
    await mcpPost(token, {
      jsonrpc: "2.0",
      id: 2,
      method: "tools/call",
      params: { name: "systemserver__listmodels", arguments: {} },
    });
  };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    idp = await startMockIdp();
    idpIssuer = idp.url;
    app = await spawnApp({});
    seed = new SeedClient(etcd, app.etcdPrefix);

    // The upstream that authorizes on the end user's claims: its
    // ProviderKey names the header the caller's token arrives in.
    const forwardingPk = await seed.createProviderKey({
      display_name: "jwt-fwd-pk",
      api_key: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
      request: { forward_jwt_header: JWT_HEADER },
    });
    // A second upstream with no opt-in — the default every ProviderKey
    // already has.
    const plainPk = await seed.createProviderKey({
      display_name: "jwt-plain-pk",
      api_key: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: FORWARDING_MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: forwardingPk.id,
    });
    await seed.createModel({
      display_name: PLAIN_MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: plainPk.id,
    });

    // An Anthropic-shaped upstream: `/v1/messages` reaches it through the
    // native dispatch path (which builds its own header context) and
    // `/v1/chat/completions` through the Anthropic bridge (which builds
    // its request by hand). Two different code paths, one upstream.
    const anthropicPk = await seed.createProviderKey({
      display_name: "jwt-anthropic-pk",
      api_key: "sk-mock-anthropic",
      api_base: `${upstream.baseUrl}/v1`,
      provider: "anthropic",
      request: { forward_jwt_header: JWT_HEADER },
    });
    await seed.createModel({
      display_name: ANTHROPIC_MODEL,
      provider: "anthropic",
      model_name: "claude-sonnet-4",
      provider_key_id: anthropicPk.id,
    });

    await seed.createPassthroughRoute({
      name: "claims-route",
      path_prefix: ROUTE_PREFIX,
      target_url: upstream.baseUrl,
      auth_mode: "gateway_key",
      credential_mode: "inject",
      provider_key_id: plainPk.id,
      forward_jwt_header: JWT_HEADER,
    });

    await seed.update("mcp_servers", randomUUID(), {
      name: "systemserver",
      type: "openapi",
      url: `${upstream.baseUrl}/v1`,
      spec: systemServerSpec(),
      forward_jwt_header: JWT_HEADER,
    });

    await seed.createOidcProvider({
      name: "poc-idp",
      issuer: idp.url,
      audiences: ["aisix-gateway"],
      jwks_uri: idp.jwksUrl,
    });

    // One identity, reachable by JWT and by its own plaintext key: the
    // API-key case must differ only in what the caller presented.
    await seed.createApiKey({
      key_hash: createHash("sha256").update(AGENT_KEY).digest("hex"),
      allowed_models: ["*"],
      allowed_routes: ["*"],
      mcp_access: { allow: ["*"] },
      jwt_subject: "poc-agent",
      jwt_provider: "poc-idp",
    });

    await waitConfigPropagation(async () => {
      const res = await chat(idp!.sign(claims()), FORWARDING_MODEL);
      await res.text();
      return res.status === 200;
    });
  }, 120_000);

  afterAll(async () => {
    await app?.stop();
    await upstream?.close();
    await idp?.close();
  });

  test("the verified token reaches an LLM upstream that opted in", async () => {
    if (!etcdReachable) return;
    const token = idp!.sign(claims());
    const mark = upstream!.receivedRequests.length;

    const res = await chat(token, FORWARDING_MODEL);
    expect(res.status).toBe(200);
    await res.text();

    // Relayed unmodified — the same token the caller presented, with no
    // claim added, removed, or rewritten.
    expect(forwardedOn(mark)).toBe(token);
  });

  test("the gateway's own credential still reaches that upstream", async () => {
    if (!etcdReachable) return;
    const mark = upstream!.receivedRequests.length;

    const res = await chat(idp!.sign(claims()), FORWARDING_MODEL);
    await res.text();

    // The caller's token is delivered BESIDE the ProviderKey's secret,
    // not instead of it: the operator pointed the field at a header of
    // its own, so the upstream still authenticates the gateway.
    const seen = since(mark).at(-1)!;
    expect(seen.headers.authorization).toBe("Bearer sk-mock");
  });

  test("an upstream nobody opted in receives no caller token", async () => {
    if (!etcdReachable) return;
    const mark = upstream!.receivedRequests.length;

    const res = await chat(idp!.sign(claims()), PLAIN_MODEL);
    expect(res.status).toBe(200);
    await res.text();

    expect(forwardedOn(mark)).toBeUndefined();
  });

  test("an API-key caller has no token to relay", async () => {
    if (!etcdReachable) return;
    const mark = upstream!.receivedRequests.length;

    // Same key, same upstream — only the credential the caller presented
    // differs, so an absent header here can only be the JWT's absence.
    const res = await chat(AGENT_KEY, FORWARDING_MODEL);
    expect(res.status).toBe(200);
    await res.text();

    expect(forwardedOn(mark)).toBeUndefined();
  });

  test("a passthrough route relays the token it authenticated with", async () => {
    if (!etcdReachable) return;
    const token = idp!.sign(claims());
    const mark = upstream!.receivedRequests.length;

    const res = await fetch(
      `${app!.proxyUrl}${ROUTE_PREFIX}/v1/chat/completions`,
      {
        method: "POST",
        headers: {
          authorization: `Bearer ${token}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "probe" }],
        }),
      },
    );
    expect(res.status).toBe(200);
    await res.text();

    // `gateway_key` consumes and strips `Authorization` to identify the
    // caller, so without the field the upstream would see no identity at
    // all — this is the hop the field exists for.
    expect(forwardedOn(mark)).toBe(token);
  });

  test("a REST system server exposed as MCP tools receives the token", async () => {
    if (!etcdReachable) return;
    const token = idp!.sign(claims());
    const mark = upstream!.receivedRequests.length;

    await callSystemServerTool(token);

    const toolCall = since(mark).find((r) => r.path.endsWith("/models"));
    expect(toolCall, "the generated tool must reach the REST upstream").toBeDefined();
    expect(toolCall!.headers[JWT_HEADER]).toBe(token);
  });

  test("the native /v1/messages path relays the token too", async () => {
    if (!etcdReachable) return;
    const token = idp!.sign(claims());
    const mark = upstream!.receivedRequests.length;

    // This endpoint does NOT go through the bridge context the chat path
    // uses — it builds its own outbound headers, which is exactly how a
    // sibling endpoint silently misses a per-request mechanism here.
    await fetch(`${app!.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${token}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: ANTHROPIC_MODEL,
        max_tokens: 16,
        messages: [{ role: "user", content: "probe" }],
      }),
    }).then((r) => r.text());

    const seen = since(mark).find((r) => r.path.endsWith("/messages"));
    expect(seen, "the request must reach the upstream").toBeDefined();
    expect(seen!.headers[JWT_HEADER]).toBe(token);
  });

  test("the Anthropic bridge relays the token on /v1/chat/completions", async () => {
    if (!etcdReachable) return;
    const token = idp!.sign(claims());
    const mark = upstream!.receivedRequests.length;

    // A different construction path again: this bridge assembles its
    // upstream request by hand rather than through the header pipeline.
    await chat(token, ANTHROPIC_MODEL).then((r) => r.text());

    const seen = since(mark).at(-1);
    expect(seen, "the request must reach the upstream").toBeDefined();
    expect(seen!.headers[JWT_HEADER]).toBe(token);
    // The gateway's own key still authenticates it, in its own slot.
    expect(seen!.headers["x-api-key"]).toBe("sk-mock-anthropic");
  });

  test("an API-key caller cannot occupy the verified-identity slot", async () => {
    if (!etcdReachable) return;
    const mark = upstream!.receivedRequests.length;

    // The upstream is told this header carries a gateway-VERIFIED
    // identity. A caller with no JWT must not be able to fill it itself.
    const res = await fetch(
      `${app!.proxyUrl}${ROUTE_PREFIX}/v1/chat/completions`,
      {
        method: "POST",
        headers: {
          authorization: `Bearer ${AGENT_KEY}`,
          "content-type": "application/json",
          [JWT_HEADER]: "forged.by.the.caller",
        },
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "probe" }],
        }),
      },
    );
    expect(res.status).toBe(200);
    await res.text();

    expect(forwardedOn(mark)).toBeUndefined();
  });
});
