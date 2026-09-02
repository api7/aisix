import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startMcpUpstream,
  startOpenAiUpstream,
  waitConfigPropagation,
  type McpUpstream,
  type OpenAiUpstream,
  type ReceivedRequest,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: `forward_client_headers` across every proxy face the gateway
// serves, exercised through a real gateway against real upstreams.
//
// The capability is auth-independent: the operator names header patterns
// per upstream, and whatever the caller sent under a matching name reaches
// that upstream verbatim. It is how a gateway fronting an INTERNAL service
// hands that service the end user's own credentials and context — the
// service keeps reading the header it always read.
//
// Three faces build their outbound headers in completely different code,
// which is why each gets its own coverage here:
//
//   1. /v1/*             — ProviderKey `request.forward_client_headers`,
//                          on the shared bridge pipeline AND on the
//                          Anthropic bridge, which assembles its request
//                          separately.
//   2. /mcp              — the `mcp_server` field, on both `type: mcp`
//                          (Streamable HTTP upstream) and `type: openapi`
//                          (generated tools calling a REST API).
//   3. /passthrough/*    — the `passthrough_route` field, where the
//                          default is the opposite (forward everything)
//                          and the list OVERRIDES a strip.
//
// The credential collision is asserted on every face: a header the
// operator named beats the credential the gateway would otherwise inject
// into that slot, and rides ALONE — two credentials on the wire would let
// the upstream choose. And the limits are asserted beside it, because they
// are why the default is "forward nothing": the gateway's own `x-aisix-*`
// namespace, and trace context under a glob.

const CALLER_KEY = "sk-fwd-headers-caller-PLAINTEXT";
const CALLER_HASH = createHash("sha256").update(CALLER_KEY).digest("hex");

const FORWARD_MODEL = "fwd-openai-model";
const PLAIN_MODEL = "fwd-plain-model";
const ANTHROPIC_MODEL = "fwd-anthropic-model";
const ROUTE_PREFIX = "/passthrough/fwd";
const CUSTOM_HEADER = "x-user-jwt";
const CALLER_CREDENTIAL = "Bearer callers-own-credential";
const GATEWAY_SECRET = "sk-mock-provider-secret";

const ANTHROPIC_BODY = {
  id: "msg_01",
  type: "message",
  role: "assistant",
  content: [{ type: "text", text: "hi" }],
  model: "claude-3-5-haiku-20241022",
  stop_reason: "end_turn",
  usage: { input_tokens: 5, output_tokens: 4 },
};

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

describe("forward_client_headers e2e: one capability across every proxy face", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let anthropicUpstream: OpenAiUpstream | undefined;
  let mcpUpstream: McpUpstream | undefined;
  let etcdReachable = false;

  const since = (mark: number): ReceivedRequest[] =>
    upstream!.receivedRequests.slice(mark);

  const bodyOf = async (res: Response, what: string): Promise<string> => {
    const text = await res.text();
    expect(
      res.ok,
      `${what} failed with ${res.status}: ${text.slice(0, 400)}`,
    ).toBe(true);
    return text;
  };

  const chat = async (
    model: string,
    headers: Record<string, string> = {},
  ): Promise<Response> =>
    fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_KEY}`,
        "content-type": "application/json",
        ...headers,
      },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: "probe" }],
      }),
    });

  const mcpPost = async (
    body: unknown,
    headers: Record<string, string> = {},
  ): Promise<Record<string, any> | undefined> => {
    const res = await fetch(`${app!.proxyUrl}/mcp`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_KEY}`,
        "content-type": "application/json",
        accept: "application/json, text/event-stream",
        ...headers,
      },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    // A handshake that fails here would surface as the header assertion
    // finding nothing at the upstream, which says nothing about why.
    if (!res.ok) throw new Error(`MCP POST ${res.status}: ${text.slice(0, 400)}`);
    if (!text) return undefined;
    try {
      return JSON.parse(text);
    } catch {
      throw new Error(`MCP POST returned unparseable JSON: ${text.slice(0, 400)}`);
    }
  };

  /** Spec-faithful per-operation handshake; the endpoint is stateless. */
  const callTool = async (
    tool: string,
    headers: Record<string, string> = {},
  ): Promise<void> => {
    await mcpPost(
      {
        jsonrpc: "2.0",
        id: 1,
        method: "initialize",
        params: {
          protocolVersion: "2025-11-25",
          capabilities: {},
          clientInfo: { name: "forward-client-headers-e2e", version: "0.1" },
        },
      },
      headers,
    );
    await mcpPost({ jsonrpc: "2.0", method: "notifications/initialized" }, headers);
    await mcpPost(
      { jsonrpc: "2.0", id: 2, method: "tools/call", params: { name: tool, arguments: { text: "hi" } } },
      headers,
    );
  };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    anthropicUpstream = await startOpenAiUpstream({ nonStreamBody: ANTHROPIC_BODY });
    mcpUpstream = await startMcpUpstream("fwd");
    app = await spawnApp({});
    const seed = new SeedClient(etcd, app.etcdPrefix);

    // The upstream the operator opted in: a custom header, the caller's
    // own credential slot, and a glob that must NOT sweep trace context.
    const forwardingPk = await seed.createProviderKey({
      display_name: "fwd-pk",
      api_key: GATEWAY_SECRET,
      api_base: `${upstream.baseUrl}/v1`,
      request: {
        forward_client_headers: [CUSTOM_HEADER, "authorization", "x-*", "trace*"],
      },
    });
    // A second upstream with no opt-in — the default every ProviderKey has.
    const plainPk = await seed.createProviderKey({
      display_name: "fwd-plain-pk",
      api_key: GATEWAY_SECRET,
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: FORWARD_MODEL,
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

    // The Anthropic bridge assembles its upstream request by hand rather
    // than through the shared pipeline — the classic place a per-request
    // mechanism goes silently missing. Its credential slot is `x-api-key`.
    const anthropicPk = await seed.createProviderKey({
      display_name: "fwd-anthropic-pk",
      api_key: "sk-mock-anthropic",
      api_base: `${anthropicUpstream.baseUrl}/v1`,
      provider: "anthropic",
      adapter: "anthropic",
      request: { forward_client_headers: [CUSTOM_HEADER, "x-api-key"] },
    });
    await seed.createModel({
      display_name: ANTHROPIC_MODEL,
      provider: "anthropic",
      model_name: "claude-sonnet-4",
      provider_key_id: anthropicPk.id,
    });

    // `gateway_key` + `inject`: the gateway CONSUMES `authorization` to
    // identify the caller and strips it, and `strip_headers` removes the
    // custom slot too. Both strips are what the list overrides.
    const routePk = await seed.createProviderKey({
      display_name: "fwd-route-pk",
      api_key: GATEWAY_SECRET,
      api_base: upstream.baseUrl,
      strip_headers: ["authorization", CUSTOM_HEADER],
    });
    await seed.createPassthroughRoute({
      name: "fwd-route",
      path_prefix: ROUTE_PREFIX,
      target_url: upstream.baseUrl,
      auth_mode: "gateway_key",
      credential_mode: "inject",
      provider_key_id: routePk.id,
      forward_client_headers: [CUSTOM_HEADER, "authorization"],
    });
    await seed.createPassthroughRoute({
      name: "fwd-route-plain",
      path_prefix: `${ROUTE_PREFIX}-plain`,
      target_url: upstream.baseUrl,
      auth_mode: "gateway_key",
      credential_mode: "inject",
      provider_key_id: routePk.id,
    });

    // A REST API exposed as MCP tools, and a real MCP upstream: two
    // different outbound builders behind one endpoint.
    await seed.update("mcp_servers", randomUUID(), {
      name: "systemserver",
      type: "openapi",
      url: `${upstream.baseUrl}/v1`,
      spec: systemServerSpec(),
      auth_type: "bearer",
      secret: "gateway-held-mcp-secret",
      forward_client_headers: [CUSTOM_HEADER, "authorization"],
    });
    await seed.update("mcp_servers", randomUUID(), {
      name: "nativeserver",
      type: "mcp",
      url: mcpUpstream.url,
      forward_client_headers: [CUSTOM_HEADER, "*"],
    });

    await seed.createApiKey({
      key_hash: CALLER_HASH,
      allowed_models: ["*"],
      allowed_routes: ["*"],
      mcp_access: { allow: ["*"] },
    });

    await waitConfigPropagation(async () => {
      const res = await chat(FORWARD_MODEL);
      await res.text();
      return res.status === 200;
    });
  }, 120_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await anthropicUpstream?.close();
    await mcpUpstream?.close();
  });

  test("a named header reaches an LLM upstream that opted in", async () => {
    if (!etcdReachable) return;
    const mark = upstream!.receivedRequests.length;

    await chat(FORWARD_MODEL, { [CUSTOM_HEADER]: "carried.verbatim" }).then((r) =>
      bodyOf(r, "/v1/chat/completions"),
    );

    expect(since(mark).at(-1)!.headers[CUSTOM_HEADER]).toBe("carried.verbatim");
  });

  test("an upstream nobody opted in receives nothing", async () => {
    if (!etcdReachable) return;
    const mark = upstream!.receivedRequests.length;

    await chat(PLAIN_MODEL, { [CUSTOM_HEADER]: "carried.verbatim" }).then((r) =>
      bodyOf(r, "/v1/chat/completions on the un-opted-in key"),
    );

    const seen = since(mark).at(-1)!;
    expect(seen.headers[CUSTOM_HEADER]).toBeUndefined();
    expect(seen.headers.authorization).toBe(`Bearer ${GATEWAY_SECRET}`);
  });

  test("a forwarded credential replaces the gateway's, and rides alone", async () => {
    if (!etcdReachable) return;
    const mark = upstream!.receivedRequests.length;

    // The caller authenticates to the GATEWAY with its own AISIX key; the
    // operator has declared that this upstream reads the caller's
    // credential rather than the ProviderKey's.
    await chat(FORWARD_MODEL).then((r) => bodyOf(r, "credential-slot forward"));

    const seen = since(mark).at(-1)!;
    expect(seen.headers.authorization).toBe(`Bearer ${CALLER_KEY}`);
    // Single-valued: not `Bearer sk-mock..., Bearer sk-fwd...`.
    expect(seen.headers.authorization).not.toContain(GATEWAY_SECRET);
  });

  test("the Anthropic bridge forwards too, into its own credential slot", async () => {
    if (!etcdReachable) return;
    const mark = anthropicUpstream!.receivedRequests.length;

    await chat(ANTHROPIC_MODEL, {
      [CUSTOM_HEADER]: "carried.verbatim",
      "x-api-key": "callers-own-api-key",
    }).then((r) => bodyOf(r, "the Anthropic bridge on /v1/chat/completions"));

    const seen = anthropicUpstream!.receivedRequests.slice(mark).at(-1)!;
    expect(seen.headers[CUSTOM_HEADER]).toBe("carried.verbatim");
    expect(seen.headers["x-api-key"]).toBe("callers-own-api-key");
    // The bridge still selects the wire format it decodes.
    expect(seen.headers["anthropic-version"]).toBeDefined();
  });

  test("a passthrough route's list overrides its strip set", async () => {
    if (!etcdReachable) return;
    const mark = upstream!.receivedRequests.length;

    // Control first: without the field, `gateway_key` strips the caller's
    // `Authorization` and `strip_headers` strips the custom slot, so the
    // upstream sees the ProviderKey's own credential instead.
    await fetch(`${app!.proxyUrl}${ROUTE_PREFIX}-plain/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_KEY}`,
        "content-type": "application/json",
        [CUSTOM_HEADER]: "carried.verbatim",
      },
      body: JSON.stringify({ model: "gpt-4o-mini", messages: [] }),
    }).then((r) => bodyOf(r, "the route with no forward"));

    const stripped = since(mark).at(-1)!;
    expect(stripped.headers[CUSTOM_HEADER]).toBeUndefined();
    expect(stripped.headers.authorization).toBe(`Bearer ${GATEWAY_SECRET}`);

    const mark2 = upstream!.receivedRequests.length;
    await fetch(`${app!.proxyUrl}${ROUTE_PREFIX}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_KEY}`,
        "content-type": "application/json",
        [CUSTOM_HEADER]: "carried.verbatim",
      },
      body: JSON.stringify({ model: "gpt-4o-mini", messages: [] }),
    }).then((r) => bodyOf(r, "the route with a forward"));

    const forwarded = since(mark2).at(-1)!;
    expect(forwarded.headers[CUSTOM_HEADER]).toBe("carried.verbatim");
    // The consumed credential slot too — and single-valued, so the
    // injected ProviderKey credential stood aside rather than joining it.
    expect(forwarded.headers.authorization).toBe(`Bearer ${CALLER_KEY}`);
    expect(forwarded.headers.authorization).not.toContain(GATEWAY_SECRET);
  });

  test("a REST system server exposed as MCP tools receives them", async () => {
    if (!etcdReachable) return;
    const mark = upstream!.receivedRequests.length;

    await callTool("systemserver__listmodels", {
      [CUSTOM_HEADER]: "carried.verbatim",
    });

    const toolCall = since(mark).find((r) => r.path.endsWith("/models"));
    expect(toolCall, "the generated tool must reach the REST upstream").toBeDefined();
    expect(toolCall!.headers[CUSTOM_HEADER]).toBe("carried.verbatim");
    // The caller's own credential took the slot `auth_type: bearer` would
    // have filled with `gateway-held-mcp-secret`.
    expect(toolCall!.headers.authorization).toBe(`Bearer ${CALLER_KEY}`);
    expect(toolCall!.headers.authorization).not.toContain("gateway-held-mcp-secret");
  });

  test("a native MCP upstream receives them, minus the session slots", async () => {
    if (!etcdReachable) return;
    const mark = mcpUpstream!.receivedHeaders.length;

    await callTool("nativeserver__echo", { [CUSTOM_HEADER]: "carried.verbatim" });

    const seen = mcpUpstream!.receivedHeaders.slice(mark);
    expect(seen.length, "the tool call must reach the MCP upstream").toBeGreaterThan(0);
    expect(seen.some((h) => h[CUSTOM_HEADER] === "carried.verbatim")).toBe(true);
    // `*` sweeps in everything else, but never the session this gateway
    // holds with the CALLER: an MCP upstream rejects a foreign value for
    // it outright, so forwarding would break the connection, not just
    // misidentify it. Every request carries the upstream's own id.
    const callerSession = seen.find(
      (h) => h["mcp-session-id"] !== undefined && h["mcp-session-id"] === "callers-own",
    );
    expect(callerSession).toBeUndefined();
  });

  test("the gateway's own namespace and trace context resist a glob", async () => {
    if (!etcdReachable) return;
    const mark = upstream!.receivedRequests.length;

    await chat(FORWARD_MODEL, {
      // Every one of these MATCHES a configured pattern — `x-*` the
      // first three, `trace*` the last — so what stops them is the rule
      // under test, not a pattern that failed to fire.
      "x-aisix-routing-tags": "spoofed-by-caller",
      "x-stainless-lang": "js",
      "x-allowed": "yes",
      traceparent: "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
    }).then((r) => bodyOf(r, "the glob case"));

    const seen = since(mark).at(-1)!;
    expect(seen.headers["x-allowed"]).toBe("yes");
    // The gateway's own assertions cannot be forged through the forward.
    expect(seen.headers["x-aisix-routing-tags"]).toBeUndefined();
    expect(seen.headers["x-stainless-lang"]).toBeUndefined();
    // Trace context needs its own name: a glob is not consent to graft the
    // caller's trace onto the provider's telemetry.
    expect(seen.headers.traceparent).toBeUndefined();
  });
});
