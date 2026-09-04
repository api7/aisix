import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startA2aUpstream,
  startMcpUpstream,
  startOpenAiUpstream,
  waitConfigPropagation,
  type A2aUpstream,
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
//   4. /a2a/*            — the `a2a_agent` field, on the JSON-RPC call,
//                          its streaming variant, and the agent-card
//                          fetch, which is an upstream hop of its own.
//
// And beside them, the sub-dispatch paths of the model kinds that do NOT
// go through the one convergence point the faces above share: an
// ensemble's panel members and its judge each reach a DIFFERENT upstream,
// so each obeys ITS OWN ProviderKey's list rather than the ensemble
// entry's — the entry has no ProviderKey at all.
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
const ENSEMBLE_MODEL = "fwd-ensemble-model";
const ENSEMBLE_MEMBER_A = "fwd-ensemble-member-a";
const ENSEMBLE_MEMBER_B = "fwd-ensemble-member-b";
const ENSEMBLE_JUDGE = "fwd-ensemble-judge";
const ROUTE_PREFIX = "/passthrough/fwd";
const A2A_AGENT = "fwdagent";
const A2A_PLAIN_AGENT = "fwdplainagent";
const A2A_GATEWAY_SECRET = "gateway-held-a2a-secret";
const GATEWAY_KEY_HEADER = "x-gw-key";
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
  let a2aUpstream: A2aUpstream | undefined;
  let etcdReachable = false;

  const since = (mark: number): ReceivedRequest[] =>
    upstream!.receivedRequests.slice(mark);

  /** How many times `name` arrived on the wire, occurrences not values. */
  const occurrences = (req: { headerNames: string[] }, name: string): number =>
    req.headerNames.filter((n) => n === name).length;

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
    // No `token`: this stub gates on the gateway's bearer when given one, and
    // the point here is to READ what arrived rather than to be refused for it.
    a2aUpstream = await startA2aUpstream();
    app = await spawnApp({});
    const seed = new SeedClient(etcd, app.etcdPrefix);

    // The upstream the operator opted in: a custom header, the caller's
    // own credential slot, and a glob that must NOT sweep trace context.
    const forwardingPk = await seed.createProviderKey({
      display_name: "fwd-pk",
      api_key: GATEWAY_SECRET,
      api_base: `${upstream.baseUrl}/v1`,
      request: {
        // `X-*` and `Trace*` deliberately in the spelling an operator's
        // own docs use: header names are case-insensitive on the wire, so
        // a matcher that compared case-sensitively would forward nothing
        // under either.
        forward_client_headers: [CUSTOM_HEADER, "authorization", "X-*", "Trace*"],
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

    // An ensemble whose panel members and judge all sit behind the
    // opted-in ProviderKey. `ProxyModelCaller::call` and the streaming
    // judge build their own dispatch contexts instead of reusing the
    // entry's, which is the shape this repo's model-kind rule calls the
    // most-repeated silent gap.
    for (const member of [ENSEMBLE_MEMBER_A, ENSEMBLE_MEMBER_B, ENSEMBLE_JUDGE]) {
      await seed.createModel({
        display_name: member,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: forwardingPk.id,
      });
    }
    await seed.createModel({
      display_name: ENSEMBLE_MODEL,
      ensemble: {
        panel: [{ model: ENSEMBLE_MEMBER_A }, { model: ENSEMBLE_MEMBER_B }],
        judge: { model: ENSEMBLE_JUDGE },
        min_responses: 2,
      },
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
    // Anthropic under `inject`: the route injects `anthropic-version`,
    // which is in no strip set, so a caller sending its own (every
    // Anthropic SDK does) is the duplicate-header case.
    const anthropicRoutePk = await seed.createProviderKey({
      display_name: "fwd-route-anthropic-pk",
      api_key: GATEWAY_SECRET,
      provider: "anthropic",
      api_base: upstream.baseUrl,
    });
    await seed.createPassthroughRoute({
      name: "fwd-route-anthropic",
      path_prefix: `${ROUTE_PREFIX}-anthropic`,
      target_url: upstream.baseUrl,
      auth_mode: "gateway_key",
      credential_mode: "inject",
      provider_key_id: anthropicRoutePk.id,
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

    // Two A2A agents behind one stub: one opted in, one with the default.
    // `"*"` beside the named entries is what proves the version announcement
    // resists a glob rather than simply never matching a pattern.
    await seed.update("a2a_agents", randomUUID(), {
      name: A2A_AGENT,
      url: a2aUpstream.url,
      protocol_version: "1.0",
      auth_type: "bearer",
      secret: A2A_GATEWAY_SECRET,
      forward_client_headers: [CUSTOM_HEADER, "authorization", "*"],
      enabled: true,
    });
    await seed.update("a2a_agents", randomUUID(), {
      name: A2A_PLAIN_AGENT,
      url: a2aUpstream.url,
      protocol_version: "1.0",
      auth_type: "bearer",
      secret: A2A_GATEWAY_SECRET,
      enabled: true,
    });

    const callerKey = await seed.createApiKey({
      key_hash: CALLER_HASH,
      allowed_models: ["*"],
      allowed_routes: ["*"],
      allowed_agents: ["*"],
      mcp_access: { allow: ["*"] },
    });

    // A route that names its OWN gateway-credential slot, and one that
    // names none. `x-*` is the same pattern on both: what differs is
    // whether the route declared a slot for it to have to leave alone.
    //
    // Their ProviderKey strips all three `x-` names below, which is what
    // makes the assertions mean anything: a passthrough route relays
    // whatever it does not strip, so a header outside the strip set
    // arrives upstream whether or not a pattern matched it.
    const slotPk = await seed.createProviderKey({
      display_name: "fwd-slot-pk",
      api_key: GATEWAY_SECRET,
      api_base: upstream.baseUrl,
      strip_headers: [
        "authorization",
        GATEWAY_KEY_HEADER,
        "x-end-user",
        CUSTOM_HEADER,
      ],
    });
    await seed.createPassthroughRoute({
      name: "fwd-route-headerkey",
      path_prefix: `${ROUTE_PREFIX}-headerkey`,
      target_url: upstream.baseUrl,
      auth_mode: "header_key",
      auth_header_name: GATEWAY_KEY_HEADER,
      credential_mode: "inject",
      provider_key_id: slotPk.id,
      identity_header: "x-end-user",
      forward_client_headers: ["x-*"],
    });
    await seed.createPassthroughRoute({
      name: "fwd-route-anon",
      path_prefix: `${ROUTE_PREFIX}-anon`,
      target_url: upstream.baseUrl,
      auth_mode: "anonymous",
      anonymous_key_id: callerKey.id,
      source_cidrs: ["127.0.0.0/8", "::1/128"],
      credential_mode: "inject",
      provider_key_id: slotPk.id,
      forward_client_headers: ["x-*"],
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
    await a2aUpstream?.close();
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
    // And ALONE. Counting occurrences rather than reading the collapsed
    // value is the whole point: node keeps only the FIRST `authorization`,
    // so a second one appended behind it would be invisible here.
    expect(occurrences(seen, "authorization")).toBe(1);
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
        "x-aisix-routing-tags": "spoofed-by-caller",
      },
      body: JSON.stringify({ model: "gpt-4o-mini", messages: [] }),
    }).then((r) => bodyOf(r, "the route with a forward"));

    const forwarded = since(mark2).at(-1)!;
    expect(forwarded.headers[CUSTOM_HEADER]).toBe("carried.verbatim");
    // A route forwards by default, so the gateway's own namespace has to
    // be refused explicitly — nothing in the strip set names it beyond
    // `x-aisix-request-id`, and a relayed copy would forge a gateway
    // assertion at the upstream.
    expect(forwarded.headers["x-aisix-routing-tags"]).toBeUndefined();
    // The consumed credential slot too. Passthrough relays the caller's
    // copy FIRST and appends the gateway's after it, so the collapsed
    // value would read correctly even if the suppression failed entirely
    // — only the occurrence count shows the ProviderKey credential really
    // stood aside rather than joining it on the wire.
    expect(forwarded.headers.authorization).toBe(`Bearer ${CALLER_KEY}`);
    expect(occurrences(forwarded, "authorization")).toBe(1);
  });

  test("a glob never sweeps the slots a route named for itself", async () => {
    if (!etcdReachable) return;
    const mark = upstream!.receivedRequests.length;

    // `header_key`: the caller's AISIX key arrives in the route's own
    // `x-gw-key`, and the end-user identity in its own `x-end-user`.
    // Both match `x-*`, and neither is on the credential list every
    // surface shares — the route schema in fact forbids those names —
    // so only the per-route rule keeps them off the wire.
    await fetch(`${app!.proxyUrl}${ROUTE_PREFIX}-headerkey/v1/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        [GATEWAY_KEY_HEADER]: CALLER_KEY,
        "x-end-user": "alice@example.com",
        [CUSTOM_HEADER]: "recovered",
      },
      body: JSON.stringify({ model: "gpt-4o-mini", messages: [] }),
    }).then((r) => bodyOf(r, "the header_key route"));

    const seen = since(mark).at(-1)!;
    expect(seen.headers[GATEWAY_KEY_HEADER]).toBeUndefined();
    expect(seen.headers["x-end-user"]).toBeUndefined();
    // The SAME `x-*` recovers a stripped header that is not a slot. All
    // three are in this ProviderKey's strip set, so the pattern was asked
    // about each of them — without that, a header the route never strips
    // arrives upstream regardless and proves nothing.
    expect(seen.headers[CUSTOM_HEADER]).toBe("recovered");
  });

  test("a route that names no slot of its own is unchanged", async () => {
    if (!etcdReachable) return;
    const mark = upstream!.receivedRequests.length;

    // `anonymous` reads no inbound credential at all, so nothing joins
    // the exact-name set and `x-*` means exactly what it always did —
    // including for a header another route would have treated as a slot.
    await fetch(`${app!.proxyUrl}${ROUTE_PREFIX}-anon/v1/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        [GATEWAY_KEY_HEADER]: "not-a-slot-on-this-route",
        [CUSTOM_HEADER]: "recovered",
      },
      body: JSON.stringify({ model: "gpt-4o-mini", messages: [] }),
    }).then((r) => bodyOf(r, "the anonymous route"));

    const seen = since(mark).at(-1)!;
    // `x-gw-key` is in this ProviderKey's strip set, so the pattern IS
    // asked about it — and answers yes, because THIS route declared no
    // slot. Widen the narrowing to a global list and this line fails.
    expect(seen.headers[GATEWAY_KEY_HEADER]).toBe("not-a-slot-on-this-route");
    expect(seen.headers[CUSTOM_HEADER]).toBe("recovered");
    // And the shared rule is untouched: the ProviderKey's credential
    // still rides alone.
    expect(seen.headers.authorization).toBe(`Bearer ${GATEWAY_SECRET}`);
    expect(occurrences(seen, "authorization")).toBe(1);
  });

  test("a passthrough route never doubles a header it injects", async () => {
    if (!etcdReachable) return;
    const mark = upstream!.receivedRequests.length;

    // `anthropic-version` is in no strip set, so the caller's own copy is
    // relayed by the default forward — the route must then not append a
    // second value behind it and leave the upstream to pick.
    await fetch(`${app!.proxyUrl}${ROUTE_PREFIX}-anthropic/v1/messages`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_KEY}`,
        "content-type": "application/json",
        "anthropic-version": "2023-01-01",
      },
      body: JSON.stringify({ model: "claude-sonnet-4", messages: [] }),
    }).then((r) => bodyOf(r, "the anthropic passthrough route"));

    const seen = since(mark).at(-1)!;
    expect(occurrences(seen, "anthropic-version")).toBe(1);
    expect(seen.headers["anthropic-version"]).toBe("2023-01-01");
    // And the gateway's own credential still authenticates the call.
    expect(seen.headers["x-api-key"]).toBe(GATEWAY_SECRET);
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

    // The caller really sends the session slots, so the assertion below
    // fails if the block is removed rather than because nothing matched.
    await callTool("nativeserver__echo", {
      [CUSTOM_HEADER]: "carried.verbatim",
      "mcp-session-id": "callers-own-session",
      "last-event-id": "callers-own-event",
    });

    const seen = mcpUpstream!.receivedHeaders.slice(mark);
    expect(seen.length, "the tool call must reach the MCP upstream").toBeGreaterThan(0);
    expect(seen.some((h) => h[CUSTOM_HEADER] === "carried.verbatim")).toBe(true);
    // `*` sweeps in everything else, but never the session this gateway
    // holds with the CALLER: it names a session the upstream never issued,
    // and rmcp refuses a foreign `mcp-session-id` outright — the
    // connection fails rather than degrades.
    for (const h of seen) {
      // `toContain`, not `toBe`: node joins a repeated header, so a
      // relayed copy could arrive as `callers-own-session,<real id>` and
      // an equality check would call that clean.
      expect(h["mcp-session-id"] ?? "").not.toContain("callers-own-session");
      expect(h["last-event-id"] ?? "").not.toContain("callers-own-event");
    }
  });

  /** A JSON-RPC call to an agent through the gateway. */
  const a2aCall = async (
    agent: string,
    method: string,
    headers: Record<string, string> = {},
  ): Promise<Response> =>
    fetch(`${app!.proxyUrl}/a2a/${agent}`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_KEY}`,
        "content-type": "application/json",
        ...headers,
      },
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: "fwd-1",
        method,
        params: {
          message: {
            role: "user",
            parts: [{ kind: "text", text: "probe" }],
            messageId: "m-fwd",
          },
        },
      }),
    });

  /** The last request the A2A stub saw on `method`, since `mark`. */
  const a2aSince = (mark: number, httpMethod: "GET" | "POST") =>
    a2aUpstream!.requests.slice(mark).filter((r) => r.httpMethod === httpMethod);

  test("an A2A agent receives a named header, and a forwarded credential rides alone", async () => {
    if (!etcdReachable) return;
    const mark = a2aUpstream!.requests.length;

    await a2aCall(A2A_AGENT, "message/send", {
      [CUSTOM_HEADER]: "carried.verbatim",
    }).then((r) => bodyOf(r, "/a2a message/send"));

    const seen = a2aSince(mark, "POST").at(-1)!;
    expect(seen.headers[CUSTOM_HEADER]).toBe("carried.verbatim");
    // The caller authenticates to the GATEWAY with its own AISIX key; the
    // operator has declared that this agent reads the caller's credential
    // rather than the one the gateway holds for it.
    expect(seen.headers.authorization).toBe(`Bearer ${CALLER_KEY}`);
    // And ALONE: node keeps only the FIRST `authorization`, so a gateway
    // credential appended behind the caller's would be invisible in the
    // collapsed value and only the occurrence count shows it stood aside.
    expect(occurrences(seen, "authorization")).toBe(1);
    expect(seen.headers.authorization).not.toContain(A2A_GATEWAY_SECRET);
  });

  test("an A2A agent nobody opted in receives nothing", async () => {
    if (!etcdReachable) return;
    const mark = a2aUpstream!.requests.length;

    await a2aCall(A2A_PLAIN_AGENT, "message/send", {
      [CUSTOM_HEADER]: "carried.verbatim",
    }).then((r) => bodyOf(r, "/a2a on the un-opted-in agent"));

    const seen = a2aSince(mark, "POST").at(-1)!;
    expect(seen.headers[CUSTOM_HEADER]).toBeUndefined();
    expect(seen.headers.authorization).toBe(`Bearer ${A2A_GATEWAY_SECRET}`);
  });

  test("the A2A agent-card fetch forwards them too", async () => {
    if (!etcdReachable) return;
    const mark = a2aUpstream!.requests.length;

    // Card discovery is an upstream hop of its own, built at a different call
    // site than the JSON-RPC one — an agent that gates discovery on the end
    // user's own credential is exactly what this capability is for.
    await fetch(
      `${app!.proxyUrl}/a2a/${A2A_AGENT}/.well-known/agent-card.json`,
      { headers: { authorization: `Bearer ${CALLER_KEY}`, [CUSTOM_HEADER]: "carried.verbatim" } },
    ).then((r) => bodyOf(r, "the A2A agent card"));

    const card = a2aSince(mark, "GET").at(-1)!;
    expect(card.headers[CUSTOM_HEADER]).toBe("carried.verbatim");
    expect(card.headers.authorization).toBe(`Bearer ${CALLER_KEY}`);
    expect(occurrences(card, "authorization")).toBe(1);
  });

  test("an A2A streaming call forwards them too", async () => {
    if (!etcdReachable) return;
    const mark = a2aUpstream!.requests.length;

    // `message/stream` opens its upstream request at a different call site
    // than the buffered one, and only this path exercises it.
    await a2aCall(A2A_AGENT, "message/stream", {
      [CUSTOM_HEADER]: "carried.verbatim",
    }).then((r) => bodyOf(r, "/a2a message/stream"));

    const seen = a2aSince(mark, "POST").at(-1)!;
    expect(seen.headers[CUSTOM_HEADER]).toBe("carried.verbatim");
    expect(seen.headers.authorization).toBe(`Bearer ${CALLER_KEY}`);
  });

  test("a glob never sweeps the A2A version the gateway announces", async () => {
    if (!etcdReachable) return;
    const mark = a2aUpstream!.requests.length;

    await a2aCall(A2A_AGENT, "message/send", {
      // Both MATCH the configured `*`, so what separates them is the rule
      // under test rather than a pattern that failed to fire.
      "a2a-version": "0.3",
      "x-allowed": "yes",
    }).then((r) => bodyOf(r, "the A2A glob case"));

    const seen = a2aSince(mark, "POST").at(-1)!;
    expect(seen.headers["x-allowed"]).toBe("yes");
    // The version is the gateway's own announcement of the agent's pinned
    // wire format. A relayed copy would let the caller pick the envelope
    // shape the agent answers in, or make it refuse the call outright.
    expect(seen.version).toBe("1.0");
    expect(occurrences(seen, "a2a-version")).toBe(1);
  });

  test("an ensemble's panel and judge each forward on their own upstream", async () => {
    if (!etcdReachable) return;
    const mark = upstream!.receivedRequests.length;

    await chat(ENSEMBLE_MODEL, { [CUSTOM_HEADER]: "carried.verbatim" }).then((r) =>
      bodyOf(r, "the ensemble entry"),
    );

    // Two panel members plus the judge, each a separate upstream call
    // built by `ProxyModelCaller::call` rather than by the single
    // dispatch chokepoint.
    const calls = since(mark);
    expect(calls.length, "panel of two plus a judge").toBe(3);
    for (const call of calls) {
      expect(call.headers[CUSTOM_HEADER]).toBe("carried.verbatim");
      // Each sub-call reads its OWN ProviderKey's list, which here also
      // names the credential slot — the ensemble entry has no
      // ProviderKey to inherit one from.
      expect(call.headers.authorization).toBe(`Bearer ${CALLER_KEY}`);
    }
  });

  test("a streamed ensemble judge forwards too", async () => {
    if (!etcdReachable) return;
    const mark = upstream!.receivedRequests.length;

    // The streaming judge is dispatched from a different call site than
    // the buffered one, and only this path exercises it.
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_KEY}`,
        "content-type": "application/json",
        [CUSTOM_HEADER]: "carried.verbatim",
      },
      body: JSON.stringify({
        model: ENSEMBLE_MODEL,
        messages: [{ role: "user", content: "probe" }],
        stream: true,
      }),
    });
    await bodyOf(res, "the streamed ensemble");

    const calls = since(mark);
    expect(calls.length).toBe(3);
    for (const call of calls) {
      expect(call.headers[CUSTOM_HEADER]).toBe("carried.verbatim");
    }
  });

  test("the gateway's own namespace and trace context resist a glob", async () => {
    if (!etcdReachable) return;
    const mark = upstream!.receivedRequests.length;

    await chat(FORWARD_MODEL, {
      // Every one of these MATCHES a configured pattern — `x-*` the
      // first three and the SigV4 trio, `trace*` the last — so what stops
      // them is the rule under test, not a pattern that failed to fire.
      "x-aisix-routing-tags": "spoofed-by-caller",
      "x-stainless-lang": "js",
      "x-allowed": "yes",
      "x-amz-security-token": "FQoGZXIvYXdzE-caller-session",
      "x-amz-date": "20260903T120000Z",
      "x-amz-content-sha256": "e3b0c44298fc1c14",
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
    // A credential slot needs its own name for the same reason, and this
    // is the shape the bug actually took: the caller's live AWS session
    // token arriving at an upstream that has no business holding one.
    // This upstream is not Bedrock, so nothing downstream would strip it.
    for (const name of ["x-amz-security-token", "x-amz-date", "x-amz-content-sha256"]) {
      expect(seen.headers[name], `${name} reached the upstream under a glob`).toBeUndefined();
    }
  });
});
