import { createHash, randomUUID } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  agentClaims,
  EtcdClient,
  pickFreePort,
  ProxyClient,
  SeedClient,
  spawnApp,
  startMcpUpstream,
  startMockIdp,
  startMockSls,
  startOpenAiUpstream,
  startRestUpstream,
  waitConfigPropagation,
  waitForSlsLog,
  type McpUpstream,
  type MockIdp,
  type MockSls,
  type OpenAiUpstream,
  type RestUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: `auth_type: oauth2_token_exchange` with `token_exchange_profile:
// entra_obo` — the gateway exchanges the caller's own verified token for
// an upstream token (Microsoft Entra ID On-Behalf-Of) and presents that
// upstream instead.
//
// A mock Entra token endpoint records every exchange it receives. Pinned:
//
//   1. The exchange form is exactly the OBO `jwt-bearer` grant with the
//      caller's JWT as `assertion`; the minted token reaches the upstream
//      and the caller's JWT never does; a second call reuses the cached
//      token; a different caller gets its own exchange.
//   2. Only a token the gateway verified through a JWT trust provider on
//      the request is a subject. An API-key caller — and a caller admitted
//      with `x-aisix-api-key` whose `Authorization` holds a perfectly valid
//      JWT the gateway never verified — gets 401 on the single-server
//      endpoint, and the token endpoint is never contacted.
//   3. The identity provider's answer maps to the status the caller sees:
//      the subject rejected → 401, the gateway's own client refused → 500,
//      the provider failing or unreachable → 503.
//   4. An upstream 401 drops the cached token and the exchange is redone,
//      once.
//   5. A `type: openapi` server exchanges the same way.
//   6. Throttling (429) and a request timeout (408) at the token endpoint
//      are the provider being unavailable (503), not the caller's token
//      being rejected.
//   7. Concurrent first requests from one caller share one exchange, and
//      its outcome; a failed exchange is not reused by the next request.
//   8. A rejection carrying a claims challenge (Conditional Access
//      step-up) reaches the caller in the 401's `WWW-Authenticate`.
//   9. An exchange slower than the server's `timeout_ms` is a 503 within
//      that timeout — on `initialize` and on a tool call, for `type: mcp`
//      and `type: openapi` alike — and the exchange still completes for
//      the requests that come after.
//  10. A re-exchange that fails after the upstream rejected the previous
//      token is recorded as a call that reached the upstream.
//
// Reference: Microsoft identity platform OAuth 2.0 On-Behalf-Of flow,
// <https://learn.microsoft.com/entra/identity-platform/v2-oauth2-on-behalf-of-flow>.

const SCOPE = "api://upstream-api/.default";
const CLIENT_ID = "gateway-client-id";
const CLIENT_SECRET = "gateway-client-secret";
const KEY_1 = "sk-obo-agent-1";
const KEY_2 = "sk-obo-agent-2";
const KEY_3 = "sk-obo-agent-3";
const hash = (key: string) => createHash("sha256").update(key).digest("hex");
const STEP_UP_CLAIMS = '{"access_token":{"capolids":{"essential":true,"values":["c1"]}}}';
// How long a `hang-` token endpoint takes, and the `timeout_ms` of the
// servers that use one.
const HANG_MS = 3000;
const SHORT_TIMEOUT_MS = 400;
const CREDENTIAL_REF = "obo";
const LOGSTORE = "obo-usage";

/** A mock Entra token endpoint: `/<behavior>/token`, counted per path. */
interface MockEntra {
  url: string;
  forms: Record<string, Record<string, string>[]>;
  hits(path: string): number;
  /** How many responses on `path` have been written in full. */
  answered(path: string): number;
  close(): Promise<void>;
}

async function startMockEntra(): Promise<MockEntra> {
  const forms: Record<string, Record<string, string>[]> = {};
  const answered: Record<string, number> = {};
  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      const path = (req.url ?? "").split("?")[0];
      const seen = (forms[path] ??= []);
      seen.push(Object.fromEntries(new URLSearchParams(raw)));
      const segment = path.split("/")[1] ?? "";
      // `slow-<behavior>` answers like `<behavior>`, late enough for
      // concurrent requests to overlap; `hang-<behavior>` later than any
      // short server timeout.
      const slow = segment.startsWith("slow-");
      const hang = segment.startsWith("hang-");
      const behavior = slow || hang ? segment.slice(5) : segment;
      const json = (status: number, body: unknown) =>
        setTimeout(
          () =>
            res
              .writeHead(status, { "content-type": "application/json" })
              .end(JSON.stringify(body), () => {
                answered[path] = (answered[path] ?? 0) + 1;
              }),
          hang ? HANG_MS : slow ? 500 : 0,
        );
      if (behavior.startsWith("mint-once") && seen.length > 1) {
        // Mints once, then is down.
        json(503, { error: "temporarily_unavailable" });
      } else if (behavior.startsWith("mint")) {
        json(200, {
          token_type: "Bearer",
          scope: SCOPE,
          expires_in: 3599,
          ext_expires_in: 3599,
          access_token: `${behavior}-${seen.length}`,
        });
      } else if (behavior === "reject") {
        json(400, {
          error: "invalid_grant",
          error_description: "AADSTS50013: Assertion failed signature validation.",
          error_codes: [50013],
        });
      } else if (behavior === "stepup") {
        json(400, {
          error: "invalid_grant",
          error_description: "AADSTS50076: multi-factor authentication required.",
          error_codes: [50076],
          suberror: "basic_action",
          claims: STEP_UP_CLAIMS,
        });
      } else if (behavior === "throttle") {
        json(429, { error: "invalid_request", error_description: "AADSTS90055: throttled." });
      } else if (behavior === "timeout") {
        json(408, { error: "invalid_request", error_description: "AADSTS: request timeout." });
      } else if (behavior === "fault") {
        json(401, {
          error: "invalid_client",
          error_description: "AADSTS7000215: Invalid client secret provided.",
        });
      } else {
        json(503, { error: "temporarily_unavailable" });
      }
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("no address");
  return {
    url: `http://127.0.0.1:${address.port}`,
    forms,
    hits: (path) => forms[path]?.length ?? 0,
    answered: (path) => answered[path] ?? 0,
    close: () => new Promise((resolve) => server.close(() => resolve())),
  };
}

function itemSpec(): Record<string, unknown> {
  return {
    openapi: "3.0.0",
    info: { title: "obo-items", version: "1" },
    paths: {
      "/items/{id}": {
        get: {
          operationId: "getitem",
          parameters: [{ name: "id", in: "path", required: true, schema: { type: "string" } }],
          responses: { "200": { description: "ok" } },
        },
      },
    },
  };
}

function restSpec(): Record<string, unknown> {
  return {
    openapi: "3.0.0",
    info: { title: "obo-rest", version: "1" },
    paths: {
      "/models": {
        get: { operationId: "listmodels", responses: { "200": { description: "ok" } } },
      },
    },
  };
}

describe("oauth2_token_exchange (entra_obo) e2e", () => {
  let app: SpawnedApp | undefined;
  let idp: MockIdp | undefined;
  let entra: MockEntra | undefined;
  let upstream: McpUpstream | undefined;
  let retryUpstream: McpUpstream | undefined;
  let rest: OpenAiUpstream | undefined;
  let rejectingUpstream: McpUpstream | undefined;
  let items: RestUpstream | undefined;
  let sls: MockSls | undefined;
  let etcdReachable = false;
  let jwt1 = "";
  let jwt2 = "";
  let jwt3 = "";

  const initialize = {
    jsonrpc: "2.0",
    id: 1,
    method: "initialize",
    params: {
      protocolVersion: "2025-11-25",
      capabilities: {},
      clientInfo: { name: "obo-e2e", version: "0.1" },
    },
  };

  const mcpPost = (path: string, body: unknown, headers: Record<string, string>) =>
    fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        accept: "application/json, text/event-stream",
        ...headers,
      },
      body: JSON.stringify(body),
    });

  const bearer = (token: string) => ({ authorization: `Bearer ${token}` });

  /** initialize → initialized → tools/call; returns each response. */
  const session = async (
    path: string,
    tool: string,
    headers: Record<string, string>,
  ): Promise<Response[]> => {
    const out: Response[] = [];
    for (const body of [
      initialize,
      { jsonrpc: "2.0", method: "notifications/initialized" },
      { jsonrpc: "2.0", id: 2, method: "tools/call", params: { name: tool, arguments: { text: "hi" } } },
    ]) {
      const res = await mcpPost(path, body, headers);
      out.push(res);
      if (!res.ok) break;
    }
    return out;
  };

  const errorCode = async (res: Response): Promise<string | undefined> =>
    ((await res.json()) as { error?: { code?: string } }).error?.code;

  /** A lone `tools/call`, with how long the gateway took to answer it. */
  const toolCall = async (path: string, name: string, args: unknown, token: string) => {
    const started = Date.now();
    const res = await mcpPost(
      path,
      { jsonrpc: "2.0", id: 3, method: "tools/call", params: { name, arguments: args } },
      bearer(token),
    );
    const text = await res.text();
    return {
      status: res.status,
      text,
      elapsed: Date.now() - started,
      requestId: res.headers.get("x-aisix-request-id") ?? "",
    };
  };

  const usageRow = (requestId: string, what: string) =>
    waitForSlsLog(sls!, LOGSTORE, (l) => l.get("request_id") === requestId, what, 15_000);

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    idp = await startMockIdp();
    entra = await startMockEntra();
    upstream = await startMcpUpstream("obo");
    retryUpstream = await startMcpUpstream("obo-retry", {
      // The first token minted for this server is "revoked".
      rejectAuthorization: (value) => value === "Bearer mint-retry-1",
    });
    rest = await startOpenAiUpstream();
    // Rejects every token, so each call is answered with a 401.
    rejectingUpstream = await startMcpUpstream("obo-401", { rejectAuthorization: () => true });
    // Accepts only its own fixed token, so an exchanged one gets a 401.
    items = await startRestUpstream();
    sls = await startMockSls();
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "LTAI_mock_ak",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock_ak_secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "obo-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "aisix-e2e-obs",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
    });

    await seed.createOidcProvider({
      name: "corp-entra",
      issuer: idp.url,
      audiences: ["aisix-gateway"],
      jwks_uri: idp.jwksUrl,
    });

    const exchange = (behavior: string) => ({
      auth_type: "oauth2_token_exchange",
      token_exchange_profile: "entra_obo",
      token_url: `${entra!.url}/${behavior}/token`,
      client_id: CLIENT_ID,
      secret: CLIENT_SECRET,
      scopes: [SCOPE],
    });
    const unreachable = await pickFreePort();
    for (const [name, behavior, url] of [
      ["obo", "mint", upstream.url],
      ["oboretry", "mint-retry", retryUpstream.url],
      ["oboreject", "reject", upstream.url],
      ["obofault", "fault", upstream.url],
      ["obodown", "down", upstream.url],
      ["obothrottle", "throttle", upstream.url],
      ["obotimeout", "timeout", upstream.url],
      ["oboslow", "slow-mint-slow", upstream.url],
      ["oboslowreject", "slow-reject", upstream.url],
      ["obostepup", "stepup", upstream.url],
      ["obo401", "mint-once", rejectingUpstream.url],
    ] as const) {
      await seed.update("mcp_servers", randomUUID(), { name, type: "mcp", url, ...exchange(behavior) });
    }
    for (const [name, behavior] of [
      ["obohang", "hang-mint-a"],
      ["obohangcall", "hang-mint-b"],
    ] as const) {
      await seed.update("mcp_servers", randomUUID(), {
        name,
        type: "mcp",
        url: upstream.url,
        timeout_ms: SHORT_TIMEOUT_MS,
        ...exchange(behavior),
      });
    }
    await seed.update("mcp_servers", randomUUID(), {
      name: "oboresthang",
      type: "openapi",
      url: `${rest.baseUrl}/v1`,
      spec: restSpec(),
      timeout_ms: SHORT_TIMEOUT_MS,
      ...exchange("hang-mint-rest"),
    });
    await seed.update("mcp_servers", randomUUID(), {
      name: "oborest401",
      type: "openapi",
      url: items.baseUrl,
      spec: itemSpec(),
      ...exchange("mint-once-rest"),
    });
    await seed.update("mcp_servers", randomUUID(), {
      name: "obounreachable",
      type: "mcp",
      url: upstream.url,
      ...exchange("x"),
      token_url: `http://127.0.0.1:${unreachable}/token`,
    });
    await seed.update("mcp_servers", randomUUID(), {
      name: "oborest",
      type: "openapi",
      url: `${rest.baseUrl}/v1`,
      spec: restSpec(),
      ...exchange("mint-rest"),
    });

    for (const [key, subject] of [
      [KEY_1, "agent-1"],
      [KEY_2, "agent-2"],
    ]) {
      await seed.createApiKey({
        key_hash: hash(key),
        allowed_models: [],
        jwt_subject: subject,
        jwt_provider: "corp-entra",
        mcp_access: { allow: ["*"] },
      });
    }
    await seed.createApiKey({
      key_hash: hash(KEY_3),
      allowed_models: [],
      jwt_subject: "agent-3",
      jwt_provider: "corp-entra",
      mcp_access: { allow: ["oborest__*"] },
    });
    jwt3 = idp.sign(agentClaims(idp.url, { sub: "agent-3" }));
    jwt1 = idp.sign(agentClaims(idp.url, { sub: "agent-1" }));
    jwt2 = idp.sign(agentClaims(idp.url, { sub: "agent-2" }));
    const proxy = new ProxyClient(app.proxyUrl, KEY_2);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  }, 120_000);

  afterAll(async () => {
    await app?.exit();
    await idp?.close();
    await entra?.close();
    await upstream?.close();
    await retryUpstream?.close();
    await rest?.close();
    await rejectingUpstream?.close();
    await items?.close();
    await sls?.close();
  });

  test("a JWT caller's token is exchanged, the result sent upstream, and cached per caller", async () => {
    if (!etcdReachable) return;
    const mark = upstream!.receivedHeaders.length;

    const responses = await session("/mcp/obo", "echo", bearer(jwt1));
    for (const res of responses) expect(res.status, await res.clone().text()).toBeLessThan(300);
    const result = (await responses[2].json()) as { result?: { isError?: boolean } };
    expect(result.result?.isError ?? false).toBe(false);

    expect(entra!.hits("/mint/token")).toBe(1);
    expect(entra!.forms["/mint/token"][0]).toEqual({
      grant_type: "urn:ietf:params:oauth:grant-type:jwt-bearer",
      assertion: jwt1,
      scope: SCOPE,
      requested_token_use: "on_behalf_of",
      client_id: CLIENT_ID,
      client_secret: CLIENT_SECRET,
    });

    const seen = upstream!.receivedHeaders.slice(mark);
    expect(seen.length).toBeGreaterThan(0);
    for (const h of seen) {
      expect(h.authorization).toBe("Bearer mint-1");
      expect(JSON.stringify(h)).not.toContain(jwt1);
    }

    // Same caller again: served from the cache.
    await session("/mcp/obo", "echo", bearer(jwt1));
    await session("/mcp", "obo__echo", bearer(jwt1));
    expect(entra!.hits("/mint/token")).toBe(1);

    // Another caller never shares that entry.
    const mark2 = upstream!.receivedHeaders.length;
    const other = await session("/mcp/obo", "echo", bearer(jwt2));
    expect(other[2].status).toBe(200);
    expect(entra!.hits("/mint/token")).toBe(2);
    expect(entra!.forms["/mint/token"][1].assertion).toBe(jwt2);
    for (const h of upstream!.receivedHeaders.slice(mark2)) {
      expect(h.authorization).toBe("Bearer mint-2");
    }
  });

  test("a caller with no verified token gets 401 and nothing is exchanged", async () => {
    if (!etcdReachable) return;
    const before = entra!.hits("/mint/token");
    const upstreamBefore = upstream!.receivedHeaders.length;

    // An API key is not a token to exchange.
    const byKey = await mcpPost("/mcp/obo", initialize, bearer(KEY_1));
    expect(byKey.status).toBe(401);
    expect(await errorCode(byKey)).toBe("token_exchange_subject_missing");

    // Nor is an Authorization the gateway never verified, valid JWT or not,
    // when the gateway key came in its own header.
    const unverified = await mcpPost("/mcp/obo", initialize, {
      "x-aisix-api-key": KEY_1,
      ...bearer(jwt1),
    });
    expect(unverified.status).toBe(401);
    expect(await errorCode(unverified)).toBe("token_exchange_subject_missing");

    // The aggregated endpoint does not fail as a whole: the call to that
    // server fails, and nothing reaches its upstream.
    const aggregated = await session("/mcp", "obo__echo", bearer(KEY_1));
    expect(aggregated.map((r) => r.status)).toEqual([200, 202, 200]);
    const call = (await aggregated[2].json()) as { error?: unknown; result?: { isError?: boolean } };
    expect(call.error ?? call.result?.isError).toBeTruthy();

    expect(entra!.hits("/mint/token")).toBe(before);
    expect(upstream!.receivedHeaders.length).toBe(upstreamBefore);
  });

  test("a server outside the caller's grant is an ACL outcome, not a sign-in challenge", async () => {
    if (!etcdReachable) return;
    const before = entra!.hits("/mint/token");
    const init = await mcpPost("/mcp/obo", initialize, bearer(jwt3));
    expect(init.status, await init.clone().text()).toBe(200);
    const list = await mcpPost(
      "/mcp/obo",
      { jsonrpc: "2.0", id: 2, method: "tools/list" },
      bearer(jwt3),
    );
    expect(list.status, await list.clone().text()).toBe(200);
    const body = (await list.json()) as { result?: { tools?: unknown[] } };
    expect(body.result?.tools).toEqual([]);
    expect(entra!.hits("/mint/token")).toBe(before);
  });

  test("the identity provider's answer maps to the caller's status", async () => {
    if (!etcdReachable) return;
    const cases: [string, number, string][] = [
      ["/mcp/oboreject", 401, "token_exchange_subject_rejected"],
      ["/mcp/obofault", 500, "token_exchange_misconfigured"],
      ["/mcp/obodown", 503, "token_exchange_unavailable"],
      ["/mcp/obounreachable", 503, "token_exchange_unavailable"],
      ["/mcp/obothrottle", 503, "token_exchange_unavailable"],
      ["/mcp/obotimeout", 503, "token_exchange_unavailable"],
    ];
    for (const [path, status, code] of cases) {
      const res = await mcpPost(path, initialize, bearer(jwt1));
      const text = await res.text();
      expect(res.status, `${path}: ${text}`).toBe(status);
      expect(JSON.parse(text).error?.code, path).toBe(code);
      // The identity provider's own wording never reaches the caller.
      expect(text).not.toContain("AADSTS");
    }
    expect(entra!.forms["/reject/token"][0].assertion).toBe(jwt1);
  });

  test("an upstream 401 drops the exchanged token and exchanges again, once", async () => {
    if (!etcdReachable) return;
    const mark = retryUpstream!.receivedHeaders.length;
    const responses = await session("/mcp/oboretry", "echo", bearer(jwt1));
    for (const res of responses) expect(res.status, await res.clone().text()).toBeLessThan(300);
    const result = (await responses[2].json()) as { result?: { isError?: boolean } };
    expect(result.result?.isError ?? false).toBe(false);

    expect(entra!.hits("/mint-retry/token")).toBe(2);
    const auths = retryUpstream!.receivedHeaders.slice(mark).map((h) => h.authorization);
    expect(auths[0]).toBe("Bearer mint-retry-1");
    expect(auths.slice(1).length).toBeGreaterThan(0);
    for (const auth of auths.slice(1)) expect(auth).toBe("Bearer mint-retry-2");
  });

  test("an OpenAPI server exchanges the same way", async () => {
    if (!etcdReachable) return;
    const mark = rest!.receivedRequests.length;
    const responses = await session("/mcp", "oborest__listmodels", bearer(jwt2));
    expect(responses[2].status).toBe(200);
    const call = rest!.receivedRequests.slice(mark).find((r) => r.path.endsWith("/models"));
    expect(call, "the generated tool must reach the REST upstream").toBeDefined();
    expect(call!.headers.authorization).toBe("Bearer mint-rest-1");
    expect(JSON.stringify(call!.headers)).not.toContain(jwt2);
    expect(entra!.forms["/mint-rest/token"][0].assertion).toBe(jwt2);
  });

  test("concurrent first requests from one caller share one exchange", async () => {
    if (!etcdReachable) return;
    const N = 8;
    const burst = (path: string) =>
      Promise.all(Array.from({ length: N }, () => mcpPost(path, initialize, bearer(jwt1))));

    const minted = await burst("/mcp/oboslow");
    for (const res of minted) expect(res.status, await res.clone().text()).toBe(200);
    expect(entra!.hits("/slow-mint-slow/token")).toBe(1);

    // A failure goes to every request that waited on it, and the next
    // request asks the provider again rather than reusing it.
    const rejected = await burst("/mcp/oboslowreject");
    for (const res of rejected) {
      expect(res.status).toBe(401);
      expect(await errorCode(res)).toBe("token_exchange_subject_rejected");
    }
    expect(entra!.hits("/slow-reject/token")).toBe(1);
    const again = await mcpPost("/mcp/oboslowreject", initialize, bearer(jwt1));
    expect(again.status).toBe(401);
    expect(entra!.hits("/slow-reject/token")).toBe(2);
  });

  test("a step-up rejection hands the provider's claims to the caller", async () => {
    if (!etcdReachable) return;
    const etcd = new EtcdClient();
    const seed = new SeedClient(etcd, app!.etcdPrefix);
    const settingsId = randomUUID();
    await seed.update("mcp_auth_settings", settingsId, { resource_url: `${app!.proxyUrl}/mcp` });
    try {
      const metadata = `${app!.proxyUrl}/.well-known/oauth-protected-resource/mcp`;
      await waitConfigPropagation(async () => {
        const res = await mcpPost("/mcp/oboreject", initialize, bearer(jwt1));
        return res.headers.get("www-authenticate") !== null;
      });

      const res = await mcpPost("/mcp/obostepup", initialize, bearer(jwt1));
      const text = await res.text();
      expect(res.status, text).toBe(401);
      expect(JSON.parse(text).error?.code).toBe("token_exchange_subject_rejected");
      expect(text).not.toContain("AADSTS");
      const encoded = Buffer.from(STEP_UP_CLAIMS, "utf8").toString("base64");
      expect(res.headers.get("www-authenticate")).toBe(
        `Bearer resource_metadata="${metadata}", error="insufficient_claims", ` +
          `error_description="Step-up authentication required; satisfy the returned claims ` +
          `challenge with the IdP and retry", claims="${encoded}"`,
      );

      // A rejection without claims keeps the plain invalid-token challenge.
      const plain = await mcpPost("/mcp/oboreject", initialize, bearer(jwt1));
      expect(plain.status).toBe(401);
      expect(plain.headers.get("www-authenticate")).toBe(
        `Bearer error="invalid_token", resource_metadata="${metadata}"`,
      );

      // The aggregated endpoint still absorbs the failure.
      const aggregated = await session("/mcp", "obostepup__echo", bearer(jwt1));
      expect(aggregated.map((r) => r.status)).toEqual([200, 202, 200]);
    } finally {
      await seed.delete("mcp_auth_settings", settingsId);
    }
  });

  test("an exchange slower than the server's timeout_ms is 503 within it, and still completes", async () => {
    if (!etcdReachable) return;
    const bound = SHORT_TIMEOUT_MS + 1500;

    // `initialize` on the single-server endpoint, which runs the exchange
    // up front.
    const started = Date.now();
    const init = await mcpPost("/mcp/obohang", initialize, bearer(jwt1));
    const initText = await init.text();
    const initElapsed = Date.now() - started;
    expect(init.status, initText).toBe(503);
    expect(JSON.parse(initText).error?.code).toBe("token_exchange_unavailable");
    expect(initElapsed).toBeLessThan(bound);

    // A tool call, which exchanges while connecting to the upstream.
    const call = await toolCall("/mcp/obohangcall", "echo", { text: "hi" }, jwt1);
    expect(call.status, call.text).toBe(503);
    expect(JSON.parse(call.text).error?.code).toBe("token_exchange_unavailable");
    expect(call.elapsed).toBeLessThan(bound);

    // The same on a REST API behind an OpenAPI server.
    const restCall = await toolCall("/mcp/oboresthang", "listmodels", {}, jwt1);
    expect(restCall.status, restCall.text).toBe(503);
    expect(JSON.parse(restCall.text).error?.code).toBe("token_exchange_unavailable");
    expect(restCall.elapsed).toBeLessThan(bound);
    const restStarted = Date.now();
    const restInit = await mcpPost("/mcp/oboresthang", initialize, bearer(jwt2));
    const restInitText = await restInit.text();
    expect(restInit.status, restInitText).toBe(503);
    expect(JSON.parse(restInitText).error?.code).toBe("token_exchange_unavailable");
    expect(Date.now() - restStarted).toBeLessThan(bound);

    // The request that gave up did not take the exchange down with it: once
    // the identity provider answers, its token is there for the next
    // request, without a second exchange. Nothing touches the gateway until
    // the identity provider has answered the abandoned exchange.
    await waitConfigPropagation(async () => entra!.answered("/hang-mint-a/token") === 1);
    await new Promise((r) => setTimeout(r, 200));
    const after = await mcpPost("/mcp/obohang", initialize, bearer(jwt1));
    expect(after.status, await after.text()).toBe(200);
    expect(entra!.hits("/hang-mint-a/token")).toBe(1);
  });

  test("a re-exchange that fails after an upstream 401 is recorded as dispatched", async () => {
    if (!etcdReachable) return;
    const unmeasured = (row: Map<string, string>) => row.get("upstream_latency_ms");

    // Nothing was contacted but the identity provider: no upstream latency.
    const before = await toolCall("/mcp/obodown", "echo", { text: "hi" }, jwt1);
    expect(before.status, before.text).toBe(503);
    expect(unmeasured(await usageRow(before.requestId, "exchange failed up front"))).toBe("0");

    // The upstream rejected the first token, and getting another failed.
    const mark = rejectingUpstream!.receivedHeaders.length;
    const mcp = await toolCall("/mcp/obo401", "echo", { text: "hi" }, jwt1);
    expect(mcp.status, mcp.text).toBe(503);
    expect(JSON.parse(mcp.text).error?.code).toBe("token_exchange_unavailable");
    expect(entra!.hits("/mint-once/token")).toBe(2);
    expect(rejectingUpstream!.receivedHeaders.slice(mark)[0]?.authorization).toBe(
      "Bearer mint-once-1",
    );
    const mcpRow = await usageRow(mcp.requestId, "mcp re-exchange failed");
    expect(Number(unmeasured(mcpRow) ?? 0)).toBeGreaterThanOrEqual(1);

    const rest = await toolCall("/mcp/oborest401", "getitem", { id: "1" }, jwt1);
    expect(rest.status, rest.text).toBe(503);
    expect(JSON.parse(rest.text).error?.code).toBe("token_exchange_unavailable");
    expect(entra!.hits("/mint-once-rest/token")).toBe(2);
    const restRow = await usageRow(rest.requestId, "openapi re-exchange failed");
    expect(Number(unmeasured(restRow) ?? 0)).toBeGreaterThanOrEqual(1);
  });
});
