import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test, vi } from "vitest";
import {
  agentClaims,
  awaitWindowHeadroom,
  EtcdClient,
  OAuthMcpUpstream,
  ProxyClient,
  SeedClient,
  signInMcpClient,
  slsLogsFor,
  spawnApp,
  startMcpUpstream,
  startMockIdp,
  startMockOidcProvider,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForLogLine,
  waitForSlsLog,
  whoami,
  type McpUpstream,
  type MockIdp,
  type MockOidcProvider,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: an OIDC provider's `tag_claims` turns claims of the verified token into
// the request's cost-attribution tags. Per tag key the claim wins over the
// API key, its team and its member, and a claim that yields nothing leaves the
// key to them. What is asserted is what a consumer of the feature sees: the
// usage row an exporter receives, the tag budget that refuses, and the `tag:`
// rate limit that counts.

const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

const CREDENTIAL_REF = "mock";
const SLS_CREDENTIALS = {
  [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "LTAI_mock_ak",
  [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock_ak_secret",
};
const LOGSTORE = "jwt-tag-claims";
const MODEL = "jtc-model";

const TEAM = randomUUID();
const MEMBER = randomUUID();

const TAG_CLAIMS = {
  department: "department",
  subsidiary: "org.subsidiary",
  division: "https://acme.com/division",
  groups: "groups",
  employee_no: "employee_no",
  tier: "tier",
};

/** The `tags.<key>` fields of a usage row, as a plain object. */
function rowTags(row: Map<string, string>): Record<string, string> {
  const tags: Record<string, string> = {};
  for (const [k, v] of row) {
    if (k.startsWith("tags.")) tags[k.slice("tags.".length)] = v;
  }
  return tags;
}

async function seedExporter(seed: SeedClient, sls: MockSls): Promise<void> {
  await seed.createObservabilityExporter({
    name: "jtc-sls",
    enabled: true,
    kind: "aliyun_sls",
    endpoint: sls.url,
    project: "aisix-e2e-obs",
    logstore: LOGSTORE,
    credential_ref: CREDENTIAL_REF,
  });
}

describe("tags from JWT claims on bearer tokens", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let upstream: OpenAiUpstream | undefined;
  let mcp: McpUpstream | undefined;
  let idp: MockIdp;
  let plainIdp: MockIdp;
  let dottedIdp: MockIdp;

  const tokenFor = (sub: string, claims: Record<string, unknown> = {}) =>
    idp.sign(agentClaims(idp.url, { sub, ...claims }));

  async function chat(
    bearer: string,
    model = MODEL,
  ): Promise<{ status: number; body: string; requestId: string }> {
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${bearer}`, "content-type": "application/json" },
      body: JSON.stringify({ model, messages: [{ role: "user", content: "hello" }] }),
    });
    return {
      status: res.status,
      body: await res.text(),
      requestId: res.headers.get("x-aisix-call-id") ?? res.headers.get("x-aisix-request-id") ?? "",
    };
  }

  /** The usage row of one chat request made with `bearer`. */
  async function tagsOfChat(bearer: string): Promise<Record<string, string>> {
    const res = await chat(bearer);
    expect(res.status, res.body).toBe(200);
    const row = await waitForSlsLog(
      sls!,
      LOGSTORE,
      (l) => l.get("request_id") === res.requestId,
      `usage row of ${res.requestId}`,
    );
    return rowTags(row);
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    idp = await startMockIdp();
    plainIdp = await startMockIdp();
    dottedIdp = await startMockIdp();
    upstream = await startOpenAiUpstream();
    mcp = await startMcpUpstream("jtc");

    app = await spawnApp({
      exposeDebugHeaders: true,
      logLevel: "info,aisix::auth=debug",
      extraEnv: SLS_CREDENTIALS,
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seedExporter(seed, sls);

    const pk = await seed.createProviderKey({
      display_name: "jtc-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.update("mcp_servers", randomUUID(), {
      display_name: "jtc",
      url: mcp.url,
      enabled: true,
    });

    await seed.update("teams", TEAM, {
      name: "Platform",
      tags: { subsidiary: "team-sub", region: "team-region" },
    });
    await seed.update("users", MEMBER, { name: "Alice", tags: { groups: "member-groups" } });

    await seed.createOidcProvider({
      name: "jtc-idp",
      issuer: idp.url,
      audiences: ["aisix-gateway"],
      jwks_uri: idp.jwksUrl,
      tag_claims: TAG_CLAIMS,
    });
    await seed.createOidcProvider({
      name: "jtc-plain",
      issuer: plainIdp.url,
      audiences: ["aisix-gateway"],
      jwks_uri: plainIdp.jwksUrl,
    });
    await seed.createOidcProvider({
      name: "jtc-dotted",
      issuer: dottedIdp.url,
      audiences: ["aisix-gateway"],
      jwks_uri: dottedIdp.jwksUrl,
      identity_claim: "https://acme.com/uid",
      bound_claims: { "https://acme.com/realm": "staff" },
    });

    await seed.createRateLimitPolicy({
      name: "jtc-per-department",
      conditions: [{ dimension: "tag:bg", operator: "==", value: "jtc-grouped" }],
      group_by: ["tag:department"],
      limits: { rpm: 1 },
    });
    await seed.createRateLimitPolicy({
      name: "jtc-capped-tier",
      conditions: [{ dimension: "tag:tier", operator: "==", value: "capped" }],
      limits: { rpm: 1 },
    });

    const reason = (message: string) => ({
      type: "billing_error",
      code: "budget_exceeded",
      message,
      scope: "tag",
      limit_usd: "10.00",
      spent_usd: "10.25",
      remaining_usd: "-0.25",
      period: "month",
    });
    for (const [value, message] of [
      ["exhausted-by-claim", "claim value out"],
      ["exhausted-on-key", "key value out"],
    ]) {
      await seed.update("budgets", randomUUID(), {
        scope: "tag",
        tag: { key: "department", value },
        period: "month",
        exhausted: {
          reason: { ...reason(message), scope_ref: `department=${value}` },
        },
      });
    }

    const bound = (subject: string, provider: string, extra: Record<string, unknown>) =>
      seed.createApiKey({
        key_hash: sha256(`sk-jtc-${subject}`),
        allowed_models: [MODEL],
        jwt_subject: subject,
        jwt_provider: provider,
        ...extra,
      });
    await bound("attributed", "jtc-idp", {
      team_id: TEAM,
      user_id: MEMBER,
      tags: { department: "key-dept" },
    });
    await bound("rate-a", "jtc-idp", { tags: { bg: "jtc-grouped" } });
    await bound("rate-b", "jtc-idp", { tags: { bg: "jtc-grouped" } });
    await bound("capped", "jtc-idp", {});
    await bound("budgeted", "jtc-idp", { tags: { department: "exhausted-on-key" } });
    await bound("mcp-caller", "jtc-idp", {
      tags: { department: "key-dept" },
      mcp_access: { allow: ["*"] },
    });
    await bound("plain", "jtc-plain", {
      team_id: TEAM,
      tags: { department: "plain-key-dept" },
    });
    await bound("dotted", "jtc-dotted", {});
    const mapped = await seed.createApiKey({
      key_hash: sha256("sk-jtc-mapped"),
      allowed_models: [MODEL],
    });
    await seed.createClaimMapping({
      name: "jtc-by-dotted-group",
      jwt_provider: "jtc-dotted",
      match: [{ claim: "https://acme.com/group", op: "exact", values: ["eng"] }],
      resolve: { api_key_id: mapped.id },
    });

    // Last: once it authenticates, everything above is in the snapshot.
    const sentinel = "sk-jtc-sentinel";
    await seed.createApiKey({ key_hash: sha256(sentinel), allowed_models: [MODEL] });
    const proxy = new ProxyClient(app.proxyUrl, sentinel);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await mcp?.close();
    await idp?.close();
    await plainIdp?.close();
    await dottedIdp?.close();
    await sls?.close();
  });

  test("mapped claims reach the usage event, over the key's, team's and member's values", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const tags = await tagsOfChat(
      tokenFor("attributed", {
        department: "claim-dept",
        org: { subsidiary: "claim-sub" },
        "https://acme.com/division": "claim-div",
        groups: "claim-groups",
        employee_no: 1042,
      }),
    );
    expect(tags).toEqual({
      department: "claim-dept",
      subsidiary: "claim-sub",
      division: "claim-div",
      groups: "claim-groups",
      employee_no: "1042",
      // No mapping names this key: the team's value stands.
      region: "team-region",
    });
  });

  test("a token without the claims leaves each key to the key, team and member", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    expect(await tagsOfChat(tokenFor("attributed"))).toEqual({
      department: "key-dept",
      subsidiary: "team-sub",
      groups: "member-groups",
      region: "team-region",
    });
  });

  const inherited = {
    subsidiary: "team-sub",
    groups: "member-groups",
    region: "team-region",
  };

  test.each<[string, unknown, string]>([
    ["a one-element array is the element", ["solo"], "solo"],
    ["a numeric one-element array is its text", [7], "7"],
    ["an array is sorted and deduplicated", ["rnd", "ops", "rnd"], "ops,rnd"],
    ["numbers and strings sort as text", [2, "1", 2, 10], "1,10,2"],
    ["the longest allowed value is kept", "x".repeat(128), "x".repeat(128)],
    ["length counts characters, not bytes", "研".repeat(128), "研".repeat(128)],
    ["whitespace inside a value is kept", "research and development", "research and development"],
    ["a comma inside an element is not treated specially", ["a,b", "c"], "a,b,c"],
  ])("%s", async (name, claim, expected) => {
    if (!etcdReachable) return;
    expect(await tagsOfChat(tokenFor("attributed", { department: claim })), name).toEqual({
      department: expected,
      ...inherited,
    });
  });

  test.each<[string, unknown]>([
    ["an object", { name: "rnd" }],
    ["a boolean", true],
    ["null", null],
    ["an empty array", []],
    ["an array holding a boolean", ["rnd", true]],
    ["an array holding an object", ["rnd", { a: 1 }]],
    ["a value over 128 characters", "x".repeat(129)],
    ["an array joining to over 128 characters", ["x".repeat(100), "y".repeat(100)]],
    ["an array holding an empty string", ["", "rnd"]],
    ["an array holding an element with outer whitespace", ["a ", "b"]],
    ["an array holding an element with a control character", ["r\nd", "ops"]],
    ["an empty string", ""],
    ["leading whitespace", " rnd"],
    ["trailing whitespace", "rnd "],
    ["trailing Unicode whitespace", "rnd　"],
    ["a control character", "r\nd"],
  ])("%s falls back rather than being truncated or trimmed", async (name, claim) => {
    if (!etcdReachable) return;
    expect(await tagsOfChat(tokenFor("attributed", { department: claim })), name).toEqual({
      department: "key-dept",
      ...inherited,
    });
  });

  test("a dropped claim is logged by provider, tag key and reason, without its value", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const secretish = "value-that-must-not-be-logged ";
    const res = await chat(tokenFor("attributed", { department: secretish }));
    expect(res.status).toBe(200);
    const line = await waitForLogLine(
      app,
      (l) =>
        l.includes("claim yields no tag value") &&
        l.includes("jtc-idp") &&
        l.includes("department") &&
        l.includes("surrounding_whitespace"),
      "the dropped-claim line",
    );
    expect(line).not.toContain("value-that-must-not-be-logged");
    expect(app.output()).not.toContain("value-that-must-not-be-logged");
  });

  test("group_by on a tag counts per claim value", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    await awaitWindowHeadroom();
    expect((await chat(tokenFor("rate-a", { department: "dept-a" }))).status).toBe(200);
    // The counter is the claim value's: another key with it is already out.
    expect((await chat(tokenFor("rate-b", { department: "dept-a" }))).status).toBe(429);
    expect((await chat(tokenFor("rate-a", { department: "dept-a" }))).status).toBe(429);
    // Another value of the claim counts separately, on the same key.
    expect((await chat(tokenFor("rate-a", { department: "dept-b" }))).status).toBe(200);
    expect((await chat(tokenFor("rate-a", { department: "dept-b" }))).status).toBe(429);
    // Without the tag the policy does not apply.
    for (let i = 0; i < 3; i += 1) {
      expect((await chat(tokenFor("rate-a"))).status).toBe(200);
    }
  });

  test("a tag condition matches the claim value", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    await awaitWindowHeadroom();
    expect((await chat(tokenFor("capped", { tier: "capped" }))).status).toBe(200);
    expect((await chat(tokenFor("capped", { tier: "capped" }))).status).toBe(429);
    for (let i = 0; i < 3; i += 1) {
      expect((await chat(tokenFor("capped", { tier: "free" }))).status).toBe(200);
    }
  });

  test("an exhausted tag budget stops the claim value, and a claim moves a key off its own", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const stopped = await chat(tokenFor("attributed", { department: "exhausted-by-claim" }));
    expect(stopped.status).toBe(429);
    expect(JSON.parse(stopped.body).error.message).toBe("claim value out");

    const onKey = await chat(tokenFor("budgeted"));
    expect(onKey.status).toBe(429);
    expect(JSON.parse(onKey.body).error.message).toBe("key value out");
    expect((await chat(tokenFor("budgeted", { department: "elsewhere" }))).status).toBe(200);
  });

  test("a tool call on /mcp with a bearer JWT carries the claim's tag", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const auth = { authorization: `Bearer ${tokenFor("mcp-caller", { department: "mcp-dept" })}` };
    const rpc = (id: number, method: string, params: Record<string, unknown>) =>
      fetch(`${app!.proxyUrl}/mcp`, {
        method: "POST",
        headers: {
          ...auth,
          "content-type": "application/json",
          accept: "application/json, text/event-stream",
        },
        body: JSON.stringify({ jsonrpc: "2.0", id, method, params }),
      });
    await (
      await rpc(1, "initialize", {
        protocolVersion: "2025-11-25",
        capabilities: {},
        clientInfo: { name: "jwt-tag-claims", version: "0.1" },
      })
    ).text();
    const res = await rpc(2, "tools/call", { name: "jtc__echo", arguments: { text: "hi" } });
    expect(res.status, await res.text()).toBe(200);
    const row = await waitForSlsLog(
      sls,
      LOGSTORE,
      (l) => l.get("operation") === "mcp" && l.get("jwt_subject") === "mcp-caller",
      "usage row of the tool call",
    );
    expect(rowTags(row)).toEqual({ department: "mcp-dept" });
  });

  test("a provider without tag_claims, and an API key, take no tag from anywhere else", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const claims = agentClaims(plainIdp.url, { sub: "plain", department: "claim-dept", tier: "capped" });
    const expected = {
      department: "plain-key-dept",
      subsidiary: "team-sub",
      region: "team-region",
    };
    expect(await tagsOfChat(plainIdp.sign(claims))).toEqual(expected);
    // The same provider's key, on the path that has no token to read.
    expect(await tagsOfChat("sk-jtc-plain")).toEqual(expected);
  });

  test("identity, bound claims and claim mappings read a claim name with dots in it", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const token = (claims: Record<string, unknown>) =>
      dottedIdp.sign(agentClaims(dottedIdp.url, { "https://acme.com/realm": "staff", ...claims }));

    const bound = await chat(token({ "https://acme.com/uid": "dotted" }));
    expect(bound.status, bound.body).toBe(200);
    const row = await waitForSlsLog(
      sls!,
      LOGSTORE,
      (l) => l.get("request_id") === bound.requestId,
      "usage row of the dotted identity",
    );
    expect(row.get("jwt_subject")).toBe("dotted");

    // The bound claim is really being read: the wrong value is refused.
    const outside = await chat(
      token({ "https://acme.com/uid": "dotted", "https://acme.com/realm": "guest" }),
    );
    expect(outside.status).toBe(403);

    const mapped = await chat(
      token({ "https://acme.com/uid": "nobody-binds-this", "https://acme.com/group": "eng" }),
    );
    expect(mapped.status, mapped.body).toBe(200);
    const unmapped = await chat(
      token({ "https://acme.com/uid": "nobody-binds-this", "https://acme.com/group": "sales" }),
    );
    expect(unmapped.status).toBe(401);
  });
});

describe("tags from JWT claims on MCP sign-in sessions", () => {
  const SERVER = "11111111-1111-4111-8111-111111111111";
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let provider: MockOidcProvider;
  let upstream: OAuthMcpUpstream;
  let seed: SeedClient;
  let providerId: string;

  const providerRow = (tagClaims: Record<string, string>) => ({
    name: "corp",
    issuer: provider.url,
    audiences: ["some-api"],
    login: { client_id: provider.clientId, scopes: ["profile"] },
    tag_claims: tagClaims,
  });

  const settled = async () => {
    const sentinel = `sk-sentinel-${randomUUID()}`;
    await seed.createApiKey({ key_hash: sha256(sentinel), allowed_models: [] });
    const proxy = new ProxyClient(app!.proxyUrl, sentinel);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    sls = await startMockSls();
    provider = await startMockOidcProvider();
    upstream = await new OAuthMcpUpstream().start();
    app = await spawnApp({ extraEnv: SLS_CREDENTIALS });
    seed = new SeedClient(etcd, app.etcdPrefix);
    await seedExporter(seed, sls);

    providerId = (await seed.createOidcProvider(providerRow({ department: "org.department" }))).id;
    const key = await seed.createApiKey({
      key_hash: sha256("sk-jtc-session"),
      allowed_models: [],
      mcp_access: { allow: ["*"] },
      tags: { department: "key-dept", cost_center: "key-cc" },
    });
    await seed.createClaimMapping({
      name: "staff",
      jwt_provider: "corp",
      match: [{ claim: "realm", op: "exact", values: ["staff"] }],
      resolve: { api_key_id: key.id },
    });
    // Last: a server known before its login provider is a relay-mode server.
    await seed.update("mcp_auth_settings", randomUUID(), { resource_url: `${app.proxyUrl}/mcp` });
    await seed.update("mcp_servers", SERVER, {
      name: "main",
      type: "mcp",
      url: upstream.mcpUrl,
      auth_type: "oauth2_authorization_code",
    });
    await settled();
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await provider?.close();
    await sls?.close();
  });

  test("a session reads the claims of its sign-in through the provider's current tag_claims", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    provider.claims = {
      sub: "sam",
      realm: "staff",
      org: { department: "signin-dept" },
      cost_center: 9001,
    };
    upstream.user = "up-sam";
    const { client } = await signInMcpClient(`${app.proxyUrl}/mcp/main`, {});
    const isSam = (l: Map<string, string>) =>
      l.get("operation") === "mcp" && l.get("jwt_subject") === "sam";

    expect(await whoami(client)).toBe("up-sam");
    const first = await waitForSlsLog(sls, LOGSTORE, isSam, "usage row of the session's tool call");
    expect(rowTags(first)).toEqual({ department: "signin-dept", cost_center: "key-cc" });

    // The next request follows the provider as it is now, reading a claim the
    // mapping at sign-in never named. The identity provider is not asked again.
    const seen = slsLogsFor(sls, LOGSTORE).filter(isSam).length;
    provider.claims = { sub: "sam", realm: "staff", cost_center: "not-this-one" };
    await seed.update("oidc_providers", providerId, providerRow({ cost_center: "cost_center" }));
    await settled();
    expect(await whoami(client)).toBe("up-sam");
    await vi.waitFor(() => expect(slsLogsFor(sls!, LOGSTORE).filter(isSam).length).toBeGreaterThan(seen), {
      timeout: 10_000,
    });
    const next = slsLogsFor(sls, LOGSTORE).filter(isSam).at(-1)!;
    expect(rowTags(next)).toEqual({ department: "key-dept", cost_center: "9001" });
    await client.close();
  });
});
