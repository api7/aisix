import { createHash } from "node:crypto";
import { createServer } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  agentClaims,
  EtcdClient,
  pickFreePort,
  SeedClient,
  spawnApp,
  startMockIdp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type MockIdp,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: JWT claim mappings — claims → existing API key (AISIX-Cloud#564).
//
// The environment trusts a mock identity provider and defines
// `claim_mappings` rules. Pinned journeys:
//
//   1. A token whose claims match a rule runs as the rule's API key —
//      that key's allowed_models apply (allowed model 200, other 403).
//   2. Rules evaluate in priority order (lower first) and the first
//      match wins; a disabled rule never matches even at the best
//      priority.
//   3. `contains` matches membership in an array claim at a nested
//      (dotted) path.
//   4. The direct `(jwt_provider, jwt_subject)` key binding stays
//      authoritative: a subject with a bound key never falls through to
//      the rules, even when its claims match one — including a DISABLED
//      binding (rules cannot re-enable a pinned identity) and an
//      AMBIGUOUS one (two keys claiming the binding fail closed).
//   5. A token matching no rule is rejected (`jwt_identity_unmapped`) —
//      never an anonymous or default pass.
//   6. A rule resolving to a nonexistent key rejects; a rule resolving
//      to a disabled key rejects with `api_key_disabled`.
//   7. Usage events carry the identity: `aisix.jwt_subject` /
//      `aisix.jwt_provider` / `aisix.jwt_claim_mapping` span attributes
//      (mapping name absent for a directly-bound subject), and plain
//      API-key requests carry none of them.
//
// References:
// - RFC 7519 (JWT) §4.1 registered claims
//   <https://datatracker.ietf.org/doc/html/rfc7519#section-4.1>

const MODEL = "cm-finance-model";
const OTHER_MODEL = "cm-admin-model";

interface OtlpReceiver {
  url: string;
  spans: Array<Record<string, string>>;
  close(): Promise<void>;
}

async function startOtlpReceiver(): Promise<OtlpReceiver> {
  const spans: Array<Record<string, string>> = [];
  const server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      try {
        const body = JSON.parse(raw);
        for (const rs of body.resourceSpans ?? []) {
          for (const ss of rs.scopeSpans ?? []) {
            for (const span of ss.spans ?? []) {
              const attrs: Record<string, string> = {};
              for (const a of span.attributes ?? []) {
                const v = a.value ?? {};
                attrs[a.key] =
                  v.stringValue ?? String(v.intValue ?? v.boolValue ?? "");
              }
              spans.push(attrs);
            }
          }
        }
      } catch {
        // ignore malformed bodies — assertions fail on missing spans
      }
      res.statusCode = 200;
      res.end("{}");
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) =>
    server.listen(port, "127.0.0.1", resolve),
  );
  return {
    url: `http://127.0.0.1:${port}/v1/traces`,
    spans,
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

async function waitForSpan(
  recv: OtlpReceiver,
  requestId: string,
  timeoutMs = 10_000,
): Promise<Record<string, string>> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const hit = recv.spans.find((a) => a["aisix.request_id"] === requestId);
    if (hit) return hit;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(`no usage span for request_id=${requestId}`);
}

function chatBody(model = MODEL): string {
  return JSON.stringify({
    model,
    messages: [{ role: "user", content: "claim mapping probe" }],
  });
}

async function chat(
  app: SpawnedApp,
  token: string,
  model = MODEL,
): Promise<Response> {
  return fetch(`${app.proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${token}`,
      "content-type": "application/json",
    },
    body: chatBody(model),
  });
}

async function errorCode(res: Response): Promise<string | undefined> {
  const body = (await res.json()) as { error?: { code?: string } };
  return body.error?.code;
}

describe("claim mapping e2e: verified claims resolve to an existing api key", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let idp: MockIdp | undefined;
  let seed: SeedClient | undefined;
  let otlp: OtlpReceiver | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    idp = await startMockIdp();
    otlp = await startOtlpReceiver();
    app = await spawnApp({});
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "cm-pk",
      api_key: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    for (const name of [MODEL, OTHER_MODEL]) {
      await seed.createModel({
        display_name: name,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
    }
    await seed.createObservabilityExporter({
      name: "cm-otlp",
      kind: "otlp_http",
      endpoint: otlp.url,
    });

    await seed.createOidcProvider({
      name: "mock-idp",
      issuer: idp.url,
      audiences: ["aisix-gateway"],
      jwks_uri: idp.jwksUrl,
    });

    // The finance policy key: MODEL only. Shared by every identity the
    // `finance-dept` rule admits.
    const financeKey = await seed.createApiKey({
      key_hash: createHash("sha256").update("sk-cm-finance").digest("hex"),
      allowed_models: [MODEL],
    });
    // The admin policy key: unrestricted.
    const adminKey = await seed.createApiKey({
      key_hash: createHash("sha256").update("sk-cm-admin").digest("hex"),
      allowed_models: ["*"],
    });
    // A disabled policy key — a rule resolving here must reject.
    const frozenKey = await seed.createApiKey({
      key_hash: createHash("sha256").update("sk-cm-frozen").digest("hex"),
      allowed_models: ["*"],
      disabled: true,
    });
    // agent-bound has a DIRECT key binding allowing OTHER_MODEL only —
    // it must never fall through to the rules even though its claims
    // also match `finance-dept`.
    await seed.createApiKey({
      key_hash: createHash("sha256").update("sk-cm-bound").digest("hex"),
      allowed_models: [OTHER_MODEL],
      jwt_subject: "agent-bound",
      jwt_provider: "mock-idp",
    });
    // agent-bound-off is bound to a DISABLED key: the binding must stay
    // authoritative (401 api_key_disabled), never fall through to a
    // matching rule — a mapping cannot re-enable a pinned identity.
    await seed.createApiKey({
      key_hash: createHash("sha256").update("sk-cm-bound-off").digest("hex"),
      allowed_models: ["*"],
      jwt_subject: "agent-bound-off",
      jwt_provider: "mock-idp",
      disabled: true,
    });
    // agent-dup is bound TWICE (a CP-invariant violation the etcd path
    // cannot rule out): the identity is ambiguous and must be rejected,
    // never resolved through the rules.
    for (const plaintext of ["sk-cm-dup-a", "sk-cm-dup-b"]) {
      await seed.createApiKey({
        key_hash: createHash("sha256").update(plaintext).digest("hex"),
        allowed_models: ["*"],
        jwt_subject: "agent-dup",
        jwt_provider: "mock-idp",
      });
    }

    // department=finance → the finance policy key.
    await seed.createClaimMapping({
      name: "finance-dept",
      jwt_provider: "mock-idp",
      priority: 100,
      match: [{ claim: "department", op: "exact", values: ["finance"] }],
      resolve: { api_key_id: financeKey.id },
    });
    // Nested array claim, better priority: platform admins win over the
    // department rule when both match.
    await seed.createClaimMapping({
      name: "platform-admins",
      jwt_provider: "mock-idp",
      priority: 50,
      match: [
        {
          claim: "realm_access.groups",
          op: "contains",
          values: ["platform-admin"],
        },
      ],
      resolve: { api_key_id: adminKey.id },
    });
    // Would beat both on priority, but is disabled — must never match.
    await seed.createClaimMapping({
      name: "disabled-rule",
      jwt_provider: "mock-idp",
      priority: 1,
      enabled: false,
      match: [{ claim: "department", op: "exact", values: ["finance"] }],
      resolve: { api_key_id: adminKey.id },
    });
    // A rule whose target key does not exist — fails closed.
    await seed.createClaimMapping({
      name: "dead-target",
      jwt_provider: "mock-idp",
      priority: 10,
      match: [{ claim: "department", op: "exact", values: ["ghost"] }],
      resolve: { api_key_id: "99999999-9999-9999-9999-999999999999" },
    });
    // A rule resolving to a disabled key — rejected at the same
    // lifecycle gate as the direct-binding path.
    await seed.createClaimMapping({
      name: "frozen-dept",
      jwt_provider: "mock-idp",
      priority: 20,
      match: [{ claim: "department", op: "exact", values: ["frozen"] }],
      resolve: { api_key_id: frozenKey.id },
    });

    // Probe the LAST-written rule (`frozen-dept`, distinguishable by
    // its error code flipping from jwt_identity_unmapped to
    // api_key_disabled): watch events apply in revision order, so the
    // final seed being live implies every earlier one is too. Probing
    // the first rule would leave a window where later rules haven't
    // landed yet.
    await waitConfigPropagation(async () => {
      const res = await chat(
        app!,
        idp!.sign(financeClaims({ department: "frozen" })),
      );
      if (res.status !== 401) {
        await res.text();
        return false;
      }
      return (await errorCode(res)) === "api_key_disabled";
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await idp?.close();
    await otlp?.close();
  });

  function financeClaims(
    overrides: Record<string, unknown> = {},
  ): Record<string, unknown> {
    return agentClaims(idp!.url, {
      sub: "dev-alice",
      department: "finance",
      ...overrides,
    });
  }

  function requireSetup(ctx: { skip: () => void }): boolean {
    if (!etcdReachable || !app || !idp || !seed || !otlp) {
      ctx.skip();
      return false;
    }
    return true;
  }

  test("matching claims run as the rule's key — its model ACL applies", async (ctx) => {
    if (!requireSetup(ctx)) return;
    const ok = await chat(app!, idp!.sign(financeClaims()));
    expect(ok.status).toBe(200);
    await ok.text();

    // The finance key allows MODEL only, so the identity cannot reach
    // OTHER_MODEL — proof the request really runs under that key, and
    // (priority 1) `disabled-rule` → admin key was skipped.
    const denied = await chat(app!, idp!.sign(financeClaims()), OTHER_MODEL);
    expect(denied.status).toBe(403);
    await denied.text();
  });

  test("lower priority wins when several rules match (nested contains)", async (ctx) => {
    if (!requireSetup(ctx)) return;
    // Claims match BOTH finance-dept (100) and platform-admins (50) —
    // the admin key wins, so OTHER_MODEL is reachable.
    const claims = financeClaims({
      sub: "dev-bob",
      realm_access: { groups: ["dev", "platform-admin"] },
    });
    const res = await chat(app!, idp!.sign(claims), OTHER_MODEL);
    expect(res.status).toBe(200);
    await res.text();
  });

  test("a directly-bound subject never falls through to the rules", async (ctx) => {
    if (!requireSetup(ctx)) return;
    // agent-bound's claims match finance-dept, but its direct binding
    // (OTHER_MODEL only) is authoritative: MODEL is denied…
    const viaRule = await chat(
      app!,
      idp!.sign(financeClaims({ sub: "agent-bound" })),
    );
    expect(viaRule.status).toBe(403);
    await viaRule.text();
    // …and OTHER_MODEL (which the finance rule's key would deny) works.
    const viaBinding = await chat(
      app!,
      idp!.sign(financeClaims({ sub: "agent-bound" })),
      OTHER_MODEL,
    );
    expect(viaBinding.status).toBe(200);
    await viaBinding.text();
  });

  test("a disabled direct binding is not re-enabled by a matching rule", async (ctx) => {
    if (!requireSetup(ctx)) return;
    // agent-bound-off's claims match finance-dept, but its disabled
    // binding stays authoritative.
    const res = await chat(
      app!,
      idp!.sign(financeClaims({ sub: "agent-bound-off" })),
    );
    expect(res.status).toBe(401);
    expect(await errorCode(res)).toBe("api_key_disabled");
  });

  test("an ambiguous direct binding fails closed, not through the rules", async (ctx) => {
    if (!requireSetup(ctx)) return;
    // agent-dup is bound to two keys; its claims match finance-dept —
    // the request must still be rejected.
    const res = await chat(app!, idp!.sign(financeClaims({ sub: "agent-dup" })));
    expect(res.status).toBe(401);
    expect(await errorCode(res)).toBe("jwt_identity_unmapped");

    const metrics = await fetch(`${app!.metricsUrl}/metrics`).then((r) =>
      r.text(),
    );
    expect(metrics).toContain('reason="jwt_binding_ambiguous"');
  });

  test("claims matching no rule are rejected, never defaulted", async (ctx) => {
    if (!requireSetup(ctx)) return;
    const res = await chat(app!, idp!.sign(financeClaims({ department: "hr" })));
    expect(res.status).toBe(401);
    expect(await errorCode(res)).toBe("jwt_identity_unmapped");
  });

  test("a rule with a dangling key target fails closed", async (ctx) => {
    if (!requireSetup(ctx)) return;
    const res = await chat(
      app!,
      idp!.sign(financeClaims({ department: "ghost" })),
    );
    expect(res.status).toBe(401);
    expect(await errorCode(res)).toBe("jwt_identity_unmapped");

    // The metric distinguishes the misconfiguration from an unmapped
    // identity so an operator can find the broken rule.
    const metrics = await fetch(`${app!.metricsUrl}/metrics`).then((r) =>
      r.text(),
    );
    expect(metrics).toContain('reason="claim_mapping_target_missing"');
  });

  test("a rule resolving to a disabled key rejects like the bound path", async (ctx) => {
    if (!requireSetup(ctx)) return;
    const res = await chat(
      app!,
      idp!.sign(financeClaims({ department: "frozen" })),
    );
    expect(res.status).toBe(401);
    expect(await errorCode(res)).toBe("api_key_disabled");
  });

  test("usage events attribute the JWT identity and the matched rule", async (ctx) => {
    if (!requireSetup(ctx)) return;
    const res = await chat(app!, idp!.sign(financeClaims()));
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-call-id");
    expect(requestId).toBeTruthy();
    await res.text();

    const span = await waitForSpan(otlp!, requestId!);
    expect(span["aisix.jwt_subject"]).toBe("dev-alice");
    expect(span["aisix.jwt_provider"]).toBe("mock-idp");
    expect(span["aisix.jwt_claim_mapping"]).toBe("finance-dept");
  });

  test("attribution rides the shared emit helper on the messages family too", async (ctx) => {
    if (!requireSetup(ctx)) return;
    // The stamping lives in one usage_attr helper wired through every
    // handler family's emit funnel; this pins a second, structurally
    // different family (Anthropic-shaped /v1/messages through the
    // OpenAI bridge) so an extractor-ordering or emitter regression on
    // the non-chat paths cannot hide behind chat-only coverage.
    const res = await fetch(`${app!.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${idp!.sign(financeClaims())}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: MODEL,
        max_tokens: 64,
        messages: [{ role: "user", content: "claim mapping messages probe" }],
      }),
    });
    expect(res.status).toBe(200);
    // /v1/messages carries the middleware-stamped x-aisix-request-id
    // (the usage event keys on the same id); x-aisix-call-id is the
    // chat-path spelling.
    const requestId = res.headers.get("x-aisix-request-id");
    expect(requestId).toBeTruthy();
    await res.text();

    const span = await waitForSpan(otlp!, requestId!);
    expect(span["aisix.jwt_subject"]).toBe("dev-alice");
    expect(span["aisix.jwt_provider"]).toBe("mock-idp");
    expect(span["aisix.jwt_claim_mapping"]).toBe("finance-dept");
  });

  test("a directly-bound identity attributes subject but no rule name", async (ctx) => {
    if (!requireSetup(ctx)) return;
    const res = await chat(
      app!,
      idp!.sign(financeClaims({ sub: "agent-bound" })),
      OTHER_MODEL,
    );
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-call-id");
    await res.text();

    const span = await waitForSpan(otlp!, requestId!);
    expect(span["aisix.jwt_subject"]).toBe("agent-bound");
    expect(span["aisix.jwt_provider"]).toBe("mock-idp");
    expect(span["aisix.jwt_claim_mapping"]).toBeUndefined();
  });

  test("plain api-key requests carry no jwt attribution", async (ctx) => {
    if (!requireSetup(ctx)) return;
    const res = await chat(app!, "sk-cm-admin", OTHER_MODEL);
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-call-id");
    await res.text();

    const span = await waitForSpan(otlp!, requestId!);
    expect(span["aisix.jwt_subject"]).toBeUndefined();
    expect(span["aisix.jwt_provider"]).toBeUndefined();
    expect(span["aisix.jwt_claim_mapping"]).toBeUndefined();
  });
});
