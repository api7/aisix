import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  agentClaims,
  EtcdClient,
  metricDelta,
  ProxyClient,
  scrapeMetrics,
  SeedClient,
  spawnApp,
  startMockIdp,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForSlsLog,
  type MetricSample,
  type MockIdp,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E for AISIX-Cloud#1389: the usage event must name the org MEMBER behind
// the request, not just the credential it arrived on.
//
// The ask is "show me Alice's 429s in the last 24h". Before this, the only
// caller identity on a usage row was `api_key_id` (plus a JWT subject), so
// answering it meant enumerating every key Alice owns, querying each, and
// merging by hand — and a member calling through both an API key and OIDC
// would still be split across two identities. `user_id` is snapshotted onto
// the event at request time rather than resolved from `api_key_id` at query
// time, so rebinding or deleting a key cannot re-attribute or erase the
// history it already produced.
//
// Read back off a real Aliyun-SLS export from a real `aisix` binary, so what
// is asserted is the row a consumer actually receives.
//
// The three probes are the claim's real content:
//   - the API-key path stamps the owning member;
//   - a JWT resolving to a DIFFERENT key owned by the SAME member stamps the
//     same member — this is the "one member, several credentials" half of
//     the report, and it is the one a join on api_key_id gets wrong;
//   - an unowned key stamps nobody, so a member filter cannot sweep up
//     traffic that was never theirs.
//
// The last case pins the OTHER half of the report's driving question. A 429
// reaches a caller two ways — the upstream throttled us, or the gateway's
// own rate limit refused the request before any upstream was contacted —
// and only the first is an upstream *attempt*. The gateway's own refusal
// leaves the failure path with zero attempts, which is a genuinely
// different code path (`chat.rs`'s `match charge` zero-attempt arm rather
// than its per-attempt loop). Both must reach the usage feed carrying the
// member, or "show me Alice's 429s" silently answers half the question.

const OWNED_PLAINTEXT = "sk-usage-member-owned";
const UNOWNED_PLAINTEXT = "sk-usage-member-unowned";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

const CREDENTIAL_REF = "mock";
const SLS_PROJECT = "aisix-e2e-obs";
const LOGSTORE = "usage-member-attribution";

const MODEL = "uma-model";
// A model whose upstream always throttles — the issue's headline scenario
// ("show me Alice's 429s") needs a 429 that actually reaches an upstream and
// therefore produces a per-attempt usage row.
const THROTTLED_MODEL = "uma-throttled";
// A member-owned key whose OWN rpm is 1, so its second request is refused
// by the gateway rather than by any upstream.
const GATEWAY_LIMITED_PLAINTEXT = "sk-usage-member-gateway-limited";
// The member every owned credential in this case belongs to. A uuid-shaped
// value because that is what cp-api projects (`api_keys.user_id`).
const ALICE = "3f1b7c62-9d4e-4a51-8f0b-2c6d5e4a1b90";

describe("usage member attribution e2e: a usage row names the org member (#1389)", () => {
  let upstream: OpenAiUpstream | undefined;
  let throttled: OpenAiUpstream | undefined;
  let sls: MockSls | undefined;
  let idp: MockIdp | undefined;
  let app: SpawnedApp | undefined;
  let etcdReachable = false;

  async function chat(
    credential: string,
    marker: string,
    model = MODEL,
  ): Promise<Response> {
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${credential}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: marker }],
      }),
    });
    await res.text();
    return res;
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    idp = await startMockIdp();
    upstream = await startOpenAiUpstream();
    throttled = await startOpenAiUpstream({
      status: 429,
      errorBody: { error: { message: "slow down", type: "rate_limit_error" } },
    });

    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "LTAI_mock_ak",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock_ak_secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);

    await seed.createObservabilityExporter({
      name: "uma-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "full",
    });

    const pk = await seed.createProviderKey({
      display_name: "uma-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });

    const throttledPk = await seed.createProviderKey({
      display_name: "uma-throttled-pk",
      secret: "sk-mock",
      api_base: `${throttled.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: THROTTLED_MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: throttledPk.id,
    });

    await seed.createOidcProvider({
      name: "uma-idp",
      issuer: idp.url,
      audiences: ["aisix-gateway"],
      jwks_uri: idp.jwksUrl,
    });

    // Alice's SECOND credential, reached over OIDC rather than a bearer
    // key. A different api_key row, the same person.
    await seed.createApiKey({
      key_hash: hash("sk-usage-member-jwt-bound"),
      allowed_models: [MODEL, THROTTLED_MODEL],
      user_id: ALICE,
      jwt_subject: "alice",
      jwt_provider: "uma-idp",
    });

    // A key owned by nobody — the shape every key created without an
    // explicit owner takes.
    await seed.createApiKey({
      key_hash: hash(UNOWNED_PLAINTEXT),
      allowed_models: [MODEL],
    });

    // Alice's third credential, rate-limited at the key level. Its 429 never
    // reaches an upstream, so it exercises the zero-attempt failure path.
    await seed.createApiKey({
      key_hash: hash(GATEWAY_LIMITED_PLAINTEXT),
      allowed_models: [MODEL],
      user_id: ALICE,
      rate_limit: { rpm: 1 },
    });

    // Written LAST so the readiness gate below implies everything above is
    // already in the snapshot (one watch, applied in revision order).
    await seed.createApiKey({
      key_hash: hash(OWNED_PLAINTEXT),
      allowed_models: [MODEL, THROTTLED_MODEL],
      user_id: ALICE,
    });

    const proxy = new ProxyClient(app.proxyUrl, OWNED_PLAINTEXT);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await throttled?.close();
    await sls?.close();
    await idp?.close();
  });

  test("an API key bound to a member stamps that member on its usage row", async (ctx) => {
    if (!etcdReachable || !app || !sls) {
      ctx.skip();
      return;
    }
    const marker = "uma-apikey-4f21";
    expect((await chat(OWNED_PLAINTEXT, marker)).status).toBe(200);

    const log = await waitForSlsLog(
      sls,
      LOGSTORE,
      (l) => (l.get("prompt") ?? "").includes(marker),
      "api-key usage row",
    );
    expect(log.get("user_id")).toBe(ALICE);
  });

  test("a JWT resolving to the member's OTHER key stamps the same member", async (ctx) => {
    if (!etcdReachable || !app || !sls || !idp) {
      ctx.skip();
      return;
    }
    const marker = "uma-jwt-8c07";
    const token = idp.sign(agentClaims(idp.url, { sub: "alice" }));
    expect((await chat(token, marker)).status).toBe(200);

    const log = await waitForSlsLog(
      sls,
      LOGSTORE,
      (l) => (l.get("prompt") ?? "").includes(marker),
      "jwt usage row",
    );
    // Same person, different credential: this is what a query-time join on
    // api_key_id cannot express in one filter.
    expect(log.get("user_id")).toBe(ALICE);
    expect(log.get("jwt_subject")).toBe("alice");
  });

  test("a key owned by nobody stamps no member", async (ctx) => {
    if (!etcdReachable || !app || !sls) {
      ctx.skip();
      return;
    }
    const marker = "uma-unowned-1d93";
    expect((await chat(UNOWNED_PLAINTEXT, marker)).status).toBe(200);

    const log = await waitForSlsLog(
      sls,
      LOGSTORE,
      (l) => (l.get("prompt") ?? "").includes(marker),
      "unowned-key usage row",
    );
    // Absent, not a placeholder: a member filter must not sweep up traffic
    // that belongs to no member.
    expect(log.get("user_id")).toBeUndefined();
  });

  test("a gateway rate-limit 429 reaches the feed with the member on it", async (ctx) => {
    if (!etcdReachable || !app || !sls) {
      ctx.skip();
      return;
    }
    const before: MetricSample[] = await scrapeMetrics(app.metricsUrl);
    // rpm=1: the first request consumes the budget, the second is refused
    // by the gateway itself — no upstream is contacted, so this event comes
    // from the zero-attempt path rather than from a failed attempt.
    expect((await chat(GATEWAY_LIMITED_PLAINTEXT, "uma-gw429-first")).status).toBe(200);
    expect((await chat(GATEWAY_LIMITED_PLAINTEXT, "uma-gw429-refused")).status).toBe(429);
    const after: MetricSample[] = await scrapeMetrics(app.metricsUrl);

    const log = await waitForSlsLog(
      sls,
      LOGSTORE,
      (l) => (l.get("prompt") ?? "").includes("uma-gw429-refused"),
      "gateway-throttled usage row",
    );
    expect(log.get("status_code")).toBe("429");
    expect(log.get("user_id")).toBe(ALICE);
    // The gateway's own refusal, not an upstream's — the two 429s are
    // distinguishable by error class, which is what lets an operator tell
    // "we throttled them" from "the provider throttled us".
    expect(log.get("error_class")).toBe("rate_limit_exceeded");
    // Nothing was sent upstream, so nothing is billed.
    expect(log.get("prompt_tokens") ?? "0").toBe("0");

    expect(
      metricDelta(before, after, "aisix_usage_events_emitted_total", {
        user_id: ALICE,
        status: "429",
      }),
    ).toBeGreaterThanOrEqual(1);
  });

  test("a member's 429 is addressable by member and by raw status code", async (ctx) => {
    if (!etcdReachable || !app || !sls) {
      ctx.skip();
      return;
    }
    const before: MetricSample[] = await scrapeMetrics(app.metricsUrl);
    const marker = "uma-throttled-2b55";
    expect((await chat(OWNED_PLAINTEXT, marker, THROTTLED_MODEL)).status).toBe(429);
    const after: MetricSample[] = await scrapeMetrics(app.metricsUrl);

    // The row an operator finds under "member = Alice, status = 429".
    const log = await waitForSlsLog(
      sls,
      LOGSTORE,
      (l) => (l.get("prompt") ?? "").includes(marker),
      "throttled usage row",
    );
    expect(log.get("user_id")).toBe(ALICE);
    expect(log.get("status_code")).toBe("429");

    // …and the same question answered on the counter, which before this
    // could only say "4xx" and could not say "whose".
    expect(
      metricDelta(before, after, "aisix_usage_events_emitted_total", {
        user_id: ALICE,
        status: "429",
      }),
    ).toBeGreaterThanOrEqual(1);
    // The status family survives beside the raw code, so a dashboard
    // written against `status_code="4xx"` keeps working unchanged.
    expect(
      metricDelta(before, after, "aisix_usage_events_emitted_total", {
        user_id: ALICE,
        status_code: "4xx",
      }),
    ).toBeGreaterThanOrEqual(1);
    // Traffic that resolved no member must not land on the member's series.
    expect(
      metricDelta(before, after, "aisix_usage_events_emitted_total", {
        user_id: "unknown",
        status: "429",
      }),
    ).toBe(0);
  });
});
