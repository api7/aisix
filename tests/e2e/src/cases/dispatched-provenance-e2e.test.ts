import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  metricDelta,
  scrapeMetrics,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";
import { harnessRequest } from "../harness/http.js";

// E2E: the `aisix_deployment_*` families read as UPSTREAM health for one
// deployment target, so an attempt that never left the gateway must not
// appear in them. Two kinds of attempt never leave: one the target's own
// rate-limit layers refuse, and — the kind this file pins — one the bridge
// rejects while still assembling the request.
//
// Counting those as `aisix_deployment_failure_responses_total` reports our
// own misconfiguration as provider degradation: an operator watching a
// target's failure rate sees the provider "failing" while the provider was
// never asked anything. The upstream request count is the ground truth
// here — it stays at zero across every measured call.
//
// Two rejection points, because they are raised in different places and a
// fix for one does not imply the other:
//
//   - an unusable secret, caught by the bridge's own `api_key()` guard
//     before any request is built;
//   - an `api_base` that does not parse, which reaches reqwest as a raw
//     string and fails at `send()` as a *builder* error, with no socket
//     ever opened.
//
// Every changed dispatch branch is exercised, not just one: the
// classification is applied at four separate call sites (streaming and
// non-streaming chat, `/v1/messages`, `/v1/responses`) and any of them can
// regress to `dispatched: true` on its own while the others stay correct.
//
// The mirror assertion matters just as much: the attempt is NOT dropped.
// It stays a real attempt in the per-attempt usage events (the rows the
// dashboard log counts), exactly as a rate-limit-refused attempt does. The
// deployment families are the only place it is excluded from.

const CALLER_PLAINTEXT = "sk-dispatch-provenance";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("dispatched provenance e2e: a pre-dispatch failure is not upstream health", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // A real, healthy upstream. It exists so "the upstream received
    // nothing" is a measurement rather than an artifact of there being
    // nowhere to send to.
    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    // Valid api_base — routing shape is fine — but a secret carrying an
    // embedded newline, which cannot be an Authorization header value.
    // The openai bridge's api_key() guard rejects it before dispatch.
    // (An empty secret is the same error class but never gets this far:
    // the admin schema's min-length check rejects it at admission.)
    const badCredPk = await seed.createProviderKey({
      display_name: "dp-badcred-pk",
      secret: "sk-live\n-injected",
      api_base: `${upstream.baseUrl}/v1`,
    });
    // Usable secret, but an api_base that is not a URL. The schema types
    // api_base as a plain string, so this is admitted; the parse failure
    // surfaces only when reqwest builds the request.
    const badUrlPk = await seed.createProviderKey({
      display_name: "dp-badurl-pk",
      secret: "sk-mock",
      api_base: "ht tp://not a url/v1",
    });

    for (const [name, pkId] of [
      ["dp-badcred", badCredPk.id],
      ["dp-badurl", badUrlPk.id],
    ] as const) {
      await seed.createModel({
        display_name: name,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pkId,
        // Every call in this file fails this target by design. With
        // cooldown on, the target would be marked after the first one and
        // later calls would be refused before an attempt is ever built —
        // removing the very attempt these tests are about.
        cooldown: { enabled: false },
      });
    }

    // Routing models, because the `aisix_deployment_*` families are emitted
    // from the Model-Group dispatch loops: a direct model builds no
    // AttemptRecord at all and would not exercise the code under test.
    // One target each keeps the assertions about this attempt rather than
    // about failover ordering.
    await seed.createModel({
      display_name: "dp-cred-group",
      routing: { strategy: "failover", targets: [{ model: "dp-badcred" }] },
    });
    await seed.createModel({
      display_name: "dp-url-group",
      routing: { strategy: "failover", targets: [{ model: "dp-badurl" }] },
    });
    // Seeded last: once this key authenticates, the whole seed set is in
    // the snapshot.
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["dp-cred-group", "dp-url-group"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  /** POST `path` as the seeded caller and return the raw status + body. */
  const post = async (
    path: string,
    body: unknown,
  ): Promise<{ status: number; text: string }> => {
    const res = await harnessRequest(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    });
    return { status: res.statusCode, text: await res.body.text() };
  };

  /**
   * Run `call` and assert it was refused without any upstream request and
   * without moving `model`'s deployment counters — while still landing in
   * the per-attempt usage events.
   */
  const expectNotAttributedToUpstream = async (
    model: string,
    wantStatus: number,
    call: () => Promise<{ status: number; text: string }>,
  ) => {
    const upstreamBaseline = upstream!.receivedRequests.length;
    const before = await scrapeMetrics(app!.metricsUrl);

    const res = await call();
    expect(res.status, `body: ${res.text}`).toBe(wantStatus);

    const after = await scrapeMetrics(app!.metricsUrl);
    const delta = (name: string, want?: Record<string, string>) =>
      metricDelta(before, after, name, want);

    // Ground truth: the provider was never asked anything.
    expect(upstream!.receivedRequests.length - upstreamBaseline).toBe(0);

    // Therefore it owes the deployment families nothing. Before this
    // change, `requests_total` and `failure_responses_total` each moved by
    // 1 — the failure counter is what made a healthy provider look like it
    // was failing.
    for (const family of [
      "aisix_deployment_requests_total",
      "aisix_deployment_failure_responses_total",
      "aisix_deployment_success_responses_total",
    ]) {
      expect(delta(family, { model })).toBe(0);
    }

    // …but the attempt itself is not lost. It is still a per-attempt usage
    // event, the same way a rate-limit-refused attempt is. A regression
    // that "fixes" the counters by dropping the attempt fails here. The
    // count is a lower bound rather than exactly 1 because a retryable
    // classification (see `/v1/responses` below) legitimately produces
    // several attempts for one call.
    expect(delta("aisix_usage_events_emitted_total")).toBeGreaterThanOrEqual(1);
  };

  const ready = async () => {
    const probe = new ProxyClient(app!.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      if (res.status !== 200) return false;
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return (
        data.some((m) => m.id === "dp-cred-group") &&
        data.some((m) => m.id === "dp-url-group")
      );
    });
  };

  // One case per changed call site. `dispatched` is decided independently
  // in each dispatch loop, so covering only one branch would let the other
  // three regress silently.
  const branches: Array<
    [string, number, () => Promise<{ status: number; text: string }>]
  > = [
    [
      "/v1/chat/completions",
      401,
      () =>
        post("/v1/chat/completions", {
          model: "dp-cred-group",
          messages: [{ role: "user", content: "hi" }],
        }),
    ],
    [
      "/v1/chat/completions (streaming)",
      401,
      () =>
        post("/v1/chat/completions", {
          model: "dp-cred-group",
          messages: [{ role: "user", content: "hi" }],
          stream: true,
        }),
    ],
    [
      "/v1/messages",
      401,
      () =>
        post("/v1/messages", {
          model: "dp-cred-group",
          max_tokens: 16,
          messages: [{ role: "user", content: "hi" }],
        }),
    ],
    // `/v1/responses` builds the auth header on its own path and reports
    // the same unusable secret as `BridgeError::Config` (500 config_error)
    // rather than the 401 authentication_error the other three give. That
    // divergence is pinned here as current behavior, not endorsed — it is a
    // status-taxonomy question of its own. It does not affect what this
    // file is about: `Config` is a pre-dispatch variant too, so the
    // deployment counters stay untouched either way. Being retryable, it
    // also produces more than one attempt per call.
    [
      "/v1/responses",
      500,
      () => post("/v1/responses", { model: "dp-cred-group", input: "hi" }),
    ],
  ];

  for (const [label, wantStatus, call] of branches) {
    test(`${label}: an unusable credential counts as no upstream request`, async (ctx) => {
      if (!etcdReachable || !app || !upstream) {
        ctx.skip();
        return;
      }
      await ready();
      await expectNotAttributedToUpstream("dp-badcred", wantStatus, call);
    });
  }

  // The second rejection point. An api_base that does not parse reaches
  // reqwest as a raw string and fails at send() as a builder error, with no
  // socket opened — so it is upstream *config* (400, like a missing
  // api_base), not a transport failure, and owes the deployment families
  // nothing either.
  test("a malformed api_base is upstream config, not a transport failure", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    await ready();
    await expectNotAttributedToUpstream("dp-badurl", 400, () =>
      post("/v1/chat/completions", {
        model: "dp-url-group",
        messages: [{ role: "user", content: "hi" }],
      }),
    );
  });
});
