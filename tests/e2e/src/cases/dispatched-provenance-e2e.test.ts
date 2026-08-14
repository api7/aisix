import { createHash } from "node:crypto";
import OpenAI, { APIError } from "openai";
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

// E2E: the `aisix_deployment_*` families read as UPSTREAM health for one
// deployment target, so an attempt that never left the gateway must not
// appear in them. Two kinds of attempt never leave: one the target's own
// rate-limit layers refuse, and — the one this test pins — one the bridge
// rejects while still assembling the request. `api_key()` validates the
// secret before building any HTTP request, so a ProviderKey whose secret
// cannot be an Authorization header value fails with
// InvalidUpstreamCredentials having contacted nobody.
//
// Counting that as `aisix_deployment_failure_responses_total` reports our
// own misconfiguration as provider degradation: an operator watching a
// target's failure rate sees the provider "failing" while the provider was
// never asked anything. The upstream request count is the ground truth
// here — it stays at zero across the measured call.
//
// The mirror assertion matters just as much: the attempt is NOT dropped.
// It stays a real attempt in the per-attempt usage events (the rows the
// dashboard log counts), exactly as a rate-limit-refused attempt does. The
// deployment families are the only place it is excluded from.

const CALLER_PLAINTEXT = "sk-dispatch-provenance";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("dispatched provenance e2e: a pre-dispatch bridge error is not upstream health", () => {
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
    const badPk = await seed.createProviderKey({
      display_name: "dp-badcred-pk",
      secret: "sk-live\n-injected",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "dp-badcred",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: badPk.id,
      // The readiness probe below fails this target by design. With
      // cooldown on, the target would be marked and the measured call
      // would be refused before an attempt is ever built — removing the
      // very attempt this test is about.
      cooldown: { enabled: false },
    });
    // Routing model, because the `aisix_deployment_*` families are emitted
    // from the Model-Group dispatch loops: a direct model builds no
    // AttemptRecord at all and would not exercise the code under test.
    // One target keeps the assertion about this attempt rather than about
    // failover ordering.
    await seed.createModel({
      display_name: "dp-virtual",
      routing: {
        strategy: "failover",
        targets: [{ model: "dp-badcred" }],
      },
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["dp-virtual"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("a credential rejected before dispatch counts as no upstream request", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // The caller key is seeded last, so it authenticating implies the whole
    // seed set is in the snapshot. Gating on `dp-virtual` being listed —
    // rather than on a call to it — keeps the behavior under test out of
    // the gate, so a regression fails on an assertion instead of timing out.
    const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      if (res.status !== 200) return false;
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return data.some((m) => m.id === "dp-virtual");
    });

    const upstreamBaseline = upstream.receivedRequests.length;
    const before = await scrapeMetrics(app.metricsUrl);

    let caught: unknown;
    try {
      await client.chat.completions.create({
        model: "dp-virtual",
        messages: [{ role: "user", content: "hi" }],
      });
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(APIError);
    if (!(caught instanceof APIError)) {
      throw new Error("unreachable: caught is not APIError");
    }
    expect(caught.status).toBe(401);

    const after = await scrapeMetrics(app.metricsUrl);
    const delta = (
      name: string,
      want?: Record<string, string> | ((l: Record<string, string>) => boolean),
    ) => metricDelta(before, after, name, want);

    // Ground truth: the provider was never asked anything.
    expect(upstream.receivedRequests.length - upstreamBaseline).toBe(0);

    // Therefore it owes the deployment families nothing. Both were 1
    // before the fix — the failure counter is what made a healthy
    // provider look like it was failing.
    expect(delta("aisix_deployment_requests_total", { model: "dp-badcred" })).toBe(0);
    expect(
      delta("aisix_deployment_failure_responses_total", { model: "dp-badcred" }),
    ).toBe(0);
    expect(
      delta("aisix_deployment_success_responses_total", { model: "dp-badcred" }),
    ).toBe(0);

    // …but the attempt itself is not lost. It is still a per-attempt usage
    // event, the same way a rate-limit-refused attempt is. A regression
    // that "fixed" the counters by dropping the attempt fails here.
    expect(delta("aisix_usage_events_emitted_total", { status_code: "4xx" })).toBe(1);

    // The caller-facing request family sees exactly one 401 request, as
    // it always did — this change is confined to the attempt families.
    expect(delta("aisix_proxy_requests_total", { status: "401" })).toBe(1);
  });
});
