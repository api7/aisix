import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  metricDelta,
  scrapeMetrics,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";
import { startMockOtlp, type CapturedSpan, type MockOtlp } from "../harness/otlp-mock.js";

// E2E for AISIX-Cloud#1111: a Model Group addressed on /v1/completions
// answered 400 "routing models can't be dispatched directly" — the handler
// dispatched the group itself instead of walking its targets the way
// /v1/chat/completions and /v1/messages do. The same gap sat on every
// other single-shot endpoint, so each one is driven here through a
// two-target failover group whose FIRST target is a failing upstream: a
// 200 served by the second target proves the group was resolved AND that
// it failed over.
//
// The same failover is then read back through telemetry (#655): one usage
// event per upstream attempt — a failed, unbilled `initial` on the first
// target and a `fallback` on the second — plus the per-attempt deployment
// and fallback counters, exactly as /v1/chat/completions reports a
// failover. The usage events are observed through the per-env OTLP
// fan-out, which exports one span per event.

const CALLER_PLAINTEXT = "sk-model-group-single-shot-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

// One body every single-shot endpoint's response parser accepts, so a
// single healthy upstream can serve the whole family.
const OK_BODY = {
  id: "video_ok_1",
  object: "video",
  status: "queued",
  progress: 0,
  created: 1,
  created_at: 1,
  model: "upstream-echo",
  text: "served-by-ok-target",
  choices: [{ index: 0, text: "served-by-ok-target", finish_reason: "stop" }],
  data: [
    { object: "embedding", index: 0, embedding: [0.1, 0.2], url: "https://example.com/i.png" },
  ],
  results: [{ index: 0, relevance_score: 0.9 }],
  usage: {
    prompt_tokens: 3,
    completion_tokens: 2,
    total_tokens: 5,
    input_tokens: 3,
    output_tokens: 2,
  },
};

const OK_UPSTREAM_MODEL = "ok-target-upstream";

type Call = () => Promise<Response>;

/** The spans carrying one request's usage events, in attempt order. */
async function attemptSpans(otlp: MockOtlp, requestId: string, count: number): Promise<CapturedSpan[]> {
  // Attempt carriers only: the export also ships the trace's structural
  // SERVER / logical spans, which share the request id but no attempt.
  const matching = () =>
    otlp.spans
      .filter(
        (s) =>
          s.attributes["aisix.request_id"] === requestId && s.attributes["aisix.attempt_index"] !== undefined,
      )
      .sort((a, b) => Number(a.attributes["aisix.attempt_index"]) - Number(b.attributes["aisix.attempt_index"]));
  const deadline = Date.now() + 10_000;
  while (Date.now() < deadline) {
    if (matching().length >= count) break;
    await new Promise((r) => setTimeout(r, 50));
  }
  // Settle past the count: a surplus event would arrive in the same export.
  await new Promise((r) => setTimeout(r, 300));
  return matching();
}

describe("Model Group dispatch on the single-shot endpoints (AISIX-Cloud#1111)", () => {
  let app: SpawnedApp | undefined;
  let etcdReachable = false;
  let bad: OpenAiUpstream | undefined;
  let ok: OpenAiUpstream | undefined;
  let otlp: MockOtlp | undefined;
  const auth = { authorization: `Bearer ${CALLER_PLAINTEXT}` };

  const json = (path: string, body: Record<string, unknown>): Call => () =>
    fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: { ...auth, "content-type": "application/json" },
      body: JSON.stringify(body),
    });

  const multipart = (path: string, fields: Record<string, string>, file: string): Call => () => {
    const form = new FormData();
    for (const [k, v] of Object.entries(fields)) form.set(k, v);
    form.set(file, new Blob([new Uint8Array([0x49, 0x44, 0x33])], { type: "application/octet-stream" }), "f.bin");
    return fetch(`${app!.proxyUrl}${path}`, { method: "POST", headers: auth, body: form });
  };

  // [label, group name, upstream path the gateway calls, the call]
  const cases: Array<[string, string, string, Call]> = [
    ["completions", "grp-completions", "/v1/completions", json("/v1/completions", { model: "grp-completions", prompt: "hi" })],
    ["embeddings", "grp-embeddings", "/v1/embeddings", json("/v1/embeddings", { model: "grp-embeddings", input: "hi" })],
    ["rerank", "grp-rerank", "/v1/rerank", json("/v1/rerank", { model: "grp-rerank", query: "q", documents: ["a", "b"] })],
    [
      "images/generations",
      "grp-images",
      "/v1/images/generations",
      json("/v1/images/generations", { model: "grp-images", prompt: "a cat" }),
    ],
    [
      "images/edits",
      "grp-images-edits",
      "/v1/images/edits",
      multipart("/v1/images/edits", { model: "grp-images-edits", prompt: "a hat" }, "image"),
    ],
    [
      "audio/transcriptions",
      "grp-transcriptions",
      "/v1/audio/transcriptions",
      multipart("/v1/audio/transcriptions", { model: "grp-transcriptions" }, "file"),
    ],
    [
      "audio/translations",
      "grp-translations",
      "/v1/audio/translations",
      multipart("/v1/audio/translations", { model: "grp-translations" }, "file"),
    ],
    [
      "audio/speech",
      "grp-speech",
      "/v1/audio/speech",
      json("/v1/audio/speech", { model: "grp-speech", input: "hello", voice: "alloy" }),
    ],
    ["videos", "grp-videos", "/v1/videos", json("/v1/videos", { model: "grp-videos", prompt: "a boat" })],
  ];

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    otlp = await startMockOtlp();
    await seed.createObservabilityExporter({
      name: "grp-single-shot-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });

    bad = await startOpenAiUpstream({
      status: 500,
      errorBody: { error: { message: "first target is down" } },
    });
    ok = await startOpenAiUpstream({ nonStreamBody: OK_BODY });

    // The videos surface appends `/v1/videos` to the key's base itself;
    // every other endpoint takes a `/v1` base.
    const pk = async (name: string, base: string) =>
      (await seed.createProviderKey({ display_name: name, secret: "sk-mock", api_base: base })).id;
    const badPk = await pk("grp-bad-pk", `${bad.baseUrl}/v1`);
    const okPk = await pk("grp-ok-pk", `${ok.baseUrl}/v1`);
    const badVideoPk = await pk("grp-bad-video-pk", bad.baseUrl);
    const okVideoPk = await pk("grp-ok-video-pk", ok.baseUrl);

    for (const [label, group] of cases) {
      const video = label === "videos";
      // Each group gets its own failing member, so one endpoint's failure
      // cannot cool down the member another endpoint's group relies on.
      await seed.createModel({
        display_name: `${group}-bad`,
        provider: "openai",
        model_name: "bad-target-upstream",
        provider_key_id: video ? badVideoPk : badPk,
      });
      await seed.createModel({
        display_name: `${group}-ok`,
        provider: "openai",
        model_name: OK_UPSTREAM_MODEL,
        provider_key_id: video ? okVideoPk : okPk,
      });
      await seed.createModel({
        display_name: group,
        routing: {
          strategy: "failover",
          targets: [{ model: `${group}-bad` }, { model: `${group}-ok` }],
          // No same-target retry, so the attempt sequence is exactly
          // initial-on-bad then fallback-on-ok.
          retries: 0,
          max_fallbacks: 1,
        },
      });
    }

    // Seeded last: once it authenticates, every model above is in the
    // snapshot (tests/e2e/AGENTS.md).
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, { headers: auth });
      await res.arrayBuffer();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await bad?.close();
    await ok?.close();
    await otlp?.close();
  });

  test.for(cases)(
    "%s: a Model Group fails over from its failing first target to the second",
    { timeout: 60_000 },
    async ([, group, upstreamPath, call], ctx) => {
      if (!etcdReachable || !app || !bad || !ok || !otlp) {
        ctx.skip();
        return;
      }
      const before = await scrapeMetrics(app.metricsUrl);
      const res = await call();
      // Read to the end: a relayed body's usage event is written when it ends.
      const text = await res.text();
      expect(res.status, `${group}: ${text}`).toBe(200);
      const requestId = res.headers.get("x-aisix-request-id");
      expect(requestId, `${group}: request id header`).toBeTruthy();

      // The failing first target was really tried...
      expect(bad.receivedRequests.some((r) => r.path === upstreamPath)).toBe(true);
      // ...and the second one served the request, with its own upstream
      // model id (a multipart body carries it as a form field).
      const served = ok.receivedRequests.filter((r) => r.path === upstreamPath);
      expect(served).toHaveLength(1);
      expect(served[0].body).toContain(OK_UPSTREAM_MODEL);

      // One usage event per attempt, sharing the request id.
      const spans = await attemptSpans(otlp, requestId!, 2);
      expect(spans, `${group}: attempt events`).toHaveLength(2);
      const [failed, winner] = spans.map((s) => s.attributes);
      expect(failed["aisix.attempt_index"]).toBe(0);
      expect(failed["aisix.attempt_kind"]).toBe("initial");
      expect(failed["aisix.attempt_model"]).toBe(`${group}-bad`);
      expect(failed["aisix.error_class"]).toBe("upstream_status");
      expect(failed["http.response.status_code"]).toBe(502);
      // A failed attempt is not billed.
      expect(failed["gen_ai.usage.input_tokens"] ?? 0).toBe(0);
      expect(failed["gen_ai.usage.output_tokens"] ?? 0).toBe(0);
      expect(winner["aisix.attempt_index"]).toBe(1);
      expect(winner["aisix.attempt_kind"]).toBe("fallback");
      expect(winner["aisix.attempt_model"]).toBe(`${group}-ok`);
      expect(winner["aisix.error_class"]).toBeUndefined();
      expect(winner["http.response.status_code"]).toBe(200);

      // The per-attempt counters, and the request series' fallback flag.
      const after = await scrapeMetrics(app.metricsUrl);
      const delta = (name: string, want: Record<string, string>) => metricDelta(before, after, name, want);
      expect(delta("aisix_deployment_failure_responses_total", { model: `${group}-bad` }), group).toBe(1);
      expect(delta("aisix_deployment_success_responses_total", { model: `${group}-ok` }), group).toBe(1);
      expect(
        delta("aisix_routing_successful_fallbacks_total", { model: group, fallback_model: `${group}-ok` }),
        group,
      ).toBe(1);
      expect(delta("aisix_proxy_requests_total", { endpoint: upstreamPath, is_fallback: "true" }), group).toBe(1);
    },
  );
});
