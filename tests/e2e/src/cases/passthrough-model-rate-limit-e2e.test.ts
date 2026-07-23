import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  awaitWindowHeadroom,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: model-level rate limiting on /passthrough/{provider}/*rest.
//
// User journey (verbatim from a field report): an operator registers a
// provider-native video model — reachable ONLY through passthrough,
// since video generation has no typed endpoint — sets `rate_limit`
// `{rpm: 1}` on the Model, and calls the provider's native
// video-synthesis endpoint through the tunnel. The gateway must count
// those calls against the model's cap and 429 past it. Pre-fix the
// tunnel enforced only the API-key-level layers, so a model cap
// configured in the dashboard was silently ignored — for
// passthrough-only models that meant no enforceable model limit
// anywhere in the product.
//
// The target model is identified from the JSON body's top-level
// `model` field — the envelope shared by OpenAI-compatible bodies and
// provider-native ones (e.g. the `model` + `input` + `parameters`
// shape of Alibaba Model Studio's video/image synthesis APIs, see
// <https://help.aliyun.com/zh/model-studio/text-to-image-v2-api-reference>).
//
// Two journeys pinned:
//
//   1. Registered body model → the Model's rpm cap gates the tunnel:
//      request #2 inside the window is 429 `rate_limit_exceeded`
//      with a Retry-After header.
//   2. Unregistered body model → no model-layer cap applies; repeated
//      calls keep flowing (the key-level layers, unset here, remain
//      the only gate — pre-fix behavior preserved).

const CALLER_PLAINTEXT = "sk-pt-mrl-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const VIDEO_MODEL = "pt-rl-video-model";
const SYNTH_PATH = "api/v1/services/aigc/video-generation/video-synthesis";

describe("passthrough e2e: body-model rate limiting on the raw tunnel", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];

  const headers = {
    authorization: `Bearer ${CALLER_PLAINTEXT}`,
    "content-type": "application/json",
  };

  const synthBody = (model: string) =>
    JSON.stringify({
      model,
      input: { prompt: "a cardboard city at night" },
      parameters: { resolution: "720P", duration: 5 },
    });

  const callSynth = async (model: string) =>
    fetch(`${app!.proxyUrl}/passthrough/alibaba/${SYNTH_PATH}`, {
      method: "POST",
      headers,
      body: synthBody(model),
    });

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const upstream = await startOpenAiUpstream({
      nonStreamBody: {
        output: { task_id: "task-pt-rl-01", task_status: "PENDING" },
        request_id: "req-pt-rl-01",
      },
    });
    upstreams.push(upstream);

    const pk = await seed.createProviderKey({
      display_name: "pt-rl-alibaba-pk",
      secret: "sk-mock-dashscope",
      api_base: upstream.baseUrl,
      provider: "alibaba",
    });
    await seed.createModel({
      display_name: VIDEO_MODEL,
      provider: "alibaba",
      model_name: VIDEO_MODEL,
      provider_key_id: pk.id,
      rate_limit: { rpm: 1 },
    });
    // Caller key carries NO rate limit of its own — the only cap in
    // play is the Model's, which is exactly the layer under test.
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });

    // Readiness probe uses an UNREGISTERED model name so it never
    // draws from the video model's rpm=1 bucket.
    await waitConfigPropagation(async () => {
      try {
        const r = await fetch(
          `${app!.proxyUrl}/passthrough/alibaba/${SYNTH_PATH}`,
          { method: "POST", headers, body: synthBody("probe-unregistered") },
        );
        if (r.status !== 200) {
          await r.text();
          return false;
        }
        const j = (await r.json()) as { output?: { task_id?: unknown } };
        return j.output?.task_id === "task-pt-rl-01";
      } catch {
        return false;
      }
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  test("registered body model: second call inside the window is 429 rate_limit_exceeded with Retry-After", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    // The limiter buckets on fixed wall-clock minutes, so a burst that
    // straddles a boundary gets a fresh allowance and the 429 assertion
    // below flaps. Keep the whole burst inside one window.
    await awaitWindowHeadroom();
    // First submission consumes the model's single rpm slot.
    const first = await callSynth(VIDEO_MODEL);
    expect(first.status).toBe(200);
    const firstJson = (await first.json()) as {
      output?: { task_id?: unknown };
    };
    expect(firstJson.output?.task_id).toBe("task-pt-rl-01");

    // Second submission must be rejected by the gateway — NOT reach
    // the upstream — with the standard rate-limit envelope.
    const upstreamCallsBefore = upstreams[0]!.receivedRequests.length;
    const second = await callSynth(VIDEO_MODEL);
    expect(second.status).toBe(429);
    expect(second.headers.get("retry-after")).toBeTruthy();
    const err = (await second.json()) as {
      error?: { type?: unknown };
    };
    expect(err.error?.type).toBe("rate_limit_exceeded");
    // The 429 was produced by the gateway: no upstream round-trip.
    expect(upstreams[0]!.receivedRequests.length).toBe(upstreamCallsBefore);

    // Task polling — the second half of the provider's async journey —
    // is a bodyless GET carrying no `model` field, so the exhausted
    // model bucket must NOT block it. A client that submitted a task
    // right before hitting the cap must still be able to poll it to
    // completion.
    for (let i = 0; i < 3; i += 1) {
      const poll = await fetch(
        `${app!.proxyUrl}/passthrough/alibaba/api/v1/tasks/task-pt-rl-01`,
        { method: "GET", headers: { authorization: headers.authorization } },
      );
      expect(poll.status).toBe(200);
      await poll.text();
    }
  });

  test("unregistered body model: repeated calls keep flowing (no model-layer cap)", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    for (let i = 0; i < 3; i += 1) {
      const r = await callSynth("some-unregistered-model");
      expect(r.status).toBe(200);
      await r.text();
    }
  });
});
