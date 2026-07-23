import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: the unified /v1/videos surface (submit → poll → fetch).
//
// User journey: an operator registers an Alibaba DashScope video model,
// the caller submits a generation task through the typed endpoint,
// polls it by the returned video id, and downloads the result via a
// 302 redirect to the provider's video URL — all with the standard
// video-object envelope (`object: "video"`, four-value status enum).
//
// Journeys pinned:
//
//   1. Submit → poll → content-302 happy path against a DashScope-
//      shaped upstream (PENDING → RUNNING → SUCCEEDED + video_url).
//   2. Model `rate_limit.rpm = 1` gates the SUBMIT endpoint: second
//      submit inside the window is a gateway-produced 429
//      `rate_limit_exceeded` with a Retry-After header and no
//      upstream round-trip.
//   3. Polling stays exempt from the exhausted model bucket — a client
//      that paid an RPM slot to submit must be able to poll the task
//      to completion.
//   4. An id the gateway never minted → 404.

const CALLER_PLAINTEXT = "sk-videos-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const HAPPY_MODEL = "videos-e2e-model";
const RL_MODEL = "videos-e2e-rl-model";
const PROBE_MODEL = "videos-e2e-probe-model";

const SUBMIT_PATH = "/api/v1/services/aigc/video-generation/video-synthesis";

describe("videos e2e: unified submit/poll/content surface", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  let probeModelId = "";
  const upstreams: OpenAiUpstream[] = [];

  const headers = {
    authorization: `Bearer ${CALLER_PLAINTEXT}`,
    "content-type": "application/json",
  };

  const submit = async (model: string, extra: Record<string, unknown> = {}) =>
    fetch(`${app!.proxyUrl}/v1/videos`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model,
        prompt: "a cardboard city at night",
        ...extra,
      }),
    });

  const getVideo = async (id: string, suffix = "") =>
    fetch(`${app!.proxyUrl}/v1/videos/${id}${suffix}`, {
      method: "GET",
      headers: { authorization: headers.authorization },
      redirect: "manual",
    });

  // The happy-path upstream serves a scripted DashScope task lifecycle:
  // submit → PENDING, first poll → RUNNING, second poll (the content
  // route) → SUCCEEDED with the video URL.
  let happyUpstream: OpenAiUpstream | undefined;
  // The rate-limit upstream always answers with a PENDING submit shape
  // (the poll route parses the same envelope, mapping to "queued").
  let rlUpstream: OpenAiUpstream | undefined;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    happyUpstream = await startOpenAiUpstream({
      scriptedResponses: [
        {
          nonStreamBody: {
            output: { task_id: "task-happy-01", task_status: "PENDING" },
            request_id: "req-submit-01",
          },
        },
        {
          nonStreamBody: {
            output: { task_id: "task-happy-01", task_status: "RUNNING" },
            request_id: "req-poll-01",
          },
        },
        {
          nonStreamBody: {
            output: {
              task_id: "task-happy-01",
              task_status: "SUCCEEDED",
              video_url: "https://cdn.example.com/videos/task-happy-01.mp4",
            },
            usage: { duration: 8 },
            request_id: "req-poll-02",
          },
        },
      ],
    });
    upstreams.push(happyUpstream);

    rlUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        output: { task_id: "task-rl-01", task_status: "PENDING" },
        request_id: "req-rl-01",
      },
    });
    upstreams.push(rlUpstream);

    // Readiness-probe upstream: unscripted, always SUCCEEDED — its only
    // job is to prove the model/PK/apikey rows have propagated without
    // consuming the scripted steps or the rpm=1 budget above.
    const probeUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        output: {
          task_id: "task-probe-01",
          task_status: "SUCCEEDED",
          video_url: "https://cdn.example.com/videos/probe.mp4",
        },
        request_id: "req-probe-01",
      },
    });
    upstreams.push(probeUpstream);

    const happyPk = await seed.createProviderKey({
      display_name: "videos-e2e-ali-pk",
      secret: "sk-mock-dashscope",
      api_base: happyUpstream.baseUrl,
      provider: "alibaba",
    });
    await seed.createModel({
      display_name: HAPPY_MODEL,
      provider: "alibaba",
      model_name: "wan-mock",
      provider_key_id: happyPk.id,
    });

    const rlPk = await seed.createProviderKey({
      display_name: "videos-e2e-rl-pk",
      secret: "sk-mock-dashscope",
      api_base: rlUpstream.baseUrl,
      provider: "alibaba",
    });
    await seed.createModel({
      display_name: RL_MODEL,
      provider: "alibaba",
      model_name: "wan-mock",
      provider_key_id: rlPk.id,
      rate_limit: { rpm: 1 },
    });

    const probePk = await seed.createProviderKey({
      display_name: "videos-e2e-probe-pk",
      secret: "sk-mock-dashscope",
      api_base: probeUpstream.baseUrl,
      provider: "alibaba",
    });
    const probeModel = await seed.createModel({
      display_name: PROBE_MODEL,
      provider: "alibaba",
      model_name: "wan-mock",
      provider_key_id: probePk.id,
    });
    probeModelId = probeModel.id;

    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });

    // Readiness: poll a synthetic id on the probe model. A completed
    // video object proves the apikey + model + provider-key rows have
    // all propagated. The probe never touches the scripted upstream or
    // the rpm=1 model. Id layout: entry-id : b64url(alias) : task-id,
    // all wrapped in base64url.
    const probeAlias = Buffer.from(PROBE_MODEL).toString("base64url");
    const probeVideoId = Buffer.from(
      `${probeModelId}:${probeAlias}:probe-task`,
    ).toString("base64url");
    await waitConfigPropagation(async () => {
      try {
        const r = await getVideo(probeVideoId);
        if (r.status !== 200) {
          await r.text();
          return false;
        }
        const j = (await r.json()) as { status?: unknown };
        return j.status === "completed";
      } catch {
        return false;
      }
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  test("happy path: submit → poll → content 302 with the video-object envelope", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    // Submit. The response is the video job object: queued, echoing the
    // requested model name, with a gateway-minted id.
    const created = await submit(HAPPY_MODEL, { seconds: "8", size: "1280x720" });
    expect(created.status).toBe(200);
    const video = (await created.json()) as Record<string, unknown>;
    expect(video.object).toBe("video");
    expect(video.status).toBe("queued");
    expect(video.model).toBe(HAPPY_MODEL);
    expect(typeof video.id).toBe("string");
    expect((video.id as string).length).toBeGreaterThan(0);
    expect(typeof video.created_at).toBe("number");
    expect(video.seconds).toBe("8");
    expect(video.size).toBe("1280x720");

    // The upstream saw a DashScope-shaped submit: async header + the
    // {model, input.prompt, parameters} envelope with mapped params.
    expect(happyUpstream!.receivedRequests.length).toBe(1);
    const sub = happyUpstream!.receivedRequests[0]!;
    expect(sub.method).toBe("POST");
    expect(sub.path).toBe(SUBMIT_PATH);
    expect(sub.headers["x-dashscope-async"]).toBe("enable");
    const wire = JSON.parse(sub.body) as {
      model?: unknown;
      input?: { prompt?: unknown };
      parameters?: { duration?: unknown; size?: unknown };
    };
    expect(wire.model).toBe("wan-mock");
    expect(wire.input?.prompt).toBe("a cardboard city at night");
    expect(wire.parameters?.duration).toBe(8);
    expect(wire.parameters?.size).toBe("1280*720");

    const id = video.id as string;

    // Poll: the RUNNING task surfaces as in_progress.
    const poll = await getVideo(id);
    expect(poll.status).toBe(200);
    const polled = (await poll.json()) as Record<string, unknown>;
    expect(polled.object).toBe("video");
    expect(polled.status).toBe("in_progress");
    expect(polled.id).toBe(id);
    expect(happyUpstream!.receivedRequests.length).toBe(2);
    expect(happyUpstream!.receivedRequests[1]!.path).toBe(
      "/api/v1/tasks/task-happy-01",
    );

    // Content: the SUCCEEDED task 302-redirects to the provider URL.
    const content = await getVideo(id, "/content");
    expect(content.status).toBe(302);
    expect(content.headers.get("location")).toBe(
      "https://cdn.example.com/videos/task-happy-01.mp4",
    );
  });

  test("model rpm=1: second submit is a gateway 429; polling stays exempt", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    // First submit consumes the model's single rpm slot.
    const first = await submit(RL_MODEL);
    expect(first.status).toBe(200);
    const firstVideo = (await first.json()) as { id?: unknown };
    const id = firstVideo.id as string;
    expect(typeof id).toBe("string");

    // Second submit inside the window: 429 rate_limit_exceeded with
    // Retry-After, produced by the gateway — no upstream round-trip.
    const upstreamCallsBefore = rlUpstream!.receivedRequests.length;
    const second = await submit(RL_MODEL);
    expect(second.status).toBe(429);
    expect(second.headers.get("retry-after")).toBeTruthy();
    const err = (await second.json()) as { error?: { type?: unknown } };
    expect(err.error?.type).toBe("rate_limit_exceeded");
    expect(rlUpstream!.receivedRequests.length).toBe(upstreamCallsBefore);

    // The exhausted model bucket must NOT gate polling — the poll route
    // is exempt from model-level limits by design.
    for (let i = 0; i < 3; i += 1) {
      const poll = await getVideo(id);
      expect(poll.status).toBe(200);
      const polled = (await poll.json()) as { object?: unknown };
      expect(polled.object).toBe("video");
    }
  });

  test("unknown video id: 404 without contacting any upstream", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    const before = upstreams.reduce(
      (n, u) => n + u.receivedRequests.length,
      0,
    );

    // An id the gateway never minted (not even valid base64url of the
    // expected shape).
    const bogus = await getVideo("not-a-real-video-id");
    expect(bogus.status).toBe(404);
    await bogus.text();

    // A well-formed encoding that names a model entry which does not
    // exist is equally a 404.
    const ghostAlias = Buffer.from("some-model").toString("base64url");
    const ghost = Buffer.from(
      `no-such-model-entry:${ghostAlias}:task-1`,
    ).toString("base64url");
    const ghostResp = await getVideo(ghost);
    expect(ghostResp.status).toBe(404);
    await ghostResp.text();

    const after = upstreams.reduce((n, u) => n + u.receivedRequests.length, 0);
    expect(after).toBe(before);
  });
});
