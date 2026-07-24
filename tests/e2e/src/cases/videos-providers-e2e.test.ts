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

// E2E: /v1/videos provider adapters beyond DashScope — Zhipu BigModel
// (CogVideoX) and Volcengine Ark (Seedance). One full user journey per
// provider against a mock upstream serving that provider's documented
// response shapes:
//
//   Zhipu:  submit → {id, task_status: PROCESSING}
//           poll   → {task_status: PROCESSING} (in_progress)
//           fetch  → {task_status: SUCCESS, video_result: [{url}]} → 302
//
//   Ark:    submit → {id}                     (queued — no status field)
//           poll   → {status: running}        (in_progress)
//           fetch  → {status: succeeded, content: {video_url}} → 302
//
// The DashScope journeys stay covered by videos-e2e.test.ts.

const CALLER_PLAINTEXT = "sk-videos-prov-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const ZHIPU_MODEL = "videos-e2e-zhipu-model";
const ARK_MODEL = "videos-e2e-ark-model";
const ZHIPU_PROBE_MODEL = "videos-e2e-zhipu-probe";
const ARK_PROBE_MODEL = "videos-e2e-ark-probe";

describe("videos e2e: zhipu + ark provider adapters", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  let zhipuProbeId = "";
  let arkProbeId = "";
  const upstreams: OpenAiUpstream[] = [];

  const headers = {
    authorization: `Bearer ${CALLER_PLAINTEXT}`,
    "content-type": "application/json",
  };

  const submit = async (model: string, extra: Record<string, unknown> = {}) =>
    fetch(`${app!.proxyUrl}/v1/videos`, {
      method: "POST",
      headers,
      body: JSON.stringify({ model, prompt: "a paper boat in the rain", ...extra }),
    });

  const getVideo = async (id: string, suffix = "") =>
    fetch(`${app!.proxyUrl}/v1/videos/${id}${suffix}`, {
      method: "GET",
      headers: { authorization: headers.authorization },
      redirect: "manual",
    });

  const syntheticId = (modelId: string, alias: string, task: string) =>
    Buffer.from(
      `${modelId}:${Buffer.from(alias).toString("base64url")}:${task}`,
    ).toString("base64url");

  let zhipuUpstream: OpenAiUpstream | undefined;
  let arkUpstream: OpenAiUpstream | undefined;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    zhipuUpstream = await startOpenAiUpstream({
      scriptedResponses: [
        {
          nonStreamBody: {
            model: "cogvideox-mock",
            id: "zp-e2e-task-1",
            request_id: "req-zp-1",
            task_status: "PROCESSING",
          },
        },
        {
          nonStreamBody: {
            model: "cogvideox-mock",
            request_id: "req-zp-2",
            task_status: "PROCESSING",
          },
        },
        {
          nonStreamBody: {
            model: "cogvideox-mock",
            request_id: "req-zp-3",
            task_status: "SUCCESS",
            video_result: [
              {
                url: "https://cdn.example.com/videos/cogvideo-e2e.mp4",
                cover_image_url: "https://cdn.example.com/videos/cover.png",
              },
            ],
          },
        },
      ],
    });
    upstreams.push(zhipuUpstream);

    arkUpstream = await startOpenAiUpstream({
      scriptedResponses: [
        { nonStreamBody: { id: "cgt-e2e-0001" } },
        {
          nonStreamBody: {
            id: "cgt-e2e-0001",
            model: "seedance-mock",
            status: "running",
            created_at: 1770000000,
            updated_at: 1770000010,
          },
        },
        {
          nonStreamBody: {
            id: "cgt-e2e-0001",
            model: "seedance-mock",
            status: "succeeded",
            content: {
              video_url: "https://cdn.example.com/videos/seedance-e2e.mp4",
            },
            usage: { completion_tokens: 108900 },
            duration: 5,
            resolution: "720p",
            created_at: 1770000000,
            updated_at: 1770000060,
          },
        },
      ],
    });
    upstreams.push(arkUpstream);

    // Unscripted probe upstreams: any poll returns a completed task in
    // the provider's own shape, so readiness probing never consumes the
    // scripted steps above.
    const zhipuProbeUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        task_status: "SUCCESS",
        video_result: [{ url: "https://cdn.example.com/probe-z.mp4" }],
      },
    });
    upstreams.push(zhipuProbeUpstream);
    const arkProbeUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cgt-probe",
        status: "succeeded",
        content: { video_url: "https://cdn.example.com/probe-a.mp4" },
      },
    });
    upstreams.push(arkProbeUpstream);

    const seedModel = async (
      name: string,
      provider: string,
      modelName: string,
      apiBase: string,
    ) => {
      const pk = await seed!.createProviderKey({
        display_name: `${name}-pk`,
        secret: "sk-mock-provider",
        api_base: apiBase,
        provider,
      });
      return seed!.createModel({
        display_name: name,
        provider,
        model_name: modelName,
        provider_key_id: pk.id,
      });
    };

    await seedModel(
      ZHIPU_MODEL,
      "zhipuai",
      "cogvideox-mock",
      zhipuUpstream.baseUrl,
    );
    await seedModel(ARK_MODEL, "volcengine", "seedance-mock", arkUpstream.baseUrl);
    const zhipuProbe = await seedModel(
      ZHIPU_PROBE_MODEL,
      "zhipuai",
      "cogvideox-mock",
      zhipuProbeUpstream.baseUrl,
    );
    const arkProbe = await seedModel(
      ARK_PROBE_MODEL,
      "volcengine",
      "seedance-mock",
      arkProbeUpstream.baseUrl,
    );
    zhipuProbeId = zhipuProbe.id;
    arkProbeId = arkProbe.id;

    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });

    // Readiness: both probe models must answer a completed poll, which
    // proves the apikey and both providers' model + provider-key rows
    // propagated.
    await waitConfigPropagation(async () => {
      try {
        const z = await getVideo(
          syntheticId(zhipuProbeId, ZHIPU_PROBE_MODEL, "probe-task"),
        );
        if (z.status !== 200) {
          await z.text();
          return false;
        }
        const zj = (await z.json()) as { status?: unknown };
        if (zj.status !== "completed") return false;

        const a = await getVideo(
          syntheticId(arkProbeId, ARK_PROBE_MODEL, "probe-task"),
        );
        if (a.status !== 200) {
          await a.text();
          return false;
        }
        const aj = (await a.json()) as { status?: unknown };
        return aj.status === "completed";
      } catch {
        return false;
      }
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  test("zhipu: submit → poll → content 302, provider-shaped wire", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    const created = await submit(ZHIPU_MODEL, { seconds: 10, size: "1920x1080" });
    expect(created.status).toBe(200);
    const video = (await created.json()) as Record<string, unknown>;
    expect(video.object).toBe("video");
    // The provider has no distinct queued state — an accepted task
    // reports PROCESSING, normalised to in_progress.
    expect(video.status).toBe("in_progress");
    expect(video.model).toBe(ZHIPU_MODEL);
    const id = video.id as string;
    expect(typeof id).toBe("string");

    // Upstream wire: the vendor's flat envelope on the documented path.
    expect(zhipuUpstream!.receivedRequests.length).toBe(1);
    const sub = zhipuUpstream!.receivedRequests[0]!;
    expect(sub.method).toBe("POST");
    expect(sub.path).toBe("/api/paas/v4/videos/generations");
    const wire = JSON.parse(sub.body) as Record<string, unknown>;
    expect(wire.model).toBe("cogvideox-mock");
    expect(wire.prompt).toBe("a paper boat in the rain");
    expect(wire.duration).toBe(10);
    expect(wire.size).toBe("1920x1080");
    expect(sub.headers["x-dashscope-async"]).toBeUndefined();

    const poll = await getVideo(id);
    expect(poll.status).toBe(200);
    const polled = (await poll.json()) as Record<string, unknown>;
    expect(polled.status).toBe("in_progress");
    expect(zhipuUpstream!.receivedRequests[1]!.path).toBe(
      "/api/paas/v4/async-result/zp-e2e-task-1",
    );

    const content = await getVideo(id, "/content");
    expect(content.status).toBe(302);
    expect(content.headers.get("location")).toBe(
      "https://cdn.example.com/videos/cogvideo-e2e.mp4",
    );
  });

  test("ark: submit → poll → content 302, provider-shaped wire", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    const created = await submit(ARK_MODEL, { seconds: "5" });
    expect(created.status).toBe(200);
    const video = (await created.json()) as Record<string, unknown>;
    expect(video.object).toBe("video");
    // The create response carries only the task id — queued.
    expect(video.status).toBe("queued");
    expect(video.model).toBe(ARK_MODEL);
    const id = video.id as string;

    // Upstream wire: content array + top-level duration on the
    // documented tasks path.
    expect(arkUpstream!.receivedRequests.length).toBe(1);
    const sub = arkUpstream!.receivedRequests[0]!;
    expect(sub.method).toBe("POST");
    expect(sub.path).toBe("/api/v3/contents/generations/tasks");
    const wire = JSON.parse(sub.body) as {
      model?: unknown;
      content?: Array<{ type?: unknown; text?: unknown }>;
      duration?: unknown;
    };
    expect(wire.model).toBe("seedance-mock");
    expect(wire.content?.[0]?.type).toBe("text");
    expect(wire.content?.[0]?.text).toBe("a paper boat in the rain");
    expect(wire.duration).toBe(5);

    const poll = await getVideo(id);
    expect(poll.status).toBe(200);
    const polled = (await poll.json()) as Record<string, unknown>;
    expect(polled.status).toBe("in_progress");
    expect(arkUpstream!.receivedRequests[1]!.path).toBe(
      "/api/v3/contents/generations/tasks/cgt-e2e-0001",
    );

    const content = await getVideo(id, "/content");
    expect(content.status).toBe(302);
    expect(content.headers.get("location")).toBe(
      "https://cdn.example.com/videos/seedance-e2e.mp4",
    );
    // Both GETs hit the poll endpoint; no extra submit ever went out.
    expect(
      arkUpstream!.receivedRequests.filter((r) => r.method === "POST").length,
    ).toBe(1);
  });
});
