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

// E2E: OpenAI Sora on /v1/videos — the FIRST Proxy-delivery consumer of the
// content-streaming proxy (AISIX-Cloud#1118 content-proxy design).
//
// Unlike the four signed-URL providers (Alibaba/Zhipu/Volcengine/Runway),
// which 302-redirect the caller to a credential-free URL, Sora's finished
// video is fetched from the provider's own authenticated content endpoint.
// The gateway therefore GETs `…/videos/{id}/content` with the provider
// bearer injected and STREAMS the bytes back — it never redirects and never
// exposes the credential.
//
//   Sora:  submit  → {id, status: queued, progress: 0}
//          poll    → {status: in_progress, progress: 40}
//          content → poll {status: completed} then GET …/content → MP4 bytes,
//                    streamed back to the client as video/mp4 with the
//                    provider bearer on the upstream leg (never leaked).
//
// Plus the upstream-error contract: a 403 on the content GET must surface as
// a JSON error envelope, NOT a video body.
//
// The four signed-URL providers' 302 journeys stay covered by
// videos-providers-e2e.test.ts (they must still 302, not proxy).

const CALLER_PLAINTEXT = "sk-videos-sora-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const PROVIDER_SECRET = "sk-sora-provider-secret";
const SORA_MODEL = "videos-e2e-sora-model";
const SORA_ERR_MODEL = "videos-e2e-sora-err-model";
const SORA_PROBE_MODEL = "videos-e2e-sora-probe";
const MP4_BYTES = "MP4-BYTES-SORA-e2e-streamed-payload";

describe("videos e2e: openai Sora content-streaming proxy", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  let soraErrModelId = "";
  let soraProbeId = "";
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
        prompt: "a paper boat in the rain",
        ...extra,
      }),
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

  let soraUpstream: OpenAiUpstream | undefined;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    // Journey upstream — request order: submit, poll (in_progress),
    // content-route poll (completed), content GET (MP4 bytes).
    soraUpstream = await startOpenAiUpstream({
      scriptedResponses: [
        {
          nonStreamBody: {
            id: "video_e2e_1",
            object: "video",
            model: "sora-2-mock",
            status: "queued",
            progress: 0,
            created_at: 1770000000,
          },
        },
        {
          nonStreamBody: {
            id: "video_e2e_1",
            object: "video",
            model: "sora-2-mock",
            status: "in_progress",
            progress: 40,
            created_at: 1770000000,
          },
        },
        {
          nonStreamBody: {
            id: "video_e2e_1",
            object: "video",
            model: "sora-2-mock",
            status: "completed",
            progress: 100,
            seconds: "8",
            created_at: 1770000000,
          },
        },
        { rawBody: MP4_BYTES, rawContentType: "video/mp4" },
      ],
    });
    upstreams.push(soraUpstream);

    // Error-case upstream — content-route poll returns completed, then the
    // content GET returns a 403 (expired download).
    const soraErrUpstream = await startOpenAiUpstream({
      scriptedResponses: [
        {
          nonStreamBody: {
            id: "video_err",
            object: "video",
            model: "sora-2-mock",
            status: "completed",
            progress: 100,
          },
        },
        {
          status: 403,
          errorBody: {
            error: { code: "expired", message: "download URL has expired" },
          },
        },
      ],
    });
    upstreams.push(soraErrUpstream);

    // Probe upstream — any poll returns a completed task, so readiness
    // probing never consumes the scripted journey steps above.
    const soraProbeUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "video_probe",
        object: "video",
        model: "sora-2-mock",
        status: "completed",
        progress: 100,
      },
    });
    upstreams.push(soraProbeUpstream);

    const seedModel = async (name: string, apiBase: string) => {
      const pk = await seed!.createProviderKey({
        display_name: `${name}-pk`,
        secret: PROVIDER_SECRET,
        api_base: apiBase,
        provider: "openai",
      });
      return seed!.createModel({
        display_name: name,
        provider: "openai",
        model_name: "sora-2-mock",
        provider_key_id: pk.id,
      });
    };

    await seedModel(SORA_MODEL, soraUpstream.baseUrl);
    const errModel = await seedModel(SORA_ERR_MODEL, soraErrUpstream.baseUrl);
    const probe = await seedModel(SORA_PROBE_MODEL, soraProbeUpstream.baseUrl);
    soraErrModelId = errModel.id;
    soraProbeId = probe.id;

    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });

    // Readiness: the probe model must answer a completed poll, proving the
    // apikey + provider-key + model rows propagated to the data plane.
    await waitConfigPropagation(async () => {
      try {
        const r = await getVideo(
          syntheticId(soraProbeId, SORA_PROBE_MODEL, "probe-task"),
        );
        if (r.status !== 200) {
          await r.text();
          return false;
        }
        const rj = (await r.json()) as { status?: unknown };
        return rj.status === "completed";
      } catch {
        return false;
      }
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  test("sora: submit → poll(progress) → content PROXY streams bytes, key not leaked", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    const created = await submit(SORA_MODEL, { seconds: 8, size: "1280x720" });
    expect(created.status).toBe(200);
    const video = (await created.json()) as Record<string, unknown>;
    expect(video.object).toBe("video");
    // The submit response carries a real status; a fresh job is queued.
    expect(video.status).toBe("queued");
    expect(video.model).toBe(SORA_MODEL);
    const id = video.id as string;
    expect(typeof id).toBe("string");

    // Submit wire shape: flat {model, prompt, seconds string, size} on the
    // documented /v1/videos path.
    expect(soraUpstream!.receivedRequests.length).toBe(1);
    const sub = soraUpstream!.receivedRequests[0]!;
    expect(sub.method).toBe("POST");
    expect(sub.path).toBe("/v1/videos");
    const wire = JSON.parse(sub.body) as Record<string, unknown>;
    expect(wire.model).toBe("sora-2-mock");
    expect(wire.prompt).toBe("a paper boat in the rain");
    // seconds is rendered as the string enum the create param types.
    expect(wire.seconds).toBe("8");
    expect(wire.size).toBe("1280x720");

    // Poll passes the real granular progress through (not the binary 0/100).
    const poll = await getVideo(id);
    expect(poll.status).toBe(200);
    const polled = (await poll.json()) as Record<string, unknown>;
    expect(polled.status).toBe("in_progress");
    expect(polled.progress).toBe(40);
    expect(soraUpstream!.receivedRequests[1]!.path).toBe(
      "/v1/videos/video_e2e_1",
    );

    // Content: PROXY delivery. The gateway streams the MP4 bytes back — no
    // 302 redirect — and the response is labelled video/mp4.
    const content = await getVideo(id, "/content");
    expect(content.status).toBe(200);
    expect(content.headers.get("content-type")).toBe("video/mp4");
    expect(content.headers.get("content-disposition")).toContain("attachment");
    const bodyText = await content.text();
    expect(bodyText).toBe(MP4_BYTES);
    // The provider credential must NEVER appear in the client-visible bytes.
    expect(bodyText).not.toContain(PROVIDER_SECRET);

    // The upstream content GET carried the provider bearer (injected by the
    // gateway) — and the client never saw it.
    const contentReq = soraUpstream!.receivedRequests.find((r) =>
      r.path.endsWith("/content"),
    );
    expect(contentReq).toBeDefined();
    expect(contentReq!.method).toBe("GET");
    expect(contentReq!.path).toBe("/v1/videos/video_e2e_1/content");
    expect(contentReq!.headers["authorization"]).toBe(
      `Bearer ${PROVIDER_SECRET}`,
    );
  });

  test("sora: upstream 403 on content GET → JSON error envelope, not a video body", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    const id = syntheticId(soraErrModelId, SORA_ERR_MODEL, "video_err");
    const content = await getVideo(id, "/content");

    // Not a 200 video stream and not a 302 — a typed error response.
    expect(content.status).not.toBe(200);
    expect(content.status).not.toBe(302);
    // The error must not be mislabelled as an MP4.
    expect(content.headers.get("content-type") ?? "").not.toContain(
      "video/mp4",
    );
    const body = (await content.json()) as {
      error?: { message?: unknown };
    };
    expect(String(body.error?.message)).toContain("download URL has expired");
  });
});
