import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type ReceivedRequest,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: an Azure deployment name may contain dots, as real Azure deployment
// names commonly do (`gpt-4.1`, `gpt-5.6-sol`). A model whose `model_name`
// is such a deployment reaches the upstream on chat completions,
// `/v1/responses` and `/v1/messages` (both bridged onto Azure chat
// completions), embeddings, image generations and audio speech: on the
// deployment route the dotted name is
// the path segment, on the `v1` route it is the body's `model`.
//
// A deployment name made only of dots would resolve as a `.`/`..` path
// segment and rewrite the URL, so the gateway refuses it without calling
// the upstream.
//
// Azure REST reference:
// https://learn.microsoft.com/en-us/azure/ai-foundry/openai/reference
// https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle

const CALLER = "sk-azure-dotted-deployment";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const SECRET = "az-resource-key";
const DEPLOYMENT = "gpt-5.6-sol";
const EMBED_DEPLOYMENT = "text-embedding-3.small";
const IMAGE_DEPLOYMENT = "gpt-image-1.5";
const SPEECH_DEPLOYMENT = "gpt-4o-mini-tts.2";

const VERSIONS = { unset: undefined, v1: "v1" } as const;
type Version = keyof typeof VERSIONS;
const versions = Object.keys(VERSIONS) as Version[];

const chatModel = (v: Version) => `az-dotted-${v}`;
const embedModel = (v: Version) => `az-dotted-embed-${v}`;
const imageModel = (v: Version) => `az-dotted-image-${v}`;
const speechModel = (v: Version) => `az-dotted-speech-${v}`;
const dotsModel = (dots: string) => `az-dots-${dots.length}`;
const DOTS = [".", ".."];

describe("azure-openai deployment names with dots", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let chat: OpenAiUpstream | undefined;
  let embeddings: OpenAiUpstream | undefined;
  let images: OpenAiUpstream | undefined;
  let speech: OpenAiUpstream | undefined;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    chat = await startOpenAiUpstream();
    embeddings = await startOpenAiUpstream({
      nonStreamBody: {
        object: "list",
        model: EMBED_DEPLOYMENT,
        data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2] }],
        usage: { prompt_tokens: 3, total_tokens: 3 },
      },
    });
    images = await startOpenAiUpstream({ nonStreamBody: { created: 1_700_000_000, data: [{ b64_json: "aGk=" }] } });
    speech = await startOpenAiUpstream({
      rawBody: Buffer.from([0x49, 0x44, 0x33, 0x04]),
      rawContentType: "audio/mpeg",
    });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    for (const [version, apiVersion] of Object.entries(VERSIONS) as [Version, string | undefined][]) {
      for (const [upstream, model, deployment] of [
        [chat, chatModel(version), DEPLOYMENT],
        [embeddings, embedModel(version), EMBED_DEPLOYMENT],
        [images, imageModel(version), IMAGE_DEPLOYMENT],
        [speech, speechModel(version), SPEECH_DEPLOYMENT],
      ] as const) {
        const pk = await seed.createProviderKey({
          display_name: `${model}-pk`,
          provider: "azure",
          adapter: "azure-openai",
          api_key: SECRET,
          api_base: upstream.baseUrl,
          ...(apiVersion ? { api_version: apiVersion } : {}),
        });
        await seed.createModel({
          display_name: model,
          provider: "azure",
          model_name: deployment,
          provider_key_id: pk.id,
        });
      }
    }
    const dotsPk = await seed.createProviderKey({
      display_name: "az-dots-pk",
      provider: "azure",
      adapter: "azure-openai",
      api_key: SECRET,
      api_base: chat.baseUrl,
    });
    for (const dots of DOTS) {
      await seed.createModel({
        display_name: dotsModel(dots),
        provider: "azure",
        model_name: dots,
        provider_key_id: dotsPk.id,
      });
    }
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER}` },
      });
      const body = (await res.json()) as { data?: { id: string }[] };
      return res.status === 200 && (body.data ?? []).some((m) => m.id === dotsModel(".."));
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await chat?.close();
    await embeddings?.close();
    await images?.close();
    await speech?.close();
  });

  async function postJson(path: string, body: unknown): Promise<Response> {
    return fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  async function sentBy(upstream: OpenAiUpstream, call: () => Promise<Response>): Promise<ReceivedRequest> {
    const before = upstream.receivedRequests.length;
    const res = await call();
    const text = await res.text();
    expect(res.status, text).toBe(200);
    const sent = upstream.receivedRequests.slice(before);
    expect(sent).toHaveLength(1);
    return sent[0]!;
  }

  function expectChatOnDeployment(sent: ReceivedRequest, version: Version) {
    if (version === "v1") {
      expect(sent.path).toBe("/openai/v1/chat/completions");
    } else {
      expect(sent.path).toBe(`/openai/deployments/${DEPLOYMENT}/chat/completions?api-version=2024-10-21`);
    }
    expect(sent.headers["api-key"]).toBe(SECRET);
    expect(JSON.parse(sent.body).model).toBe(DEPLOYMENT);
  }

  test.each(versions)("chat completions on a dotted deployment (%s)", async (version) => {
    if (!etcdReachable || !app || !chat) return;
    const sent = await sentBy(chat, () =>
      postJson("/v1/chat/completions", {
        model: chatModel(version),
        messages: [{ role: "user", content: "hi" }],
      }),
    );
    expectChatOnDeployment(sent, version);
  });

  test.each(versions)("/v1/responses on a dotted deployment (%s)", async (version) => {
    if (!etcdReachable || !app || !chat) return;
    const sent = await sentBy(chat, () => postJson("/v1/responses", { model: chatModel(version), input: "hi" }));
    expectChatOnDeployment(sent, version);
  });

  test.each(versions)("/v1/messages on a dotted deployment (%s)", async (version) => {
    if (!etcdReachable || !app || !chat) return;
    const sent = await sentBy(chat, () =>
      postJson("/v1/messages", {
        model: chatModel(version),
        max_tokens: 16,
        messages: [{ role: "user", content: "hi" }],
      }),
    );
    expectChatOnDeployment(sent, version);
  });

  test.each(versions)("embeddings on a dotted deployment (%s)", async (version) => {
    if (!etcdReachable || !app || !embeddings) return;
    const sent = await sentBy(embeddings, () =>
      postJson("/v1/embeddings", { model: embedModel(version), input: "hello" }),
    );
    if (version === "v1") {
      expect(sent.path).toBe("/openai/v1/embeddings");
    } else {
      expect(sent.path).toBe(`/openai/deployments/${EMBED_DEPLOYMENT}/embeddings?api-version=2024-10-21`);
    }
    expect(JSON.parse(sent.body).model).toBe(EMBED_DEPLOYMENT);
  });

  test.each(versions)("image generations on a dotted deployment (%s)", async (version) => {
    if (!etcdReachable || !app || !images) return;
    const sent = await sentBy(images, () =>
      postJson("/v1/images/generations", { model: imageModel(version), prompt: "a lighthouse" }),
    );
    if (version === "v1") {
      expect(sent.path).toBe("/openai/v1/images/generations");
      expect(JSON.parse(sent.body).model).toBe(IMAGE_DEPLOYMENT);
    } else {
      expect(sent.path).toBe(`/openai/deployments/${IMAGE_DEPLOYMENT}/images/generations?api-version=2025-04-01-preview`);
    }
  });

  test.each(versions)("audio speech on a dotted deployment (%s)", async (version) => {
    if (!etcdReachable || !app || !speech) return;
    const sent = await sentBy(speech, () =>
      postJson("/v1/audio/speech", { model: speechModel(version), input: "hello", voice: "alloy" }),
    );
    if (version === "v1") {
      expect(sent.path).toBe("/openai/v1/audio/speech");
    } else {
      expect(sent.path).toBe(`/openai/deployments/${SPEECH_DEPLOYMENT}/audio/speech?api-version=2025-04-01-preview`);
    }
    expect(JSON.parse(sent.body).model).toBe(SPEECH_DEPLOYMENT);
  });

  test.each(DOTS)("a deployment name of only dots (%j) is refused without calling Azure", async (dots) => {
    if (!etcdReachable || !app || !chat) return;
    const before = chat.receivedRequests.length;
    for (const [path, body] of [
      ["/v1/chat/completions", { model: dotsModel(dots), messages: [{ role: "user", content: "hi" }] }],
      ["/v1/responses", { model: dotsModel(dots), input: "hi" }],
      ["/v1/messages", { model: dotsModel(dots), max_tokens: 16, messages: [{ role: "user", content: "hi" }] }],
      ["/v1/embeddings", { model: dotsModel(dots), input: "hello" }],
    ] as const) {
      const res = await postJson(path, body);
      const text = await res.text();
      expect(res.status, `${path}: ${text}`).toBe(400);
      expect(text, path).toContain("deployment name");
    }
    expect(chat.receivedRequests.length - before).toBe(0);
  });
});
