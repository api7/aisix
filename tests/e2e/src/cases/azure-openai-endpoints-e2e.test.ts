import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForSlsLog,
  type MockSls,
  type OpenAiUpstream,
  type ReceivedRequest,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: an `azure-openai` Provider Key serves chat, embeddings, image
// generations and edits, and the three audio routes on Azure's own URLs,
// with the secret in the `api-key` header. The key's `api_version` picks the
// route:
//
// - unset: the deployment route with each operation's default version
//   (`2024-10-21` for chat, embeddings, transcriptions and translations;
//   `2025-04-01-preview` for speech and the image operations);
// - a dated value: the deployment route with that version on every one;
// - `v1`: `/openai/v1/<operation>` with no query, the deployment named in
//   the body's `model`.
//
// The image operations take no `model` on the deployment route.
//
// Azure REST reference:
// https://learn.microsoft.com/en-us/azure/ai-foundry/openai/reference
// https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle

const CALLER = "sk-azure-endpoints";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const SECRET = "az-resource-key";
const CREDENTIAL_REF = "azmock";
const LOGSTORE = "azure-endpoints";
const DATED = "2025-03-01-preview";

const IMAGE_BODY = {
  created: 1_700_000_000,
  data: [{ b64_json: "aGk=" }],
  usage: {
    input_tokens: 14,
    input_tokens_details: { text_tokens: 10, image_tokens: 4 },
    output_tokens: 196,
    total_tokens: 210,
  },
};

const KINDS = ["chat", "embeddings", "images", "transcribe", "speech"] as const;
type Kind = (typeof KINDS)[number];

const VERSIONS = { unset: undefined, dated: DATED, v1: "v1" } as const;
type Version = keyof typeof VERSIONS;

const deployment = (kind: Kind) => `dep-${kind}`;
const modelName = (kind: Kind, version: Version) => `az-${kind}-${version}`;

describe("azure-openai keys dispatch every endpoint on Azure's routes", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  const upstreams = {} as Record<Kind, OpenAiUpstream>;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstreams.chat = await startOpenAiUpstream();
    upstreams.embeddings = await startOpenAiUpstream({
      nonStreamBody: {
        object: "list",
        model: "text-embedding-3-small",
        data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2] }],
        usage: { prompt_tokens: 3, total_tokens: 3 },
      },
    });
    upstreams.images = await startOpenAiUpstream({ nonStreamBody: IMAGE_BODY });
    upstreams.transcribe = await startOpenAiUpstream({ nonStreamBody: { text: "hello world" } });
    upstreams.speech = await startOpenAiUpstream({
      rawBody: Buffer.from([0x49, 0x44, 0x33, 0x04]),
      rawContentType: "audio/mpeg",
    });
    sls = await startMockSls();

    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "LTAI_mock_ak",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock_ak_secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "az-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "aisix-e2e-obs",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
    });
    for (const kind of KINDS) {
      for (const [version, apiVersion] of Object.entries(VERSIONS) as [Version, string | undefined][]) {
        const pk = await seed.createProviderKey({
          display_name: `az-${kind}-${version}-pk`,
          provider: "azure",
          adapter: "azure-openai",
          api_key: SECRET,
          api_base: upstreams[kind].baseUrl,
          ...(apiVersion ? { api_version: apiVersion } : {}),
        });
        await seed.createModel({
          display_name: modelName(kind, version),
          provider: "azure",
          model_name: deployment(kind),
          provider_key_id: pk.id,
        });
      }
    }
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER}` },
      });
      await res.text();
      return res.status === 200;
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await sls?.close();
    await Promise.all(Object.values(upstreams).map((u) => u.close()));
  });

  async function postJson(path: string, body: unknown): Promise<Response> {
    return fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  async function postForm(path: string, form: FormData): Promise<Response> {
    return fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}` },
      body: form,
    });
  }

  /** The single upstream request `call` produced on `kind`'s upstream. */
  async function sentBy(kind: Kind, call: () => Promise<Response>): Promise<{ res: Response; sent: ReceivedRequest; text: string }> {
    const before = upstreams[kind].receivedRequests.length;
    const res = await call();
    const text = await res.text();
    expect(res.status, text).toBe(200);
    const sent = upstreams[kind].receivedRequests.slice(before);
    expect(sent).toHaveLength(1);
    return { res, sent: sent[0]!, text };
  }

  function expectAzureAuth(sent: ReceivedRequest) {
    expect(sent.headers["api-key"]).toBe(SECRET);
    expect(sent.headers["authorization"]).toBeUndefined();
  }

  const expectedPath = (kind: Kind, path: string, version: Version, defaultVersion: string) => {
    if (version === "v1") return `/openai/v1/${path}`;
    const v = version === "dated" ? DATED : defaultVersion;
    return `/openai/deployments/${deployment(kind)}/${path}?api-version=${v}`;
  };

  const versions: Version[] = ["unset", "dated", "v1"];

  test.each(versions)("chat completions (%s)", async (version) => {
    if (!etcdReachable || !app) return;
    const { sent } = await sentBy("chat", () =>
      postJson("/v1/chat/completions", {
        model: modelName("chat", version),
        messages: [{ role: "user", content: "hi" }],
      }),
    );
    expect(sent.path).toBe(expectedPath("chat", "chat/completions", version, "2024-10-21"));
    expectAzureAuth(sent);
    expect(JSON.parse(sent.body).model).toBe(deployment("chat"));
  });

  test.each(versions)("embeddings (%s)", async (version) => {
    if (!etcdReachable || !app) return;
    const { text, sent } = await sentBy("embeddings", () =>
      postJson("/v1/embeddings", { model: modelName("embeddings", version), input: "hello" }),
    );
    expect(sent.path).toBe(expectedPath("embeddings", "embeddings", version, "2024-10-21"));
    expectAzureAuth(sent);
    expect(JSON.parse(sent.body).model).toBe(deployment("embeddings"));
    expect(JSON.parse(text).data[0].embedding).toEqual([0.1, 0.2]);
  });

  test.each(versions)("image generations (%s)", async (version) => {
    if (!etcdReachable || !app) return;
    const { text, sent } = await sentBy("images", () =>
      postJson("/v1/images/generations", {
        model: modelName("images", version),
        prompt: "a lighthouse",
        size: "1024x1024",
        quality: "low",
      }),
    );
    expect(sent.path).toBe(expectedPath("images", "images/generations", version, "2025-04-01-preview"));
    expectAzureAuth(sent);
    const body = JSON.parse(sent.body);
    if (version === "v1") expect(body.model).toBe(deployment("images"));
    else expect("model" in body).toBe(false);
    // The client's parameters go through as sent; the answer comes back as
    // Azure sent it.
    expect(body).toMatchObject({ prompt: "a lighthouse", size: "1024x1024", quality: "low" });
    expect(JSON.parse(text)).toEqual(IMAGE_BODY);
  });

  test.each(versions)("image edits (%s)", async (version) => {
    if (!etcdReachable || !app) return;
    const { sent } = await sentBy("images", () => {
      const form = new FormData();
      form.set("model", modelName("images", version));
      form.set("prompt", "add a hat");
      form.append("image", new Blob([new Uint8Array([0x89, 0x50, 0x4e, 0x47])], { type: "image/png" }), "cat.png");
      return postForm("/v1/images/edits", form);
    });
    expect(sent.path).toBe(expectedPath("images", "images/edits", version, "2025-04-01-preview"));
    expectAzureAuth(sent);
    expect(sent.headers["content-type"]).toContain("multipart/form-data");
    expect(sent.body).toContain("add a hat");
    expect(sent.body).not.toContain(modelName("images", version));
    if (version === "v1") {
      expect(sent.body).toMatch(new RegExp(`name="model"\\r\\n\\r\\n${deployment("images")}\\r\\n`));
    } else {
      expect(sent.body).not.toContain('name="model"');
    }
  });

  for (const [route, path] of [
    ["transcriptions", "audio/transcriptions"],
    ["translations", "audio/translations"],
  ] as const) {
    test.each(versions)(`audio ${route} (%s)`, async (version) => {
      if (!etcdReachable || !app) return;
      const { sent } = await sentBy("transcribe", () => {
        const form = new FormData();
        form.set("model", modelName("transcribe", version));
        form.set("file", new Blob([new Uint8Array([0x49, 0x44, 0x33])], { type: "audio/mpeg" }), "a.mp3");
        return postForm(`/v1/${path}`, form);
      });
      expect(sent.path).toBe(expectedPath("transcribe", path, version, "2024-10-21"));
      expectAzureAuth(sent);
      expect(sent.body).toMatch(new RegExp(`name="model"\\r\\n\\r\\n${deployment("transcribe")}\\r\\n`));
    });
  }

  test.each(versions)("audio speech (%s)", async (version) => {
    if (!etcdReachable || !app) return;
    const { sent } = await sentBy("speech", () =>
      postJson("/v1/audio/speech", { model: modelName("speech", version), input: "hello", voice: "alloy" }),
    );
    expect(sent.path).toBe(expectedPath("speech", "audio/speech", version, "2025-04-01-preview"));
    expectAzureAuth(sent);
    expect(JSON.parse(sent.body)).toMatchObject({ model: deployment("speech"), input: "hello", voice: "alloy" });
  });

  test("an Azure image generation records its token usage", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await postJson("/v1/images/generations", {
      model: modelName("images", "unset"),
      prompt: "a usage marker",
    });
    await res.text();
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    expect(requestId).not.toBe("");
    const row = await waitForSlsLog(
      sls,
      LOGSTORE,
      (l) => l.get("request_id") === requestId,
      "azure image usage row",
      15_000,
    );
    expect(Number(row.get("prompt_tokens"))).toBe(14);
    expect(Number(row.get("completion_tokens"))).toBe(196);
    expect(Number(row.get("total_tokens"))).toBe(210);
    expect(Number(row.get("prompt_image_tokens"))).toBe(4);
    expect(row.get("requested_model")).toBe(modelName("images", "unset"));
  });
});
