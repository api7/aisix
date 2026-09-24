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

// E2E for AISIX-Cloud#1125: a wildcard Model (`wfam/*`) serves any name a
// caller mints under it, so a metric labelled with the caller's string
// grows one series per invented name. Every typed endpoint must label a
// wildcard-served success with the row's own name — the value `/v1/videos`
// was fixed to first — and never with the caller's. This drives the whole
// family, each through its own minted alias, and reads the whole metrics
// dump afterwards — chat included (`wildcard-identity` pins its streamed
// series), and the jobs surface, whose routing model arrives as a query
// or header hint or inside a caller-supplied gateway id.

const CALLER_PLAINTEXT = "sk-wildcard-metric-family-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

// One body every endpoint's response parser accepts (see the Model Group
// spec for the same trick).
const OK_BODY = {
  id: "video_ok_1",
  object: "video",
  status: "queued",
  progress: 0,
  created: 1,
  created_at: 1,
  model: "upstream-echo",
  text: "served",
  choices: [{ index: 0, text: "served", finish_reason: "stop", message: { role: "assistant", content: "served" } }],
  content: [{ type: "text", text: "served" }],
  role: "assistant",
  type: "message",
  stop_reason: "end_turn",
  output: [
    {
      type: "message",
      id: "msg_1",
      role: "assistant",
      status: "completed",
      content: [{ type: "output_text", text: "served", annotations: [] }],
    },
  ],
  data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2], url: "https://example.com/i.png" }],
  results: [{ index: 0, relevance_score: 0.9 }],
  usage: {
    prompt_tokens: 3,
    completion_tokens: 2,
    total_tokens: 5,
    input_tokens: 3,
    output_tokens: 2,
  },
};

type Call = () => Promise<Response>;

describe("wildcard-served success metrics label as the row across the endpoint family (AISIX-Cloud#1125)", () => {
  let app: SpawnedApp | undefined;
  let etcdReachable = false;
  let upstream: OpenAiUpstream | undefined;
  const auth = { authorization: `Bearer ${CALLER_PLAINTEXT}` };

  const json = (path: string, body: Record<string, unknown>): Call => () =>
    fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: { ...auth, "content-type": "application/json" },
      body: JSON.stringify(body),
    });

  const get = (path: string): Call => () => fetch(`${app!.proxyUrl}${path}`, { headers: auth });

  // A gateway-minted jobs id, `aisix-<base64url("<raw>;model,<model>")>`,
  // is sent back by the caller — so the model inside it is caller-supplied.
  const routedId = (raw: string, model: string) =>
    `aisix-${Buffer.from(`${raw};model,${model}`).toString("base64url")}`;

  const multipart = (path: string, fields: Record<string, string>, file: string): Call => () => {
    const form = new FormData();
    for (const [k, v] of Object.entries(fields)) form.set(k, v);
    form.set(file, new Blob([new Uint8Array([0x49, 0x44, 0x33])], { type: "application/octet-stream" }), "f.bin");
    return fetch(`${app!.proxyUrl}${path}`, { method: "POST", headers: auth, body: form });
  };

  // Every alias carries the `minted` marker, so one assertion over the
  // dump covers every label they could have leaked into.
  const cases: Array<[string, Call]> = [
    [
      "chat/completions",
      json("/v1/chat/completions", { model: "wfam/minted-chat", messages: [{ role: "user", content: "hi" }] }),
    ],
    ["completions", json("/v1/completions", { model: "wfam/minted-completions", prompt: "hi" })],
    ["embeddings", json("/v1/embeddings", { model: "wfam/minted-embeddings", input: "hi" })],
    ["rerank", json("/v1/rerank", { model: "wfam/minted-rerank", query: "q", documents: ["a"] })],
    ["images/generations", json("/v1/images/generations", { model: "wfam/minted-images", prompt: "a cat" })],
    ["images/edits", multipart("/v1/images/edits", { model: "wfam/minted-edits", prompt: "a hat" }, "image")],
    ["audio/transcriptions", multipart("/v1/audio/transcriptions", { model: "wfam/minted-transcriptions" }, "file")],
    ["audio/translations", multipart("/v1/audio/translations", { model: "wfam/minted-translations" }, "file")],
    ["audio/speech", json("/v1/audio/speech", { model: "wfam/minted-speech", input: "hello", voice: "alloy" })],
    [
      "responses",
      json("/v1/responses", { model: "wfam/minted-responses", input: "hi" }),
    ],
    [
      "messages",
      json("/v1/messages", {
        model: "wfam/minted-messages",
        max_tokens: 16,
        messages: [{ role: "user", content: "hi" }],
      }),
    ],
    ["videos", json("/v1/videos", { model: "wvid/minted-videos", prompt: "a boat" })],
    [
      "files",
      multipart("/v1/files?model=wfam/minted-files", { purpose: "batch" }, "file"),
    ],
    [
      "batches",
      json("/v1/batches?model=wfam/minted-batches", {
        input_file_id: "file-raw",
        endpoint: "/v1/chat/completions",
        completion_window: "24h",
      }),
    ],
    [
      "fine_tuning/jobs",
      json("/v1/fine_tuning/jobs?model=wfam/minted-finetune", { model: "base-model", training_file: "file-raw" }),
    ],
    ["files/:id", get(`/v1/files/${routedId("file-raw", "wfam/minted-file-id")}`)],
    ["batches/:id", get(`/v1/batches/${routedId("batch-raw", "wfam/minted-batch-id")}`)],
  ];

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    upstream = await startOpenAiUpstream({ nonStreamBody: OK_BODY });

    const pk = await seed.createProviderKey({
      display_name: "wfam-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    // The videos surface appends `/v1/videos` to the key's base itself.
    const videoPk = await seed.createProviderKey({
      display_name: "wvid-pk",
      secret: "sk-mock",
      api_base: upstream.baseUrl,
    });
    await seed.createModel({
      display_name: "wfam/*",
      provider: "openai",
      model_name: "*",
      provider_key_id: pk.id,
    });
    await seed.createModel({
      display_name: "wvid/*",
      provider: "openai",
      model_name: "*",
      provider_key_id: videoPk.id,
    });

    // Seeded last: once it authenticates, both rows are in the snapshot
    // (tests/e2e/AGENTS.md).
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, { headers: auth });
      await res.arrayBuffer();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("no caller-minted alias becomes a metric label value", { timeout: 60_000 }, async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    for (const [label, call] of cases) {
      const res = await call();
      const text = await res.text();
      expect(res.status, `${label}: ${text}`).toBe(200);
    }

    const metrics = await (await fetch(`${app.metricsUrl}/metrics`)).text();
    // The rows the traffic was served by are the only model identities.
    expect(metrics).toContain('model="wfam/*"');
    expect(metrics).toContain('model="wvid/*"');
    const leaked = metrics.split("\n").filter((l) => l.includes("minted-"));
    expect(leaked).toEqual([]);
  });
});
