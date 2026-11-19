import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: `allow_all_models: true` grants a key every model in the
// environment, including one created after the key was written, and it
// takes precedence over an empty `allowed_model_ids` (the shape the
// control plane projects, so an older gateway that ignores the flag denies
// instead). The same key document without the flag denies every model.
//
// Reference: OpenAI Chat Completions, Models and Files API specs
// (https://platform.openai.com/docs/api-reference/chat/create,
// https://platform.openai.com/docs/api-reference/models/list,
// https://platform.openai.com/docs/api-reference/files/list).

const hash = (plaintext: string) =>
  createHash("sha256").update(plaintext).digest("hex");

const KEYS = {
  all: "sk-aam-e2e-all",
  emptyIds: "sk-aam-e2e-empty-ids",
  sentinel: "sk-aam-e2e-sentinel",
};

const LATE_MODEL = "aam-late";

describe("api key allow_all_models: every model, including ones added later", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  const proxy = (plaintext: string) =>
    new ProxyClient(app!.proxyUrl, plaintext);

  const chat = (plaintext: string, model: string) =>
    proxy(plaintext).chat({
      model,
      messages: [{ role: "user", content: "hello" }],
    });

  const listedNames = async (plaintext: string) => {
    const listed = await proxy(plaintext).listModels();
    expect(listed.status).toBe(200);
    return (listed.body as { data: { id: string }[] }).data.map((m) => m.id);
  };

  // No `model` hint: the jobs surface falls back to the first direct model
  // the key may use, which is the candidate filter under test.
  const listFiles = (plaintext: string) =>
    fetch(`${app!.proxyUrl}/v1/files`, {
      headers: { authorization: `Bearer ${plaintext}` },
    });

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "aam-e2e-pk",
      api_key: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });

    await seed.createApiKey({
      key_hash: hash(KEYS.all),
      allow_all_models: true,
      allowed_model_ids: [],
    });
    await seed.createApiKey({
      key_hash: hash(KEYS.emptyIds),
      allowed_model_ids: [],
    });
    await seed.createApiKey({
      key_hash: hash(KEYS.sentinel),
      allowed_models: ["*"],
    });
    await waitConfigPropagation(
      async () => (await proxy(KEYS.sentinel).listModels()).status === 200,
    );

    // The model is created only after every key has loaded, so a grant
    // computed when the key was written could not cover it.
    await seed.createModel({
      display_name: LATE_MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await waitConfigPropagation(async () =>
      (await listedNames(KEYS.sentinel)).includes(LATE_MODEL),
    );
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("the flagged key reaches and lists a model created after it", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    expect((await chat(KEYS.all, LATE_MODEL)).status).toBe(200);
    expect(await listedNames(KEYS.all)).toEqual([LATE_MODEL]);
  });

  test("the flagged key is a candidate for the jobs surface's model fallback", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();

    const before = upstream.receivedRequests.length;
    expect((await listFiles(KEYS.all)).status).toBe(200);
    expect(
      upstream.receivedRequests
        .slice(before)
        .some((r) => r.method === "GET" && r.path.startsWith("/v1/files")),
    ).toBe(true);
  });

  test("the same document without the flag denies every model", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // The model exists, so a 403 is authorization and not "not found".
    expect((await chat(KEYS.emptyIds, LATE_MODEL)).status).toBe(403);
    expect(await listedNames(KEYS.emptyIds)).toEqual([]);
    expect((await listFiles(KEYS.emptyIds)).status).toBe(400);
  });
});
