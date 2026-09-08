import { createHash, randomUUID } from "node:crypto";
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

// E2E: an API key may grant models by resource id (`allowed_model_ids`)
// instead of by name (`allowed_models`). The id form is authoritative
// when present, resolves to whatever name the model currently carries,
// and grants nothing for an id that names no model.
//
// Reference: OpenAI Chat Completions and Models API specs
// (https://platform.openai.com/docs/api-reference/chat/create,
// https://platform.openai.com/docs/api-reference/models/list).

const hash = (plaintext: string) =>
  createHash("sha256").update(plaintext).digest("hex");

const KEYS = {
  idsOnly: "sk-ami-e2e-ids-only",
  wildcard: "sk-ami-e2e-wildcard",
  rename: "sk-ami-e2e-rename",
  conflict: "sk-ami-e2e-conflict",
  partial: "sk-ami-e2e-partial",
  nothing: "sk-ami-e2e-nothing",
  sentinel: "sk-ami-e2e-sentinel",
};

describe("api key allowed_model_ids: grants follow the model id, not its name", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  let renameModelId = "";
  let providerKeyId = "";

  const proxy = (plaintext: string) =>
    new ProxyClient(app!.proxyUrl, plaintext);

  const chat = (plaintext: string, model: string) =>
    proxy(plaintext).chat({
      model,
      messages: [{ role: "user", content: "hello" }],
    });

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "ami-e2e-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    providerKeyId = pk.id;
    const model = (display_name: string) =>
      seed!.createModel({
        display_name,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });

    const alpha = await model("ami-alpha");
    const beta = await model("ami-beta");
    const wildcard = await model("ami-gpt-*");
    const renamed = await model("ami-before");
    renameModelId = renamed.id;

    // Every caller key is seeded after every model, and the sentinel key
    // last of all, so gating on the sentinel implies the whole seed set
    // has landed without exercising any behavior under test.
    await seed.createApiKey({
      key_hash: hash(KEYS.idsOnly),
      allowed_model_ids: [alpha.id],
    });
    await seed.createApiKey({
      key_hash: hash(KEYS.wildcard),
      allowed_model_ids: [wildcard.id],
    });
    await seed.createApiKey({
      key_hash: hash(KEYS.rename),
      allowed_model_ids: [renamed.id],
    });
    await seed.createApiKey({
      key_hash: hash(KEYS.conflict),
      allowed_models: ["ami-beta"],
      allowed_model_ids: [alpha.id],
    });
    await seed.createApiKey({
      key_hash: hash(KEYS.partial),
      allowed_model_ids: [alpha.id, randomUUID()],
    });
    await seed.createApiKey({ key_hash: hash(KEYS.nothing) });
    await seed.createApiKey({
      key_hash: hash(KEYS.sentinel),
      allowed_models: ["*"],
    });

    await waitConfigPropagation(
      async () => (await proxy(KEYS.sentinel).listModels()).status === 200,
    );
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("a key granted by id reaches that model and no other", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    expect((await chat(KEYS.idsOnly, "ami-alpha")).status).toBe(200);
    // `ami-beta` exists in the snapshot, so a 403 here is authorization
    // and not "model not found".
    expect((await chat(KEYS.idsOnly, "ami-beta")).status).toBe(403);
  });

  test("an id naming a wildcard model grants every name its pattern covers", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    expect((await chat(KEYS.wildcard, "ami-gpt-4o")).status).toBe(200);
    expect((await chat(KEYS.wildcard, "ami-alpha")).status).toBe(403);
  });

  test("both fields present: the ids decide and the names are ignored", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    expect((await chat(KEYS.conflict, "ami-alpha")).status).toBe(200);
    expect((await chat(KEYS.conflict, "ami-beta")).status).toBe(403);
  });

  test("an id that names no model grants nothing and leaves the rest intact", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    expect((await chat(KEYS.partial, "ami-alpha")).status).toBe(200);
    expect((await chat(KEYS.partial, "ami-beta")).status).toBe(403);
    // The key itself still authenticates — an unusable entry must not
    // cost the caller its credential.
    expect((await proxy(KEYS.partial).listModels()).status).toBe(200);
  });

  test("a key granting no models at all still authenticates and reaches nothing", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // 200 (not 401) proves the document loaded and the key authenticates;
    // the empty listing and the 403 prove it grants no model.
    const listed = await proxy(KEYS.nothing).listModels();
    expect(listed.status).toBe(200);
    expect((listed.body as { data: unknown[] }).data).toEqual([]);
    expect((await chat(KEYS.nothing, "ami-alpha")).status).toBe(403);
  });

  // Last: the rename mutates shared state, and the model it renames is
  // used by no other case.
  test("renaming a model moves the grant with it, key document untouched", async (ctx) => {
    if (!etcdReachable || !app || !seed) return ctx.skip();

    expect((await chat(KEYS.rename, "ami-before")).status).toBe(200);

    // Same etcd key (same resource id), new display name. The api key
    // document is not rewritten anywhere in this test.
    await seed.update("models", renameModelId, {
      display_name: "ami-after",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: providerKeyId,
    });
    // Gate on the rename landing, observed through the unrestricted
    // sentinel key rather than through the key under test.
    await waitConfigPropagation(async () => {
      const listed = await proxy(KEYS.sentinel).listModels();
      if (listed.status !== 200) return false;
      const names = (listed.body as { data: { id: string }[] }).data.map(
        (m) => m.id,
      );
      return names.includes("ami-after") && !names.includes("ami-before");
    });

    expect((await chat(KEYS.rename, "ami-after")).status).toBe(200);
    // No model carries the old name any more, so the old name is simply
    // unknown — the grant did not stay behind on it.
    expect((await chat(KEYS.rename, "ami-before")).status).toBe(404);
  });
});
