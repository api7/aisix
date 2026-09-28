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

// E2E: an Azure OpenAI provider key whose `api_base` carries userinfo is
// used as configured — the request reaches the upstream, with the
// userinfo turned into whatever the HTTP client makes of it.

const CALLER = "sk-upstream-url-userinfo";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const MODEL = "azure-userinfo";

describe("configured upstream URL with userinfo is used as configured", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "azure-userinfo-pk",
      provider: "azure-openai",
      adapter: "azure-openai",
      secret: "az-key",
      api_base: upstream.baseUrl.replace("http://", "http://proxy-user:proxy-pw@"),
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "azure-openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: [MODEL] });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("the chat request reaches the upstream", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();
    const client = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => (await client.listModels()).status === 200);

    const res = await client.chat({ model: MODEL, messages: [{ role: "user", content: "hi" }] });
    expect(res.status, JSON.stringify(res.body)).toBe(200);
    const hit = upstream.receivedRequests.find((r) => r.path.includes("/chat/completions"));
    expect(hit, JSON.stringify(upstream.receivedRequests.map((r) => r.path))).toBeDefined();
  });
});
