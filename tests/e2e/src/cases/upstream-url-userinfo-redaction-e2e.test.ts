import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  waitConfigPropagation,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: an Azure OpenAI provider key whose `api_base` embeds userinfo is
// refused with a 400 that tells the caller why — and the base it quotes
// must not carry the credential, because the message is returned to
// whoever made the API call, not only to the operator.

const CALLER_PLAINTEXT = "sk-userinfo-redaction";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");
const CREDENTIAL = "hunter2-e2e";

describe("configured upstream URL userinfo is redacted e2e", () => {
  let app: SpawnedApp | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const bases: Record<string, string> = {
      "azure-override": `https://proxy-user:${CREDENTIAL}@azure-proxy.invalid/openai`,
      "azure-canonical": `https://proxy-user:${CREDENTIAL}@acme.openai.azure.com`,
    };
    for (const [name, api_base] of Object.entries(bases)) {
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        provider: "azure-openai",
        adapter: "azure-openai",
        secret: "az-key",
        api_base,
      });
      await seed.createModel({
        display_name: name,
        provider: "azure-openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
    }
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: Object.keys(bases),
    });
  });

  afterAll(async () => {
    await app?.exit();
  });

  test.for([
    ["azure-override", "https://***@azure-proxy.invalid/openai"],
    ["azure-canonical", "https://***@acme.openai.azure.com"],
  ] as const)("%s answers 400 without echoing the credential", async ([model, shown], ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const client = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const res = await client.listModels();
      return res.status === 200;
    });

    const res = await client.chat({
      model,
      messages: [{ role: "user", content: "hi" }],
    });
    expect(res.status).toBe(400);
    const message =
      (res.body as { error?: { message?: string } }).error?.message ?? "";
    expect(message).toContain("userinfo");
    expect(message).toContain(shown);
    expect(JSON.stringify(res.body)).not.toContain(CREDENTIAL);
  });
});
