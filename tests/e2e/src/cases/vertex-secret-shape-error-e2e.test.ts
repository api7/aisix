import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { EtcdClient, ProxyClient, SeedClient, spawnApp, waitConfigPropagation, type SpawnedApp } from "../harness/index.js";

// E2E: a Vertex AI provider key whose secret is valid JSON but carries a field
// of the wrong type — the service-account key submitted as a JSON string
// rather than the key object — is refused with an error that names that
// field, not one claiming the secret is not JSON.

const CALLER = "sk-vertex-secret-shape";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");

const SA_KEY = {
  type: "service_account",
  client_email: "sa@p.iam.gserviceaccount.com",
  private_key: "-----BEGIN PRIVATE KEY-----\nMIIB\n-----END PRIVATE KEY-----\n",
  token_uri: "http://127.0.0.1:9/token",
};

describe("vertex provider key with a mistyped secret field", () => {
  let app: SpawnedApp | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "vertex-secret-shape-pk",
      provider: "google",
      adapter: "vertex",
      secret: JSON.stringify({ project: "p", region: "us-central1", service_account_json: JSON.stringify(SA_KEY) }),
      api_base: "http://127.0.0.1:9",
    });
    await seed.createModel({
      display_name: "vertex-secret-shape",
      provider: "google",
      model_name: "gemini-2.5-flash",
      provider_key_id: pk.id,
    });
    // Seeded last: this key authenticating implies the whole seed set landed.
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: ["vertex-secret-shape"] });
    const probe = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
  });

  test("the error names service_account_json and the shape it must have", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify({ model: "vertex-secret-shape", messages: [{ role: "user", content: "hi" }] }),
    });
    const text = await res.text();
    expect(res.status, text).toBe(401);
    const message = JSON.parse(text).error.message as string;
    expect(message).toContain("service_account_json must be a JSON object");
    expect(message).not.toContain("must be valid JSON");
    // No secret material is echoed back.
    expect(message).not.toContain("BEGIN PRIVATE KEY");
  });
});
