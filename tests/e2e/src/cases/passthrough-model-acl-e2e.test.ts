import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: generic passthrough enforces the caller's model ACL (#449).
// Pre-fix, /passthrough/{provider}/* picked the first Model matching the
// provider and lent its credentials to ANY valid API key, regardless of
// the key's allowed_models — so a low-privilege key could reach a
// provider's upstream credentials it was never granted. The gateway now
// requires the key to be allowed to access a model of that provider
// before injecting the provider credential.

const sha = (s: string) => createHash("sha256").update(s).digest("hex");
const ALLOWED = "sk-pt-acl-allowed";
const DENIED = "sk-pt-acl-denied";

describe("passthrough model ACL (#449)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    upstream = await startOpenAiUpstream({});
    app = await spawnApp();
    const admin = new AdminClient(app.adminUrl, app.adminKey);
    const pk = await admin.createProviderKey({
      display_name: "pt-acl-pk",
      secret: "sk-openai-mock",
      api_base: upstream.baseUrl,
    });
    await admin.createModel({
      display_name: "pt-acl-model",
      provider: "openai",
      model_name: "gpt-x",
      provider_key_id: pk.id,
    });
    // ALLOWED key may use the openai model; DENIED key may only use an
    // unrelated model name (no openai model in its ACL).
    await admin.createApiKey({ key_hash: sha(ALLOWED), allowed_models: ["pt-acl-model"] });
    await admin.createApiKey({ key_hash: sha(DENIED), allowed_models: ["unrelated-model"] });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  const callPassthrough = (key: string) =>
    fetch(`${app!.proxyUrl}/passthrough/openai/v1/files`, {
      method: "GET",
      headers: { authorization: `Bearer ${key}` },
    });

  test("key without access to a provider model is rejected (#449)", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    await waitConfigPropagation(async () => (await callPassthrough(ALLOWED)).ok);

    const denied = await callPassthrough(DENIED);
    expect(
      denied.status,
      "key with no openai model in its ACL must not reach openai passthrough creds",
    ).toBe(403);

    const allowed = await callPassthrough(ALLOWED);
    expect(allowed.status, "key allowed for an openai model may use openai passthrough").toBe(200);
  });
});
