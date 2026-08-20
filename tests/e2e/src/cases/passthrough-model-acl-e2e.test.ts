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

// E2E: passthrough routes are an explicit grant on the API key
// (`allowed_routes`, mirroring `allowed_agents`). The
// removed implicit tunnel derived authorization from the key's model ACL
// (#449); routes carry their own dimension: a key with no grant for the
// route's name must not reach the route's upstream credential, whatever
// its model ACL says.

const sha = (s: string) => createHash("sha256").update(s).digest("hex");
const ALLOWED = "sk-ptr-acl-allowed";
const DENIED = "sk-ptr-acl-denied";

describe("passthrough route ACL (allowed_routes)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    upstream = await startOpenAiUpstream({});
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "ptr-acl-pk",
      secret: "sk-openai-mock",
      api_base: "http://unused-on-routes",
    });
    await seed.createPassthroughRoute({
      name: "ptr-acl-tunnel",
      path_prefix: "/passthrough/openai",
      target_url: upstream.baseUrl,
      provider_key_id: pk.id,
    });
    // ALLOWED names the route; DENIED has every model but no route grant —
    // model ACL must not leak into route authorization.
    await seed.createApiKey({
      key_hash: sha(ALLOWED),
      allowed_models: [],
      allowed_routes: ["ptr-acl-tunnel"],
    });
    await seed.createApiKey({
      key_hash: sha(DENIED),
      allowed_models: ["*"],
      allowed_routes: ["unrelated-route"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  const callRoute = (key: string) =>
    fetch(`${app!.proxyUrl}/passthrough/openai/v1/files`, {
      method: "GET",
      headers: { authorization: `Bearer ${key}` },
    });

  test("key without the route grant is rejected; granted key passes", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    await waitConfigPropagation(async () => (await callRoute(ALLOWED)).ok);

    const denied = await callRoute(DENIED);
    expect(
      denied.status,
      "a key whose allowed_routes does not name the route must not reach its upstream credential",
    ).toBe(403);
    const body = (await denied.json()) as { error?: { type?: unknown } };
    expect(body.error?.type).toBe("permission_denied");

    const allowed = await callRoute(ALLOWED);
    expect(allowed.status, "the granted key may use the route").toBe(200);
  });
});
