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

// E2E: a passthrough target is an outbound boundary. An upstream redirect
// must be relayed to the caller, not followed by the gateway: the next hop
// is neither path-checked against target_url nor entitled to its credential.

const CALLER_PLAINTEXT = "sk-passthrough-redirect-boundary-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");
const PROVIDER_SECRET = "sk-passthrough-redirect-boundary-provider";

const SAME_ORIGIN_ROUTE = "/passthrough/redirect-same-origin";
const CROSS_ORIGIN_ROUTE = "/passthrough/redirect-cross-origin";
const SAME_ORIGIN_LOCATION = "/outside-target/same-origin";

describe("passthrough redirect boundary e2e", () => {
  let app: SpawnedApp | undefined;
  let sameOriginUpstream: OpenAiUpstream | undefined;
  let crossOriginUpstream: OpenAiUpstream | undefined;
  let crossOriginDestination: OpenAiUpstream | undefined;
  let etcdReachable = false;

  const post = (path: string) =>
    fetch(`${app!.proxyUrl}${path}/attempt`, {
      method: "POST",
      redirect: "manual",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "text/plain",
      },
      body: "redirect boundary probe",
    });

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sameOriginUpstream = await startOpenAiUpstream({
      status: 307,
      responseHeaders: { location: SAME_ORIGIN_LOCATION },
    });
    crossOriginDestination = await startOpenAiUpstream();
    crossOriginUpstream = await startOpenAiUpstream({
      status: 308,
      responseHeaders: {
        location: `${crossOriginDestination.baseUrl}/outside-target/cross-origin`,
      },
    });

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const providerKey = await seed.createProviderKey({
      display_name: "passthrough-redirect-boundary-pk",
      secret: PROVIDER_SECRET,
      api_base: sameOriginUpstream.baseUrl,
    });
    await seed.createPassthroughRoute({
      name: "passthrough-redirect-same-origin",
      path_prefix: SAME_ORIGIN_ROUTE,
      target_url: `${sameOriginUpstream.baseUrl}/bounded`,
      auth_mode: "gateway_key",
      credential_mode: "inject",
      provider_key_id: providerKey.id,
    });
    await seed.createPassthroughRoute({
      name: "passthrough-redirect-cross-origin",
      path_prefix: CROSS_ORIGIN_ROUTE,
      target_url: `${crossOriginUpstream.baseUrl}/bounded`,
      auth_mode: "gateway_key",
      credential_mode: "inject",
      provider_key_id: providerKey.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [],
      allowed_routes: ["*"],
    });

    // The API key is seeded last, so a successful auth-only request proves
    // both redirect routes have reached the gateway snapshot.
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  }, 120_000);

  afterAll(async () => {
    await app?.exit();
    await sameOriginUpstream?.close();
    await crossOriginUpstream?.close();
    await crossOriginDestination?.close();
  });

  test("relays a same-origin 307 outside target_url without requesting it", async (ctx) => {
    if (!etcdReachable || !sameOriginUpstream) {
      ctx.skip();
      return;
    }

    const before = sameOriginUpstream.receivedRequests.length;
    const response = await post(SAME_ORIGIN_ROUTE);

    expect(response.status).toBe(307);
    expect(response.headers.get("location")).toBe(SAME_ORIGIN_LOCATION);
    await response.text();

    // The initial request stayed under the configured /bounded mount. A
    // redirect-following client would make an additional /outside-target
    // request on this same origin, escaping that mount unchecked.
    const requests = sameOriginUpstream.receivedRequests.slice(before);
    expect(requests).toHaveLength(1);
    expect(requests[0]?.path).toBe("/bounded/attempt");
    expect(requests[0]?.headers.authorization).toBe(`Bearer ${PROVIDER_SECRET}`);
  });

  test("relays a cross-origin 308 without handing its credential to the target", async (ctx) => {
    if (!etcdReachable || !crossOriginUpstream || !crossOriginDestination) {
      ctx.skip();
      return;
    }

    const sourceBefore = crossOriginUpstream.receivedRequests.length;
    const destinationBefore = crossOriginDestination.receivedRequests.length;
    const location = `${crossOriginDestination.baseUrl}/outside-target/cross-origin`;
    const response = await post(CROSS_ORIGIN_ROUTE);

    expect(response.status).toBe(308);
    expect(response.headers.get("location")).toBe(location);
    await response.text();

    const sourceRequests = crossOriginUpstream.receivedRequests.slice(sourceBefore);
    expect(sourceRequests).toHaveLength(1);
    expect(sourceRequests[0]?.path).toBe("/bounded/attempt");
    expect(sourceRequests[0]?.headers.authorization).toBe(`Bearer ${PROVIDER_SECRET}`);
    expect(crossOriginDestination.receivedRequests.length).toBe(destinationBefore);
  });
});
