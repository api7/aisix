import { createHash } from "node:crypto";
import OpenAI from "openai";
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

// E2E: the gateway runs with the admin listener switched off
// (`admin.enabled = false`), the shape it takes once the Admin API is
// removed. Resources are seeded straight to etcd — never the Admin API —
// and the full request path must still work, with the metrics/status
// listener as the only feedback surface. This pins that the admin
// listener is genuinely not required to serve traffic.

const CALLER_PLAINTEXT = "sk-admin-off-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("admin disabled: gateway serves seeded-via-etcd config with no admin listener", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    // No admin listener is bound — spawnApp gates readiness on the proxy
    // and the metrics listener instead of the admin health endpoint.
    app = await spawnApp({ admin: false });
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "admin-off-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "admin-off-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["admin-off-model"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("a request seeded only through etcd succeeds with the admin listener off", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // A 200 means the ProviderKey + Model + ApiKey — all written to etcd,
    // none through the Admin API — propagated into the snapshot and the
    // proxy dispatched to the upstream.
    let responded = false;
    await waitConfigPropagation(async () => {
      try {
        const r = await client.chat.completions.create({
          model: "admin-off-model",
          messages: [{ role: "user", content: "admin-off-probe" }],
        });
        responded = r.choices[0]?.message.role === "assistant";
        return responded;
      } catch {
        return false;
      }
    });
    expect(responded).toBe(true);
    expect(upstream!.receivedRequests.length).toBeGreaterThan(0);
  });

  test("the metrics/status listener still reports the applied configuration", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    // The load-observability contract is the operational feedback that
    // replaces admin reads: it is served on the metrics listener, so it
    // stays available with the admin listener off.
    const res = await fetch(`${app.metricsUrl}/status/config`);
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      state?: string;
      applied?: { resource_counts?: Record<string, number> };
    };
    expect(typeof body.state).toBe("string");
    expect(body.applied?.resource_counts?.models).toBeGreaterThanOrEqual(1);
  });

  test("no admin listener is bound", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    // With `admin.enabled = false` the admin port is never bound, so a
    // connection to it is refused. (The port was reserved by the harness
    // but nothing listens on it.)
    await expect(
      fetch(`${app.adminUrl}/admin/v1/health`),
    ).rejects.toThrow();
  });
});
