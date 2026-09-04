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

// A gateway whose configuration set does not fit in one gRPC message must
// still boot on it. The data plane reads its whole prefix in a single
// range response at startup, so the transport's decode ceiling is crossed
// by how MANY resources an environment has, not by any one of them being
// large — and crossing it does not heal: the supervisor backs off and
// re-issues the identical oversized range forever, while the snapshot
// cache keeps serving whatever config last fit.
//
// The fixture is therefore many small models, the shape a deployment
// actually grows into, and everything is seeded BEFORE the gateway starts
// so the oversized read is the bootstrap range.

const CALLER_PLAINTEXT = "sk-large-config-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const SERVING_MODEL = "large-config-model";

/**
 * Bulk models seeded beside the serving one. Sized so the raw key+value
 * bytes alone exceed 4 MiB; `seededBytes` below asserts that rather than
 * trusting the arithmetic here.
 */
const BULK_MODEL_COUNT = 20_000;

interface StatusConfig {
  state: string;
  applied?: { resource_counts: Record<string, number> };
  last_failure: { last_error_kind: string; last_error: string } | null;
}

describe("etcd bootstrap: a configuration set larger than one gRPC message", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcd: EtcdClient | undefined;
  let etcdReachable = false;
  let seededBytes = 0;

  beforeAll(async () => {
    etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();

    // A prefix of our own, seeded before the gateway exists, and handed
    // to it on spawn so its first range read is the oversized one.
    const prefix = `/aisix-e2e-${randomUUID()}`;
    const seed = new SeedClient(etcd, prefix);

    const pk = await seed.createProviderKey({
      display_name: "large-config-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: SERVING_MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });

    const bulk: Array<[string, string]> = [];
    for (let i = 0; i < BULK_MODEL_COUNT; i++) {
      const key = `${prefix}/models/${randomUUID()}`;
      const value = JSON.stringify({
        display_name: `bulk-model-${i}`,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
      seededBytes += Buffer.byteLength(key) + Buffer.byteLength(value);
      bulk.push([key, value]);
    }
    await etcd.putMany(bulk);

    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [SERVING_MODEL],
    });

    app = await spawnApp({ etcdPrefix: prefix });
  }, 300_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  }, 60_000);

  test("the whole set loads, none of it dropped", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    // Key and value bytes only: protobuf framing adds to this, so it is a
    // lower bound on the range response the gateway has to decode. Guards
    // the fixture — a shrunken one would leave the test green against the
    // very ceiling it exists to cross.
    expect(seededBytes).toBeGreaterThan(4 * 1024 * 1024);

    // Wait for the config cycle to SETTLE, either way. Waiting only for
    // success would turn a regression into a timeout; a gateway that
    // cannot decode the range reports its failure within a second and
    // then loops on it, so the assertions below run on a real verdict.
    let status: StatusConfig | undefined;
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.metricsUrl}/status/config`);
      expect(res.status).toBe(200);
      status = (await res.json()) as StatusConfig;
      return status.state === "synced" || status.last_failure !== null;
    });

    expect(status?.state, JSON.stringify(status?.last_failure)).toBe("synced");
    expect(status?.applied?.resource_counts.models).toBe(BULK_MODEL_COUNT + 1);
  }, 120_000);

  test("a model from that set serves traffic", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    const chat = await proxy.chat({
      model: SERVING_MODEL,
      messages: [{ role: "user", content: "does the oversized config serve?" }],
    });
    expect(chat.status, JSON.stringify(chat.body)).toBe(200);
  }, 60_000);
});
