import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startMockControlPlane,
  startOpenAiUpstream,
  waitConfigPropagation,
  type MockControlPlane,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: what a managed gateway sends to `/dp/telemetry` stays bounded however
// much text the caller or the upstream writes into a request's usage event.
//
// The control plane keeps the first 256 characters of a requested model name
// and the first 8 KiB of other free text, and reads at most a few MiB per
// telemetry POST. A batch it cannot take is lost whole, so one caller's
// oversized value must not travel in a usage event, and a flush of many large
// events must go out as several POSTs that together deliver every event once.

const hash = (plaintext: string) => createHash("sha256").update(plaintext).digest("hex");
const KEY = "sk-tcb-e2e-caller";
const MAX_POST_BYTES = 4 * 1024 * 1024;

interface WireEvent {
  request_id: string;
  requested_model?: string;
  provider_request_id?: string;
  provider_model_version?: string;
  service_tier?: string;
  finish_reason?: string;
}

describe("usage events sent to the control plane are bounded", () => {
  let cp: MockControlPlane | undefined;
  let app: SpawnedApp | undefined;
  const upstreams: OpenAiUpstream[] = [];
  let etcdReachable = false;

  const events = (): WireEvent[] =>
    cp!.telemetryBodies.flatMap((body) => (JSON.parse(body) as { events: WireEvent[] }).events);

  // Upstream metadata that JSON-escapes to six bytes a character.
  const noisy = "\u0001".repeat(16 * 1024);

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    const plain = await startOpenAiUpstream();
    const verbose = await startOpenAiUpstream({
      nonStreamBody: {
        id: `chatcmpl-${noisy}`,
        object: "chat.completion",
        created: 1,
        model: `gpt-${noisy}`,
        service_tier: `tier-${noisy}`,
        choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: `why-${noisy}` }],
        usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
      },
    });
    upstreams.push(plain, verbose);
    cp = await startMockControlPlane();
    app = await spawnApp({ extra: { managed: cp.managed }, awaitProxyListener: false });
    const seed = new SeedClient(etcd, `${app.etcdPrefix}/${cp.envId}`);

    const pkPlain = await seed.createProviderKey({ display_name: "tcb-pk-plain", secret: "sk-mock", api_base: `${plain.baseUrl}/v1` });
    const pkVerbose = await seed.createProviderKey({ display_name: "tcb-pk-verbose", secret: "sk-mock", api_base: `${verbose.baseUrl}/v1` });
    await seed.createModel({ display_name: "tcb/*", provider: "openai", model_name: "*", provider_key_id: pkPlain.id });
    await seed.createModel({ display_name: "tcb-verbose", provider: "openai", model_name: "gpt-4o-mini", provider_key_id: pkVerbose.id });
    await seed.createApiKey({ key_hash: hash(KEY), allowed_models: ["*"] });

    await waitConfigPropagation(async () => {
      try {
        return (await new ProxyClient(app!.proxyUrl, KEY).listModels()).status === 200;
      } catch (e) {
        // The proxy listener binds once the first configuration applies;
        // anything else is a real failure.
        const err = e as { code?: string; cause?: { code?: string } };
        if ((err.code ?? err.cause?.code) === "ECONNREFUSED") return false;
        throw e;
      }
    });
  });

  afterAll(async () => {
    await app?.exit();
    await cp?.close();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  test("a very long model name is served as sent and recorded as its first 256 characters", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const model = `tcb/${"m".repeat(64 * 1024)}`;
    const upstream = upstreams[0]!;
    const before = upstream.receivedRequests.length;

    const res = await new ProxyClient(app.proxyUrl, KEY).chat({ model, messages: [{ role: "user", content: "hi" }] });
    expect(res.status).toBe(200);
    expect(upstream.receivedRequests.length).toBe(before + 1);
    expect((JSON.parse(upstream.receivedRequests[before]!.body) as { model: string }).model).toBe(model.slice(4));

    await waitConfigPropagation(
      async () => events().some((e) => e.requested_model?.startsWith("tcb/mmm")),
      20_000,
    );
    const recorded = events().filter((e) => e.requested_model?.startsWith("tcb/mmm"));
    expect(recorded).toHaveLength(1);
    expect(recorded[0]!.requested_model).toBe(model.slice(0, 256));
  }, 60_000);

  test("a flush larger than one POST carries is split, and every event arrives exactly once", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const client = new ProxyClient(app.proxyUrl, KEY);
    const firstPost = cp!.telemetryBodies.length;
    const count = 100;

    const responses = await Promise.all(
      Array.from({ length: count }, () =>
        client.chat({ model: "tcb-verbose", messages: [{ role: "user", content: "hi" }] }),
      ),
    );
    expect(responses.map((r) => r.status)).toEqual(Array(count).fill(200));

    const verbose = () => events().filter((e) => e.requested_model === "tcb-verbose");
    await waitConfigPropagation(async () => verbose().length >= count, 30_000);
    // One more flush interval: a POST that re-delivered events would land here.
    await new Promise((resolve) => setTimeout(resolve, 6_000));

    const recorded = verbose();
    expect(recorded).toHaveLength(count);
    expect(new Set(recorded.map((e) => e.request_id)).size).toBe(count);
    for (const e of recorded) {
      // The upstream's metadata arrives, clipped, rather than costing the row.
      for (const [field, prefix] of [
        ["provider_request_id", "chatcmpl-"],
        ["provider_model_version", "gpt-"],
        ["service_tier", "tier-"],
        ["finish_reason", "why-"],
      ] as const) {
        const value = e[field];
        expect(value?.startsWith(prefix), field).toBe(true);
        expect(Buffer.byteLength(value!), field).toBeLessThanOrEqual(8 * 1024);
      }
    }

    const posts = cp!.telemetryBodies.slice(firstPost);
    const sizes = posts.map((body) => Buffer.byteLength(body));
    for (const size of sizes) expect(size).toBeLessThanOrEqual(MAX_POST_BYTES);
    // The events did not fit one POST, so the bound actually split them.
    expect(sizes.reduce((a, b) => a + b, 0)).toBeGreaterThan(MAX_POST_BYTES);
    const ids = cp!.telemetryBatchIds.slice(firstPost);
    expect(new Set(ids).size).toBe(ids.length);
    expect(cp!.telemetryAnswers.slice(firstPost).every((status) => status === 200)).toBe(true);
  }, 90_000);
});
