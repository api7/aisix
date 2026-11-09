import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  scrapeMetrics,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";
import { startMockOtlp, type MockOtlp } from "../harness/otlp-mock.js";

// #901: under thread-per-core, an exporter's delivery pipeline is a long-lived
// task on the control runtime, not on the worker whose request first emitted
// an event for it. Each pipeline is one task that stays alive after its first
// delivery, so three exporters add at least three tasks to the control runtime.

const CALLER_PLAINTEXT = "sk-exporter-runtime-901-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const MODEL = "exporter-runtime-model";

describe("exporter pipelines run on the control runtime (#901)", () => {
  let etcdReachable = false;
  let upstream: OpenAiUpstream | undefined;
  let app: SpawnedApp | undefined;
  const receivers: MockOtlp[] = [];

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    upstream = await startOpenAiUpstream();
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(receivers.map((r) => r.close()));
    await upstream?.close();
  });

  async function controlTasks(target: SpawnedApp): Promise<number> {
    const samples = await scrapeMetrics(target.metricsUrl);
    const control = samples.find(
      (s) => s.name === "aisix_runtime_alive_tasks" && s.labels.runtime === "control",
    );
    expect(control, "aisix_runtime_alive_tasks{runtime=control}").toBeDefined();
    return control!.value;
  }

  test(
    "three exporters first sighted on a worker add their tasks to the control runtime",
    async (ctx) => {
      if (!etcdReachable || !upstream) {
        ctx.skip();
        return;
      }
      receivers.push(await startMockOtlp(), await startMockOtlp(), await startMockOtlp());

      app = await spawnApp({ threadPerCore: true });
      const seed = new SeedClient(new EtcdClient(), app.etcdPrefix);
      for (const [i, recv] of receivers.entries()) {
        await seed.createObservabilityExporter({
          name: `runtime-${i}`,
          enabled: true,
          kind: "otlp_http",
          endpoint: recv.url,
        });
      }
      const pk = await seed.createProviderKey({
        display_name: "exporter-runtime-pk",
        secret: "sk-mock-exporter-runtime",
        api_base: `${upstream.baseUrl}/v1`,
      });
      await seed.createModel({
        display_name: MODEL,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
      await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: [MODEL] });
      const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
      await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);

      const before = await controlTasks(app);

      const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({ model: MODEL, messages: [{ role: "user", content: "hi" }] }),
      });
      await res.arrayBuffer();
      expect(res.status).toBe(200);

      const deadline = Date.now() + 10_000;
      while (receivers.some((r) => r.posts < 1)) {
        if (Date.now() >= deadline) {
          throw new Error(`OTLP POSTs: ${receivers.map((r) => r.posts).join(", ")}`);
        }
        await new Promise((r) => setTimeout(r, 50));
      }

      expect(await controlTasks(app)).toBeGreaterThanOrEqual(before + 3);
    },
    60_000,
  );
});
