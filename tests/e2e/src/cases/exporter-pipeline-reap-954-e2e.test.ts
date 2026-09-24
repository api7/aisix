import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForLogLine,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";
import { startMockOtlp, type MockOtlp } from "../harness/otlp-mock.js";

// #954: deleting or disabling an observability exporter stops its delivery
// pipeline. Each exporter gets a pipeline — a worker task and its queue — the
// first time a request is exported to it. Deleting the exporter already
// stopped it RECEIVING events, but the worker lived until the process exited,
// so a gateway whose exporters churn accumulated them, and the delivery-health
// report the gateway sends its control plane kept listing exporters that no
// longer exist.
//
// The pipeline is stopped by the configuration change itself: the spec sends
// no request between the delete and the assertions, so a stop that waited for
// traffic would never be observed. A third exporter that stays configured is
// the control — it must keep its pipeline and keep delivering.

const CALLER_PLAINTEXT = "sk-exporter-reap-954-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const MODEL = "exporter-reap-model";

describe("a deleted or disabled exporter's pipeline is stopped (#954)", () => {
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

  async function chat(target: SpawnedApp, content: string): Promise<Response> {
    const res = await fetch(`${target.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({ model: MODEL, messages: [{ role: "user", content }] }),
    });
    await res.arrayBuffer();
    return res;
  }

  async function waitForPosts(recv: MockOtlp, atLeast: number, what: string) {
    const deadline = Date.now() + 10_000;
    while (recv.posts < atLeast) {
      if (Date.now() >= deadline) {
        throw new Error(`${what}: ${recv.posts} OTLP POSTs, wanted ${atLeast}`);
      }
      await new Promise((r) => setTimeout(r, 50));
    }
  }

  test(
    "deleting one exporter and disabling another stops both pipelines without any traffic",
    async (ctx) => {
      if (!etcdReachable || !upstream) {
        ctx.skip();
        return;
      }
      const [keepRecv, goneRecv, offRecv] = await Promise.all([
        startMockOtlp(),
        startMockOtlp(),
        startMockOtlp(),
      ]);
      receivers.push(keepRecv, goneRecv, offRecv);

      // `info`: the pipeline lifecycle lines this spec reads are info-level.
      app = await spawnApp({ logLevel: "info" });
      const seed = new SeedClient(new EtcdClient(), app.etcdPrefix);
      const otlp = (name: string, recv: MockOtlp) => ({
        name,
        enabled: true,
        kind: "otlp_http",
        endpoint: recv.url,
      });
      await seed.createObservabilityExporter(otlp("reap-keep", keepRecv));
      const gone = await seed.createObservabilityExporter(otlp("reap-gone", goneRecv));
      const off = await seed.createObservabilityExporter(otlp("reap-off", offRecv));
      const pk = await seed.createProviderKey({
        display_name: "exporter-reap-pk",
        secret: "sk-mock-exporter-reap",
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

      // One request starts all three pipelines.
      expect((await chat(app, "start")).status).toBe(200);
      await waitForPosts(keepRecv, 1, "reap-keep before the change");
      await waitForPosts(goneRecv, 1, "reap-gone before the change");
      await waitForPosts(offRecv, 1, "reap-off before the change");

      await seed.delete("observability_exporters", gone.id);
      await seed.update("observability_exporters", off.id, {
        ...off.value,
        enabled: false,
      });

      for (const name of ["reap-gone", "reap-off"]) {
        await waitForLogLine(
          app,
          (l) => l.includes("stopping removed exporter pipeline") && l.includes(name),
          `the stop line for ${name}`,
        );
        // The worker itself exits, on whichever stop signal it sees first.
        // Under thread-per-core it runs on the worker runtime of the request
        // that started it, not on the one that reacted to the change.
        await waitForLogLine(
          app,
          (l) =>
            /sink pipeline(: channel closed, exiting| shutting down)/.test(l) &&
            l.includes(`sink=${name}`),
          `the worker exit line for ${name}`,
        );
      }

      // The exporter that is still configured keeps delivering, through the
      // pipeline it already had.
      const after = await chat(app, "after");
      expect(after.status).toBe(200);
      await waitForPosts(keepRecv, 2, "reap-keep after the change");
      // Barrier: this request's access-log line is queued after anything the
      // configuration change logged, so a stop line for the control exporter
      // would be visible by now.
      const afterId = after.headers.get("x-aisix-request-id") ?? "";
      await waitForLogLine(
        app,
        (l) => l.includes("proxy request completed") && l.includes(`request_id="${afterId}"`),
        `the access-log line for ${afterId}`,
      );
      expect(
        app
          .output()
          .split("\n")
          .filter((l) => l.includes("stopping removed exporter pipeline") && l.includes("reap-keep")),
      ).toEqual([]);
    },
    60_000,
  );
});
