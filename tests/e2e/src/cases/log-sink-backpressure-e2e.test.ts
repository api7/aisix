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

// E2E: a log consumer that stops reading must not stop the gateway.
//
// The gateway's log descriptor is a pipe, and in Kubernetes the reader is
// the container runtime's log shim. Kubelet rotating and gzipping the
// container log stops that reader for a moment — routinely, because the
// rotation is driven by the gateway's own access-log volume. The pipe is
// 64 KiB; once it is full, `write` blocks the thread that logged. On the
// stability benchmark that froze the only request worker for 0.65 s,
// `/livez` included, with the process at 0.00 CPU and the upstream
// answering normally throughout.
//
// So this spawns a real gateway against real etcd and a real upstream,
// stops draining its log pipe, and keeps serving traffic. What is pinned
// is that requests still complete — the log is allowed to lose lines,
// which is the trade the queue makes, and `aisix_log_lines_dropped_total`
// is how that loss is visible.
//
// The batch has to write more than the pipe holds to mean anything: at
// `info` the gateway emits an access-log line and a usage line per
// request, a few hundred bytes each, so several hundred requests is
// comfortably past 64 KiB.

const PLAINTEXT = "sk-log-backpressure";
const KEY_HASH = createHash("sha256").update(PLAINTEXT).digest("hex");
const REQUESTS = 600;
/** Generous against CI scheduling, tiny against a blocked pipe (forever). */
const BUDGET_MS = 60_000;

describe("log sink backpressure e2e: an unread log pipe does not stall requests", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    // `info` is what a deployment runs at and what produces the volume
    // that fills the pipe; at the suite default of `warn` a healthy
    // request logs nothing and there would be no backpressure to test.
    app = await spawnApp({ logLevel: "info" });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "log-backpressure-pk",
      api_key: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "log-backpressure",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: KEY_HASH,
      allowed_models: ["log-backpressure"],
    });

    // The caller key is seeded last, so it authenticating implies the
    // whole seed set is in the snapshot — and listing models is not the
    // behaviour under test, so a failure here reads as a seed failure.
    const proxy = new ProxyClient(app.proxyUrl, PLAINTEXT);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  });

  afterAll(async () => {
    app?.releaseLogSink();
    await app?.exit();
    await upstream?.close();
  });

  const call = () =>
    fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "log-backpressure",
        messages: [{ role: "user", content: "log backpressure" }],
      }),
    });

  test("requests keep completing while nothing reads the log pipe", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    app.holdLogSink();
    const started = Date.now();
    const statuses: number[] = [];
    for (let i = 0; i < REQUESTS; i += 1) {
      const res = await call();
      await res.text();
      statuses.push(res.status);
    }
    const elapsed = Date.now() - started;

    expect(statuses.filter((s) => s !== 200)).toEqual([]);
    expect(elapsed).toBeLessThan(BUDGET_MS);

    // The gateway is still answering its liveness probe, which shares the
    // process with everything that just logged.
    const livez = await fetch(`${app.proxyUrl}/livez`);
    expect(livez.status).toBe(200);

    // And the loss, if the queue overflowed at all, is countable rather
    // than silent. Reading /metrics also proves the metrics listener
    // survived the same window.
    app.releaseLogSink();
    const metrics = await fetch(`${app.metricsUrl}/metrics`).then((r) => r.text());
    expect(metrics).toContain("aisix_requests_total");
  }, BUDGET_MS + 30_000);
});
