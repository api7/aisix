import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  ProxyClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

const CALLER_PLAINTEXT = "sk-prometheus-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("prometheus metrics e2e", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      nonStreamBody: responseBody(),
    });
    // Held-back: asserts the admin listener's health endpoint, so it keeps
    // admin bound (the suite default is now admin-off).
    app = await spawnApp({ admin: true });
    seed = new SeedClient(etcd, app.etcdPrefix);

    await configureOpenAi(seed, upstream, "prometheus-gpt");
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("scrape contains AISIX-native request and token metrics", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const probe = await proxy.chat({
        model: "prometheus-gpt",
        messages: [{ role: "user", content: "ready" }],
      });
      return probe.status === 200;
    });

    const { status, body } = await proxy.chat({
      model: "prometheus-gpt",
      messages: [{ role: "user", content: "metrics" }],
    });
    expect(status, JSON.stringify(body)).toBe(200);

    const scrape = await fetch(`${app.metricsUrl}/metrics`);
    expect(scrape.status).toBe(200);
    const text = await scrape.text();

    expect(text).toContain("aisix_proxy_requests_total");
    expect(text).toContain("aisix_llm_requests_total");
    expect(text).toContain("aisix_llm_input_tokens_total");
    expect(text).toContain("aisix_llm_output_tokens_total");
    expect(text).toContain("aisix_llm_total_tokens_total");
    expect(text).toContain("aisix_proxy_in_flight_requests");
    expect(text).toMatch(
      /aisix_proxy_requests_total\{[^}]*endpoint="\/v1\/chat\/completions"[^}]*model="prometheus-gpt"[^}]*status="200"/,
    );
    expect(text).toMatch(
      /aisix_llm_requests_total\{[^}]*endpoint="\/v1\/chat\/completions"[^}]*model="prometheus-gpt"[^}]*status="200"/,
    );
    expect(text).toContain('team_id="unknown"');
    expect(text).toContain('user_id="unknown"');
    expect(text).not.toContain("owner_id=");
  });

  test("custom prometheus path is used for scrapes", async (ctx) => {
    if (!etcdReachable) {
      ctx.skip();
      return;
    }

    const customUpstream = await startOpenAiUpstream({
      nonStreamBody: responseBody(),
    });
    const customApp = await spawnApp({ admin: true, prometheusPath: "/custom-metrics" });
    try {
      const customSeed = new SeedClient(new EtcdClient(), customApp.etcdPrefix);
      await configureOpenAi(customSeed, customUpstream, "prometheus-custom-gpt");
      const proxy = new ProxyClient(customApp.proxyUrl, CALLER_PLAINTEXT);
      await waitConfigPropagation(async () => {
        const probe = await proxy.chat({
          model: "prometheus-custom-gpt",
          messages: [{ role: "user", content: "ready" }],
        });
        return probe.status === 200;
      });

      const defaultScrape = await fetch(`${customApp.metricsUrl}/metrics`);
      expect(defaultScrape.status).toBe(404);

      const scrape = await fetch(`${customApp.metricsUrl}/custom-metrics`);
      expect(scrape.status).toBe(200);
      const text = await scrape.text();
      expect(text).toMatch(
        /aisix_proxy_requests_total\{[^}]*endpoint="\/v1\/chat\/completions"[^}]*model="prometheus-custom-gpt"/,
      );
      expect(text).toContain("aisix_llm_total_tokens_total");
    } finally {
      await customApp.exit();
      await customUpstream.close();
    }
  });

  // Issue #408: a successful /v1/chat/completions request must bump
  // `aisix_usage_events_emitted_total{handler="chat", status_code="2xx",
  // inbound_protocol="openai"}` on the DP's /metrics endpoint. Pre-#408
  // the gateway emitted UsageEvents to the sink + OTLP fan-out but
  // had no DP-side prometheus counter, so a regression that dropped
  // emission was invisible to e2e (the harness has no cp-api / OTLP
  // receiver in the loop).
  test("usage_events_emitted counter increments on successful chat (#408)", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    // Reuse the same model the first test configured. The counter
    // accumulates across tests within the same describe block, so we
    // snapshot the pre-call value and assert a delta rather than an
    // absolute count.
    await waitConfigPropagation(async () => {
      const probe = await proxy.chat({
        model: "prometheus-gpt",
        messages: [{ role: "user", content: "ready" }],
      });
      return probe.status === 200;
    });

    const before = await fetch(`${app.metricsUrl}/metrics`).then((r) => r.text());
    const beforeCount = parseUsageEmittedCount(before, "chat", "2xx", "openai");

    const { status } = await proxy.chat({
      model: "prometheus-gpt",
      messages: [{ role: "user", content: "for-#408-counter" }],
    });
    expect(status).toBe(200);

    const after = await fetch(`${app.metricsUrl}/metrics`).then((r) => r.text());
    expect(after).toContain("aisix_usage_events_emitted_total");
    // The counter line must carry all three #408 labels — handler,
    // bucketed status_code, inbound_protocol. A regression that
    // dropped any label (cardinality compromise, mis-spelled key)
    // would surface here.
    expect(after).toMatch(
      /aisix_usage_events_emitted_total\{[^}]*handler="chat"[^}]*\}/,
    );
    expect(after).toMatch(
      /aisix_usage_events_emitted_total\{[^}]*status_code="2xx"[^}]*\}/,
    );
    expect(after).toMatch(
      /aisix_usage_events_emitted_total\{[^}]*inbound_protocol="openai"[^}]*\}/,
    );
    // Status codes MUST be bucketed (2xx / 4xx / 5xx) — raw "200"
    // would explode cardinality at ~1000 series per handler×protocol.
    expect(after).not.toMatch(
      /aisix_usage_events_emitted_total\{[^}]*status_code="200"/,
    );

    const afterCount = parseUsageEmittedCount(after, "chat", "2xx", "openai");
    expect(afterCount - beforeCount).toBeGreaterThanOrEqual(1);
  });

  test("disabled prometheus leaves the metrics port unbound", async (ctx) => {
    if (!etcdReachable) {
      ctx.skip();
      return;
    }

    const disabledApp = await spawnApp({ admin: true, prometheus: false });
    try {
      // No listener at all: the fetch must fail at the connection level,
      // not return an HTTP error.
      await expect(fetch(`${disabledApp.metricsUrl}/metrics`)).rejects.toThrow();
    } finally {
      await disabledApp.exit();
    }
  });

  // The dedicated metrics listener is the ONLY scrape surface — the same
  // mechanism in standalone and managed mode. This pins the topology
  // end-to-end: the metrics port (distinct from admin) serves the
  // AISIX-native series after real proxy traffic, no admin routes leak
  // onto it, and the admin listener does not serve `/metrics` at all.
  test("metrics listener is the only scrape surface", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    // A different port from the admin listener.
    expect(app.metricsUrl).not.toBe(app.adminUrl);

    // Drive traffic so the AISIX-native series exist regardless of
    // which tests ran before this one.
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const probe = await proxy.chat({
        model: "prometheus-gpt",
        messages: [{ role: "user", content: "ready" }],
      });
      return probe.status === 200;
    });

    const scrape = await fetch(`${app.metricsUrl}/metrics`);
    expect(scrape.status).toBe(200);
    const text = await scrape.text();
    expect(text).toContain("aisix_proxy_requests_total");
    expect(text).toContain("aisix_llm_total_tokens_total");

    // The metrics listener carries ONLY metrics — admin routes are not
    // mounted there.
    const adminProbe = await fetch(`${app.metricsUrl}/admin/v1/health`);
    expect(adminProbe.status).toBe(404);

    // And the admin listener no longer hosts the scrape endpoint.
    const adminScrape = await fetch(`${app.adminUrl}/metrics`);
    expect(adminScrape.status).toBe(404);
  });
});

function responseBody() {
  return {
    id: "chatcmpl-prom-1",
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: "gpt-4o-mini",
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: "hello" },
        finish_reason: "stop",
      },
    ],
    usage: { prompt_tokens: 11, completion_tokens: 13, total_tokens: 24 },
  };
}

/**
 * Extract the integer value of one `aisix_usage_events_emitted_total`
 * label combination from a prometheus scrape. Returns 0 if the line
 * is absent (the metric only appears once an emission has happened).
 * Labels may appear in any order in the scrape output; we match each
 * by name independently.
 */
function parseUsageEmittedCount(
  scrape: string,
  handler: string,
  statusCode: string,
  inboundProtocol: string,
): number {
  for (const line of scrape.split("\n")) {
    if (!line.startsWith("aisix_usage_events_emitted_total{")) continue;
    if (!line.includes(`handler="${handler}"`)) continue;
    if (!line.includes(`status_code="${statusCode}"`)) continue;
    if (!line.includes(`inbound_protocol="${inboundProtocol}"`)) continue;
    const valueStr = line.split("}").at(-1)?.trim() ?? "";
    const v = parseInt(valueStr, 10);
    if (!Number.isNaN(v)) return v;
  }
  return 0;
}

async function configureOpenAi(
  seed: SeedClient,
  upstream: OpenAiUpstream,
  modelName: string,
) {
  const pk = await seed.createProviderKey({
    display_name: `${modelName}-pk`,
    secret: "sk-mock",
    api_base: `${upstream.baseUrl}/v1`,
  });
  await seed.createModel({
    display_name: modelName,
    provider: "openai",
    model_name: "gpt-4o-mini",
    provider_key_id: pk.id,
  });
  await seed.createApiKey({
    key_hash: CALLER_KEY_HASH,
    allowed_models: [modelName],
  });
}
