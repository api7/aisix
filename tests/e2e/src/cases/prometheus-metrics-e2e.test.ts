import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
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
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      nonStreamBody: responseBody(),
    });
    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    await configureOpenAi(admin, upstream, "prometheus-gpt");
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

    const scrape = await fetch(`${app.adminUrl}/metrics`);
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
    expect(text).toContain('owner_id="unknown"');
  });

  test("custom prometheus path is used for scrapes", async (ctx) => {
    if (!etcdReachable) {
      ctx.skip();
      return;
    }

    const customUpstream = await startOpenAiUpstream({
      nonStreamBody: responseBody(),
    });
    const customApp = await spawnApp({ prometheusPath: "/custom-metrics" });
    try {
      const customAdmin = new AdminClient(customApp.adminUrl, customApp.adminKey);
      await configureOpenAi(customAdmin, customUpstream, "prometheus-custom-gpt");
      const proxy = new ProxyClient(customApp.proxyUrl, CALLER_PLAINTEXT);
      await waitConfigPropagation(async () => {
        const probe = await proxy.chat({
          model: "prometheus-custom-gpt",
          messages: [{ role: "user", content: "ready" }],
        });
        return probe.status === 200;
      });

      const defaultScrape = await fetch(`${customApp.adminUrl}/metrics`);
      expect(defaultScrape.status).toBe(404);

      const scrape = await fetch(`${customApp.adminUrl}/custom-metrics`);
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

  test("disabled prometheus endpoint is not mounted", async (ctx) => {
    if (!etcdReachable) {
      ctx.skip();
      return;
    }

    const disabledApp = await spawnApp({ prometheus: false });
    try {
      const scrape = await fetch(`${disabledApp.adminUrl}/metrics`);
      expect(scrape.status).toBe(404);
      expect(await scrape.text()).not.toContain("aisix_");
    } finally {
      await disabledApp.exit();
    }
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

async function configureOpenAi(
  admin: AdminClient,
  upstream: OpenAiUpstream,
  modelName: string,
) {
  const pk = await admin.createProviderKey({
    display_name: `${modelName}-pk`,
    secret: "sk-mock",
    api_base: `${upstream.baseUrl}/v1`,
  });
  await admin.createModel({
    display_name: modelName,
    provider: "openai",
    model_name: "gpt-4o-mini",
    provider_key_id: pk.id,
  });
  await admin.createApiKey({
    key_hash: CALLER_KEY_HASH,
    allowed_models: [modelName],
  });
}
