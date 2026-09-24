import { createHash } from "node:crypto";
import { createServer, type Server, type ServerResponse } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient, ProxyClient, SeedClient, pickFreePort, spawnApp, startOpenAiUpstream,
  waitConfigPropagation, type OpenAiUpstream, type SpawnedApp,
} from "../harness/index.js";
import { scrapeMetrics, sumMetric } from "../harness/metrics.js";

const KEY = "sk-latency-lifetime";
const E2E = "aisix_request_e2e_latency_seconds";
const TTFT = "aisix_request_ttft_seconds";
const contexts = [
  { endpoint: "chat/completions", provider: "openai", adapter: "openai" },
  { endpoint: "messages", provider: "anthropic", adapter: "anthropic" },
  { endpoint: "messages", provider: "deepseek", adapter: "openai" },
  { endpoint: "responses", provider: "openai", adapter: "openai" },
  { endpoint: "responses", provider: "deepseek", adapter: "openai" },
  { endpoint: "chat/completions", provider: "ensemble", adapter: "openai" },
];

describe("request latency labels survive buffering and configuration reloads", () => {
  let reachable = false;
  const apps: SpawnedApp[] = [];
  const upstreams: OpenAiUpstream[] = [];
  const servers: Server[] = [];
  beforeAll(async () => { reachable = await new EtcdClient().ping(); });
  afterAll(async () => {
    await Promise.all(apps.map((app) => app.exit()));
    await Promise.all(upstreams.map((upstream) => upstream.close()));
    await Promise.all(servers.map((server) => new Promise<void>((resolve, reject) => {
      server.closeAllConnections();
      server.close((error) => error ? reject(error) : resolve());
    })));
  });

  async function setup() {
    const app = await spawnApp({ extraEnv: {
      AISIX_OBSERVABILITY__METRICS__LABELS: JSON.stringify({
        ...Object.fromEntries([E2E, TTFT].map((metric) => [metric,
          ["endpoint", "model", "upstream_model", "status_class", "streaming", "api_key_id", "team_id", "user_id", "user_name"],
        ])),
      }),
    } });
    apps.push(app);
    return { app, seed: new SeedClient(new EtcdClient(), app.etcdPrefix) };
  }

  async function authorize(app: SpawnedApp, seed: SeedClient, models: string[]) {
    const apiKey = await seed.createApiKey({
      team_id: "original-team", user_id: "original-user", user_name: "Original User",
      key_hash: createHash("sha256").update(KEY).digest("hex"), allowed_models: models,
    });
    const client = new ProxyClient(app.proxyUrl, KEY);
    await waitConfigPropagation(async () => (await client.listModels()).status === 200);
    return { client, apiKey };
  }

  function request(app: SpawnedApp, endpoint: string, model: string) {
    return fetch(`${app.proxyUrl}/v1/${endpoint}`, {
      method: "POST", headers: { authorization: `Bearer ${KEY}`, "content-type": "application/json" },
      body: JSON.stringify({ model, stream: true, max_tokens: 100,
        ...(endpoint === "responses" ? { input: "hello" } : { messages: [{ role: "user", content: "hello" }] }),
      }),
    });
  }

  test("buffered Responses keep streaming=true for both allowed and blocked output", async (ctx) => {
    if (!reachable) return ctx.skip();
    const { app, seed } = await setup();
    await seed.createGuardrail({ name: "block-secret", kind: "keyword", enabled: true,
      hook_point: "output", enforcement_mode: "block", patterns: [{ kind: "literal", value: "SECRET" }],
    });
    for (const [model, text] of [["buffer-allowed", "hello"], ["buffer-blocked", "SECRET"]]) {
      const response = { id: "resp-buffer", model, status: "completed",
        output: [{ type: "message", role: "assistant", content: [{ type: "output_text", text }] }],
        usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
      };
      const upstream = await startOpenAiUpstream({ streamEvents: [
        JSON.stringify({ type: "response.created", response: { ...response, status: "in_progress", output: [] } }),
        JSON.stringify({ type: "response.output_text.delta", delta: text }),
        JSON.stringify({ type: "response.completed", response }),
      ] });
      upstreams.push(upstream);
      const pk = await seed.createProviderKey({ display_name: model, provider: "openai",
        adapter: "openai", secret: "test", api_base: `${upstream.baseUrl}/v1`,
      });
      await seed.createModel({ display_name: model, model_name: model, provider: "openai", provider_key_id: pk.id });
    }
    await authorize(app, seed, ["buffer-allowed", "buffer-blocked"]);
    for (const [model, status] of [["buffer-allowed", 200], ["buffer-blocked", 422]] as const) {
      const response = await request(app, "responses", model);
      const body = await response.text();
      expect(response.status, body).toBe(status);
      const samples = await scrapeMetrics(app.metricsUrl);
      expect(sumMetric(samples, `${E2E}_count`, { model, streaming: "true" })).toBe(1);
      expect(sumMetric(samples, `${E2E}_count`, { model, streaming: "false" })).toBe(0);
    }
  });

  test("deleting models and API keys during native and bridged streams retains request labels", async (ctx) => {
    if (!reachable) return ctx.skip();
    const { app, seed } = await setup();
    const pending: Array<{ res: ServerResponse; terminal: unknown[] }> = [];
    const server = createServer(async (req, res) => {
      let raw = "";
      for await (const chunk of req) raw += chunk.toString();
      const input = JSON.parse(raw);
      if (!input.stream) {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ id: "panel", model: "upstream",
          choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
          usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
        }));
        return;
      }
      // Ensure the gateway measures a non-zero upstream TTFT.
      await new Promise((resolve) => setTimeout(resolve, 20));
      res.writeHead(200, { "content-type": "text/event-stream" });
      if (req.url === "/v1/messages") {
        res.write(`data: ${JSON.stringify({ type: "message_start", message: { id: "msg-lifetime", model: "upstream", type: "message", role: "assistant", content: [], usage: { input_tokens: 10, output_tokens: 0 } } })}\n\n`);
        pending.push({ res, terminal: [
          { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "ok" } },
          { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 5 } },
          { type: "message_stop" },
        ] });
      } else if (req.url === "/v1/responses") {
        res.write(`data: ${JSON.stringify({ type: "response.created", response: { id: "resp-lifetime", model: "upstream", status: "in_progress", output: [] } })}\n\n`);
        pending.push({ res, terminal: [{ type: "response.completed", response: { id: "resp-lifetime", model: "upstream", status: "completed", output: [], usage: { input_tokens: 10, output_tokens: 5 } } }] });
      } else {
        res.write(`data: ${JSON.stringify({ id: "chat-lifetime", model: "upstream", choices: [{ index: 0, delta: { content: "ok" }, finish_reason: null }] })}\n\n`);
        pending.push({ res, terminal: [{ id: "chat-lifetime", model: "upstream", choices: [{ index: 0, delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 } }] });
      }
    });
    servers.push(server);
    const port = await pickFreePort();
    await new Promise<void>((resolve, reject) => {
      server.once("error", reject);
      server.listen(port, "127.0.0.1", resolve);
    });
    const models = [];
    const allowed: string[] = [];
    const requested: string[] = [];
    for (const [index, context] of contexts.entries()) {
      const ensemble = context.provider === "ensemble";
      const provider = ensemble ? "openai" : context.provider;
      const pk = await seed.createProviderKey({ display_name: `lifetime-${index}`,
        provider, adapter: context.adapter,
        secret: "test", api_base: `http://127.0.0.1:${port}/v1`,
      });
      const displayName = ensemble ? "lifetime-member" : `lifetime-${index}/*`;
      const model = await seed.createModel({ display_name: displayName, model_name: "*",
        provider, provider_key_id: pk.id,
      });
      models.push(model);
      if (ensemble) {
        models.push(await seed.createModel({ display_name: "lifetime-ensemble", ensemble: {
          panel: [{ model: displayName }], judge: { model: displayName }, min_responses: 1,
        } }));
      }
      allowed.push(ensemble ? "lifetime-ensemble" : displayName);
      requested.push(ensemble ? "lifetime-ensemble" : `lifetime-${index}/caller-name-${index}`);
    }
    const { client, apiKey } = await authorize(app, seed, allowed);
    const requests = contexts.map((context, index) => request(app, context.endpoint, requested[index]));
    await expect.poll(() => pending.length).toBe(contexts.length);
    // Wait for client headers too: each request has entered its relay before the reload.
    const responses = await Promise.all(requests);
    for (const model of models) await seed.delete("models", model.id);
    await waitConfigPropagation(async () => {
      const response = await client.listModels();
      return response.status === 200 && (response.body as { data: unknown[] }).data.length === 0;
    });
    await seed.delete("api_keys", apiKey.id);
    await waitConfigPropagation(async () => (await client.listModels()).status === 401);
    for (const { res, terminal } of pending) {
      for (const event of terminal) res.write(`data: ${JSON.stringify(event)}\n\n`);
      res.end("data: [DONE]\n\n");
    }
    for (const response of responses) {
      expect(response.status, await response.text()).toBe(200);
    }
    await expect.poll(async () => sumMetric(await scrapeMetrics(app.metricsUrl), `${E2E}_count`)).toBe(contexts.length);
    const samples = await scrapeMetrics(app.metricsUrl);
    for (const [index, context] of contexts.entries()) {
      for (const metric of [E2E, TTFT]) {
        expect(sumMetric(samples, `${metric}_count`, {
          endpoint: `/v1/${context.endpoint}`, model: allowed[index],
          upstream_model: context.provider === "ensemble" ? "unknown" : "*", streaming: "true",
          api_key_id: apiKey.id, team_id: "original-team", user_id: "original-user", user_name: "Original User",
        }), `${metric} ${context.endpoint} ${context.provider}`).toBe(1);
      }
    }
    expect(samples.filter((s) => s.name.startsWith(`${E2E}_`)).every((s) =>
      !s.labels.model.includes("caller-name") && !s.labels.upstream_model.includes("caller-name"),
    )).toBe(true);
  });
});
