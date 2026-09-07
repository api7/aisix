import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { gunzipSync } from "node:zlib";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient, ProxyClient, SeedClient, pickFreePort, spawnApp,
  waitConfigPropagation, type SpawnedApp,
} from "../harness/index.js";

const KEY = "sk-cache-write-usage-test";
const surfaces = ["chat/completions", "messages", "responses", "completions", "responses-bridge"];
const cases = surfaces.flatMap((surface) => (surface === "completions" ? [false] : [false, true]).flatMap((stream) =>
  [37, 0, undefined].map((write) => ({
    surface, stream, write,
    model: `write-${surface.replaceAll("/", "-")}-${stream}-${write ?? "absent"}`,
  })),
));

describe("OpenAI cache-write usage survives every supported entry point", () => {
  let app: SpawnedApp | undefined;
  let server: Server | undefined;
  let reachable = false;
  const logs: Record<string, unknown>[] = [];
  let intakeError: unknown;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    reachable = await etcd.ping();
    if (!reachable) return;
    server = createServer(async (req, res) => {
      const parts: Buffer[] = [];
      for await (const part of req) parts.push(Buffer.from(part));
      const bytes = Buffer.concat(parts);
      if (req.url === "/api/v2/logs") {
        try { logs.push(...JSON.parse(gunzipSync(bytes).toString())); }
        catch (error) { intakeError = error; }
        res.writeHead(202).end();
        return;
      }
      const request = JSON.parse(bytes.toString());
      const write = request.model.endsWith("absent") ? undefined
        : request.model.endsWith("-37") ? 37 : 0;
      const details = { cached_tokens: 19, ...(write === undefined ? {} : { cache_write_tokens: write }) };
      const responses = req.url === "/v1/responses";
      const usage = responses
        ? { input_tokens: 101, output_tokens: 11, total_tokens: 112, input_tokens_details: details }
        : { prompt_tokens: 101, completion_tokens: 11, total_tokens: 112, prompt_tokens_details: details };
      const response = responses ? {
        id: "resp-write", object: "response", model: request.model, status: "completed",
        output: [{ type: "message", role: "assistant", content: [{ type: "output_text", text: "ok" }] }], usage,
      } : {
        id: "chatcmpl-write", object: "chat.completion", created: 1, model: request.model,
        choices: [{ index: 0, text: "ok", message: { role: "assistant", content: "ok" }, finish_reason: "stop" }], usage,
      };
      if (!request.stream) {
        res.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify(response));
        return;
      }
      res.writeHead(200, { "content-type": "text/event-stream" });
      const events = responses ? [
        { type: "response.created", response: { ...response, status: "in_progress", usage: null } },
        { type: "response.completed", response },
      ] : [
        { ...response, usage: undefined, choices: [{ index: 0, text: "ok", delta: { content: "ok" }, finish_reason: null }] },
        { ...response, usage: undefined, choices: [{ index: 0, text: "", delta: {}, finish_reason: "stop" }] },
        { ...response, choices: [], usage },
      ];
      for (const event of events) res.write(`data: ${JSON.stringify(event)}\n\n`);
      if (!responses) res.write("data: [DONE]\n\n");
      res.end();
    });
    const port = await pickFreePort();
    await new Promise<void>((resolve) => server!.listen(port, "127.0.0.1", resolve));
    app = await spawnApp({ extraEnv: { DD_CRED_WRITE_API_KEY: "test" } });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "cache-write", kind: "datadog", enabled: true,
      site: `127.0.0.1:${port}`, credential_ref: "write", service: "cache-write-test", content_mode: "metadata_only",
    });
    for (const provider of ["openai", "deepseek"]) {
      const pk = await seed.createProviderKey({
        display_name: `cache-write-${provider}`, provider, adapter: "openai",
        secret: "test", api_base: `http://127.0.0.1:${port}/v1`,
      });
      for (const scenario of cases.filter((c) => (c.surface === "responses-bridge") === (provider === "deepseek"))) {
        await seed.createModel({
          display_name: scenario.model, model_name: scenario.model, provider, provider_key_id: pk.id,
        });
      }
    }
    await seed.createApiKey({
      key_hash: createHash("sha256").update(KEY).digest("hex"),
      allowed_models: cases.map((c) => c.model),
    });
    const client = new ProxyClient(app.proxyUrl, KEY);
    await waitConfigPropagation(async () => (await client.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await new Promise<void>((resolve, reject) => {
      if (!server) return resolve();
      server.close((error) => error ? reject(error) : resolve());
    });
  });

  test("logs retain a raw value, explicit zero, and absence without changing token accounting", async (ctx) => {
    if (!reachable) return ctx.skip();
    for (const scenario of cases) {
      const endpoint = scenario.surface === "responses-bridge" ? "responses" : scenario.surface;
      const response = await fetch(`${app!.proxyUrl}/v1/${endpoint}`, {
        method: "POST",
        headers: { authorization: `Bearer ${KEY}`, "content-type": "application/json" },
        body: JSON.stringify({
          model: scenario.model, stream: scenario.stream, max_tokens: 100,
          ...(endpoint === "responses" ? { input: "hello" }
            : endpoint === "completions" ? { prompt: "hello" }
            : { messages: [{ role: "user", content: "hello" }] }),
          ...(endpoint === "chat/completions" && scenario.stream ? { stream_options: { include_usage: true } } : {}),
        }),
      });
      const body = await response.text();
      expect(response.status, `${scenario.model}: ${body}`).toBe(200);
      if (!scenario.stream && endpoint !== "messages") {
        const usage = JSON.parse(body).usage;
        const details = endpoint === "responses" ? usage.input_tokens_details : usage.prompt_tokens_details;
        expect(details.cache_write_tokens, scenario.model).toBe(scenario.write);
        expect(usage.total_tokens, scenario.model).toBe(112);
      }
    }
    await expect.poll(() => {
      if (intakeError) throw intakeError;
      return cases.every((c) => logs.some((l) => l["aisix.requested_model"] === c.model));
    }, { timeout: 15_000 }).toBe(true);
    for (const scenario of cases) {
      const event = logs.find((l) => l["aisix.requested_model"] === scenario.model)!;
      expect(event["aisix.cache_write_tokens"], scenario.model).toBe(scenario.write);
      expect(event["gen_ai.usage.input_tokens"], scenario.model).toBe(101);
      expect(event["gen_ai.usage.output_tokens"], scenario.model).toBe(11);
      expect(event["aisix.cached_prompt_tokens"], scenario.model).toBe(19);
      expect(event["aisix.cache_creation_tokens"], scenario.model).toBeUndefined();
    }
  });
});
