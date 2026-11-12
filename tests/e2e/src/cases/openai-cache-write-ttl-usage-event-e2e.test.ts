import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { gunzipSync } from "node:zlib";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient, ProxyClient, SeedClient, pickFreePort, spawnApp,
  waitConfigPropagation, type SpawnedApp,
} from "../harness/index.js";

// E2E: an OpenAI-shape prompt-cache write is reported on the usage event
// split by cache retention whenever the upstream reports the split, so a
// 12-hour write can be priced at its own rate. `cache_write_tokens` stays the
// total; `cache_write_30m_tokens` / `cache_write_12h_tokens` are its parts,
// present only together and only when they add up to it.
//
// The caller's prompt-cache settings (`prompt_cache_options`, a content
// block's `prompt_cache_breakpoint`) reach the upstream as sent.
//
// Reference:
//   - <https://platform.openai.com/docs/guides/prompt-caching>
//   - <https://platform.openai.com/docs/api-reference/responses/object>

const KEY = "sk-cache-write-ttl-test";
const WRITE = 7000;
const WRITE_30M = 2000;
const WRITE_12H = 5000;

// The model name tells the upstream which split to report.
type Split = "split" | "mismatch" | "single" | "none";
const SPLITS: Record<Split, Record<string, number> | undefined> = {
  split: { ttl_30m_tokens: WRITE_30M, ttl_12h_tokens: WRITE_12H },
  mismatch: { ttl_30m_tokens: 1000, ttl_12h_tokens: 1000 },
  single: { ttl_12h_tokens: WRITE },
  none: undefined,
};

type Surface = "responses" | "chat/completions" | "completions";
const cases: Array<{ surface: Surface; stream: boolean; split: Split; model: string }> = [];
for (const [surface, streams, splits] of [
  ["responses", [false, true], ["split", "mismatch", "single", "none"]],
  ["chat/completions", [false, true], ["split", "none"]],
  ["completions", [false], ["split"]],
] as const) {
  for (const stream of streams) {
    for (const split of splits) {
      cases.push({
        surface, stream, split,
        model: `ttl-${surface.replaceAll("/", "-")}-${stream}-${split}`,
      });
    }
  }
}

describe("OpenAI cache-write retention split on the usage event", () => {
  let app: SpawnedApp | undefined;
  let server: Server | undefined;
  let reachable = false;
  const logs: Record<string, unknown>[] = [];
  const sent = new Map<string, Record<string, unknown>>();
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
      sent.set(request.model, request);
      const split = (Object.keys(SPLITS) as Split[]).find((s) => request.model.endsWith(`-${s}`)) ?? "none";
      const details = {
        cached_tokens: 300,
        cache_write_tokens: WRITE,
        ...(SPLITS[split] ? { cache_write_tokens_details: SPLITS[split] } : {}),
      };
      const responses = req.url === "/v1/responses";
      const usage = responses
        ? { input_tokens: 8000, output_tokens: 5, total_tokens: 8005, input_tokens_details: details }
        : { prompt_tokens: 8000, completion_tokens: 5, total_tokens: 8005, prompt_tokens_details: details };
      const response = responses ? {
        id: "resp-ttl", object: "response", model: request.model, status: "completed",
        output: [{ type: "message", role: "assistant", content: [{ type: "output_text", text: "ok" }] }], usage,
      } : {
        id: "chatcmpl-ttl", object: "chat.completion", created: 1, model: request.model,
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
        { ...response, usage: undefined, choices: [{ index: 0, delta: { content: "ok" }, finish_reason: null }] },
        { ...response, usage: undefined, choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
        { ...response, choices: [], usage },
      ];
      for (const event of events) res.write(`data: ${JSON.stringify(event)}\n\n`);
      if (!responses) res.write("data: [DONE]\n\n");
      res.end();
    });
    const port = await pickFreePort();
    await new Promise<void>((resolve, reject) => {
      server!.once("error", reject);
      server!.listen(port, "127.0.0.1", resolve);
    });
    app = await spawnApp({ extraEnv: { DD_CRED_TTL_API_KEY: "test" } });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "cache-write-ttl", kind: "datadog", enabled: true,
      site: `127.0.0.1:${port}`, credential_ref: "ttl", service: "cache-write-ttl-test", content_mode: "metadata_only",
    });
    const pk = await seed.createProviderKey({
      display_name: "cache-write-ttl-openai", provider: "openai", adapter: "openai",
      secret: "test", api_base: `http://127.0.0.1:${port}/v1`,
    });
    for (const c of cases) {
      await seed.createModel({ display_name: c.model, model_name: c.model, provider: "openai", provider_key_id: pk.id });
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

  test("the split rides the event only when both lifetimes add up to the write", async (ctx) => {
    if (!reachable) return ctx.skip();
    const breakpoint = { mode: "explicit", ttl: "12h" };
    const cacheOptions = { ttl: "12h" };
    for (const c of cases) {
      const body = c.surface === "responses"
        ? {
          input: [{ role: "user", content: [{ type: "input_text", text: "hello", prompt_cache_breakpoint: breakpoint }] }],
          prompt_cache_options: cacheOptions,
        }
        : c.surface === "completions"
          ? { prompt: "hello" }
          : {
            messages: [{ role: "user", content: [{ type: "text", text: "hello", prompt_cache_breakpoint: breakpoint }] }],
            prompt_cache_options: cacheOptions,
            ...(c.stream ? { stream_options: { include_usage: true } } : {}),
          };
      const response = await fetch(`${app!.proxyUrl}/v1/${c.surface}`, {
        method: "POST",
        headers: { authorization: `Bearer ${KEY}`, "content-type": "application/json" },
        body: JSON.stringify({ model: c.model, stream: c.stream, max_tokens: 16, ...body }),
      });
      expect(response.status, `${c.model}: ${await response.text()}`).toBe(200);
      if (c.surface !== "completions") {
        const upstream = sent.get(c.model)!;
        expect(upstream.prompt_cache_options, c.model).toEqual(cacheOptions);
        const block = c.surface === "responses"
          ? (upstream.input as any)[0].content[0]
          : (upstream.messages as any)[0].content[0];
        expect(block.prompt_cache_breakpoint, c.model).toEqual(breakpoint);
      }
    }
    await expect.poll(() => {
      if (intakeError) throw intakeError;
      return cases.every((c) => logs.some((l) => l["aisix.requested_model"] === c.model));
    }, { timeout: 15_000 }).toBe(true);
    for (const c of cases) {
      const event = logs.find((l) => l["aisix.requested_model"] === c.model)!;
      expect(event["aisix.cache_write_tokens"], c.model).toBe(WRITE);
      expect(event["aisix.cached_prompt_tokens"], c.model).toBe(300);
      expect(event["gen_ai.usage.input_tokens"], c.model).toBe(8000);
      if (c.split === "split") {
        expect(event["aisix.cache_write_30m_tokens"], c.model).toBe(WRITE_30M);
        expect(event["aisix.cache_write_12h_tokens"], c.model).toBe(WRITE_12H);
      } else {
        expect(event["aisix.cache_write_30m_tokens"], c.model).toBeUndefined();
        expect(event["aisix.cache_write_12h_tokens"], c.model).toBeUndefined();
      }
      expect(event["aisix.cache_creation_5m_tokens"], c.model).toBeUndefined();
      expect(event["aisix.cache_creation_1h_tokens"], c.model).toBeUndefined();
    }
  });
});
