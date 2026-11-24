import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { EtcdClient, ProxyClient, SeedClient, spawnApp, waitConfigPropagation, type SpawnedApp } from "../harness/index.js";

const key = "sk-gemini-client";
const providerKey = "sk-chat-upstream";
const model = "agent-model";
const remoteModel = "upstream-model";
const short = `/v1beta/models/${model}`;
const full = `/v1beta1/projects/local/locations/us-central1/publishers/google/models/${model}`;
const agent = {
  systemInstruction: { role: "system", parts: [{ text: "Use tools." }] },
  contents: [{ role: "user", parts: [{ text: "Read the file." }] }],
  generationConfig: { maxOutputTokens: 4096, thinkingConfig: { includeThoughts: true, thinkingLevel: "HIGH" } },
  tools: [{ functionDeclarations: [{ name: "read_file", description: "Read a file",
    parameters: { type: "OBJECT", minProperties: "1", maxProperties: "2", properties: {
      path: { type: "STRING", minLength: "1", maxLength: "128" },
      flags: { type: "ARRAY", minItems: "1", maxItems: "3", items: { anyOf: [{ type: "STRING", minLength: "2" }] } },
    }, required: ["path"] } }] }],
};
const usage = { prompt_tokens: 11, completion_tokens: 7, total_tokens: 18, completion_tokens_details: { reasoning_tokens: 2 } };
const meta = { id: "chatcmpl-123", model: remoteModel, created: 1234 };
const call = { id: "up-1", type: "function", function: { name: "read_file", arguments: '{"path":"README"}' },
  extra_content: { google: { thought_signature: "YWJj" } } };
const received: Array<{ url: string; auth?: string; body: Record<string, any> }> = [];

describe("Gemini generation through a Chat Completions provider", () => {
  let app: SpawnedApp;
  let upstream: Server;
  let release: (() => void) | undefined;
  let etcdReady = false;
  beforeAll(async () => {
    const etcd = new EtcdClient();
    if (!(await etcd.ping())) return;
    etcdReady = true;
    upstream = createServer(async (req, res) => {
      const chunks: Buffer[] = [];
      for await (const chunk of req) chunks.push(chunk);
      const body = JSON.parse(Buffer.concat(chunks).toString());
      received.push({ url: req.url!, auth: req.headers.authorization, body });
      if (body.messages[0]?.content === "quota") {
        res.writeHead(429, { "content-type": "application/json", "retry-after": "9" });
        res.end(JSON.stringify({ error: { type: "rate_limit_error", message: "try again later" } }));
        return;
      }
      if (!body.stream) {
        const reply = body.messages.some((m: any) => m.role === "tool");
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ ...meta, object: "chat.completion", usage, choices: [{ index: 0,
          message: reply ? { role: "assistant", content: "done" } : { role: "assistant", content: null, tool_calls: [call] },
          finish_reason: reply ? "stop" : "tool_calls" }] }));
        return;
      }
      res.writeHead(200, { "content-type": "text/event-stream" });
      const event = (delta: object, finish_reason: string | null = null) => res.write(`data: ${JSON.stringify({
        ...meta, object: "chat.completion.chunk", choices: [{ index: 0, delta, finish_reason }],
      })}\n\n`);
      const resume = new Promise<void>((resolve) => { release = resolve; });
      event({ role: "assistant", content: "I will read it." });
      // The test client opens this latch only after it has read the first Gemini
      // event. A gateway that waits for EOF can never complete the exchange.
      await resume;
      event({ tool_calls: [{ ...call, index: 0, function: { name: "read_file", arguments: '{"pa' } }] });
      event({ tool_calls: [{ index: 0, function: { arguments: 'th":"README"}' } }] });
      event({}, "tool_calls");
      res.end(`data: ${JSON.stringify({ ...meta, object: "chat.completion.chunk", choices: [], usage })}\n\ndata: [DONE]\n\n`);
    });
    await new Promise<void>(r => upstream.listen(0, "127.0.0.1", r));
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({ display_name: "agent-key", provider: "openai", adapter: "openai",
      api_base: `http://127.0.0.1:${(upstream.address() as AddressInfo).port}/v1`, api_key: providerKey });
    for (const name of [model, "forbidden-model"]) await seed.createModel({ display_name: name, provider: "openai", model_name: remoteModel, provider_key_id: pk.id });
    await seed.createApiKey({ key_hash: createHash("sha256").update(key).digest("hex"), allowed_models: [model] });
    const probe = new ProxyClient(app.proxyUrl, key);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
  });
  afterAll(async () => {
    release?.();
    if (app) await app.exit();
    if (upstream) await new Promise<void>(r => upstream.close(() => r()));
  });

  function post(path: string, body: object, auth = { authorization: `Bearer ${key}` } as Record<string, string>) {
    return fetch(app.proxyUrl + path, { method: "POST", headers: { "content-type": "application/json", ...auth }, body: JSON.stringify(body) });
  }

  test("streams text before the upstream finishes, then replays an ID-less tool result", async ctx => {
    if (!etcdReady) return ctx.skip();
    const response = await post(short + ":streamGenerateContent?alt=sse", agent);
    expect(response.status).toBe(200);
    expect(response.headers.get("content-type")).toContain("text/event-stream");
    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let wire = "";
    try {
      while (!wire.includes("I will read it.")) {
        const { value, done } = await reader.read();
        if (done) throw new Error("Gemini stream ended before the first text");
        wire += decoder.decode(value, { stream: true });
      }
    } finally { release?.(); }
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      wire += decoder.decode(value, { stream: true });
    }
    const frames = wire.split("\n").filter(l => l.startsWith("data:")).map(l => JSON.parse(l.slice(5)));
    const toolParts = frames.flatMap(f => f.candidates.flatMap((c:any) => c.content.parts)).filter(p => p.functionCall);
    expect(toolParts).toEqual([{ functionCall: { id: "up-1", name: "read_file", args: { path: "README" } }, thoughtSignature: "YWJj" }]);
    expect(frames.at(-1)).toMatchObject({ candidates: [{ finishReason: "STOP" }], usageMetadata: {
      promptTokenCount: 11, candidatesTokenCount: 5, thoughtsTokenCount: 2, totalTokenCount: 18 } });
    expect(received.at(-1)).toMatchObject({ url: "/v1/chat/completions", auth: `Bearer ${providerKey}`, body: {
      model: remoteModel, max_tokens: 4096, reasoning_effort: "high",
      messages: [{ role: "system", content: "Use tools." }, { role: "user", content: "Read the file." }],
      tools: [{ type: "function", function: { name: "read_file", parameters: {
        type: "object", minProperties: 1, maxProperties: 2, properties: {
          path: { type: "string", minLength: 1, maxLength: 128 },
          flags: { type: "array", minItems: 1, maxItems: 3, items: { anyOf: [{ type: "string", minLength: 2 }] } },
        }, required: ["path"],
      } } }],
    } });
    const replay = await post(full + ":generateContent", { ...agent, contents: [...agent.contents,
      { role: "model", parts: [{ functionCall: { name: "read_file", args: { path: "README" } }, thoughtSignature: toolParts[0].thoughtSignature }] },
      { role: "user", parts: [{ functionResponse: { name: "read_file", response: { output: "file contents" } } }] },
    ] }, { "x-goog-api-key": key });
    expect(replay.status).toBe(200);
    expect(await replay.json()).toMatchObject({ candidates: [{ content: { role: "model", parts: [{ text: "done" }] }, finishReason: "STOP" }] });
    const history = received.at(-1)!.body.messages;
    expect(history[2].tool_calls).toMatchObject([{ type: "function", function: call.function, extra_content: call.extra_content }]);
    expect(history[2].tool_calls[0].id).toEqual(expect.any(String));
    expect(history[3]).toMatchObject({ role: "tool", tool_call_id: history[2].tool_calls[0].id, content: '{"output":"file contents"}' });
    const metrics = await fetch(`${app.metricsUrl}/metrics`).then(r => r.text());
    expect(metrics.split("\n").some(line => line.startsWith("aisix_llm_requests_total{") &&
      line.includes('inbound_protocol="gemini"') && line.includes('endpoint="/v1beta/models/:model:streamGenerateContent"'))).toBe(true);
  });

  test("unary tool calls and Google error envelopes", async ctx => {
    if (!etcdReady) return ctx.skip();
    const ok = await post(short + ":generateContent", agent, { "x-goog-api-key": key });
    expect(await ok.json()).toMatchObject({ candidates: [{ content: { parts: [
      { functionCall: { name: "read_file", args: { path: "README" }, id: "up-1" }, thoughtSignature: "YWJj" },
    ] } }] });
    const before = received.length;
    for (const [path, auth, status] of [
      [short, {}, 401], ["/v1beta/models/forbidden-model", { authorization: `Bearer ${key}` }, 403],
    ] as const) {
      const denied = await post(path + ":generateContent", agent, auth as Record<string, string>);
      expect(denied.status).toBe(status);
      expect(await denied.json()).toMatchObject({ error: { code: status, status: status === 401 ? "UNAUTHENTICATED" : "PERMISSION_DENIED" } });
    }
    expect(received).toHaveLength(before);
    const failed = await post(short + ":generateContent", { contents: [{ parts: [{ text: "quota" }] }] });
    expect(failed.status).toBe(429);
    expect(await failed.json()).toMatchObject({ error: { code: 429, status: "RESOURCE_EXHAUSTED", message: "try again later" } });
  });
});
