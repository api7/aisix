import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient, ProxyClient, SeedClient, spawnApp, startOpenAiUpstream, waitConfigPropagation,
  type OpenAiUpstream, type SpawnedApp,
} from "../harness/index.js";

const caller = "gemini-content-client";
const ask = { contents: [{ parts: [{ text: "Summarize the report." }] }] };
const execution = (label: string) => [
  { executableCode: { language: "PYTHON", code: `print("${label}")` }, thoughtSignature: "c2ln" },
  { code_execution_result: { outcome: "OUTCOME_OK", output: `Result: ${label}` } },
  { executable_code: { language: "PYTHON", code: `# ${label}` } },
  { codeExecutionResult: { outcome: "OUTCOME_OK", output: label } },
];
const native = { candidates: [{ content: { role: "model", parts: execution("June report") }, finishReason: "STOP" }] };
const images = [
  { type: "image_url", image_url: { url: "data:image/png;base64,aW1hZ2U=" } },
  { type: "image_url", image_url: { url: "https://media.example/sample.webp" } },
];
const media = { audio: { id: "recording", data: "YXVkaW8=", transcript: "Reading." }, images };
const meta = { id: "chat-content", model: "upstream", created: 1 };
const chat = { ...meta, choices: [{ index: 0, message: { role: "assistant", content: "Media.", ...media }, finish_reason: "stop" }] };
const event = (delta: object, finish_reason: string | null = null) =>
  JSON.stringify({ ...meta, choices: [{ index: 0, delta, finish_reason }] });

describe("Gemini generation content", () => {
  const ups: Record<string, OpenAiUpstream> = {};
  let moderation: OpenAiUpstream | undefined;
  let app: SpawnedApp | undefined;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    if (!(await etcd.ping())) return;
    app = await spawnApp({ exposeDebugHeaders: true });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    moderation = await startOpenAiUpstream({ nonStreamBody: { results: [{ flagged: false, categories: {}, category_scores: {} }] } });
    ups.native = await startOpenAiUpstream({ nonStreamBody: native });
    ups.nativesse = await startOpenAiUpstream({ streamEvents: [JSON.stringify(native)] });
    ups.chat = await startOpenAiUpstream({ nonStreamBody: chat });
    ups.chatsse = await startOpenAiUpstream({ streamEvents: [event(media), event({}, "stop"), "[DONE]"] });
    ups.azure = await startOpenAiUpstream({ nonStreamBody: chat });
    ups.azuresse = await startOpenAiUpstream({ streamEvents: [event(media), event({}, "stop"), "[DONE]"] });
    const tool = (args: string) => ({ tool_calls: [{ index: 0, id: "batch-id", function: { name: "read_report", arguments: args } }] });
    ups.audio = await startOpenAiUpstream({ nonStreamBody: { ...meta, choices: [{ index: 0, message: { role: "assistant", content: null, audio: media.audio }, finish_reason: "stop" }] } });
    ups.tools = await startOpenAiUpstream({ streamEvents: [event(tool('{"page":')), event(tool("2}")), event({}, "tool_calls"), "[DONE]"] });
    const mask = await seed.createGuardrail({ name: "report-mask", kind: "pii", enabled: true, hook_point: "both",
      custom_patterns: [{ name: "report", regex: "June report|report_title", action: "mask" }] }, { attach: false });
    const review = await seed.createGuardrail({ name: "report-review", kind: "openai_moderation", enabled: true,
      hook_point: "input", api_key: "review-client", endpoint: moderation.baseUrl }, { attach: false });
    const names = await seed.createGuardrail({ name: "report-name", kind: "keyword", enabled: true,
      hook_point: "input", input_messages: "latest_turn", patterns: [{ kind: "literal", value: "catalog_review" }] }, { attach: false });
    for (const [alias, up] of Object.entries(ups)) {
      const isNative = alias.startsWith("native");
      const provider = alias.startsWith("azure") ? "azure" : "openai";
      const pk = await seed.createProviderKey({ display_name: "content-" + alias, api_key: "upstream-key",
        provider, adapter: provider === "azure" ? "azure-openai" : "openai", api_base: up.baseUrl, apis: isNative ? { gemini: {} } : {},
        request: isNative ? { param_constraints: { temperature_max: 0.5 } }
          : { default_body_fields: { audio: { format: "mp3", voice: "alloy" } } } });
      const model = await seed.createModel({ display_name: alias, model_name: "report", provider, provider_key_id: pk.id });
      await seed.attachGuardrailToModel(mask.id, model.id);
      if (alias === "native") await seed.attachGuardrailToModel(review.id, model.id);
      if (alias === "native" || alias === "chat") await seed.attachGuardrailToModel(names.id, model.id);
    }
    await seed.createCachePolicy({ name: "content-policy", enabled: true, applies_to: "all" });
    await seed.createApiKey({ key_hash: createHash("sha256").update(caller).digest("hex"), allowed_models: Object.keys(ups) });
    const probe = new ProxyClient(app.proxyUrl, caller);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all([...Object.values(ups), moderation].map(up => up?.close()));
  });

  function post(alias: string, body: object = ask) {
    const stream = alias.endsWith("sse") || alias === "tools";
    return fetch(app!.proxyUrl + "/v1beta/models/" + alias + ":" + (stream ? "streamGenerateContent?alt=sse" : "generateContent"), {
      method: "POST", headers: { "content-type": "application/json", authorization: "Bearer " + caller }, body: JSON.stringify(body),
    });
  }

  async function parts(res: Response) {
    expect(res.status).toBe(200);
    if (res.headers.get("content-type")?.includes("text/event-stream")) {
      return (await res.text()).split(/\r?\n/).filter(l => l.startsWith("data:"))
        .flatMap(l => JSON.parse(l.slice(5)).candidates.flatMap((c: { content: { parts: unknown[] } }) => c.content.parts));
    }
    return (await res.json()).candidates[0].content.parts;
  }

  test.for(["native", "nativesse"])("masks ordered code and results in %s responses", async (alias, ctx) => {
    if (!app) return ctx.skip();
    expect(await parts(await post(alias))).toEqual(execution("[REPORT_REDACTED]"));
  });

  test("projects and rewrites native code and results on input", async ctx => {
    if (!app) return ctx.skip();
    const req = { contents: [{ role: "model", parts: execution("June report") }] };
    expect((await post("native", req)).status).toBe(200);
    expect(JSON.parse(ups.native.receivedRequests.at(-1)!.body).contents[0].parts).toEqual(execution("[REPORT_REDACTED]"));
    const input = JSON.parse(moderation!.receivedRequests.at(-1)!.body).input;
    expect(input).toContain('print("June report")');
    expect(input).toContain("Result: June report");
  });

  test.for(["native", "chat"])("masks joined text in the %s outbound request", async (alias, ctx) => {
    if (!app) return ctx.skip();
    const req = { contents: [{ parts: [{ text: "June ", thoughtSignature: "c2ln" }, { text: "report" }] }] };
    expect((await post(alias, req)).status).toBe(200);
    const sent = JSON.parse(ups[alias].receivedRequests.at(-1)!.body);
    if (alias === "chat") expect(sent.messages[0].content).toBe("[REPORT_REDACTED]");
    else {
      expect(sent.contents[0].parts).toHaveLength(2);
      expect(sent.contents[0].parts[0].thoughtSignature).toBe("c2ln");
      expect(sent.contents[0].parts.map((p: {text: string}) => p.text).join("")).toBe("[REPORT_REDACTED]");
    }
  });

  test("constrains quoted temperatures without changing in-range values", async ctx => {
    if (!app) return ctx.skip();
    for (const [config, temperature, expected] of [
      ["generationConfig", "0.85", 0.5], ["generation_config", "0.25", "0.25"],
    ] as const) {
      expect((await post("native", { ...ask, [config]: { temperature, topK: 3 } })).status).toBe(200);
      expect(JSON.parse(ups.native.receivedRequests.at(-1)!.body)[config]).toEqual({ temperature: expected, topK: 3 });
    }
  });

  test("accepts optional null system instructions and rejects invalid values", async ctx => {
    if (!app) return ctx.skip();
    for (const system of ["systemInstruction", "system_instruction"]) {
      expect((await post("native", { ...ask, [system]: null })).status).toBe(200);
      expect(JSON.parse(ups.native.receivedRequests.at(-1)!.body)[system]).toBeNull();
      expect((await post("chat", { ...ask, [system]: null })).status).toBe(200);
      expect((await post("chat", { ...ask, [system]: 3 })).status).toBe(400);
    }
  });

  test.for(["functionCall", "function_response"])("applies the input name policy to a replayed %s", async (field, ctx) => {
    if (!app) return ctx.skip();
    const call = field === "functionCall";
    const role = call ? "model" : "user";
    const part = { [field]: { name: "catalog_review", id: "review", ...(call ? { args: {} } : { response: { ok: true } }) } };
    const earlier = call ? [] : [{ role: "model", parts: [{ functionCall: { name: "catalog_review", id: "review", args: {} } }] }];
    const before = ups.native.receivedRequests.length;
    expect((await post("native", { contents: [...earlier, { role, parts: [part] }] })).status).toBe(422);
    expect(ups.native.receivedRequests.length).toBe(before);
  });

  test("keeps explicit and generated IDs distinct when matching tool results", async ctx => {
    if (!app) return ctx.skip();
    const req = { contents: [{ role: "model", parts: [
      { functionCall: { name: "report_title", args: { page: 1, topic: "June report" } } },
      { functionCall: { name: "report_title", id: "call_0", args: { page: 2 } } },
    ] }, { role: "user", parts: [
      { functionResponse: { name: "report_title", id: "call_0", response: { page: 2 } } },
      { functionResponse: { name: "report_title", response: { page: 1 } } },
    ] }] };
    expect((await post("chat", req)).status).toBe(200);
    const [assistant, second, first] = JSON.parse(ups.chat.receivedRequests.at(-1)!.body).messages;
    const [a, b] = assistant.tool_calls;
    expect(a.function.name).toBe("report_title");
    expect(JSON.parse(a.function.arguments).topic).toBe("[REPORT_REDACTED]");
    expect(a.id).toBeTruthy();
    expect(a.id).not.toBe(b.id);
    expect(b.id).toBe("call_0");
    expect([second.tool_call_id, first.tool_call_id]).toEqual([b.id, a.id]);
  });

  test("accumulates tool arguments when a Chat stream repeats the call ID", async ctx => {
    if (!app) return ctx.skip();
    expect(await parts(await post("tools"))).toEqual([{ functionCall: { id: "batch-id", name: "read_report", args: { page: 2 } } }]);
  });

  test("resolves the original audio format on repeated requests with a cache policy", async ctx => {
    if (!app) return ctx.skip();
    const req = { ...ask, generationConfig: { responseModalities: ["AUDIO"], temperature: 0 } };
    for (let i = 0; i < 2; i++) {
      const res = await post("audio", req);
      expect(res.headers.get("x-aisix-cache")).toBe("miss");
      expect(await parts(res)).toEqual([{ inlineData: { mimeType: "audio/mpeg", data: "YXVkaW8=" } }]);
    }
    const cached = await fetch(app.proxyUrl + "/v1/chat/completions", {
      method: "POST", headers: { "content-type": "application/json", authorization: "Bearer " + caller },
      body: JSON.stringify({ model: "audio", messages: [{ role: "user", content: ask.contents[0].parts[0].text }],
        stream: false, modalities: ["audio"], temperature: 0 }),
    });
    expect(cached.headers.get("x-aisix-cache")).toBe("hit");
    expect((await cached.json()).choices[0].message.audio).toEqual(media.audio);
  });

  test.for(["chat", "chatsse", "azure", "azuresse"])("returns images and audio in the %s caller's Gemini response", async (alias, ctx) => {
    if (!app) return ctx.skip();
    const output = await parts(await post(alias, { ...ask, generationConfig: { responseModalities: ["TEXT", "AUDIO"] } }));
    expect(JSON.parse(ups[alias].receivedRequests.at(-1)!.body).audio.format).toBe("mp3");
    expect(output).toEqual(expect.arrayContaining([
      { inlineData: { mimeType: "image/png", data: "aW1hZ2U=" } },
      { fileData: { fileUri: "https://media.example/sample.webp", mimeType: "image/webp" } },
      { inlineData: { mimeType: "audio/mpeg", data: "YXVkaW8=" } },
    ]));
  });
});
