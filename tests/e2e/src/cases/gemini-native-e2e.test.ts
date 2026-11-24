import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient, ProxyClient, SeedClient, spawnApp, startOpenAiUpstream,
  waitConfigPropagation, type OpenAiUpstream, type SpawnedApp,
} from "../harness/index.js";

const CALLER = "native-gemini-caller";
const UPSTREAM_KEY = "native-gemini-provider-key";
const CLIENT_CREDENTIALS = {
  "x-aisix-api-key": CALLER,
  authorization: "Bearer client-upstream-token",
  "x-goog-api-key": "client-upstream-google-key",
};
const ASK = {
  contents: [{ role: "user", parts: [{ text: "Check the weather." }] }],
  tools: [{ functionDeclarations: [{ name: "weather", parameters: {
    type: "OBJECT", properties: { city: { type: "STRING" } },
  } }] }],
  generationConfig: { temperature: 0, candidateCount: 2 },
  futureRequestField: { enabled: true },
};
const CALL = {
  functionCall: { name: "weather", args: { city: "Singapore" } },
  thoughtSignature: "c2lnbmF0dXJl",
};
const CALL_CONTENT = {
  role: "model", parts: [
    { thought: true, text: "Plan", thoughtSignature: "YQo=" },
    { text: "Checking. " }, CALL, { text: "One moment." }, { futurePart: ["opaque"] },
  ],
};
const USAGE = {
  promptTokenCount: 12, candidatesTokenCount: 3, thoughtsTokenCount: 2,
  cachedContentTokenCount: 5, totalTokenCount: 17, futureUsageField: "reported",
};
const ANSWER = {
  candidates: [
    { index: 0, content: CALL_CONTENT, finishReason: "STOP", futureCandidateField: 7 },
    { index: 1, content: { role: "model", parts: [{ text: "另一项" }] }, finishReason: "STOP" },
  ],
  usageMetadata: USAGE, responseId: "google-response", modelVersion: "gemini-release",
  futureResponseField: [1, 2],
};
const TOOL_RESULT = {
  ...ASK,
  contents: [
    ...ASK.contents, CALL_CONTENT,
    { role: "user", parts: [{ functionResponse: { name: "weather", response: { temperature: 31 } } }] },
  ],
};
const FINAL = {
  candidates: [{ content: { role: "model", parts: [{ text: "31 °C." }] }, finishReason: "STOP" }],
  usageMetadata: USAGE,
};
const STREAM = [
  { candidates: ANSWER.candidates, responseId: "s1", futureMetadata: ["preserve"] },
  { usageMetadata: USAGE, modelVersion: "gemini-release" },
];
const GOOGLE_ERROR = {
  error: {
    code: 400, status: "INVALID_ARGUMENT", message: "Invalid function declaration.",
    details: [{ "@type": "type.googleapis.com/google.rpc.BadRequest", fieldViolations: [
      { field: "tools[0]", description: "Invalid parameter type" },
    ] }],
  },
};

describe("native Gemini generation", () => {
  let app: SpawnedApp | undefined;
  const upstreams: Record<string, OpenAiUpstream> = {};

  beforeAll(async () => {
    const etcd = new EtcdClient();
    if (!(await etcd.ping())) return;
    app = await spawnApp({ exposeDebugHeaders: true });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    upstreams.tools = await startOpenAiUpstream({
      scriptedResponses: [{ nonStreamBody: ANSWER }, { nonStreamBody: FINAL }],
    });
    upstreams.stream = await startOpenAiUpstream({
      streamEvents: STREAM.map((frame) => JSON.stringify(frame)),
    });
    upstreams.streamerror = await startOpenAiUpstream({
      streamEvents: [STREAM[0], GOOGLE_ERROR].map((v) => JSON.stringify(v)),
    });
    upstreams.error = await startOpenAiUpstream({ status: 400, errorBody: GOOGLE_ERROR });
    upstreams.busy = await startOpenAiUpstream({ status: 503, errorBody: {
      error: { code: 503, status: "UNAVAILABLE", message: "Try later", details: [] },
    } });
    upstreams.mask = await startOpenAiUpstream({ nonStreamBody: {
      candidates: [{ content: { parts: [{ text: "owner@example.com" }, {
        functionCall: { name: "notify", args: { to: "owner@example.com" } },
        thoughtSignature: "c2lnbmF0dXJl",
      }] }, finishReason: "STOP" }], usageMetadata: USAGE,
    } });
    upstreams.maskstream = await startOpenAiUpstream({ streamEvents: [
      { candidates: [{ index: 1, content: { parts: [{ text: "owner@" }] } }] },
      { candidates: [{ index: 1, content: { parts: [{ text: "example.com" }, {
        functionCall: { name: "notify", args: { to: "second@example.com" } },
      }] }, finishReason: "STOP" }], usageMetadata: USAGE },
    ].map((v) => JSON.stringify(v)) });
    upstreams.legacy = await startOpenAiUpstream({});
    upstreams.undeclared = await startOpenAiUpstream({});
    upstreams.nativeonly = await startOpenAiUpstream({ nonStreamBody: ANSWER });
    upstreams.anthropic = await startOpenAiUpstream({});
    const guard = await seed.createGuardrail({
      name: "mask-email", kind: "pii", enabled: true, hook_point: "both",
      detectors: [{ type: "email", action: "mask" }],
    }, { attach: false });
    for (const [alias, up] of Object.entries(upstreams)) {
      const root = up.baseUrl + "/v1beta";
      const override = alias === "tools";
      const chat = ["legacy", "undeclared", "anthropic"].includes(alias);
      const surfaces = alias === "legacy" ? {} : {
        apis: chat ? {} : { gemini: override ? { base: root } : {} },
      };
      const pk = await seed.createProviderKey({
        display_name: "native-key-" + alias, api_key: UPSTREAM_KEY,
        ...(alias === "anthropic" ? { provider: "anthropic", adapter: "anthropic" } : {}),
        ...(alias === "nativeonly" ? { provider: "byo", adapter: undefined } : {}),
        api_base: chat || override ? up.baseUrl + "/v1beta/openai" : root, ...surfaces,
        request: {
          forward_client_headers: alias === "tools" || alias === "legacy"
            ? ["authorization", "x-goog-api-key", "x-client-tag"]
            : alias === "stream" ? ["*"] : [],
          ...(alias === "mask" ? { param_constraints: { temperature_max: 0.5 } } : {}),
        },
      });
      const model = await seed.createModel({
        display_name: alias, provider: "openai", provider_key_id: pk.id,
        model_name: "gemini-upstream",
      });
      if (alias === "mask" || alias === "maskstream") {
        await seed.attachGuardrailToModel(guard.id, model.id);
      }
    }
    for (const [name, targets] of [
      ["mixed-chat", ["anthropic", "undeclared"]],
      ["mixed-cache", ["undeclared", "nativeonly"]],
    ] as const) {
      await seed.createModel({
        display_name: name, routing: { strategy: "failover", targets: targets.map(model => ({ model })) },
      });
      await seed.createCachePolicy({ name: name + "-policy", enabled: true, applies_to: "model:" + name });
    }
    await seed.createModel({
      display_name: "ensemble",
      ensemble: { panel: [{ model: "anthropic" }, { model: "undeclared" }], judge: { model: "legacy" } },
    });
    await seed.createApiKey({
      key_hash: createHash("sha256").update(CALLER).digest("hex"),
      allowed_models: [...Object.keys(upstreams), "mixed-chat", "mixed-cache", "ensemble"],
    });
    const proxy = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(Object.values(upstreams).map((u) => u.close()));
  });

  function call(alias: string, method: string, body: unknown) {
    return fetch(app!.proxyUrl + "/v1beta/models/" + alias + ":" + method, {
      method: "POST", headers: {
        "content-type": "application/json", ...CLIENT_CREDENTIALS,
        "x-client-tag": "antigravity",
        "user-agent": "test-agent/1", "x-goog-api-client": "gl-test/1",
      }, body: JSON.stringify(body),
    });
  }

  test("preserves native parts, unknown fields and an unsigned client tool result", async (ctx) => {
    if (!app) return ctx.skip();
    for (const [request, expected] of [[ASK, ANSWER], [TOOL_RESULT, FINAL]] as const) {
      const res = await call("tools", "generateContent", request);
      expect(res.status).toBe(200);
      expect(await res.json()).toEqual(expected);
      const sent = upstreams.tools.receivedRequests.at(-1)!;
      expect(sent.method).toBe("POST");
      expect(sent.path).toBe("/v1beta/models/gemini-upstream:generateContent");
      expect(JSON.parse(sent.body)).toEqual(request);
      expect(sent.headers["x-goog-api-key"]).toBe(CLIENT_CREDENTIALS["x-goog-api-key"]);
      expect(sent.headers.authorization).toBe(CLIENT_CREDENTIALS.authorization);
      expect(sent.headers["x-aisix-api-key"]).toBeUndefined();
      expect(sent.headers["user-agent"]).toBe("test-agent/1");
      expect(sent.headers["x-goog-api-client"]).toBe("gl-test/1");
      expect(sent.headers["x-client-tag"]).toBe("antigravity");
    }
  });

  test("relays multi-candidate SSE frames and their usage-only terminal frame", async (ctx) => {
    if (!app) return ctx.skip();
    const res = await call("stream", "streamGenerateContent?alt=sse", TOOL_RESULT);
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toMatch(/text\/event-stream/);
    const frames = (await res.text()).split(/\r?\n/).filter((l) => l.startsWith("data:"))
      .map((line) => JSON.parse(line.slice(5)));
    expect(frames).toEqual(STREAM);
    const [sent] = upstreams.stream.receivedRequests;
    expect(sent.headers["x-goog-api-key"]).toBe(UPSTREAM_KEY);
    expect(sent.headers.authorization).toBeUndefined();
    expect(sent.headers["x-aisix-api-key"]).toBeUndefined();
    expect(sent.path).toBe("/v1beta/models/gemini-upstream:streamGenerateContent?alt=sse");
    expect(JSON.parse(sent.body)).toEqual(TOOL_RESULT);
  });

  test.for(["", "?alt=json", "?alt=json&alt=sse"])("requires SSE for streaming requests with %s", async (query, ctx) => {
    if (!app) return ctx.skip();
    const before = upstreams.stream.receivedRequests.length;
    const res = await call("stream", "streamGenerateContent" + query, ASK);
    expect(res.status).toBe(400);
    expect(await res.json()).toMatchObject({
      error: { status: "INVALID_ARGUMENT", message: expect.stringContaining("alt=sse") },
    });
    expect(upstreams.stream.receivedRequests).toHaveLength(before);
  });

  test.for(["legacy", "undeclared"])("a %s key uses chat without a native declaration", async (alias, ctx) => {
    if (!app) return ctx.skip();
    const res = await call(alias, "generateContent", ASK);
    expect(res.status).toBe(200);
    const answer = await res.json();
    expect(JSON.stringify(answer.candidates)).toContain("mock reply");
    const [sent] = upstreams[alias].receivedRequests;
    expect(sent.path).toBe("/v1beta/openai/chat/completions");
    expect(sent.headers.authorization).toBe(alias === "legacy" ? CLIENT_CREDENTIALS.authorization : "Bearer " + UPSTREAM_KEY);
    expect(sent.headers["x-goog-api-key"]).toBe(alias === "legacy" ? CLIENT_CREDENTIALS["x-goog-api-key"] : undefined);
    expect(sent.headers["x-aisix-api-key"]).toBeUndefined();
  });

  test("applies input and output masks to native text and tool arguments", async (ctx) => {
    if (!app) return ctx.skip();
    const ask = { contents: [{ parts: [{ text: "Write to owner@example.com" }] }], generationConfig: { temperature: 1 } };
    const res = await call("mask", "generateContent", ask);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.candidates[0].content.parts).toEqual([
      { text: "[EMAIL_REDACTED]" },
      { functionCall: { name: "notify", args: { to: "[EMAIL_REDACTED]" } }, thoughtSignature: "c2lnbmF0dXJl" },
    ]);
    const sent = JSON.parse(upstreams.mask.receivedRequests[0].body);
    expect(sent.contents[0].parts[0].text).toBe("Write to [EMAIL_REDACTED]");
    expect(sent.generationConfig.temperature).toBe(0.5);
  });

  test("a native-only key works without a Chat bridge", async (ctx) => {
    if (!app) return ctx.skip();
    const res = await call("nativeonly", "generateContent", ASK);
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual(ANSWER);
  });

  test("a Gemini request selects only compatible targets in its routing plan", async (ctx) => {
    if (!app) return ctx.skip();
    const body = { contents: [{ parts: [{ text: "hello" }] }] };
    for (const name of ["anthropic", "ensemble"]) {
      const res = await call(name, "generateContent", body);
      expect(res.status).toBe(400);
      expect(await res.json()).toMatchObject({ error: { code: 400, status: "INVALID_ARGUMENT" } });
    }
    const before = upstreams.undeclared.receivedRequests.length;
    const fallback = await call("mixed-chat", "generateContent", ASK);
    expect(fallback.status).toBe(200);
    expect(JSON.stringify((await fallback.json()).candidates)).toContain("mock reply");
    expect(upstreams.undeclared.receivedRequests).toHaveLength(before + 1);
    expect(upstreams.anthropic.receivedRequests).toHaveLength(0);
  });

  test.for(["mixed-cache", "mixed-chat"])("the %s Gemini plan skips a warm Chat cache", async (alias, ctx) => {
    if (!app) return ctx.skip();
    const text = "Which wire is cached?";
    for (const expected of ["miss", "hit"]) {
      const res = await fetch(app.proxyUrl + "/v1/chat/completions", {
        method: "POST", headers: { "content-type": "application/json", ...CLIENT_CREDENTIALS },
        body: JSON.stringify({ model: alias, stream: false, messages: [{ role: "user", content: text }] }),
      });
      expect(res.status).toBe(200);
      expect(res.headers.get("x-aisix-cache")).toBe(expected);
      await res.json();
    }
    const before = upstreams.undeclared.receivedRequests.length;
    for (let i = 0; i < 2; i++) {
      const res = await call(alias, "generateContent", { contents: [{ parts: [{ text }] }] });
      expect(res.status).toBe(200);
      expect(res.headers.get("x-aisix-cache")).toBeNull();
      expect(JSON.stringify((await res.json()).candidates)).toContain("mock reply");
    }
    expect(upstreams.undeclared.receivedRequests).toHaveLength(before + 2);
  });

  test("masks a later stream candidate across frames and masks its tool arguments", async (ctx) => {
    if (!app) return ctx.skip();
    const res = await call("maskstream", "streamGenerateContent?alt=sse", ASK);
    expect(res.status).toBe(200);
    const frames = (await res.text()).split(/\r?\n/).filter((line) => line.startsWith("data:"))
      .map((line) => JSON.parse(line.slice(5)));
    expect(frames).toHaveLength(2);
    const parts = frames.flatMap(f => f.candidates[0].content.parts);
    expect(parts.map(p => p.text ?? "").join("")).toBe("[EMAIL_REDACTED]");
    expect(parts.at(-1).functionCall.args.to).toBe("[EMAIL_REDACTED]");
    expect(frames.map(f => f.candidates[0].index)).toEqual([1, 1]);
  });

  test("returns Google's in-band error after the first native event", async (ctx) => {
    if (!app) return ctx.skip();
    const res = await call("streamerror", "streamGenerateContent?alt=sse", ASK);
    expect(res.status).toBe(200);
    const frames = (await res.text()).split(/\r?\n/).filter((line) => line.startsWith("data:"))
      .map((line) => JSON.parse(line.slice(5)));
    expect(frames).toEqual([STREAM[0], GOOGLE_ERROR]);
  });

  test("keeps the Google service error's HTTP status consistent with its JSON body", async (ctx) => {
    if (!app) return ctx.skip();
    const res = await call("busy", "generateContent", ASK);
    expect(res.status).toBe(503);
    expect(await res.json()).toEqual({
      error: { code: 503, status: "UNAVAILABLE", message: "Try later", details: [] },
    });
  });

  test.for(["generateContent", "streamGenerateContent?alt=sse"])(
    "%s preserves the upstream Google error and its details", async (method, ctx) => {
      if (!app) return ctx.skip();
      const res = await call("error", method, ASK);
      expect(res.status).toBe(400);
      expect(await res.json()).toEqual(GOOGLE_ERROR);
    },
  );
});
