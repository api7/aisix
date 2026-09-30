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
import { startMockOtlp, type MockOtlp } from "../harness/otlp-mock.js";

// E2E: what a passthrough route's guardrails read, per detected envelope.
//
// Output: a streamed reply is scanned for its generated text and tool-call
// arguments on every envelope — Anthropic Messages text and tool input
// deltas (carried on the chat envelope), chat tool-call arguments, and
// Responses function-call argument deltas. Generated reasoning is not
// scanned. The same extraction feeds the hold-back cap (#513): a stream
// whose frames outweigh `max_buffer_bytes` while its content does not is
// released.
//
// Input: an Anthropic Messages body is scanned in every slot the typed
// `/v1/messages` route scans (system prompt, tool results), and a Responses
// body in every item slot the typed `/v1/responses` route scans (a replayed
// tool call's arguments).

const CALLER = "sk-pt-scan-coverage";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const OUT_LIT = "outputleakliteral";
const IN_LIT = "inputleakliteral";
const ESCAPED_BLOCK = "BLOCKME";
const ESCAPED_CJK = "中文";
const ESCAPED_BLOCK_JSON = String.raw`{"state":"\u0042LOCKME","state":"clean"}`;
const ESCAPED_CJK_JSON = String.raw`{"query":"\u4e2d\u6587"}`;
const SAFE_ESCAPED_JSON = String.raw`{"state":"\u0063lean","state":"safe"}`;
const deepEscapedBlockJSON = (depth: number) =>
  `${'{"v":'.repeat(depth)}"${String.raw`\u0042LOCKME`}"${'}'.repeat(depth)}`;
// Above serde_json's default recursion limit. It remains valid JSON and the
// provider receives it verbatim, so Raw guardrails must still decode the leaf.
const DEEP_ESCAPED_BLOCK_JSON = deepEscapedBlockJSON(160);
const CAP = 1_000;

const anthropicEvents = (blocks: Array<Record<string, unknown>>) => [
  JSON.stringify({
    type: "message_start",
    message: {
      id: "msg_pt",
      type: "message",
      role: "assistant",
      content: [],
      model: "claude-3-5-haiku-20241022",
      stop_reason: null,
      usage: { input_tokens: 5, output_tokens: 1 },
    },
  }),
  ...blocks.map((delta) => JSON.stringify({ type: "content_block_delta", index: 0, delta })),
  JSON.stringify({ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 9 } }),
  JSON.stringify({ type: "message_stop" }),
];

const chatChunk = (delta: Record<string, unknown>) =>
  JSON.stringify({ id: "c", object: "chat.completion.chunk", choices: [{ index: 0, delta, finish_reason: null }] });

const STREAMS: Record<string, string[]> = {
  "anthropic-text": anthropicEvents([{ type: "text_delta", text: `here: ${OUT_LIT}` }]),
  "anthropic-tool": anthropicEvents([{ type: "input_json_delta", partial_json: `{"q":"${OUT_LIT}"}` }]),
  "anthropic-thinking": anthropicEvents([
    { type: "thinking_delta", thinking: `considering ${OUT_LIT}` },
    { type: "text_delta", text: "visible answer" },
  ]),
  "chat-tool": [
    chatChunk({ role: "assistant" }),
    chatChunk({ tool_calls: [{ index: 0, id: "call_1", type: "function", function: { name: "f", arguments: "" } }] }),
    chatChunk({ tool_calls: [{ index: 0, function: { arguments: `{"q":"${OUT_LIT}"}` } }] }),
    "[DONE]",
  ],
  "responses-tool": [
    JSON.stringify({ type: "response.created", response: { id: "r", status: "in_progress", output: [] } }),
    JSON.stringify({ type: "response.function_call_arguments.delta", item_id: "fc", output_index: 0, delta: `{"q":"${OUT_LIT}"}` }),
    JSON.stringify({ type: "response.completed", response: { id: "r", status: "completed", output: [] } }),
  ],
  // ~600 bytes of text over 60 frames: the frames outweigh the cap, the
  // text they carry does not.
  "chat-many-frames": [
    chatChunk({ role: "assistant" }),
    ...Array.from({ length: 60 }, () => chatChunk({ content: "0123456789" })),
    "[DONE]",
  ],
};

describe("passthrough guardrail scan coverage", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  const upstreams: Record<string, OpenAiUpstream> = {};
  const otlps: MockOtlp[] = [];
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);
    for (const [name, streamEvents] of Object.entries(STREAMS)) {
      upstreams[name] = await startOpenAiUpstream({ streamEvents });
    }
    upstreams.input = await startOpenAiUpstream({
      nonStreamBody: { id: "c", object: "chat.completion", choices: [] },
    });
    upstreams["raw-output"] = await startOpenAiUpstream({
      rawBody: ESCAPED_BLOCK_JSON,
      rawContentType: "application/json",
    });
    upstreams["raw-stream"] = await startOpenAiUpstream({
      rawStreamFrames: [`data: ${ESCAPED_BLOCK_JSON}\n\n`, "data: [DONE]\n\n"],
    });
    upstreams["raw-deep-output"] = await startOpenAiUpstream({
      rawBody: DEEP_ESCAPED_BLOCK_JSON,
      rawContentType: "application/json",
    });
    upstreams["raw-deep-stream"] = await startOpenAiUpstream({
      rawStreamFrames: [`data: ${DEEP_ESCAPED_BLOCK_JSON}\n\n`, "data: [DONE]\n\n"],
    });
    upstreams["raw-safe-output"] = await startOpenAiUpstream({
      rawBody: SAFE_ESCAPED_JSON,
      rawContentType: "application/json",
    });
    upstreams["raw-safe-stream"] = await startOpenAiUpstream({
      rawStreamFrames: [`data: ${SAFE_ESCAPED_JSON}\n\n`],
    });
    const pk = await seed.createProviderKey({
      display_name: "pt-scan-pk",
      secret: "sk-mock",
      api_base: upstreams.input.baseUrl,
    });
    for (const [name, upstream] of Object.entries(upstreams)) {
      await seed.createPassthroughRoute({
        name: `pt-scan-${name}`,
        path_prefix: `/pt-scan-${name}`,
        target_url: upstream.baseUrl,
        provider_key_id: pk.id,
      });
    }
    await seed.createGuardrail({
      name: "pt-scan-output",
      enabled: true,
      hook_point: "output",
      kind: "keyword",
      patterns: [
        { kind: "literal", value: OUT_LIT },
        { kind: "literal", value: ESCAPED_BLOCK },
      ],
    });
    await seed.createGuardrail({
      name: "pt-scan-input",
      enabled: true,
      hook_point: "input",
      kind: "keyword",
      patterns: [
        { kind: "literal", value: IN_LIT },
        { kind: "literal", value: ESCAPED_BLOCK },
        { kind: "literal", value: ESCAPED_CJK },
      ],
    });
    // Folds the output chain's hold-back cap down to CAP, fail-closed.
    await seed.createGuardrail({
      name: "pt-scan-cap",
      enabled: true,
      hook_point: "output",
      kind: "pii",
      detectors: [{ type: "email", action: "block" }],
      max_buffer_bytes: CAP,
      on_buffer_exceeded: "fail_closed",
    });
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: [], allowed_routes: ["*"] });
    const proxy = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(Object.values(upstreams).map((u) => u.close()));
    await Promise.all(otlps.map((o) => o.close()));
  });

  const call = (route: string, path: string, body: Record<string, unknown>) =>
    fetch(`${app!.proxyUrl}/pt-scan-${route}${path}`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify(body),
    });
  const callRaw = (route: string, path: string, body: string) =>
    fetch(`${app!.proxyUrl}/pt-scan-${route}${path}`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body,
    });
  const anthropicBody = { model: "claude-3-5-haiku-20241022", max_tokens: 64, stream: true, messages: [{ role: "user", content: "go" }] };
  const chatBody = { model: "gpt-4o-mini", stream: true, messages: [{ role: "user", content: "go" }] };
  const responsesBody = { model: "gpt-4o-mini", stream: true, input: "go" };

  const ready = (ctx: { skip: () => void }) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return false;
    }
    return true;
  };

  const expectBlocked = async (res: Response) => {
    expect(res.status).toBe(200);
    const body = await res.text();
    expect(body).toContain("event: error");
    expect(body).toContain("content_filter");
    expect(body).not.toContain(OUT_LIT);
  };

  test.for([
    ["anthropic-text", "/v1/messages", anthropicBody],
    ["anthropic-tool", "/v1/messages", anthropicBody],
    ["chat-tool", "/v1/chat/completions", chatBody],
    ["responses-tool", "/v1/responses", responsesBody],
  ] as const)("output: %s is scanned", async ([route, path, body], ctx) => {
    if (!ready(ctx)) return;
    await expectBlocked(await call(route, path, body));
  });

  test("output: generated thinking is not scanned", async (ctx) => {
    if (!ready(ctx)) return;
    const res = await call("anthropic-thinking", "/v1/messages", anthropicBody);
    expect(res.status).toBe(200);
    const body = await res.text();
    expect(body).not.toContain("event: error");
    expect(body).toContain("visible answer");
  });

  test("hold-back cap: frames over the cap, content under it, is released", async (ctx) => {
    if (!ready(ctx)) return;
    const res = await call("chat-many-frames", "/v1/chat/completions", chatBody);
    expect(res.status).toBe(200);
    const body = await res.text();
    expect(body).not.toContain("output_buffer_exceeded");
    expect(body.split("0123456789").length - 1).toBe(60);
  });

  test.for([
    [
      "anthropic system prompt",
      { model: "claude-3-5-haiku-20241022", max_tokens: 64, system: `be ${IN_LIT}`, messages: [{ role: "user", content: "go" }] },
    ],
    [
      "anthropic tool result",
      {
        model: "claude-3-5-haiku-20241022",
        max_tokens: 64,
        messages: [
          { role: "user", content: "go" },
          { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "f", input: {} }] },
          {
            role: "user",
            content: [{ type: "tool_result", tool_use_id: "t1", content: [{ type: "text", text: `found ${IN_LIT}` }] }],
          },
        ],
      },
    ],
    [
      "responses replayed tool call",
      {
        model: "gpt-4o-mini",
        input: [
          { role: "user", content: "go" },
          { type: "function_call", call_id: "c1", name: "f", arguments: `{"q":"${IN_LIT}"}` },
        ],
      },
    ],
  ] as const)("input: %s is scanned", async ([, body], ctx) => {
    if (!ready(ctx)) return;
    const before = upstreams.input!.receivedRequests.length;
    const res = await call("input", "/v1/any", body);
    expect(res.status).toBe(422);
    expect(await res.text()).toContain("pt-scan-input");
    expect(upstreams.input!.receivedRequests.length).toBe(before);
  });

  test.for([
    [
      "system-one state",
      {
        messages: [{ role: "user", content: "clean" }],
        state: { query: IN_LIT },
      },
    ],
    [
      "rerank query and documents",
      {
        messages: [{ role: "user", content: "clean" }],
        query: IN_LIT,
        documents: ["clean"],
      },
    ],
  ] as const)("input: forwarded %s is scanned", async ([, body], ctx) => {
    if (!ready(ctx)) return;
    const before = upstreams.input!.receivedRequests.length;
    const res = await call("input", "/v1/any", body);
    expect(res.status).toBe(422);
    expect(await res.text()).toContain("pt-scan-input");
    expect(upstreams.input!.receivedRequests.length).toBe(before);
  });

  test("input: a duplicate forwarded field is scanned", async (ctx) => {
    if (!ready(ctx)) return;
    const before = upstreams.input!.receivedRequests.length;
    const body = String.raw`{"messages":[{"role":"user","content":"clean"}],"state":{"query":"${IN_LIT}"},"state":"clean"}`;
    const res = await callRaw("input", "/v1/any", body);
    expect(res.status).toBe(422);
    expect(await res.text()).toContain("pt-scan-input");
    expect(upstreams.input!.receivedRequests.length).toBe(before);
  });

  test.for([
    ["ASCII", ESCAPED_BLOCK_JSON],
    ["CJK", ESCAPED_CJK_JSON],
  ] as const)("input: raw JSON %s escapes are decoded before scanning", async ([, body], ctx) => {
    if (!ready(ctx)) return;
    const before = upstreams.input!.receivedRequests.length;
    const res = await callRaw("input", "/v1/any", body);
    expect(res.status).toBe(422);
    expect(await res.text()).toContain("pt-scan-input");
    expect(upstreams.input!.receivedRequests.length).toBe(before);
  });

  test("input: deep raw JSON escapes are decoded before scanning", async (ctx) => {
    if (!ready(ctx)) return;
    const before = upstreams.input!.receivedRequests.length;
    const res = await callRaw("input", "/v1/any", DEEP_ESCAPED_BLOCK_JSON);
    expect(res.status).toBe(422);
    expect(await res.text()).toContain("pt-scan-input");
    expect(upstreams.input!.receivedRequests.length).toBe(before);
  });

  test("input: safe raw JSON keeps its original bytes upstream", async (ctx) => {
    if (!ready(ctx)) return;
    const before = upstreams.input!.receivedRequests.length;
    const res = await callRaw("input", "/v1/any", SAFE_ESCAPED_JSON);
    expect(res.status).toBe(200);
    await res.text();
    expect(upstreams.input!.receivedRequests.length).toBe(before + 1);
    expect(upstreams.input!.receivedRequests.at(-1)!.body).toBe(SAFE_ESCAPED_JSON);
  });

  test("output: raw JSON escapes are decoded before scanning", async (ctx) => {
    if (!ready(ctx)) return;
    const before = upstreams["raw-output"]!.receivedRequests.length;
    const res = await callRaw("raw-output", "/v1/any", String.raw`{"state":"clean"}`);
    expect(res.status).toBe(422);
    const body = await res.text();
    expect(body).toContain("pt-scan-output");
    expect(body).not.toContain(ESCAPED_BLOCK);
    expect(upstreams["raw-output"]!.receivedRequests.length).toBe(before + 1);
  });

  test("output: safe raw JSON keeps its original bytes downstream", async (ctx) => {
    if (!ready(ctx)) return;
    const before = upstreams["raw-safe-output"]!.receivedRequests.length;
    const res = await callRaw("raw-safe-output", "/v1/any", SAFE_ESCAPED_JSON);
    expect(res.status).toBe(200);
    expect(await res.text()).toBe(SAFE_ESCAPED_JSON);
    expect(upstreams["raw-safe-output"]!.receivedRequests.length).toBe(before + 1);
  });

  test("output: deep raw JSON escapes are decoded before scanning", async (ctx) => {
    if (!ready(ctx)) return;
    const before = upstreams["raw-deep-output"]!.receivedRequests.length;
    const res = await callRaw("raw-deep-output", "/v1/any", String.raw`{"state":"clean"}`);
    expect(res.status).toBe(422);
    const body = await res.text();
    expect(body).toContain("pt-scan-output");
    expect(body).not.toContain(ESCAPED_BLOCK);
    expect(upstreams["raw-deep-output"]!.receivedRequests.length).toBe(before + 1);
  });

  test("output: raw SSE JSON escapes are decoded before scanning", async (ctx) => {
    if (!ready(ctx)) return;
    const before = upstreams["raw-stream"]!.receivedRequests.length;
    const res = await callRaw("raw-stream", "/v1/any", String.raw`{"state":"clean"}`);
    expect(res.status).toBe(200);
    const body = await res.text();
    expect(body).toContain("event: error");
    expect(body).toContain("content_filter");
    expect(body).not.toContain(ESCAPED_BLOCK);
    expect(upstreams["raw-stream"]!.receivedRequests.length).toBe(before + 1);
  });

  test("output: safe raw SSE JSON keeps its original bytes downstream", async (ctx) => {
    if (!ready(ctx)) return;
    const before = upstreams["raw-safe-stream"]!.receivedRequests.length;
    const res = await callRaw("raw-safe-stream", "/v1/any", SAFE_ESCAPED_JSON);
    expect(res.status).toBe(200);
    expect(await res.text()).toBe(`data: ${SAFE_ESCAPED_JSON}\n\n`);
    expect(upstreams["raw-safe-stream"]!.receivedRequests.length).toBe(before + 1);
  });

  test("telemetry: Raw buffered and SSE responses retain their original JSON source", async (ctx) => {
    if (!ready(ctx)) return;

    const otlp = await startMockOtlp();
    otlps.push(otlp);
    await seed!.createObservabilityExporter({
      name: "pt-scan-raw-source-capture",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
      content_mode: "full",
      content_max_bytes: 4_096,
    });
    const proxy = new ProxyClient(app!.proxyUrl, CALLER);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);

    const completionFor = async (route: string) => {
      const deadline = Date.now() + 10_000;
      while (Date.now() < deadline) {
        const span = otlp.spans.find(
          (candidate) =>
            candidate.attributes["aisix.passthrough.route_name"] === `pt-scan-${route}` &&
            candidate.attributes["gen_ai.completion"] === SAFE_ESCAPED_JSON,
        );
        if (span) return span;
        await new Promise((resolve) => setTimeout(resolve, 50));
      }
      throw new Error(`no raw-source OTLP completion for ${route}`);
    };

    const buffered = await callRaw("raw-safe-output", "/v1/any", SAFE_ESCAPED_JSON);
    expect(buffered.status).toBe(200);
    expect(await buffered.text()).toBe(SAFE_ESCAPED_JSON);
    await completionFor("raw-safe-output");

    const streamed = await callRaw("raw-safe-stream", "/v1/any", SAFE_ESCAPED_JSON);
    expect(streamed.status).toBe(200);
    expect(await streamed.text()).toBe(`data: ${SAFE_ESCAPED_JSON}\n\n`);
    await completionFor("raw-safe-stream");
    expect(otlp.parseFailures).toEqual([]);
  });

  test("output: deep raw SSE JSON escapes are decoded before scanning", async (ctx) => {
    if (!ready(ctx)) return;
    const before = upstreams["raw-deep-stream"]!.receivedRequests.length;
    const res = await callRaw("raw-deep-stream", "/v1/any", String.raw`{"state":"clean"}`);
    expect(res.status).toBe(200);
    const body = await res.text();
    expect(body).toContain("event: error");
    expect(body).toContain("content_filter");
    expect(body).not.toContain(ESCAPED_BLOCK);
    expect(upstreams["raw-deep-stream"]!.receivedRequests.length).toBe(before + 1);
  });
});
