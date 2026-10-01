import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForSlsLog,
  type MockSls,
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
const RAW_BLOCK_SSE = String.raw`"\u0042LOCKME"`;
const RAW_SAFE_SSE = String.raw`"\u0063lean"`;
const RAW_PREFIX_SSE = `"FOR"`;
const RAW_UNEVALUABLE_SSE = String.raw`{"state":"safe"}`;
const RAW_SUFFIX_SSE = `"BIDDEN"`;
const RAW_HELD_BLOCK_SSE = `"FORBIDDEN"`;
const deepEscapedBlockJSON = (depth: number) =>
  `${'{"v":'.repeat(depth)}"${String.raw`\u0042LOCKME`}"${'}'.repeat(depth)}`;
const deepLiteralBlockJSON = (depth: number) =>
  `${'{"v":'.repeat(depth)}"${ESCAPED_BLOCK}"${'}'.repeat(depth)}`;
const ANTHROPIC_TOOL_RESULT_PREFIX = '[{"type":"tool_result","tool_use_id":"t","content":';
const ANTHROPIC_TOOL_RESULT_TEXT = `[{"type":"text","text":"${ESCAPED_BLOCK}"}]`;
const ANTHROPIC_TOOL_RESULT_SUFFIX = "}]";
const deeplyNestedAnthropicToolResultRequest = (depth: number, model: string) => {
  const content = [
    ANTHROPIC_TOOL_RESULT_PREFIX.repeat(depth),
    ANTHROPIC_TOOL_RESULT_TEXT,
    ANTHROPIC_TOOL_RESULT_SUFFIX.repeat(depth),
  ].join("");
  return `{"model":"${model}","max_tokens":64,"messages":[{"role":"user","content":${content}}]}`;
};
// Above serde_json's default recursion limit. It remains valid JSON and the
// provider receives it verbatim, so Raw guardrails must still decode the leaf.
const DEEP_ESCAPED_BLOCK_JSON = deepEscapedBlockJSON(160);
// Mirrors `json_splice::MAX_JSON_DEPTH`: one deeper is unscannable and must
// use the resolved guardrail failure policy instead of falling back to escapes.
const JSON_DEPTH_CAP = 4_096;
const OVER_DEPTH_ESCAPED_BLOCK_JSON = deepEscapedBlockJSON(JSON_DEPTH_CAP + 1);
const OVER_DEPTH_OPAQUE_BLOCK_JSON = deepLiteralBlockJSON(JSON_DEPTH_CAP + 1);
const AT_DEPTH_ANTHROPIC_TOOL_RESULT_INPUT = deeplyNestedAnthropicToolResultRequest(
  JSON_DEPTH_CAP,
  "nested-tool-result-at-depth-cap",
);
const OVER_DEPTH_ANTHROPIC_TOOL_RESULT_INPUT = deeplyNestedAnthropicToolResultRequest(
  JSON_DEPTH_CAP + 1,
  "nested-tool-result-fail-closed",
);
const OVER_DEPTH_ANTHROPIC_TOOL_RESULT_FAIL_OPEN_INPUT = deeplyNestedAnthropicToolResultRequest(
  JSON_DEPTH_CAP + 1,
  "nested-tool-result-fail-open",
);
const OVER_DEPTH_CHAT_INPUT = `{"model":"gpt-4o-mini","messages":[{"role":"user","content":"go","metadata":${OVER_DEPTH_ESCAPED_BLOCK_JSON}}]}`;
const OVER_DEPTH_RESPONSES_INPUT = `{"model":"gpt-4o-mini","input":[{"role":"user","metadata":${OVER_DEPTH_ESCAPED_BLOCK_JSON},"content":[{"type":"input_text","text":"go"}]}]}`;
const OVER_DEPTH_CHAT_OPAQUE_INPUT = `{"model":"gpt-4o-mini","messages":[{"role":"user","content":[{"type":"image","source":{"data":"image","metadata":${OVER_DEPTH_OPAQUE_BLOCK_JSON}}},{"type":"text","text":"go"}]}]}`;
const OVER_DEPTH_RESPONSES_OPAQUE_INPUT = `{"model":"gpt-4o-mini","input":[{"role":"user","content":[{"type":"input_image","image_url":{"url":"https://example.invalid/image","metadata":${OVER_DEPTH_OPAQUE_BLOCK_JSON}}},{"type":"input_text","text":"go"}]}]}`;
const OVER_DEPTH_ANTHROPIC_TOOL_OUTPUT = `{"type":"message","content":[{"type":"tool_use","id":"tool_1","name":"lookup","input":${OVER_DEPTH_ESCAPED_BLOCK_JSON}}]}`;
const CAP = 1_000;
const SPLIT_BLOCK = "FORBIDDEN";
const SPLIT_BLOCK_REGEX = String.raw`FOR\s*BIDDEN`;
const KNOWN_CHAT_OUTPUT = String.raw`{"model":"routing-only","choices":[{"message":{"content":"\u0042LOCKME","metadata":{"note":"${OUT_LIT}"}}}],"choices":[{"message":{"content":"clean"}}]}`;
const KNOWN_RESPONSES_OUTPUT = String.raw`{"output":[{"type":"message","content":[{"type":"output_text","text":"\u0042LOCKME","metadata":{"note":"${OUT_LIT}"}}]}],"output":[{"type":"message","content":[{"type":"output_text","text":"clean"}]}]}`;
const KNOWN_CHAT_STREAM = `${String.raw`data: {"choices":[{"index":0,"delta":{"content":"\u0042LOCKME"}},{"index":1,"delta":{"content":"clean"}}]}`}\n\n`;
const KNOWN_RESPONSES_STREAM = `${String.raw`data: {"type":"response.output_text.delta","item_id":"known","output_index":0,"content_index":0,"delta":"\u0042LOCKME","delta":"clean","metadata":{"note":"${OUT_LIT}"}}`}\n\n`;
const SPLIT_RESPONSES_STREAM = [
  `data: {"type":"response.output_text.delta","item_id":"same","output_index":0,"content_index":0,"delta":"FOR"}\n\n`,
  `data: {"type":"response.output_text.delta","item_id":"same","output_index":0,"content_index":0,"delta":"noise","delta":"BIDDEN"}\n\n`,
  `data: {"type":"response.output_item.done","item":{"id":"same","type":"message"}}\n\n`,
  "data: [DONE]\n\n",
];
const DISTINCT_ITEMS_RESPONSES_STREAM = [
  `data: {"type":"response.output_text.delta","item_id":"first","output_index":0,"content_index":0,"delta":"FOR"}\n\n`,
  `data: {"type":"response.output_text.delta","item_id":"second","output_index":1,"content_index":0,"delta":"BIDDEN"}\n\n`,
  "data: [DONE]\n\n",
];
const RESPONSES_REASONING_STREAM = [
  `data: ${JSON.stringify({ type: "response.reasoning_text.done", text: OUT_LIT })}\n\n`,
  `data: ${JSON.stringify({ type: "response.content_part.done", part: { type: "reasoning_text", text: OUT_LIT } })}\n\n`,
  `data: ${JSON.stringify({ type: "response.output_item.done", item: { id: "reasoning", type: "reasoning", summary: [{ type: "summary_text", text: OUT_LIT }] } })}\n\n`,
  `data: ${JSON.stringify({ type: "response.output_text.delta", item_id: "message", output_index: 0, content_index: 0, delta: "clean" })}\n\n`,
  `data: ${JSON.stringify({ type: "response.completed", response: { output: [{ type: "reasoning", summary: [{ type: "summary_text", text: OUT_LIT }] }, { type: "message", content: [{ type: "output_text", text: "clean" }] }] } })}\n\n`,
  "data: [DONE]\n\n",
];

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
      rawStreamFrames: [`data: ${RAW_BLOCK_SSE}\n\n`, "data: [DONE]\n\n"],
    });
    upstreams["raw-deep-output"] = await startOpenAiUpstream({
      rawBody: DEEP_ESCAPED_BLOCK_JSON,
      rawContentType: "application/json",
    });
    upstreams["raw-over-depth-output"] = await startOpenAiUpstream({
      rawBody: OVER_DEPTH_ESCAPED_BLOCK_JSON,
      rawContentType: "application/json",
    });
    upstreams["completions-over-depth-output"] = await startOpenAiUpstream({
      rawBody: OVER_DEPTH_ESCAPED_BLOCK_JSON,
      rawContentType: "application/json",
    });
    upstreams["anthropic-over-depth-tool-output"] = await startOpenAiUpstream({
      rawBody: OVER_DEPTH_ANTHROPIC_TOOL_OUTPUT,
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
      rawStreamFrames: [`data: ${RAW_SAFE_SSE}\n\n`, "data: [DONE]\n\n"],
    });
    upstreams["known-chat-buffered-output"] = await startOpenAiUpstream({
      rawBody: KNOWN_CHAT_OUTPUT,
      rawContentType: "application/json",
    });
    upstreams["known-responses-buffered-output"] = await startOpenAiUpstream({
      rawBody: KNOWN_RESPONSES_OUTPUT,
      rawContentType: "application/json",
    });
    upstreams["known-chat-stream-output"] = await startOpenAiUpstream({
      rawStreamFrames: [KNOWN_CHAT_STREAM],
    });
    upstreams["known-responses-stream-output"] = await startOpenAiUpstream({
      rawStreamFrames: [KNOWN_RESPONSES_STREAM],
    });
    upstreams["split-responses-stream-output"] = await startOpenAiUpstream({
      rawStreamFrames: SPLIT_RESPONSES_STREAM,
    });
    upstreams["distinct-responses-stream-output"] = await startOpenAiUpstream({
      rawStreamFrames: DISTINCT_ITEMS_RESPONSES_STREAM,
    });
    upstreams["known-responses-reasoning-stream"] = await startOpenAiUpstream({
      rawStreamFrames: RESPONSES_REASONING_STREAM,
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
        { kind: "regex", value: SPLIT_BLOCK_REGEX },
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
  test.for([
    [
      "chat",
      "known-chat-buffered-output",
      `{"model":"gpt-4o-mini","messages":[{"role":"user","content":"go"}]}`,
    ],
    [
      "Responses",
      "known-responses-buffered-output",
      `{"model":"gpt-4o-mini","input":"go"}`,
    ],
  ] as const)("output: known %s envelope is source-scanned when buffered", async ([, route, body], ctx) => {
    if (!ready(ctx)) return;
    const upstream = upstreams[route];
    if (!upstream) throw new Error(`missing ${route} upstream`);
    const before = upstream.receivedRequests.length;
    const res = await callRaw(route, "/v1/any", body);
    expect(res.status).toBe(422);
    const response = await res.text();
    expect(response).toContain("pt-scan-output");
    expect(response).not.toContain(ESCAPED_BLOCK);
    expect(response).not.toContain(String.raw`\u0042LOCKME`);
    expect(upstream.receivedRequests.length).toBe(before + 1);
  });

  test.for([
    [
      "chat",
      "known-chat-stream-output",
      `{"model":"gpt-4o-mini","stream":true,"messages":[{"role":"user","content":"go"}]}`,
    ],
    [
      "Responses",
      "known-responses-stream-output",
      `{"model":"gpt-4o-mini","stream":true,"input":"go"}`,
    ],
  ] as const)("output: known %s envelope is source-scanned when streamed", async ([, route, body], ctx) => {
    if (!ready(ctx)) return;
    const upstream = upstreams[route];
    if (!upstream) throw new Error(`missing ${route} upstream`);
    const before = upstream.receivedRequests.length;
    const res = await callRaw(route, "/v1/any", body);
    expect(res.status).toBe(200);
    const response = await res.text();
    expect(response).toContain("event: error");
    expect(response).toContain("content_filter");
    expect(response).not.toContain(ESCAPED_BLOCK);
    expect(response).not.toContain(String.raw`\u0042LOCKME`);
    expect(upstream.receivedRequests.length).toBe(before + 1);
  });

  test("output: visible deltas remain contiguous across stream metadata", async (ctx) => {
    if (!ready(ctx)) return;
    const route = "split-responses-stream-output";
    const upstream = upstreams[route];
    if (!upstream) throw new Error(`missing ${route} upstream`);
    const before = upstream.receivedRequests.length;
    const res = await callRaw(
      route,
      "/v1/any",
      `{"model":"gpt-4o-mini","stream":true,"input":"go"}`,
    );
    expect(res.status).toBe(200);
    const response = await res.text();
    expect(response).toContain("event: error");
    expect(response).toContain("content_filter");
    expect(response).not.toContain(SPLIT_BLOCK);
    expect(upstream.receivedRequests.length).toBe(before + 1);
  });

  test("output: visible deltas from distinct Responses items never concatenate", async (ctx) => {
    if (!ready(ctx)) return;
    const route = "distinct-responses-stream-output";
    const upstream = upstreams[route];
    if (!upstream) throw new Error(`missing ${route} upstream`);
    const before = upstream.receivedRequests.length;
    const res = await callRaw(
      route,
      "/v1/any",
      `{"model":"gpt-4o-mini","stream":true,"input":"go"}`,
    );
    expect(res.status).toBe(200);
    const response = await res.text();
    expect(response).not.toContain("event: error");
    expect(response).toContain("FOR");
    expect(response).toContain("BIDDEN");
    expect(upstream.receivedRequests.length).toBe(before + 1);
  });

  test("output: generated Responses reasoning frames stay out of scope", async (ctx) => {
    if (!ready(ctx)) return;
    const route = "known-responses-reasoning-stream";
    const upstream = upstreams[route];
    if (!upstream) throw new Error(`missing ${route} upstream`);
    const before = upstream.receivedRequests.length;
    const res = await callRaw(
      route,
      "/v1/any",
      `{"model":"gpt-4o-mini","stream":true,"input":"go"}`,
    );
    expect(res.status).toBe(200);
    const response = await res.text();
    expect(response).not.toContain("event: error");
    expect(response).toContain(OUT_LIT);
    expect(response).toContain("clean");
    expect(upstream.receivedRequests.length).toBe(before + 1);
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
    [
      "duplicate chat messages",
      String.raw`{"model":"gpt-4o-mini","messages":[{"role":"user","content":"\u0069nputleakliteral","metadata":{"note":"${IN_LIT}"}}],"messages":[{"role":"user","content":"clean"}]}`,
    ],
    [
      "duplicate Responses input",
      String.raw`{"model":"gpt-4o-mini","input":"\u0069nputleakliteral","input":"clean","metadata":{"note":"${IN_LIT}"}}`,
    ],
  ] as const)("input: known envelope %s is source-scanned", async ([, body], ctx) => {
    if (!ready(ctx)) return;
    const before = upstreams.input!.receivedRequests.length;
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

  test("input: Raw JSON beyond the scanner depth cap fails closed", async (ctx) => {
    if (!ready(ctx)) return;
    const before = upstreams.input!.receivedRequests.length;
    const res = await callRaw("input", "/v1/any", OVER_DEPTH_ESCAPED_BLOCK_JSON);
    expect(res.status).toBe(422);
    const body = await res.text();
    expect(body).toContain("guardrail_unavailable");
    expect(body).toContain("unscannable_body");
    expect(body).not.toContain(ESCAPED_BLOCK);
    expect(upstreams.input!.receivedRequests.length).toBe(before);
  });

  test("input: nested Anthropic tool results at the scanner depth cap are blocked", async (ctx) => {
    if (!ready(ctx)) return;
    const before = upstreams.input!.receivedRequests.length;
    const res = await callRaw("input", "/v1/any", AT_DEPTH_ANTHROPIC_TOOL_RESULT_INPUT);
    expect(res.status).toBe(422);
    const body = await res.text();
    expect(body).toContain("pt-scan-input");
    expect(body).not.toContain("guardrail_unavailable");
    expect(body).not.toContain("unscannable_body");
    expect(body).not.toContain(ESCAPED_BLOCK);
    expect(upstreams.input!.receivedRequests.length).toBe(before);
  });

  test("input: nested Anthropic tool results beyond the scanner depth cap fail closed", async (ctx) => {
    if (!ready(ctx)) return;
    const before = upstreams.input!.receivedRequests.length;
    const res = await callRaw("input", "/v1/any", OVER_DEPTH_ANTHROPIC_TOOL_RESULT_INPUT);
    expect(res.status).toBe(422);
    const body = await res.text();
    expect(body).toContain("guardrail_unavailable");
    expect(body).toContain("unscannable_body");
    expect(body).not.toContain(ESCAPED_BLOCK);
    expect(upstreams.input!.receivedRequests.length).toBe(before);
  });

  test.for([
    ["Chat", OVER_DEPTH_CHAT_INPUT],
    ["Responses", OVER_DEPTH_RESPONSES_INPUT],
  ] as const)("input: %s envelope beyond the scanner depth cap fails closed", async ([, body], ctx) => {
    if (!ready(ctx)) return;
    const before = upstreams.input!.receivedRequests.length;
    const res = await callRaw("input", "/v1/any", body);
    expect(res.status).toBe(422);
    const response = await res.text();
    expect(response).toContain("guardrail_unavailable");
    expect(response).toContain("unscannable_body");
    expect(response).not.toContain(ESCAPED_BLOCK);
    expect(upstreams.input!.receivedRequests.length).toBe(before);
  });

  test.for([
    ["Chat", OVER_DEPTH_CHAT_OPAQUE_INPUT],
    ["Responses", OVER_DEPTH_RESPONSES_OPAQUE_INPUT],
  ] as const)("input: %s opaque media beyond the scanner depth cap stays out of scope", async ([, body], ctx) => {
    if (!ready(ctx)) return;
    const before = upstreams.input!.receivedRequests.length;
    const res = await callRaw("input", "/v1/any", body);
    expect(res.status).toBe(200);
    await res.text();
    expect(upstreams.input!.receivedRequests.length).toBe(before + 1);
    expect(upstreams.input!.receivedRequests.at(-1)!.body).toBe(body);
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

  test("output: Raw JSON beyond the scanner depth cap fails closed", async (ctx) => {
    if (!ready(ctx)) return;
    const before = upstreams["raw-over-depth-output"]!.receivedRequests.length;
    const res = await callRaw("raw-over-depth-output", "/v1/any", String.raw`{"state":"clean"}`);
    expect(res.status).toBe(422);
    const body = await res.text();
    expect(body).toContain("guardrail_unavailable");
    expect(body).toContain("unscannable_body");
    expect(body).not.toContain(ESCAPED_BLOCK);
    expect(upstreams["raw-over-depth-output"]!.receivedRequests.length).toBe(before + 1);
  });

  test("output: completions JSON beyond the scanner depth cap fails closed", async (ctx) => {
    if (!ready(ctx)) return;
    const before = upstreams["completions-over-depth-output"]!.receivedRequests.length;
    const res = await callRaw(
      "completions-over-depth-output",
      "/v1/completions",
      `{"model":"gpt-4o-mini","prompt":"go"}`,
    );
    expect(res.status).toBe(422);
    const body = await res.text();
    expect(body).toContain("guardrail_unavailable");
    expect(body).toContain("unscannable_body");
    expect(body).not.toContain(ESCAPED_BLOCK);
    expect(upstreams["completions-over-depth-output"]!.receivedRequests.length).toBe(before + 1);
  });

  test("output: Anthropic tool input beyond the scanner depth cap fails closed", async (ctx) => {
    if (!ready(ctx)) return;
    const route = "anthropic-over-depth-tool-output";
    const upstream = upstreams[route];
    if (!upstream) throw new Error(`missing ${route} upstream`);
    const before = upstream.receivedRequests.length;
    const res = await callRaw(
      route,
      "/v1/any",
      `{"model":"gpt-4o-mini","messages":[{"role":"user","content":"go"}]}`,
    );
    expect(res.status).toBe(422);
    const response = await res.text();
    expect(response).toContain("guardrail_unavailable");
    expect(response).toContain("unscannable_body");
    expect(response).not.toContain(ESCAPED_BLOCK);
    expect(upstream.receivedRequests.length).toBe(before + 1);
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

  test("output: safe bare-string Raw SSE keeps its original bytes downstream", async (ctx) => {
    if (!ready(ctx)) return;
    const before = upstreams["raw-safe-stream"]!.receivedRequests.length;
    const res = await callRaw("raw-safe-stream", "/v1/any", SAFE_ESCAPED_JSON);
    expect(res.status).toBe(200);
    expect(await res.text()).toBe(`data: ${RAW_SAFE_SSE}\n\ndata: [DONE]\n\n`);
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
    const requestUntilCaptured = async (
      route: string,
      expectedBody: string,
      expectedCompletion: string,
    ) => {
      // A healthy `/v1/models` reply proves only caller authentication. It
      // does not prove this newly added exporter has reached the snapshot, so
      // retry the real route until its real completion reaches the OTLP
      // receiver. Each attempt must still preserve the exact client bytes.
      const deadline = Date.now() + 30_000;
      let last = "no response";
      while (Date.now() < deadline) {
        const response = await callRaw(route, "/v1/any", SAFE_ESCAPED_JSON);
        const body = await response.text();
        last = `${response.status}: ${body}`;
        if (response.status !== 200 || body !== expectedBody) {
          await new Promise((resolve) => setTimeout(resolve, 50));
          continue;
        }
        const span = otlp.spans.find(
          (candidate) =>
            candidate.attributes["aisix.passthrough.route_name"] === `pt-scan-${route}` &&
            candidate.attributes["gen_ai.completion"] === expectedCompletion,
        );
        if (span) return span;
        await new Promise((resolve) => setTimeout(resolve, 50));
      }
      throw new Error(`no raw-source OTLP completion for ${route}; last response ${last}`);
    };

    await requestUntilCaptured("raw-safe-output", SAFE_ESCAPED_JSON, SAFE_ESCAPED_JSON);
    await requestUntilCaptured(
      "raw-safe-stream",
      `data: ${RAW_SAFE_SSE}\n\ndata: [DONE]\n\n`,
      RAW_SAFE_SSE,
    );
    expect(otlp.parseFailures).toEqual([]);
  });

  test("output: deep Raw JSON SSE is refused as an unevaluable carrier", async (ctx) => {
    if (!ready(ctx)) return;
    const before = upstreams["raw-deep-stream"]!.receivedRequests.length;
    const res = await callRaw("raw-deep-stream", "/v1/any", String.raw`{"state":"clean"}`);
    expect(res.status).toBe(200);
    const body = await res.text();
    expect(body).toContain("event: error");
    expect(body).toContain("guardrail_unavailable");
    expect(body).toContain("unscannable_body");
    expect(body).not.toContain(ESCAPED_BLOCK);
    expect(upstreams["raw-deep-stream"]!.receivedRequests.length).toBe(before + 1);
  });
});

// This is intentionally a separate DP: the main suite has an env-scoped
// blocking row, so it cannot demonstrate the live fail-open policy on either
// an unevaluable Raw input or output stream.
describe("passthrough Raw stream unevaluable-output fail-open", () => {
  const caller = "sk-pt-scan-fail-open";
  const callerHash = createHash("sha256").update(caller).digest("hex");
  const route = "pt-scan-fail-open";
  const depthInputRoute = "pt-scan-depth-fail-open-input";
  const logstore = "pt-scan-fail-open";
  const credentialRef = "pt_scan_open";
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let depthInputUpstream: OpenAiUpstream | undefined;
  let sls: MockSls | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    upstream = await startOpenAiUpstream({
      rawStreamFrames: [
        `data: ${RAW_PREFIX_SSE}\n\n`,
        `data: ${RAW_UNEVALUABLE_SSE}\n\n`,
        `data: ${RAW_SUFFIX_SSE}\n\n`,
        "data: [DONE]\n\n",
      ],
    });
    depthInputUpstream = await startOpenAiUpstream({
      rawBody: SAFE_ESCAPED_JSON,
      rawContentType: "application/json",
    });
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${credentialRef.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${credentialRef.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "pt-scan-fail-open-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "aisix-e2e-obs",
      logstore,
      credential_ref: credentialRef,
    });
    const providerKey = await seed.createProviderKey({
      display_name: "pt-scan-fail-open-pk",
      secret: "sk-mock",
      api_base: upstream.baseUrl,
    });
    await seed.createPassthroughRoute({
      name: route,
      path_prefix: "/pt-scan-fail-open",
      target_url: upstream.baseUrl,
      provider_key_id: providerKey.id,
    });
    await seed.createPassthroughRoute({
      name: depthInputRoute,
      path_prefix: `/${depthInputRoute}`,
      target_url: depthInputUpstream.baseUrl,
      provider_key_id: providerKey.id,
    });
    await seed.createGuardrail({
      name: "pt-scan-depth-fail-open-input",
      enabled: true,
      hook_point: "input",
      fail_open: true,
      kind: "keyword",
      patterns: [{ kind: "literal", value: ESCAPED_BLOCK }],
    });
    await seed.createGuardrail({
      name: "pt-scan-fail-open-output",
      enabled: true,
      enforcement_mode: "monitor",
      hook_point: "output",
      fail_open: true,
      kind: "keyword",
      patterns: [{ kind: "literal", value: "FOR" }],
    });
    await seed.createGuardrail({
      name: "pt-scan-fail-open-boundary",
      enabled: true,
      enforcement_mode: "monitor",
      hook_point: "output",
      fail_open: true,
      kind: "keyword",
      patterns: [{ kind: "regex", value: SPLIT_BLOCK_REGEX }],
    });
    await seed.createApiKey({ key_hash: callerHash, allowed_models: [], allowed_routes: ["*"] });
    const proxy = new ProxyClient(app.proxyUrl, caller);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  }, 90_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await depthInputUpstream?.close();
    await sls?.close();
  });

  test("forwards Raw JSON beyond the depth cap only under input fail_open", async (ctx) => {
    if (!etcdReachable || !app || !sls || !depthInputUpstream) return ctx.skip();

    const before = depthInputUpstream.receivedRequests.length;
    const res = await fetch(`${app.proxyUrl}/${depthInputRoute}/v1/any`, {
      method: "POST",
      headers: { authorization: `Bearer ${caller}`, "content-type": "application/json" },
      body: OVER_DEPTH_ESCAPED_BLOCK_JSON,
    });
    expect(res.status).toBe(200);
    expect(await res.text()).toBe(SAFE_ESCAPED_JSON);
    expect(depthInputUpstream.receivedRequests.length).toBe(before + 1);

    const log = await waitForSlsLog(
      sls,
      logstore,
      (entry) => entry.get("passthrough_route_name") === depthInputRoute,
      "fail-open depth-capped passthrough usage event",
    );
    expect(log.get("guardrail_blocked") ?? "false").not.toBe("true");
    expect(log.get("guardrail_bypassed_reason")).toBe("unscannable_body");
  });

  test("forwards nested Anthropic tool results beyond the depth cap only under input fail_open", async (ctx) => {
    if (!etcdReachable || !app || !sls || !depthInputUpstream) return ctx.skip();

    const before = depthInputUpstream.receivedRequests.length;
    const res = await fetch(`${app.proxyUrl}/${depthInputRoute}/v1/any`, {
      method: "POST",
      headers: { authorization: `Bearer ${caller}`, "content-type": "application/json" },
      body: OVER_DEPTH_ANTHROPIC_TOOL_RESULT_FAIL_OPEN_INPUT,
    });
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    expect(requestId).toBeTruthy();
    expect(await res.text()).toBe(SAFE_ESCAPED_JSON);
    expect(depthInputUpstream.receivedRequests.length).toBe(before + 1);
    expect(depthInputUpstream.receivedRequests.at(-1)!.body).toBe(
      OVER_DEPTH_ANTHROPIC_TOOL_RESULT_FAIL_OPEN_INPUT,
    );

    const log = await waitForSlsLog(
      sls,
      logstore,
      (entry) =>
        entry.get("passthrough_route_name") === depthInputRoute &&
        entry.get("request_id") === requestId,
      "fail-open nested-tool-result passthrough usage event",
    );
    expect(log.get("guardrail_blocked") ?? "false").not.toBe("true");
    expect(log.get("guardrail_bypassed_reason")).toBe("unscannable_body");
  });

  test("starts a new live scan epoch after an unkeyable Raw SSE object", async (ctx) => {
    if (!etcdReachable || !app || !sls || !upstream) return ctx.skip();

    const before = upstream.receivedRequests.length;
    const res = await fetch(`${app.proxyUrl}/pt-scan-fail-open/v1/any`, {
      method: "POST",
      headers: { authorization: `Bearer ${caller}`, "content-type": "application/json" },
      body: `{"model":"fail-open-raw","stream":true,"state":"go"}`,
    });
    expect(res.status).toBe(200);
    const body = await res.text();
    expect(body).not.toContain("event: error");
    expect(body).toBe(
      `data: ${RAW_PREFIX_SSE}\n\ndata: ${RAW_UNEVALUABLE_SSE}\n\ndata: ${RAW_SUFFIX_SSE}\n\ndata: [DONE]\n\n`,
    );
    expect(upstream.receivedRequests.length).toBe(before + 1);

    const log = await waitForSlsLog(
      sls,
      logstore,
      (entry) => entry.get("passthrough_route_name") === route,
      "fail-open passthrough usage event",
    );
    expect(log.get("guardrail_blocked") ?? "false").not.toBe("true");
    expect(log.get("guardrail_bypassed_reason")).toBe("unscannable_body");
    const hits = JSON.parse(log.get("guardrail_monitor_hits") ?? "[]") as Array<{
      action: string;
      hook: string;
      guardrail_name: string;
    }>;
    expect(hits).toContainEqual(
      expect.objectContaining({
        action: "would_block",
        hook: "output",
        guardrail_name: "pt-scan-fail-open-output",
      }),
    );
    expect(hits).not.toContainEqual(
      expect.objectContaining({
        action: "would_block",
        hook: "output",
        guardrail_name: "pt-scan-fail-open-boundary",
      }),
    );
  });
});

// `fail_open` only permits an unevaluable frame to pass on a live stream. A
// blocking guardrail still holds its prefix until it scans clean, so that
// prefix must never be released just because the next frame is unevaluable.
describe("passthrough Raw stream held fail-open continuity", () => {
  const caller = "sk-pt-scan-held-fail-open";
  const callerHash = createHash("sha256").update(caller).digest("hex");
  const route = "pt-scan-held-fail-open";
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      rawStreamFrames: [
        `data: ${RAW_HELD_BLOCK_SSE}\n\n`,
        `data: ${RAW_UNEVALUABLE_SSE}\n\n`,
        "data: [DONE]\n\n",
      ],
    });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const providerKey = await seed.createProviderKey({
      display_name: "pt-scan-held-fail-open-pk",
      secret: "sk-mock",
      api_base: upstream.baseUrl,
    });
    await seed.createPassthroughRoute({
      name: route,
      path_prefix: "/pt-scan-held-fail-open",
      target_url: upstream.baseUrl,
      provider_key_id: providerKey.id,
    });
    await seed.createGuardrail({
      name: "pt-scan-held-fail-open-output",
      enabled: true,
      hook_point: "output",
      fail_open: true,
      kind: "keyword",
      patterns: [{ kind: "literal", value: "FORBIDDEN" }],
    });
    await seed.createApiKey({ key_hash: callerHash, allowed_models: [], allowed_routes: ["*"] });
    const proxy = new ProxyClient(app.proxyUrl, caller);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  }, 90_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("refuses a held prefix before an unevaluable Raw SSE object can release it", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();

    const before = upstream.receivedRequests.length;
    const res = await fetch(`${app.proxyUrl}/pt-scan-held-fail-open/v1/any`, {
      method: "POST",
      headers: { authorization: `Bearer ${caller}`, "content-type": "application/json" },
      body: `{"model":"held-fail-open-raw","stream":true,"state":"go"}`,
    });
    expect(res.status).toBe(200);
    const body = await res.text();
    expect(body).toContain("event: error");
    expect(body).toContain("guardrail_unavailable");
    expect(body).toContain("unscannable_body");
    expect(body).not.toContain("FORBIDDEN");
    expect(upstream.receivedRequests.length).toBe(before + 1);
  });
});
