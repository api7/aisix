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
  const upstreams: Record<string, OpenAiUpstream> = {};
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    for (const [name, streamEvents] of Object.entries(STREAMS)) {
      upstreams[name] = await startOpenAiUpstream({ streamEvents });
    }
    upstreams.input = await startOpenAiUpstream({
      nonStreamBody: { id: "c", object: "chat.completion", choices: [] },
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
      patterns: [{ kind: "literal", value: OUT_LIT }],
    });
    await seed.createGuardrail({
      name: "pt-scan-input",
      enabled: true,
      hook_point: "input",
      kind: "keyword",
      patterns: [{ kind: "literal", value: IN_LIT }],
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
  });

  const call = (route: string, path: string, body: Record<string, unknown>) =>
    fetch(`${app!.proxyUrl}/pt-scan-${route}${path}`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify(body),
    });
  const anthropicBody = { model: "claude-3-5-haiku-20241022", max_tokens: 64, stream: true, messages: [{ role: "user", content: "go" }] };
  const chatBody = { model: "gpt-4o-mini", stream: true, messages: [{ role: "user", content: "go" }] };
  const responsesBody = { model: "gpt-4o-mini", stream: true, input: "go" };

  const ready = (ctx: { skip: () => void }) => {
    if (!etcdReachable || !app) {
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
});
