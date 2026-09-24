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
import type { OpenAiUpstreamOptions } from "../harness/upstream-openai.js";

// E2E for the `/v1/responses` → chat-completions bridge as an agent client
// (the Codex CLI) drives it against a non-OpenAI model.
//
// Five contracts.
//
// 1. Every Response object carries the full top-level field set the
//    Responses API defines — the non-streaming body and the `response` of
//    every lifecycle event — echoing the request's own settings. It used to
//    carry seven keys, so a client validating the object rejected it.
//
// 2. A stream that fails part-way ends with the flat `error` event AND a
//    `response.failed` carrying `error.code` / `error.message`. With only
//    the flat event, a client that does not read it saw a dropped
//    connection and retried the turn without ever showing why. A guardrail
//    stop reports `invalid_prompt` there, which clients do not retry. A
//    transport drop AFTER the upstream's finish reason completes normally.
//
// 3. An upstream stream that carries nothing (a bare `[DONE]`, a usage-only
//    frame) fails with a retryable code instead of completing with an empty
//    output that the client takes as a finished turn. A failure that is the
//    first thing the client receives — that one, a connection dropped before
//    the first chunk, a held-back output blocked by a guardrail — still
//    opens the stream with `response.created` and `response.in_progress`:
//    the OpenAI SDKs' `responses.stream()` helper rejects a stream that
//    does not, and its caller never sees the real error.
//
// 4. Replayed `reasoning` items reach the upstream as `reasoning_content` on
//    the assistant message they belong to. They used to be dropped.
//
// 5. `namespace` tools reach the model as flattened function tools, and a
//    call to one comes back as a `function_call` naming the sub-tool and its
//    `namespace`. They used to be dropped, so the model never saw them.
//
// The bridge is reached by declaring an EMPTY `apis` map on the provider
// key — the operator saying "this OpenAI-compatible endpoint has no
// `/v1/responses`".

const CALLER_PLAINTEXT = "sk-responses-bridge-codex-parity";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");
const HEADERS = {
  authorization: `Bearer ${CALLER_PLAINTEXT}`,
  "content-type": "application/json",
};

// Every top-level member the Responses API requires of a Response object.
const RESPONSE_MEMBERS = [
  "id", "object", "created_at", "completed_at", "status", "incomplete_details",
  "model", "previous_response_id", "instructions", "output", "error", "tools",
  "tool_choice", "truncation", "parallel_tool_calls", "text", "top_p",
  "presence_penalty", "frequency_penalty", "top_logprobs", "temperature",
  "reasoning", "usage", "max_output_tokens", "max_tool_calls", "store",
  "background", "service_tier", "metadata", "safety_identifier",
  "prompt_cache_key",
];

const FORBIDDEN_OUTPUT = "forbidden-output-word";

function chunk(delta: Record<string, unknown>, finish: string | null = null, usage?: unknown): string {
  return JSON.stringify({
    id: "chatcmpl-codex-parity",
    object: "chat.completion.chunk",
    model: "relay-mini",
    choices: [{ index: 0, delta, finish_reason: finish }],
    ...(usage ? { usage } : {}),
  });
}

const USAGE = { prompt_tokens: 12, completion_tokens: 3, total_tokens: 15 };

const NAMESPACE_TOOL = {
  type: "namespace",
  name: "multi_agent_v1",
  description: "Tools for spawning and managing sub-agents.",
  tools: [
    {
      type: "function",
      name: "spawn_agent",
      description: "Spawn a sub-agent.",
      strict: false,
      parameters: {
        type: "object",
        properties: { task: { type: "string" } },
        required: ["task"],
      },
    },
  ],
};
const SHELL_TOOL = {
  type: "function",
  name: "exec_command",
  parameters: { type: "object", properties: { cmd: { type: "string" } } },
};

const SPAWN_CALL = {
  id: "call_spawn",
  type: "function",
  function: { name: "multi_agent_v1__spawn_agent", arguments: '{"task":"x"}' },
};

// Each upstream serves one canned reply; the model name selects it.
const UPSTREAMS: Record<string, OpenAiUpstreamOptions> = {
  "parity-text": {
    nonStreamBody: {
      id: "chatcmpl-text",
      object: "chat.completion",
      created: 1_700_000_000,
      model: "relay-mini",
      choices: [{ index: 0, message: { role: "assistant", content: "hello" }, finish_reason: "stop" }],
      usage: USAGE,
    },
  },
  "parity-text-stream": {
    streamEvents: [chunk({ content: "hel" }), chunk({ content: "lo" }), chunk({}, "stop", USAGE), "[DONE]"],
  },
  // Content, then the connection drops before any finish reason.
  "parity-cut-before-finish": {
    streamEvents: [chunk({ content: "half an ans" }), chunk({ content: "wer" }), chunk({}, "stop", USAGE), "[DONE]"],
    disconnectAfterEvents: 1,
    // Lets each write reach the socket before the connection is destroyed.
    eventDelayMs: 50,
  },
  // Finish reason sent, then the connection drops before usage / [DONE].
  "parity-cut-after-finish": {
    streamEvents: [chunk({ content: "whole answer" }), chunk({}, "stop"), chunk({}, null, USAGE), "[DONE]"],
    disconnectAfterEvents: 2,
    eventDelayMs: 50,
  },
  "parity-in-band-error": {
    streamEvents: [
      chunk({ content: "par" }),
      JSON.stringify({
        error: {
          message: "This model's maximum context length is 8192 tokens.",
          type: "invalid_request_error",
          code: "context_length_exceeded",
        },
      }),
    ],
  },
  "parity-empty-stream": { streamEvents: ["[DONE]"] },
  // Headers, then the connection drops before the first chunk.
  "parity-cut-before-first-chunk": {
    streamEvents: [chunk({ content: "never sent" }), chunk({}, "stop", USAGE), "[DONE]"],
    disconnectAfterEvents: 0,
    // Lets the headers reach the socket before the connection is destroyed.
    firstEventDelayMs: 50,
  },
  "parity-usage-only-stream": { streamEvents: [chunk({}, null, USAGE), "[DONE]"] },
  "parity-output-blocked": {
    streamEvents: [chunk({ content: `this carries ${FORBIDDEN_OUTPUT}` }), chunk({}, "stop", USAGE), "[DONE]"],
  },
  "parity-namespace-call": {
    nonStreamBody: {
      id: "chatcmpl-ns",
      object: "chat.completion",
      created: 1_700_000_000,
      model: "relay-mini",
      choices: [
        { index: 0, message: { role: "assistant", content: null, tool_calls: [SPAWN_CALL] }, finish_reason: "tool_calls" },
      ],
      usage: USAGE,
    },
  },
  "parity-namespace-call-stream": {
    streamEvents: [
      chunk({ role: "assistant", tool_calls: [{ index: 0, ...SPAWN_CALL, function: { ...SPAWN_CALL.function, arguments: "" } }] }),
      chunk({ tool_calls: [{ index: 0, function: { arguments: '{"task":"x"}' } }] }),
      chunk({}, "tool_calls", USAGE),
      "[DONE]",
    ],
  },
};

interface SseFrame {
  event: string;
  data: Record<string, any>;
}

function parseSse(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  for (const block of text.split("\n\n")) {
    let event = "";
    const dataLines: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event: ")) event = line.slice(7).trim();
      else if (line.startsWith("data: ")) dataLines.push(line.slice(6));
    }
    const payload = dataLines.join("\n").trim();
    if (!payload || payload === "[DONE]") continue;
    frames.push({ event, data: JSON.parse(payload) });
  }
  return frames;
}

function expectFullResponse(r: Record<string, any>): void {
  const missing = RESPONSE_MEMBERS.filter((k) => !(k in r));
  expect(missing, `Response object lacks ${missing.join(", ")}`).toEqual([]);
}

const FAILED_BEFORE_ANY_OUTPUT = ["response.created", "response.in_progress", "error", "response.failed"];

/** A stream whose failure was the first thing to reach the client: opened,
 *  then failed, one Response throughout, numbered from zero. */
function expectOpenedThenFailed(frames: SseFrame[]): [Record<string, any>, Record<string, any>] {
  expect(frames.map((f) => f.data.type)).toEqual(FAILED_BEFORE_ANY_OUTPUT);
  expect(frames.map((f) => f.event)).toEqual(FAILED_BEFORE_ANY_OUTPUT);
  expect(frames.map((f) => f.data.sequence_number)).toEqual([0, 1, 2, 3]);
  const [created, inProgress, error, failed] = frames.map((f) => f.data);
  for (const opening of [created!, inProgress!]) {
    expectFullResponse(opening.response);
    expect(opening.response.status).toBe("in_progress");
    expect(opening.response.output).toEqual([]);
  }
  expect(failed!.response.id).toBe(created!.response.id);
  expect(failed!.response.status).toBe("failed");
  expect(failed!.response.error.message).toBe(error!.message);
  return [error!, failed!];
}

describe("/v1/responses bridge: what an agent client needs from a non-OpenAI model", () => {
  let app: SpawnedApp | undefined;
  const upstreams: Record<string, OpenAiUpstream> = {};
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    for (const [model, opts] of Object.entries(UPSTREAMS)) {
      const upstream = await startOpenAiUpstream(opts);
      upstreams[model] = upstream;
      const pk = await seed.createProviderKey({
        display_name: `${model}-pk`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
        apis: {},
      });
      const created = await seed.createModel({
        display_name: model,
        provider: "openai",
        model_name: "relay-compat-x",
        provider_key_id: pk.id,
      });
      if (model === "parity-output-blocked") {
        const guardrail = await seed.createGuardrail(
          {
            name: "parity-output-block",
            enabled: true,
            hook_point: "output",
            kind: "keyword",
            patterns: [{ kind: "literal", value: FORBIDDEN_OUTPUT }],
          },
          { attach: false },
        );
        await seed.attachGuardrailToModel(guardrail.id, created.id);
      }
    }
    // Seeded last: the key authenticating implies the whole seed set is in
    // the snapshot.
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: Object.keys(UPSTREAMS),
    });
    const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    for (const u of Object.values(upstreams)) await u.close();
  });

  async function responses(body: Record<string, unknown>): Promise<Response> {
    return fetch(`${app!.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify(body),
    });
  }

  async function streamFrames(model: string, extra: Record<string, unknown> = {}): Promise<SseFrame[]> {
    const res = await responses({ model, input: "hi", stream: true, ...extra });
    expect(res.status).toBe(200);
    return parseSse(await res.text());
  }

  /** The last upstream request body a model's mock received. */
  function dispatched(model: string): Record<string, any> {
    return JSON.parse(upstreams[model]!.receivedRequests.at(-1)!.body);
  }

  // ── 1. Full Response object ────────────────────────────────────────

  test("non-streaming: the Response carries every member, echoing the request's settings", async (ctx) => {
    if (!etcdReachable || !app) return void ctx.skip();
    const res = await responses({
      model: "parity-text",
      instructions: "be terse",
      input: "hi",
      tools: [SHELL_TOOL],
      tool_choice: "auto",
      parallel_tool_calls: false,
      temperature: 0.3,
      store: false,
      prompt_cache_key: "session-1",
      max_output_tokens: null,
      previous_response_id: null,
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as Record<string, any>;
    expectFullResponse(body);
    expect(body.status).toBe("completed");
    expect(body.instructions).toBe("be terse");
    expect(body.parallel_tool_calls).toBe(false);
    expect(body.temperature).toBe(0.3);
    expect(body.store).toBe(false);
    expect(body.prompt_cache_key).toBe("session-1");
    expect(body.tools).toEqual([{ ...SHELL_TOOL, description: null, strict: null }]);
    expect(body.error).toBeNull();
    expect(body.max_output_tokens).toBeNull();
    expect(typeof body.completed_at).toBe("number");
    expect(body.output[0].content[0].text).toBe("hello");
  });

  test("streaming: created, in_progress and completed all carry the full Response", async (ctx) => {
    if (!etcdReachable || !app) return void ctx.skip();
    const frames = await streamFrames("parity-text-stream", { instructions: "sys" });
    const lifecycle = frames.filter((f) => f.data.response !== undefined);
    expect(lifecycle.map((f) => f.data.type)).toEqual([
      "response.created",
      "response.in_progress",
      "response.completed",
    ]);
    for (const f of lifecycle) {
      expectFullResponse(f.data.response);
      expect(f.data.response.instructions).toBe("sys");
    }
    expect(lifecycle[0]!.data.response.usage).toBeNull();
    expect(lifecycle[2]!.data.response.usage.input_tokens).toBe(12);
  });

  // ── 2. response.failed ──────────────────────────────────────────────

  test("a stream cut before the finish ends with the error frame, then response.failed", async (ctx) => {
    if (!etcdReachable || !app) return void ctx.skip();
    const frames = await streamFrames("parity-cut-before-finish");
    const types = frames.map((f) => f.data.type);
    expect(types).toContain("response.output_text.delta");
    expect(types.slice(-2)).toEqual(["error", "response.failed"]);
    expect(types).not.toContain("response.completed");
    const [error, failed] = frames.slice(-2).map((f) => f.data);
    expect(failed!.response.status).toBe("failed");
    expect(failed!.response.error).toEqual({ code: error!.code, message: error!.message });
    expect(failed!.response.output).toEqual([]);
    expect(failed!.response.usage).toBeNull();
    expectFullResponse(failed!.response);
    // The two events continue the stream's own numbering.
    expect(frames.map((f) => f.data.sequence_number)).toEqual(frames.map((_, i) => i));
  });

  test("an upstream's own context-length code reaches response.failed verbatim", async (ctx) => {
    if (!etcdReachable || !app) return void ctx.skip();
    const frames = await streamFrames("parity-in-band-error");
    const failed = frames.at(-1)!.data;
    expect(failed.type).toBe("response.failed");
    expect(failed.response.error.code).toBe("context_length_exceeded");
    expect(failed.response.error.message).toContain("maximum context length");
    // The flat frame keeps the gateway's own code.
    expect(frames.at(-2)!.data.code).toBe("upstream_in_band_error");
  });

  test("an output-guardrail block ends with response.failed carrying invalid_prompt", async (ctx) => {
    if (!etcdReachable || !app) return void ctx.skip();
    const frames = await streamFrames("parity-output-blocked");
    // Held back and never released: the stream is opened, then fails.
    const [error, failed] = expectOpenedThenFailed(frames);
    expect(error.code).toBe("content_filter");
    expect(failed.response.error.code).toBe("invalid_prompt");
    expect(JSON.stringify(frames)).not.toContain(FORBIDDEN_OUTPUT);
  });

  test("a connection dropped after the finish reason still completes the response", async (ctx) => {
    if (!etcdReachable || !app) return void ctx.skip();
    const frames = await streamFrames("parity-cut-after-finish");
    const types = frames.map((f) => f.data.type);
    expect(types.at(-1)).toBe("response.completed");
    expect(types).not.toContain("error");
    const completed = frames.at(-1)!.data.response;
    expect(completed.output[0].content[0].text).toBe("whole answer");
    // The usage frame never arrived: the reported usage is the estimate.
    expect(completed.usage.input_tokens).toBeGreaterThan(0);
    expect(completed.usage.output_tokens).toBeGreaterThan(0);
  });

  // ── 3. Empty upstream stream ────────────────────────────────────────

  for (const model of ["parity-empty-stream", "parity-usage-only-stream"]) {
    test(`${model}: a stream that carried nothing fails with a retryable code`, async (ctx) => {
      if (!etcdReachable || !app) return void ctx.skip();
      const frames = await streamFrames(model);
      const [error, failed] = expectOpenedThenFailed(frames);
      expect(error.code).toBe("upstream_error");
      expect(error.message).toContain("empty stream");
      expect(failed.response.error.code).toBe("upstream_error");
    });
  }

  test("a connection dropped before the first chunk opens the stream, then fails", async (ctx) => {
    if (!etcdReachable || !app) return void ctx.skip();
    const frames = await streamFrames("parity-cut-before-first-chunk");
    const [error, failed] = expectOpenedThenFailed(frames);
    expect(error.code).toBe("transport_error");
    expect(failed.response.error.code).toBe("transport_error");
  });

  // ── 4. Replayed reasoning ───────────────────────────────────────────

  test("replayed reasoning and tool calls go back on the assistant turn that made them", async (ctx) => {
    if (!etcdReachable || !app) return void ctx.skip();
    const reasoning = (text: string) => ({
      type: "reasoning",
      id: `rs_${text.length}`,
      summary: [{ type: "summary_text", text }],
      content: null,
      encrypted_content: null,
    });
    // Codex's own history order: reasoning, the model's text, the calls it
    // made in that turn, their results.
    const res = await responses({
      model: "parity-text",
      input: [
        { type: "message", role: "user", content: [{ type: "input_text", text: "list files" }] },
        reasoning("I should run ls"),
        { type: "message", role: "assistant", content: [{ type: "output_text", text: "Running ls." }] },
        { type: "function_call", call_id: "c1", name: "exec_command", arguments: '{"cmd":"ls"}' },
        { type: "function_call_output", call_id: "c1", output: "a.txt" },
        reasoning("interrupted before answering"),
        { type: "message", role: "user", content: [{ type: "input_text", text: "stop" }] },
      ],
      tools: [SHELL_TOOL],
    });
    expect(res.status).toBe(200);
    const messages = dispatched("parity-text").messages as Record<string, any>[];
    expect(messages.map((m) => m.role)).toEqual(["user", "assistant", "tool", "assistant", "user"]);
    // One turn: the text, the call, and the reasoning behind both.
    expect(messages[1]).toMatchObject({ content: "Running ls.", reasoning_content: "I should run ls" });
    expect(messages[1].tool_calls.map((tc: any) => tc.id)).toEqual(["c1"]);
    // Reasoning no answer followed is still passed back, on a turn of its
    // own that carries nothing else.
    expect(messages[3].reasoning_content).toBe("interrupted before answering");
    expect(messages[3].content).toBe("");
    expect(messages[3].tool_calls).toBeUndefined();
    // Never as visible content.
    for (const m of messages) expect(JSON.stringify(m.content ?? "")).not.toContain("I should run ls");
  });

  // ── 5. Namespace tools ──────────────────────────────────────────────

  test("namespace sub-tools reach the model flattened, and a replayed call uses that name", async (ctx) => {
    if (!etcdReachable || !app) return void ctx.skip();
    const res = await responses({
      model: "parity-namespace-call",
      input: [
        { type: "message", role: "user", content: "spawn a helper" },
        { type: "function_call", call_id: "c0", name: "spawn_agent", namespace: "multi_agent_v1", arguments: '{"task":"y"}' },
        { type: "function_call_output", call_id: "c0", output: "agent-1" },
      ],
      tools: [SHELL_TOOL, NAMESPACE_TOOL, { type: "web_search", external_web_access: true }],
    });
    expect(res.status).toBe(200);
    const sent = dispatched("parity-namespace-call");
    expect(sent.tools.map((t: any) => t.function.name)).toEqual([
      "exec_command",
      "multi_agent_v1__spawn_agent",
    ]);
    expect(sent.tools[1].function.description).toBe(
      "Tools for spawning and managing sub-agents.\n\nSpawn a sub-agent.",
    );
    expect(sent.tools[1].function.parameters).toEqual(NAMESPACE_TOOL.tools[0]!.parameters);
    expect(sent.messages[1].tool_calls[0].function.name).toBe("multi_agent_v1__spawn_agent");

    const body = (await res.json()) as Record<string, any>;
    expect(body.output).toHaveLength(1);
    expect(body.output[0]).toMatchObject({
      type: "function_call",
      call_id: "call_spawn",
      name: "spawn_agent",
      namespace: "multi_agent_v1",
      arguments: '{"task":"x"}',
    });
  });

  test("streaming: a namespace call is announced and closed as the sub-tool with its namespace", async (ctx) => {
    if (!etcdReachable || !app) return void ctx.skip();
    const frames = await streamFrames("parity-namespace-call-stream", { tools: [NAMESPACE_TOOL] });
    const items = frames
      .filter((f) => ["response.output_item.added", "response.output_item.done"].includes(f.data.type))
      .map((f) => f.data.item);
    expect(items).toHaveLength(2);
    for (const item of items) {
      expect(item).toMatchObject({ type: "function_call", name: "spawn_agent", namespace: "multi_agent_v1" });
    }
    const completed = frames.at(-1)!.data;
    expect(completed.type).toBe("response.completed");
    expect(completed.response.output[0]).toMatchObject({
      name: "spawn_agent",
      namespace: "multi_agent_v1",
      arguments: '{"task":"x"}',
    });
  });
});
