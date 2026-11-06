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

// E2E for the `/v1/responses` → chat-completions bridge driven the way the
// Codex CLI's lite mode drives it: no top-level `tools`, the tools sent in an
// `additional_tools` input item instead, all of them in the `functions`
// namespace, the shell tool (`exec`) a freeform `custom` tool.
//
// The tools must reach the chat upstream, so the model can call them rather
// than writing its calls out as text, and a call the model makes must come
// back as the item Codex dispatches on: the declared name with its
// `namespace`, as a `custom_tool_call` for the freeform tool and a
// `function_call` for the function. Both directions used to fail: the item
// was dropped as an empty message, and a namespace's `custom` members were
// dropped even when declared at the top level.
//
// The bridge is reached by declaring an EMPTY `apis` map on the provider
// key, so the request cannot be served by a native `/v1/responses`.

const CALLER = "sk-responses-bridge-additional-tools";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");

const EXEC_TOOL = {
  type: "custom",
  name: "exec",
  description: "Run a shell command.",
  format: { type: "grammar", syntax: "lark", definition: "start: /.+/" },
};
const WAIT_TOOL = {
  type: "function",
  name: "wait",
  description: "Wait for a running command.",
  strict: false,
  parameters: { type: "object", properties: { ms: { type: "number" } }, required: ["ms"] },
};
const FUNCTIONS_NAMESPACE = {
  type: "namespace",
  name: "functions",
  description: "Tools available to the agent.",
  tools: [EXEC_TOOL, WAIT_TOOL],
};
const ADDITIONAL_TOOLS = {
  type: "additional_tools",
  role: "developer",
  id: "at_00000000-0000-5000-8000-000000000000",
  tools: [FUNCTIONS_NAMESPACE],
};
const USER_TURN = { type: "message", role: "user", content: [{ type: "input_text", text: "list the files" }] };

const USAGE = { prompt_tokens: 20, completion_tokens: 6, total_tokens: 26 };
const EXEC_CALL = { id: "call_exec", type: "function", function: { name: "functions__exec", arguments: '{"content":"ls -la"}' } };
const WAIT_CALL = { id: "call_wait", type: "function", function: { name: "functions__wait", arguments: '{"ms":10}' } };

function chunk(delta: Record<string, unknown>, finish: string | null = null, usage?: unknown): string {
  return JSON.stringify({
    id: "chatcmpl-additional-tools",
    object: "chat.completion.chunk",
    model: "relay-mini",
    choices: [{ index: 0, delta, finish_reason: finish }],
    ...(usage ? { usage } : {}),
  });
}

const UPSTREAMS: Record<string, OpenAiUpstreamOptions> = {
  "lite-tools": {
    nonStreamBody: {
      id: "chatcmpl-lite",
      object: "chat.completion",
      created: 1_700_000_000,
      model: "relay-mini",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: null, tool_calls: [EXEC_CALL, WAIT_CALL] },
          finish_reason: "tool_calls",
        },
      ],
      usage: USAGE,
    },
  },
  "lite-tools-stream": {
    streamEvents: [
      chunk({ role: "assistant", tool_calls: [{ index: 0, ...EXEC_CALL, function: { name: "functions__exec", arguments: "" } }] }),
      chunk({ tool_calls: [{ index: 0, function: { arguments: '{"content":' } }] }),
      chunk({ tool_calls: [{ index: 0, function: { arguments: '"ls -la"}' } }] }),
      chunk({ tool_calls: [{ index: 1, ...WAIT_CALL, function: { name: "functions__wait", arguments: "" } }] }),
      chunk({ tool_calls: [{ index: 1, function: { arguments: '{"ms":10}' } }] }),
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

/** The call items the caller must get back, whatever their ids. */
const EXPECTED_CALLS = [
  { type: "custom_tool_call", call_id: "call_exec", name: "exec", namespace: "functions", input: "ls -la", status: "completed" },
  { type: "function_call", call_id: "call_wait", name: "wait", namespace: "functions", arguments: '{"ms":10}', status: "completed" },
];

describe("/v1/responses bridge: tools declared in additional_tools input items", () => {
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
      await seed.createModel({
        display_name: model,
        provider: "openai",
        model_name: "relay-compat-x",
        provider_key_id: pk.id,
      });
    }
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: Object.keys(UPSTREAMS) });
    const probe = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    for (const u of Object.values(upstreams)) await u.close();
  });

  async function responses(body: Record<string, unknown>): Promise<Response> {
    return fetch(`${app!.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  function dispatched(model: string): Record<string, any> {
    return JSON.parse(upstreams[model]!.receivedRequests.at(-1)!.body);
  }

  function toolNames(body: Record<string, any>): string[] {
    return (body.tools ?? []).map((t: { function: { name: string } }) => t.function.name);
  }

  /** What every lite-mode request must put on the chat wire. */
  function expectLiteToolsDispatched(sent: Record<string, any>): void {
    expect(toolNames(sent)).toEqual(["functions__exec", "functions__wait"]);
    const [exec, wait] = sent.tools;
    expect(exec.type).toBe("function");
    // The freeform tool takes its input as one string.
    expect(exec.function.parameters.type).toBe("object");
    expect(Object.values(exec.function.parameters.properties)).toEqual([
      expect.objectContaining({ type: "string" }),
    ]);
    expect(exec.function.description).toContain("Run a shell command.");
    expect(wait.function.parameters).toEqual(WAIT_TOOL.parameters);
    expect(sent.tool_choice).toBe("auto");
    expect(sent.parallel_tool_calls).toBe(true);
    // The declaration item is not a turn of the conversation.
    const serialized = JSON.stringify(sent.messages);
    expect(serialized).not.toContain("additional_tools");
    expect(serialized).not.toContain("Wait for a running command.");
  }

  test("non-streaming: the tools reach the upstream and the calls come back under their declared names", async (ctx) => {
    if (!etcdReachable || !app) return void ctx.skip();
    const res = await responses({
      model: "lite-tools",
      input: [ADDITIONAL_TOOLS, USER_TURN],
      tool_choice: "auto",
      parallel_tool_calls: true,
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as Record<string, any>;
    const sent = dispatched("lite-tools");
    expectLiteToolsDispatched(sent);
    expect(sent.messages.map((m: { role: string }) => m.role)).toEqual(["user"]);
    expect(body.output).toEqual(EXPECTED_CALLS.map((c) => ({ ...c, id: expect.any(String) })));
  });

  test("a replayed namespace call goes upstream under the name the model was offered", async (ctx) => {
    if (!etcdReachable || !app) return void ctx.skip();
    const res = await responses({
      model: "lite-tools",
      input: [
        ADDITIONAL_TOOLS,
        USER_TURN,
        { type: "custom_tool_call", call_id: "call_prev", name: "exec", namespace: "functions", input: "pwd" },
        { type: "custom_tool_call_output", call_id: "call_prev", output: "/work" },
      ],
      tool_choice: "auto",
      parallel_tool_calls: true,
    });
    expect(res.status).toBe(200);
    const sent = dispatched("lite-tools");
    expectLiteToolsDispatched(sent);
    const assistant = sent.messages.find((m: { role: string }) => m.role === "assistant");
    expect(assistant.tool_calls).toHaveLength(1);
    expect(assistant.tool_calls[0].id).toBe("call_prev");
    expect(assistant.tool_calls[0].function.name).toBe("functions__exec");
    expect(Object.values(JSON.parse(assistant.tool_calls[0].function.arguments))).toEqual(["pwd"]);
  });

  test("streaming: each call is announced and closed under its declared name and namespace", async (ctx) => {
    if (!etcdReachable || !app) return void ctx.skip();
    const res = await responses({
      model: "lite-tools-stream",
      stream: true,
      input: [ADDITIONAL_TOOLS, USER_TURN],
      tool_choice: "auto",
      parallel_tool_calls: true,
    });
    expect(res.status).toBe(200);
    const frames = parseSse(await res.text());
    expectLiteToolsDispatched(dispatched("lite-tools-stream"));

    const added = frames.filter((f) => f.data.type === "response.output_item.added").map((f) => f.data.item);
    expect(added).toEqual([
      expect.objectContaining({ type: "custom_tool_call", name: "exec", namespace: "functions", call_id: "call_exec" }),
      expect.objectContaining({ type: "function_call", name: "wait", namespace: "functions", call_id: "call_wait" }),
    ]);
    const done = frames.filter((f) => f.data.type === "response.output_item.done").map((f) => f.data.item);
    expect(done).toEqual(EXPECTED_CALLS.map((c) => ({ ...c, id: expect.any(String) })));
    // Each item keeps the id it was announced under.
    expect(done.map((i) => i.id)).toEqual(added.map((i) => i.id));
    const inputDone = frames.find((f) => f.data.type === "response.custom_tool_call_input.done");
    expect(inputDone?.data.input).toBe("ls -la");
    const completed = frames.find((f) => f.data.type === "response.completed");
    expect(completed?.data.response.output).toEqual(done);
  });

  test("a tool re-declared with a new schema in a later item is offered once, with the new schema", async (ctx) => {
    if (!etcdReachable || !app) return void ctx.skip();
    // What Codex appends to the history when the shell tool's schema changes:
    // the earlier declaration stays, the new one follows.
    const exec2 = {
      ...EXEC_TOOL,
      description: "Run a shell command in the sandbox.",
      format: { type: "grammar", syntax: "lark", definition: "start: CMD\nCMD: /[a-z ]+/" },
    };
    const res = await responses({
      model: "lite-tools",
      input: [
        ADDITIONAL_TOOLS,
        USER_TURN,
        { type: "additional_tools", role: "developer", tools: [{ ...FUNCTIONS_NAMESPACE, tools: [exec2] }] },
      ],
      tool_choice: "auto",
    });
    expect(res.status).toBe(200);
    const sent = dispatched("lite-tools");
    expect(toolNames(sent)).toEqual(["functions__exec", "functions__wait"]);
    const exec = sent.tools[0].function;
    expect(exec.description).toContain("Run a shell command in the sandbox.");
    expect(exec.description).toContain("CMD: /[a-z ]+/");
    expect(exec.description).not.toContain("start: /.+/");
    // The call still comes back under the declared name and namespace.
    const body = (await res.json()) as Record<string, any>;
    expect(body.output).toEqual(EXPECTED_CALLS.map((c) => ({ ...c, id: expect.any(String) })));
  });

  test("additional_tools items and top-level tools all reach the upstream", async (ctx) => {
    if (!etcdReachable || !app) return void ctx.skip();
    const res = await responses({
      model: "lite-tools",
      tools: [{ type: "function", name: "top_level", parameters: { type: "object" } }],
      input: [
        ADDITIONAL_TOOLS,
        USER_TURN,
        { type: "additional_tools", role: "developer", tools: [{ type: "custom", name: "later" }] },
      ],
      tool_choice: "auto",
    });
    expect(res.status).toBe(200);
    expect(toolNames(dispatched("lite-tools"))).toEqual(["top_level", "functions__exec", "functions__wait", "later"]);
  });
});
