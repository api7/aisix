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
// Codex CLI drives it when the model supports tool search: every MCP tool is
// deferred, Codex declares a `tool_search` tool it executes itself, and a
// search shows up in the history as a `tool_search_call` plus a
// `tool_search_output` whose namespace members are marked `defer_loading`.
//
// The chat model must be offered the search as a function and every tool a
// search loaded, see the search history as a tool call and its result, and
// its calls must come back as the items Codex dispatches on: a
// `tool_search_call` it runs on the client, or a `function_call` naming the
// loaded tool with its namespace. The bridge used to drop all of it silently.
//
// The bridge is reached by declaring an EMPTY `apis` map on the provider
// key, so the request cannot be served by a native `/v1/responses`.

const CALLER = "sk-responses-bridge-tool-search";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");

const SEARCH_PARAMETERS = {
  type: "object",
  properties: {
    query: { type: "string", description: "Search query for deferred tools." },
    limit: { type: "number", description: "Maximum number of tools to return. Defaults to 8." },
  },
  required: ["query"],
  additionalProperties: false,
};
const TOOL_SEARCH = {
  type: "tool_search",
  execution: "client",
  description: "# Tool discovery\n\nSearches over deferred tool metadata and exposes matching tools for the next model call.",
  parameters: SEARCH_PARAMETERS,
};
const SHELL_TOOL = {
  type: "function",
  name: "shell",
  description: "Run a shell command.",
  strict: false,
  parameters: { type: "object", properties: { command: { type: "string" } }, required: ["command"] },
};
const READ_PAGE = {
  type: "function",
  name: "read_page",
  description: "Read one page of the docs.",
  strict: false,
  defer_loading: true,
  parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
};
const SEARCH_PAGES = {
  type: "function",
  name: "search_pages",
  description: "Search the docs.",
  strict: false,
  defer_loading: true,
  parameters: { type: "object", properties: { q: { type: "string" } }, required: ["q"] },
};
const DOCS_NAMESPACE = {
  type: "namespace",
  name: "mcp__docs",
  description: "Tools of the docs MCP server.",
  tools: [READ_PAGE, SEARCH_PAGES],
};
const USER_TURN = { type: "message", role: "user", content: [{ type: "input_text", text: "find the install page" }] };
const SEARCH_HISTORY = [
  {
    type: "tool_search_call",
    call_id: "call_search_prev",
    execution: "client",
    status: "completed",
    arguments: { query: "docs pages", limit: 8 },
  },
  {
    type: "tool_search_output",
    call_id: "call_search_prev",
    execution: "client",
    status: "completed",
    tools: [DOCS_NAMESPACE],
  },
];
const LITE_INPUT = [
  { type: "additional_tools", role: "developer", tools: [SHELL_TOOL, TOOL_SEARCH] },
  USER_TURN,
  ...SEARCH_HISTORY,
];

const USAGE = { prompt_tokens: 30, completion_tokens: 8, total_tokens: 38 };
const SEARCH_CALL = {
  id: "call_search",
  type: "function",
  function: { name: "tool_search", arguments: '{"query":"install","limit":4}' },
};
const READ_CALL = {
  id: "call_read",
  type: "function",
  function: { name: "mcp__docs__read_page", arguments: '{"path":"install.md"}' },
};

function chunk(delta: Record<string, unknown>, finish: string | null = null, usage?: unknown): string {
  return JSON.stringify({
    id: "chatcmpl-tool-search",
    object: "chat.completion.chunk",
    model: "relay-mini",
    choices: [{ index: 0, delta, finish_reason: finish }],
    ...(usage ? { usage } : {}),
  });
}

function completion(toolCall: unknown): OpenAiUpstreamOptions {
  return {
    nonStreamBody: {
      id: "chatcmpl-tool-search",
      object: "chat.completion",
      created: 1_700_000_000,
      model: "relay-mini",
      choices: [
        { index: 0, message: { role: "assistant", content: null, tool_calls: [toolCall] }, finish_reason: "tool_calls" },
      ],
      usage: USAGE,
    },
  };
}

const UPSTREAMS: Record<string, OpenAiUpstreamOptions> = {
  "search-call": completion(SEARCH_CALL),
  "loaded-call": completion(READ_CALL),
  "search-call-stream": {
    streamEvents: [
      chunk({ role: "assistant", tool_calls: [{ index: 0, ...SEARCH_CALL, function: { name: "tool_search", arguments: "" } }] }),
      chunk({ tool_calls: [{ index: 0, function: { arguments: '{"query":' } }] }),
      chunk({ tool_calls: [{ index: 0, function: { arguments: '"install","limit":4}' } }] }),
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

const EXPECTED_SEARCH_CALL = {
  type: "tool_search_call",
  id: expect.any(String),
  call_id: "call_search",
  execution: "client",
  arguments: { query: "install", limit: 4 },
  status: "completed",
};

describe("/v1/responses bridge: Codex's client-executed tool search", () => {
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

  /** What every request carrying the search and its history must put on the chat wire. */
  function expectSearchDispatched(sent: Record<string, any>): void {
    const tools: Array<{ type: string; function: Record<string, any> }> = sent.tools ?? [];
    expect(tools.map((t) => t.function.name)).toEqual([
      "shell",
      "tool_search",
      "mcp__docs__read_page",
      "mcp__docs__search_pages",
    ]);
    expect(tools.every((t) => t.type === "function")).toBe(true);
    const search = tools[1]!.function;
    expect(search.description).toBe(TOOL_SEARCH.description);
    expect(search.parameters).toEqual(SEARCH_PARAMETERS);
    const read = tools[2]!.function;
    expect(read.description).toContain("Read one page of the docs.");
    expect(read.parameters).toEqual(READ_PAGE.parameters);
    expect(JSON.stringify(tools)).not.toContain("defer_loading");

    // The search the client ran: an assistant call to `tool_search`, then its result.
    const messages: Array<Record<string, any>> = sent.messages;
    expect(messages.map((m) => m.role)).toEqual(["user", "assistant", "tool"]);
    const [call] = messages[1]!.tool_calls;
    expect(call.id).toBe("call_search_prev");
    expect(call.function.name).toBe("tool_search");
    expect(JSON.parse(call.function.arguments)).toEqual({ query: "docs pages", limit: 8 });
    expect(messages[2]!.tool_call_id).toBe("call_search_prev");
    expect(messages[2]!.content).toContain("mcp__docs__read_page");
    expect(messages[2]!.content).toContain("mcp__docs__search_pages");
  }

  test("non-streaming: the search and the tools it loaded reach the upstream, and a search call comes back as one", async (ctx) => {
    if (!etcdReachable || !app) return void ctx.skip();
    const res = await responses({ model: "search-call", input: LITE_INPUT, tool_choice: "auto" });
    expect(res.status).toBe(200);
    expectSearchDispatched(dispatched("search-call"));
    const body = (await res.json()) as Record<string, any>;
    expect(body.output).toEqual([EXPECTED_SEARCH_CALL]);
  });

  test("a search declared at the top level works the same way", async (ctx) => {
    if (!etcdReachable || !app) return void ctx.skip();
    const res = await responses({
      model: "search-call",
      tools: [SHELL_TOOL, TOOL_SEARCH],
      input: [USER_TURN, ...SEARCH_HISTORY],
      tool_choice: "auto",
    });
    expect(res.status).toBe(200);
    expectSearchDispatched(dispatched("search-call"));
    const body = (await res.json()) as Record<string, any>;
    expect(body.output).toEqual([EXPECTED_SEARCH_CALL]);
  });

  test("a deferred tool is offered only once a search has loaded it", async (ctx) => {
    if (!etcdReachable || !app) return void ctx.skip();
    const res = await responses({
      model: "search-call",
      tools: [SHELL_TOOL, TOOL_SEARCH, DOCS_NAMESPACE],
      input: [USER_TURN],
      tool_choice: "auto",
    });
    expect(res.status).toBe(200);
    const sent = dispatched("search-call");
    expect(sent.tools.map((t: { function: { name: string } }) => t.function.name)).toEqual(["shell", "tool_search"]);
  });

  test("a call to a loaded tool comes back under its name and namespace", async (ctx) => {
    if (!etcdReachable || !app) return void ctx.skip();
    const res = await responses({ model: "loaded-call", input: LITE_INPUT, tool_choice: "auto" });
    expect(res.status).toBe(200);
    expectSearchDispatched(dispatched("loaded-call"));
    const body = (await res.json()) as Record<string, any>;
    expect(body.output).toEqual([
      {
        type: "function_call",
        id: expect.any(String),
        call_id: "call_read",
        name: "read_page",
        namespace: "mcp__docs",
        arguments: '{"path":"install.md"}',
        status: "completed",
      },
    ]);
  });

  test("streaming: a search call is announced, then closed with its arguments", async (ctx) => {
    if (!etcdReachable || !app) return void ctx.skip();
    const res = await responses({ model: "search-call-stream", stream: true, input: LITE_INPUT, tool_choice: "auto" });
    expect(res.status).toBe(200);
    const frames = parseSse(await res.text());
    expectSearchDispatched(dispatched("search-call-stream"));

    const added = frames.filter((f) => f.data.type === "response.output_item.added").map((f) => f.data.item);
    expect(added).toEqual([
      expect.objectContaining({ type: "tool_search_call", call_id: "call_search", execution: "client", status: "in_progress" }),
    ]);
    const done = frames.filter((f) => f.data.type === "response.output_item.done").map((f) => f.data.item);
    expect(done).toEqual([EXPECTED_SEARCH_CALL]);
    expect(done[0].id).toBe(added[0].id);
    // The search is not a function call: no function-call argument events.
    expect(frames.some((f) => String(f.data.type).startsWith("response.function_call_arguments"))).toBe(false);
    const completed = frames.find((f) => f.data.type === "response.completed");
    expect(completed?.data.response.output).toEqual(done);
  });
});
