import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startMcpUpstream,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForSlsLog,
  type McpUpstream,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E (#1027): the local guardrail kinds (`keyword`, `pii`) judge the same
// slots their mask rewrites — one string value at a time, with object keys as
// slots of their own — so `block`, a monitor-mode `would_mask`, and an
// enforced `mask` cannot disagree about a request.
//
// Pinned contract, against a real gateway + etcd + real upstreams:
//   - /mcp: a rule anchored to a whole value (`^\d{4}$`) blocks on the tool
//     arguments and on the tool result, for both kinds;
//   - /mcp: a monitor-mode mask rule reports `would_mask` counts equal to the
//     counts the same rule masks when enforced, in both directions;
//   - /mcp: a pii mask rule that matches an argument KEY or a JSON number
//     blocks — neither has a rewrite channel, and forwarding it would
//     defeat the rule; block rules see numbers in arguments and results;
//   - chat: an anchored block rule matching ONE message of a multi-turn
//     request blocks;
//   - chat `tool_calls` and responses `function_call` arguments: the rule
//     that masks an argument value, set to block, blocks on that same value.

const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

const PIN = "^\\d{4}$";

const CREDENTIAL_REF = "mock";
const SLS_PROJECT = "aisix-e2e-obs";
const LOGSTORE = "segment-parity-events";

interface RpcReply {
  status: number;
  json?: {
    result?: {
      content?: Array<{ type: string; text?: string }>;
      structuredContent?: Record<string, unknown>;
      isError?: boolean;
    };
    error?: { code: number; message: string };
  };
}

interface MonitorHit {
  action: string;
  hook: string;
  guardrail_name: string;
  counts?: Record<string, number>;
}

describe("guardrail segment parity e2e: /mcp", () => {
  const KEY = "sk-segment-parity-mcp";
  let app: SpawnedApp | undefined;
  let upstream: McpUpstream | undefined;
  let sls: MockSls | undefined;
  let etcdReachable = false;

  const post = async (body: unknown): Promise<RpcReply> => {
    const res = await fetch(`${app!.proxyUrl}/mcp`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${KEY}`,
        "content-type": "application/json",
        accept: "application/json, text/event-stream",
      },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    let json: RpcReply["json"];
    try {
      json = text ? (JSON.parse(text) as RpcReply["json"]) : undefined;
    } catch {
      json = undefined;
    }
    return { status: res.status, json };
  };

  const callTool = async (
    name: string,
    args: Record<string, unknown>,
  ): Promise<RpcReply> => {
    await post({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: {
        protocolVersion: "2025-11-25",
        capabilities: {},
        clientInfo: { name: "segment-parity-e2e", version: "0.1" },
      },
    });
    return post({
      jsonrpc: "2.0",
      id: 3,
      method: "tools/call",
      params: { name, arguments: args },
    });
  };

  /** The exported usage event of the one tool call routed to `server`. */
  const eventFor = (server: string) =>
    waitForSlsLog(
      sls!,
      LOGSTORE,
      (log) => log.get("mcp_server_name") === server,
      `usage event for mcp server ${server}`,
    );

  const countsOf = (raw: string | undefined): Record<string, number> =>
    raw ? (JSON.parse(raw) as Record<string, number>) : {};

  const wouldMask = (raw: string | undefined, hook: string): Record<string, number> => {
    const hits = raw ? (JSON.parse(raw) as MonitorHit[]) : [];
    const hit = hits.find((h) => h.action === "would_mask" && h.hook === hook);
    return hit?.counts ?? {};
  };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startMcpUpstream("t", { structuredTool: true, numericTool: true });
    sls = await startMockSls();
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "sls-segment-parity",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });

    // One MCP server per guardrail, all pointing at the same upstream, so
    // each rule governs exactly the calls routed to its own server.
    const pinPattern = (action: string) => [{ name: "pin", regex: PIN, action }];
    const guarded: Array<[string, Record<string, unknown>]> = [
      ["kwin", { kind: "keyword", hook_point: "input", patterns: [{ kind: "regex", value: PIN }] }],
      ["kwout", { kind: "keyword", hook_point: "output", patterns: [{ kind: "regex", value: PIN }] }],
      ["piin", { kind: "pii", hook_point: "input", custom_patterns: pinPattern("block") }],
      ["piout", { kind: "pii", hook_point: "output", custom_patterns: pinPattern("block") }],
      [
        "keymask",
        { kind: "pii", hook_point: "input", detectors: [{ type: "email", action: "mask" }] },
      ],
      [
        "kwlitin",
        { kind: "keyword", hook_point: "input", patterns: [{ kind: "literal", value: "1234" }] },
      ],
      [
        "kwlitout",
        { kind: "keyword", hook_point: "output", patterns: [{ kind: "literal", value: "1234" }] },
      ],
      ["numask", { kind: "pii", hook_point: "input", custom_patterns: pinPattern("mask") }],
      ["menfin", { kind: "pii", hook_point: "input", custom_patterns: pinPattern("mask") }],
      [
        "mmonin",
        {
          kind: "pii",
          hook_point: "input",
          enforcement_mode: "monitor",
          custom_patterns: pinPattern("mask"),
        },
      ],
      ["menfout", { kind: "pii", hook_point: "output", custom_patterns: pinPattern("mask") }],
      [
        "mmonout",
        {
          kind: "pii",
          hook_point: "output",
          enforcement_mode: "monitor",
          custom_patterns: pinPattern("mask"),
        },
      ],
    ];
    for (const [server, body] of guarded) {
      const serverId = randomUUID();
      await seed.update("mcp_servers", serverId, {
        display_name: server,
        url: upstream.url,
        enabled: true,
      });
      const g = await seed.createGuardrail(
        { name: `seg-${server}`, enabled: true, ...body },
        { attach: false },
      );
      await seed.update("guardrail_attachments", randomUUID(), {
        guardrail_id: g.id,
        scope_type: "mcp_server",
        scope_id: serverId,
        priority: 100,
      });
    }

    // Caller key LAST: its authenticating implies every row above landed.
    await seed.createApiKey({
      key_hash: sha256(KEY),
      allowed_models: [],
      mcp_access: { allow: ["*"] },
    });
    const proxy = new ProxyClient(app.proxyUrl, KEY);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  }, 90_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await sls?.close();
  });

  for (const kind of ["kw", "pi"]) {
    test(`${kind}: an anchored block rule blocks a bare argument value`, async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();

      const blocked = await callTool(`${kind}in__echo`, { text: "1234" });
      expect(blocked.status).toBe(200);
      expect(blocked.json?.result?.isError).toBe(true);
      expect(blocked.json?.result?.content?.[0]?.text).toContain(`seg-${kind}in`);

      // Anchored to the whole value: the same digits inside a longer value
      // do not match.
      const passed = await callTool(`${kind}in__echo`, { text: "pin 1234" });
      expect(passed.json?.result?.isError).toBeFalsy();
      expect(passed.json?.result?.content?.[0]?.text).toBe("t:pin 1234");
    });

    test(`${kind}: an anchored block rule blocks a bare tool-result value`, async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();

      // `lookup` answers with a constant text block plus the argument as a
      // structuredContent value — two values, only one of them the digits.
      const blocked = await callTool(`${kind}out__lookup`, { text: "1234" });
      expect(blocked.status).toBe(200);
      expect(blocked.json?.result?.isError).toBe(true);
      expect(blocked.json?.result?.structuredContent).toBeUndefined();

      const passed = await callTool(`${kind}out__lookup`, { text: "pin 1234" });
      expect(passed.json?.result?.isError).toBeFalsy();
      expect(passed.json?.result?.structuredContent).toEqual({
        record: { note: "pin 1234" },
      });
    });
  }

  // A JSON number cannot be rewritten without changing its type, so it is
  // judged like a key: block rules see it, and a pii mask hit on it blocks.
  for (const server of ["kwin", "piin", "kwlitin"]) {
    test(`${server}: a block rule blocks a numeric argument`, async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();

      const blocked = await callTool(`${server}__echo`, { text: "hi", pin: 1234 });
      expect(blocked.status).toBe(200);
      expect(blocked.json?.result?.isError).toBe(true);
      expect(blocked.json?.result?.content?.[0]?.text).toContain(`seg-${server}`);

      const passed = await callTool(`${server}__echo`, { text: "hi", pin: 12 });
      expect(passed.json?.result?.isError).toBeFalsy();
    });
  }

  for (const server of ["kwout", "piout", "kwlitout"]) {
    test(`${server}: a block rule blocks a numeric tool-result value`, async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();

      const blocked = await callTool(`${server}__count`, { n: 1234 });
      expect(blocked.status).toBe(200);
      expect(blocked.json?.result?.isError).toBe(true);
      expect(blocked.json?.result?.structuredContent).toBeUndefined();

      const passed = await callTool(`${server}__count`, { n: 12 });
      expect(passed.json?.result?.isError).toBeFalsy();
      expect(passed.json?.result?.structuredContent).toEqual({ count: 12 });
    });
  }

  test("a pii mask rule matching a numeric argument blocks", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    const blocked = await callTool("numask__echo", { text: "hi", pin: 1234 });
    expect(blocked.json?.result?.isError).toBe(true);
    expect(blocked.json?.result?.content?.[0]?.text).toContain("seg-numask");

    // The same value as a string is rewritable, so it is masked instead.
    const masked = await callTool("numask__echo", { text: "1234" });
    expect(masked.json?.result?.isError).toBeFalsy();
    expect(masked.json?.result?.content?.[0]?.text).toBe("t:[PIN_REDACTED]");
  });

  test("a pii mask rule matching an argument key blocks; the same match in a value is masked", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();

    const blocked = await callTool("keymask__echo", {
      text: "approve it",
      "alice@example.com": "approved",
    });
    expect(blocked.status).toBe(200);
    expect(blocked.json?.result?.isError).toBe(true);
    expect(blocked.json?.result?.content?.[0]?.text).toContain("seg-keymask");

    const masked = await callTool("keymask__echo", { text: "mail alice@example.com" });
    expect(masked.json?.result?.isError).toBeFalsy();
    expect(masked.json?.result?.content?.[0]?.text).toBe("t:mail [EMAIL_REDACTED]");
  });

  test("monitor would_mask counts equal the enforced mask counts, both directions", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();

    // Input: two argument values, each exactly four digits.
    const args = { text: "1234", code: "5678" };
    const enfIn = await callTool("menfin__echo", args);
    const monIn = await callTool("mmonin__echo", args);
    expect(enfIn.json?.result?.isError).toBeFalsy();
    expect(monIn.json?.result?.isError).toBeFalsy();

    // Output: the digits come back as one structuredContent value beside a
    // clean text block (the output-only rules leave the arguments alone).
    const enfOut = await callTool("menfout__lookup", { text: "1234" });
    const monOut = await callTool("mmonout__lookup", { text: "1234" });
    expect(enfOut.json?.result?.structuredContent).toEqual({
      record: { note: "[PIN_REDACTED]" },
    });
    // Monitor never rewrites.
    expect(monOut.json?.result?.structuredContent).toEqual({ record: { note: "1234" } });

    const enforcedIn = countsOf((await eventFor("menfin")).get("redacted_entity_counts"));
    const previewIn = wouldMask((await eventFor("mmonin")).get("guardrail_monitor_hits"), "input");
    expect(enforcedIn).toEqual({ pin: 2 });
    expect(previewIn).toEqual(enforcedIn);

    const enforcedOut = countsOf((await eventFor("menfout")).get("redacted_entity_counts"));
    const previewOut = wouldMask(
      (await eventFor("mmonout")).get("guardrail_monitor_hits"),
      "output",
    );
    expect(enforcedOut).toEqual({ pin: 1 });
    expect(previewOut).toEqual(enforcedOut);
  });
});

describe("guardrail segment parity e2e: chat and responses", () => {
  const KEY = "sk-segment-parity-llm";
  let app: SpawnedApp | undefined;
  let chatUpstream: OpenAiUpstream | undefined;
  let toolUpstream: OpenAiUpstream | undefined;
  let respUpstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  const TOOL_CALL = {
    id: "call_1",
    type: "function",
    function: { name: "unlock", arguments: '{"pin":"1234","door":"front"}' },
  };

  const post = (path: string, body: unknown) =>
    fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: { authorization: `Bearer ${KEY}`, "content-type": "application/json" },
      body: JSON.stringify(body),
    });

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    const chatBody = (message: Record<string, unknown>) => ({
      id: "cmpl-seg",
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: "gpt-4o-mini",
      choices: [{ index: 0, message, finish_reason: "stop" }],
      usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
    });
    chatUpstream = await startOpenAiUpstream({
      nonStreamBody: chatBody({ role: "assistant", content: "ok" }),
    });
    toolUpstream = await startOpenAiUpstream({
      nonStreamBody: chatBody({ role: "assistant", content: null, tool_calls: [TOOL_CALL] }),
    });
    respUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "resp_seg",
        object: "response",
        created_at: Math.floor(Date.now() / 1000),
        status: "completed",
        model: "gpt-4o-mini",
        output: [
          {
            id: "msg_seg",
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "ok" }],
          },
        ],
        usage: { input_tokens: 4, output_tokens: 2, total_tokens: 6 },
      },
    });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const models: Array<[string, OpenAiUpstream, Record<string, unknown>]> = [
      ["seg-kw-block", chatUpstream, { kind: "keyword", hook_point: "input", patterns: [{ kind: "regex", value: PIN }] }],
      ["seg-pin-block", chatUpstream, { kind: "pii", hook_point: "input", custom_patterns: [{ name: "pin", regex: PIN, action: "block" }] }],
      ["seg-pin-mask", chatUpstream, { kind: "pii", hook_point: "input", custom_patterns: [{ name: "pin", regex: PIN, action: "mask" }] }],
      ["seg-pin-block-out", toolUpstream, { kind: "pii", hook_point: "output", custom_patterns: [{ name: "pin", regex: PIN, action: "block" }] }],
      ["seg-pin-mask-out", toolUpstream, { kind: "pii", hook_point: "output", custom_patterns: [{ name: "pin", regex: PIN, action: "mask" }] }],
      ["seg-resp-block", respUpstream, { kind: "pii", hook_point: "input", custom_patterns: [{ name: "pin", regex: PIN, action: "block" }] }],
      ["seg-resp-mask", respUpstream, { kind: "pii", hook_point: "input", custom_patterns: [{ name: "pin", regex: PIN, action: "mask" }] }],
    ];
    for (const [name, upstream, guardrail] of models) {
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      const model = await seed.createModel({
        display_name: name,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
      const g = await seed.createGuardrail(
        { name: `${name}-guard`, enabled: true, ...guardrail },
        { attach: false },
      );
      await seed.attachGuardrailToModel(g.id as string, model.id as string);
    }

    await seed.createApiKey({
      key_hash: sha256(KEY),
      allowed_models: models.map(([name]) => name),
    });
    const proxy = new ProxyClient(app.proxyUrl, KEY);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  }, 90_000);

  afterAll(async () => {
    await app?.exit();
    await chatUpstream?.close();
    await toolUpstream?.close();
    await respUpstream?.close();
  });

  const conversation = (last: string) => [
    { role: "user", content: "what should I type?" },
    { role: "assistant", content: "type your pin" },
    { role: "user", content: last },
  ];

  test("chat: an anchored block rule matching one message of a multi-turn request blocks", async (ctx) => {
    if (!etcdReachable || !app || !chatUpstream) return ctx.skip();

    for (const model of ["seg-kw-block", "seg-pin-block"]) {
      const before = chatUpstream.receivedRequests.length;
      const blocked = await post("/v1/chat/completions", {
        model,
        messages: conversation("1234"),
      });
      expect(blocked.status, model).toBe(422);
      expect(chatUpstream.receivedRequests.length).toBe(before);

      const passed = await post("/v1/chat/completions", {
        model,
        messages: conversation("it is 1234"),
      });
      expect(passed.status, model).toBe(200);
    }
  });

  test("chat tool_calls arguments: the value the mask rewrites is the value the block refuses", async (ctx) => {
    if (!etcdReachable || !app || !chatUpstream) return ctx.skip();

    const messages = [
      { role: "user", content: "open the front door" },
      { role: "assistant", content: null, tool_calls: [TOOL_CALL] },
      { role: "tool", tool_call_id: "call_1", content: "done" },
    ];
    const before = chatUpstream.receivedRequests.length;
    const blocked = await post("/v1/chat/completions", { model: "seg-pin-block", messages });
    expect(blocked.status).toBe(422);
    expect(chatUpstream.receivedRequests.length).toBe(before);

    const masked = await post("/v1/chat/completions", { model: "seg-pin-mask", messages });
    expect(masked.status).toBe(200);
    const sent = JSON.parse(chatUpstream.receivedRequests.at(-1)!.body) as {
      messages: Array<{ tool_calls?: Array<{ function: { arguments: string } }> }>;
    };
    const args = JSON.parse(sent.messages[1].tool_calls![0].function.arguments) as Record<
      string,
      string
    >;
    expect(args).toEqual({ pin: "[PIN_REDACTED]", door: "front" });
  });

  test("chat response tool_calls arguments: block and mask agree on the output hook", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    const req = { messages: [{ role: "user", content: "open the front door" }] };
    const blocked = await post("/v1/chat/completions", { model: "seg-pin-block-out", ...req });
    expect(blocked.status).toBe(422);

    const masked = await post("/v1/chat/completions", { model: "seg-pin-mask-out", ...req });
    expect(masked.status).toBe(200);
    const body = (await masked.json()) as {
      choices: Array<{ message: { tool_calls: Array<{ function: { arguments: string } }> } }>;
    };
    const args = JSON.parse(body.choices[0].message.tool_calls[0].function.arguments) as Record<
      string,
      string
    >;
    expect(args).toEqual({ pin: "[PIN_REDACTED]", door: "front" });
  });

  test("responses function_call arguments: the value the mask rewrites is the value the block refuses", async (ctx) => {
    if (!etcdReachable || !app || !respUpstream) return ctx.skip();

    const input = [
      { role: "user", content: "open the front door" },
      {
        type: "function_call",
        call_id: "call_1",
        name: "unlock",
        arguments: '{"pin":"1234","door":"front"}',
      },
      { type: "function_call_output", call_id: "call_1", output: "done" },
    ];
    const before = respUpstream.receivedRequests.length;
    const blocked = await post("/v1/responses", { model: "seg-resp-block", input });
    expect(blocked.status).toBe(422);
    expect(respUpstream.receivedRequests.length).toBe(before);

    const masked = await post("/v1/responses", { model: "seg-resp-mask", input });
    expect(masked.status).toBe(200);
    const sent = JSON.parse(respUpstream.receivedRequests.at(-1)!.body) as {
      input: Array<{ type?: string; arguments?: string }>;
    };
    const call = sent.input.find((i) => i.type === "function_call")!;
    expect(JSON.parse(call.arguments!)).toEqual({ pin: "[PIN_REDACTED]", door: "front" });
  });
});
