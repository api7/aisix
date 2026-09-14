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

// E2E for the REPLY direction of the `/v1/responses` → chat-completions
// bridge, plus the buffered `/v1/chat/completions` usage block.
//
// Two contracts.
//
// 1. A caller that registered a `custom` (freeform) tool gets the model's
//    call back as a `custom_tool_call` item carrying the freeform `input`.
//    The request side reaches a chat upstream by offering that tool as a
//    function taking one string, so the reply arrives as an ordinary chat
//    tool call — and used to be handed back as a `function_call` item whose
//    `arguments` were the gateway's own single-string wrapper. A client
//    written against the Responses API then never matched the call to the
//    tool it had registered. Streaming carries the same item, with the
//    `response.custom_tool_call_input.*` events rather than the
//    `response.function_call_arguments.*` ones.
//
// 2. On a buffered `/v1/chat/completions` whose upstream reports no `usage`,
//    the client-visible usage block is the same local estimate the usage
//    record gets. It used to be left at the upstream's zeros while the
//    dashboard billed the estimate, so a caller had no way to reconcile the
//    two.
//
// The chat bridge is reached by declaring an EMPTY `apis` map on the
// provider key — the operator saying "this OpenAI-compatible endpoint has
// no `/v1/responses`".

const CALLER_PLAINTEXT = "sk-responses-bridge-custom-tool-reply";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const HEADERS = {
  authorization: `Bearer ${CALLER_PLAINTEXT}`,
  "content-type": "application/json",
};

const CREDENTIAL_REF = "e2e";
const MOCK_AK_ID = "mock-ak-id";
const MOCK_AK_SECRET = "mock-ak-secret";
const SLS_PROJECT = "custom-tool-reply-proj";
const LOGSTORE = "custom-tool-reply-store";

const PATCH_INPUT = '*** Begin Patch\n@@ a.txt\n-old\n+new\n*** End Patch';

/** The chat tool call an upstream makes against the translated tool. */
const TOOL_CALL = {
  id: "call_ct1",
  type: "function",
  function: {
    name: "apply_patch",
    arguments: JSON.stringify({ content: PATCH_INPUT }),
  },
};

const CHAT_TOOL_CALL_REPLY = {
  id: "chatcmpl-custom-tool",
  object: "chat.completion",
  created: 1_700_000_000,
  model: "relay-mini",
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: null, tool_calls: [TOOL_CALL] },
      finish_reason: "tool_calls",
    },
  ],
  usage: { prompt_tokens: 12, completion_tokens: 8, total_tokens: 20 },
};

/** The same call, streamed: id + name first, arguments in two fragments. */
const CHAT_TOOL_CALL_STREAM_EVENTS = [
  JSON.stringify({
    id: "chatcmpl-custom-tool-stream",
    object: "chat.completion.chunk",
    model: "relay-mini",
    choices: [
      {
        index: 0,
        delta: {
          role: "assistant",
          tool_calls: [
            {
              index: 0,
              id: "call_ct1",
              type: "function",
              function: { name: "apply_patch", arguments: "" },
            },
          ],
        },
        finish_reason: null,
      },
    ],
  }),
  JSON.stringify({
    id: "chatcmpl-custom-tool-stream",
    object: "chat.completion.chunk",
    model: "relay-mini",
    choices: [
      {
        index: 0,
        delta: {
          tool_calls: [
            {
              index: 0,
              function: {
                arguments: JSON.stringify({ content: PATCH_INPUT }).slice(0, 20),
              },
            },
          ],
        },
        finish_reason: null,
      },
    ],
  }),
  JSON.stringify({
    id: "chatcmpl-custom-tool-stream",
    object: "chat.completion.chunk",
    model: "relay-mini",
    choices: [
      {
        index: 0,
        delta: {
          tool_calls: [
            {
              index: 0,
              function: {
                arguments: JSON.stringify({ content: PATCH_INPUT }).slice(20),
              },
            },
          ],
        },
        finish_reason: null,
      },
    ],
  }),
  JSON.stringify({
    id: "chatcmpl-custom-tool-stream",
    object: "chat.completion.chunk",
    model: "relay-mini",
    choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }],
    usage: { prompt_tokens: 12, completion_tokens: 8, total_tokens: 20 },
  }),
  "[DONE]",
];

/** A buffered chat 200 with no `usage` block at all. */
const CHAT_REPLY_NO_USAGE = {
  id: "chatcmpl-no-usage",
  object: "chat.completion",
  created: 1_700_000_000,
  model: "relay-mini",
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: "Hello world" },
      finish_reason: "stop",
    },
  ],
};

const CUSTOM_TOOL = {
  type: "custom",
  name: "apply_patch",
  description: "Edit a file",
};

interface SseFrame {
  event: string;
  data: Record<string, any>;
}

/** Parse a Responses-API SSE body into its `event:` / `data:` frames. */
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

describe("bridged custom tool calls come back as custom_tool_call items", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let toolUpstream: OpenAiUpstream | undefined;
  let toolStreamUpstream: OpenAiUpstream | undefined;
  let noUsageUpstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    toolUpstream = await startOpenAiUpstream({
      nonStreamBody: CHAT_TOOL_CALL_REPLY,
    });
    toolStreamUpstream = await startOpenAiUpstream({
      streamEvents: CHAT_TOOL_CALL_STREAM_EVENTS,
    });
    noUsageUpstream = await startOpenAiUpstream({
      nonStreamBody: CHAT_REPLY_NO_USAGE,
    });

    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: MOCK_AK_ID,
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: MOCK_AK_SECRET,
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "custom-tool-reply-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
    });

    // `apis: {}` — no `/v1/responses` on this endpoint, so `/v1/responses`
    // traffic takes the chat bridge.
    for (const [model, upstream] of [
      ["custom-tool-chat", toolUpstream],
      ["custom-tool-chat-stream", toolStreamUpstream],
    ] as const) {
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

    const noUsagePk = await seed.createProviderKey({
      display_name: "custom-tool-no-usage-pk",
      secret: "sk-mock",
      api_base: `${noUsageUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "custom-tool-no-usage",
      provider: "openai",
      model_name: "relay-compat-x",
      provider_key_id: noUsagePk.id,
    });

    // Seeded last: the key authenticating implies the whole seed set is
    // in the snapshot.
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [
        "custom-tool-chat",
        "custom-tool-chat-stream",
        "custom-tool-no-usage",
      ],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await toolUpstream?.close();
    await toolStreamUpstream?.close();
    await noUsageUpstream?.close();
  });

  async function ready(): Promise<void> {
    const probe = new ProxyClient(app!.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const r = await probe.listModels();
      return r.status === 200;
    });
  }

  test("/v1/responses non-streaming: the reply is a custom_tool_call item carrying the freeform input", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    await ready();

    const res = await fetch(`${app.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({
        model: "custom-tool-chat",
        input: "patch the file",
        max_output_tokens: 64,
        tools: [CUSTOM_TOOL],
        tool_choice: { type: "custom", name: "apply_patch" },
      }),
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as Record<string, any>;

    // Pre-fix this was a `function_call` item whose `arguments` were the
    // gateway's own `{"content":…}` wrapper.
    expect(body.output).toHaveLength(1);
    const item = body.output[0];
    expect(item.type).toBe("custom_tool_call");
    expect(item.call_id).toBe("call_ct1");
    expect(item.name).toBe("apply_patch");
    expect(item.input).toBe(PATCH_INPUT);
    expect(item.status).toBe("completed");
    expect(item.id).toMatch(/^ctc_/);
    expect(item).not.toHaveProperty("arguments");
  });

  test("/v1/responses streaming: the custom tool streams its own input events, never the function-call ones", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    await ready();

    const res = await fetch(`${app.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({
        model: "custom-tool-chat-stream",
        input: "patch the file",
        max_output_tokens: 64,
        stream: true,
        tools: [CUSTOM_TOOL],
        tool_choice: { type: "custom", name: "apply_patch" },
      }),
    });
    expect(res.status).toBe(200);
    const frames = parseSse(await res.text());
    const types = frames.map((f) => f.data.type);

    expect(types).not.toContain("response.function_call_arguments.delta");
    expect(types).not.toContain("response.function_call_arguments.done");

    const added = frames.find(
      (f) =>
        f.data.type === "response.output_item.added" &&
        f.data.item?.type === "custom_tool_call",
    );
    expect(added, `no custom_tool_call output_item.added in ${types}`).toBeDefined();
    expect(added!.data.item.name).toBe("apply_patch");
    expect(added!.data.item.call_id).toBe("call_ct1");
    expect(added!.data.item.input).toBe("");
    const itemId = added!.data.item.id as string;
    expect(itemId).toMatch(/^ctc_/);

    // Exactly one delta, carrying the unwrapped input — the two upstream
    // argument fragments are the wrapper's JSON and are buffered, not
    // relayed.
    const deltas = frames.filter(
      (f) => f.data.type === "response.custom_tool_call_input.delta",
    );
    expect(deltas).toHaveLength(1);
    expect(deltas[0]!.data.delta).toBe(PATCH_INPUT);
    expect(deltas[0]!.data.item_id).toBe(itemId);

    const done = frames.filter(
      (f) => f.data.type === "response.custom_tool_call_input.done",
    );
    expect(done).toHaveLength(1);
    expect(done[0]!.data.input).toBe(PATCH_INPUT);

    const itemDone = frames.find(
      (f) =>
        f.data.type === "response.output_item.done" &&
        f.data.item?.type === "custom_tool_call",
    );
    expect(itemDone!.data.item.input).toBe(PATCH_INPUT);
    expect(itemDone!.data.item.status).toBe("completed");

    const completed = frames.at(-1)!;
    expect(completed.data.type).toBe("response.completed");
    expect(completed.data.response.output[0].type).toBe("custom_tool_call");
    expect(completed.data.response.output[0].input).toBe(PATCH_INPUT);

    // The event order for the one item, and unbroken sequence numbers.
    const own = types.filter((t: string) =>
      [
        "response.output_item.added",
        "response.custom_tool_call_input.delta",
        "response.custom_tool_call_input.done",
        "response.output_item.done",
      ].includes(t),
    );
    expect(own).toEqual([
      "response.output_item.added",
      "response.custom_tool_call_input.delta",
      "response.custom_tool_call_input.done",
      "response.output_item.done",
    ]);
    expect(frames.map((f) => f.data.sequence_number)).toEqual(
      frames.map((_, i) => i),
    );
  });

  test("/v1/chat/completions non-streaming: the client usage block equals the usage record", async (ctx) => {
    if (!etcdReachable || !app || !sls) {
      ctx.skip();
      return;
    }
    await ready();

    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({
        model: "custom-tool-no-usage",
        messages: [{ role: "user", content: "hi" }],
      }),
    });
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    expect(requestId).not.toBe("");
    const body = (await res.json()) as Record<string, any>;
    expect(body.choices[0].message.content).toBe("Hello world");

    const log = await waitForSlsLog(
      sls,
      LOGSTORE,
      (l) => l.get("request_id") === requestId,
      `usage row for ${requestId}`,
      15_000,
    );
    // The record was estimated locally — the upstream reported nothing.
    expect(log.get("usage_estimated")).toBe("true");

    // Pre-fix the client read 0/0 here while the record carried the
    // estimate below.
    expect(body.usage.prompt_tokens).toBe(Number(log.get("prompt_tokens")));
    expect(body.usage.completion_tokens).toBe(
      Number(log.get("completion_tokens")),
    );
    expect(body.usage.prompt_tokens).toBeGreaterThan(0);
    expect(body.usage.completion_tokens).toBeGreaterThan(0);
    expect(body.usage.total_tokens).toBe(
      body.usage.prompt_tokens + body.usage.completion_tokens,
    );
  });
});
