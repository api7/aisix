import { createHash } from "node:crypto";
import OpenAI from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: OpenAI client → OpenAI upstream — `tool_calls` round-trip on
// both the non-streaming and streaming paths.
//
// Pins the two-issue contract pair the gateway-internal wire types
// previously dropped (api7/ai-gateway#220 + #202):
//
//   1. Non-streaming: when the upstream returns
//      `choices[].message.tool_calls`, the caller MUST observe the
//      same array on its response (id, function.name,
//      function.arguments preserved verbatim) and
//      `choices[].finish_reason === "tool_calls"`.
//   2. Streaming: when the upstream emits SSE chunks where
//      `choices[].delta.tool_calls[i].function.arguments` arrives
//      across multiple chunks (the standard OpenAI fragmenting
//      convention), the caller MUST be able to concatenate the
//      arguments fragments into the original JSON without loss.
//
// Reference contracts:
// - OpenAI tools spec:
//   <https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools>
// - OpenAI streaming choices[].delta.tool_calls shape:
//   <https://platform.openai.com/docs/api-reference/chat-streaming/streaming>
//
// Source-blind discipline: this spec asserts on the SDK-observed
// shape only. A regression in the gateway's deserializer or its
// response-projection layer would surface as `tool_calls ===
// undefined` on the SDK side.

const CALLER_PLAINTEXT = "sk-openai-tools-rt-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const TOOLS_PARAM = [
  {
    type: "function" as const,
    function: {
      name: "get_weather",
      description: "Get the current weather in a given location.",
      parameters: {
        type: "object",
        properties: {
          location: {
            type: "string",
            description: "The city and state, e.g. Beijing",
          },
          unit: {
            type: "string",
            enum: ["celsius", "fahrenheit"],
          },
        },
        required: ["location"],
      },
    },
  },
];

// Helper: a vanilla content response used for the readiness probe.
// The probe sends NO tools so we don't accidentally consume a
// scripted tool_calls response on a non-test call.
const READY_PROBE_BODY = {
  id: "chatcmpl-ready-probe",
  object: "chat.completion",
  created: 0,
  model: "gpt-4o-mini",
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: "ready" },
      finish_reason: "stop",
    },
  ],
  usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
};

// Tool_calls non-streaming response — issue #220's customer contract.
const TOOL_CALLS_NONSTREAM_BODY = {
  id: "chatcmpl-tool-call-01",
  object: "chat.completion",
  created: 1_700_000_000,
  model: "gpt-4o-mini",
  choices: [
    {
      index: 0,
      message: {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: "call_abc123",
            type: "function",
            function: {
              name: "get_weather",
              arguments: '{"location":"Beijing","unit":"celsius"}',
            },
          },
        ],
      },
      finish_reason: "tool_calls",
    },
  ],
  usage: { prompt_tokens: 12, completion_tokens: 9, total_tokens: 21 },
};

// Streaming tool_calls — issue #202's contract. OpenAI's documented
// fragmenting: chunk 1 carries id/type/function.name + first
// arguments fragment; subsequent chunks carry arguments fragments
// only (same index, same id, no name re-sent). Terminal chunk
// carries the finish_reason.
const TOOL_CALLS_STREAM_EVENTS = [
  // Chunk 1 — role + first tool_calls element with id/type/name +
  // start of arguments.
  JSON.stringify({
    id: "chatcmpl-tool-stream-01",
    object: "chat.completion.chunk",
    created: 1_700_000_001,
    model: "gpt-4o-mini",
    choices: [
      {
        index: 0,
        delta: {
          role: "assistant",
          tool_calls: [
            {
              index: 0,
              id: "call_str789",
              type: "function",
              function: {
                name: "get_weather",
                arguments: '{"loc',
              },
            },
          ],
        },
        finish_reason: null,
      },
    ],
  }),
  // Chunk 2 — arguments fragment continues. Per OpenAI's convention,
  // only `index` + `function.arguments` are present.
  JSON.stringify({
    id: "chatcmpl-tool-stream-01",
    object: "chat.completion.chunk",
    created: 1_700_000_001,
    model: "gpt-4o-mini",
    choices: [
      {
        index: 0,
        delta: {
          tool_calls: [
            { index: 0, function: { arguments: 'ation":"Beij' } },
          ],
        },
        finish_reason: null,
      },
    ],
  }),
  // Chunk 3 — closing arguments fragment.
  JSON.stringify({
    id: "chatcmpl-tool-stream-01",
    object: "chat.completion.chunk",
    created: 1_700_000_001,
    model: "gpt-4o-mini",
    choices: [
      {
        index: 0,
        delta: {
          tool_calls: [
            { index: 0, function: { arguments: 'ing","unit":"celsius"}' } },
          ],
        },
        finish_reason: null,
      },
    ],
  }),
  // Terminal chunk — finish_reason = "tool_calls" + empty delta.
  JSON.stringify({
    id: "chatcmpl-tool-stream-01",
    object: "chat.completion.chunk",
    created: 1_700_000_001,
    model: "gpt-4o-mini",
    choices: [
      { index: 0, delta: {}, finish_reason: "tool_calls" },
    ],
  }),
  "[DONE]",
];

describe("OpenAI tools round-trip (#220 + #202)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    // Scripted upstream — each test gets its own scripted step. The
    // readiness probe in the first test consumes step 0 (vanilla
    // content). The non-stream test consumes step 1 (tool_calls).
    // The stream test consumes step 2 (multi-chunk SSE). Order
    // matters: the readiness probe always runs first.
    upstream = await startOpenAiUpstream({
      scriptedResponses: [
        { nonStreamBody: READY_PROBE_BODY },
        { nonStreamBody: TOOL_CALLS_NONSTREAM_BODY },
        { streamEvents: TOOL_CALLS_STREAM_EVENTS },
      ],
      // Static fallback for any over-the-script calls (e.g. retry,
      // extra readiness probe). Same vanilla shape so the test
      // doesn't accidentally consume tool_calls bytes on the
      // wrong call.
      nonStreamBody: READY_PROBE_BODY,
    });
    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "openai-tools-rt-pk",
      secret: "sk-openai-mock",
      api_base: upstream.baseUrl,
    });
    await admin.createModel({
      display_name: "openai-tools-rt",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["openai-tools-rt"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("non-streaming: message.tool_calls survives upstream → gateway → caller (#220)", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // Probe consumes scripted step 0.
    await waitConfigPropagation(async () => {
      try {
        const r = await client.chat.completions.create({
          model: "openai-tools-rt",
          messages: [{ role: "user", content: "ready-probe" }],
        });
        return r.choices[0]?.message.role === "assistant";
      } catch {
        return false;
      }
    });

    // Tool call consumes scripted step 1.
    const completion = await client.chat.completions.create({
      model: "openai-tools-rt",
      messages: [{ role: "user", content: "What is the weather in Beijing?" }],
      tools: TOOLS_PARAM,
      tool_choice: "auto",
    });

    // Caller-side contract: assistant message carries tool_calls,
    // not content text; finish_reason is "tool_calls".
    expect(
      completion.choices[0]?.message.tool_calls,
      "tool_calls dropped end-to-end — #220 regression",
    ).toBeDefined();
    expect(completion.choices[0]?.message.tool_calls).toHaveLength(1);

    const toolCall = completion.choices[0]?.message.tool_calls?.[0];
    expect(toolCall?.id).toBe("call_abc123");
    expect(toolCall?.type).toBe("function");
    expect(toolCall?.function.name).toBe("get_weather");
    // Arguments string round-trip — gateway must NOT re-parse or
    // re-stringify the JSON; OpenAI's contract is verbatim
    // arguments preservation so the agent loop's JSON.parse on
    // the SDK side sees the same bytes the upstream emitted.
    expect(toolCall?.function.arguments).toBe(
      '{"location":"Beijing","unit":"celsius"}',
    );

    expect(completion.choices[0]?.finish_reason).toBe("tool_calls");
    // OpenAI's `message.content` is documented as `string | null`
    // (<https://platform.openai.com/docs/api-reference/chat/object>).
    // On a tool_calls response the upstream returns `content: null` to
    // signal "the assistant chose to call a tool, no text reply"; the
    // gateway must surface exactly `null`, not `""` (#395).
    expect(completion.choices[0]?.message.content).toBeNull();
  });

  test("streaming: delta.tool_calls fragments survive upstream → gateway → caller (#202)", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // Stream consumes scripted step 2. We iterate every chunk and
    // collect both the per-chunk deltas and a reconstructed
    // arguments string. A regression that drops tool_calls would
    // show up as `delta.tool_calls === undefined` on every chunk.
    const stream = await client.chat.completions.create({
      model: "openai-tools-rt",
      messages: [{ role: "user", content: "What is the weather in Beijing?" }],
      tools: TOOLS_PARAM,
      tool_choice: "auto",
      stream: true,
    });

    interface ToolCallFragment {
      index?: number;
      id?: string;
      type?: string;
      function?: {
        name?: string;
        arguments?: string;
      };
    }
    let argsBuf = "";
    let firstChunkWithToolCalls: ToolCallFragment | null = null;
    let finishReason: string | null = null;
    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta;
      const fragments = (delta as { tool_calls?: ToolCallFragment[] } | undefined)
        ?.tool_calls;
      if (fragments && fragments.length > 0) {
        if (firstChunkWithToolCalls === null) {
          firstChunkWithToolCalls = fragments[0] ?? null;
        }
        for (const f of fragments) {
          if (f.function?.arguments) {
            argsBuf += f.function.arguments;
          }
        }
      }
      if (chunk.choices[0]?.finish_reason) {
        finishReason = chunk.choices[0].finish_reason;
      }
    }

    expect(
      firstChunkWithToolCalls,
      "delta.tool_calls dropped on every streaming chunk — #202 regression",
    ).not.toBeNull();
    // The first tool_calls chunk MUST carry id + type + function.name
    // so the SDK can attribute subsequent argument fragments to the
    // same call. Without these the SDK can't dispatch.
    expect(firstChunkWithToolCalls?.id).toBe("call_str789");
    expect(firstChunkWithToolCalls?.type).toBe("function");
    expect(firstChunkWithToolCalls?.function?.name).toBe("get_weather");

    // Reconstructed arguments JSON must parse + match upstream's
    // intent verbatim. Concatenation order is preserved by the SDK
    // because chunks are delivered in-order over SSE.
    expect(argsBuf).toBe('{"location":"Beijing","unit":"celsius"}');
    const parsed = JSON.parse(argsBuf) as { location: string; unit: string };
    expect(parsed.location).toBe("Beijing");
    expect(parsed.unit).toBe("celsius");

    expect(finishReason).toBe("tool_calls");
  });
});
