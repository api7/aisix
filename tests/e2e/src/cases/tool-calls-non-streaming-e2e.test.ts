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

// E2E: tool_calls round-trip on the non-streaming chat path.
//
// The OpenAI Chat Completions API supports two complementary
// halves of function calling:
//   - request side: caller sends `tools: [{type:"function", ...}]`
//     plus optional `tool_choice`
//   - response side: assistant returns
//     `message.tool_calls: [{id, type, function: {name, arguments}}]`
//     and `finish_reason: "tool_calls"`
//
// This file pins the non-streaming round-trip end-to-end:
//
//   1. Caller's `tools` and `tool_choice` arrive at upstream
//      verbatim — same array, same order, same JSON Schema.
//   2. Upstream's `tool_calls` and `finish_reason` reach the caller
//      verbatim — id stable, function.arguments byte-equal,
//      finish_reason=="tool_calls".
//
// The streaming sibling of this contract is held back at PR #186 /
// issue #202 — `ChatChunk` does not model `tool_calls`, so the
// streaming path silently drops them. The non-streaming path uses
// a different code path that does NOT round-trip through
// `ChatChunk`, so it should pass cleanly today.
//
// Reference:
//   - OpenAI Chat Completions API
//     <https://platform.openai.com/docs/api-reference/chat/create>
//     (request `tools`, response `message.tool_calls`)
//   - OpenAI Node SDK ChatCompletionTool / ChatCompletionMessageToolCall

const CALLER_PLAINTEXT = "sk-tool-nonstream-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const TOOL_CALL_ID = "call_nonstream_round_trip_1";
const TOOL_NAME = "get_weather";
// Canonical arguments string the upstream emits and the caller
// must observe byte-equal. Use a non-trivial nested JSON so a
// regression that re-serialized (whitespace / key order) would
// surface immediately.
const CANONICAL_ARGS_STRING =
  '{"location":"Beijing","unit":"celsius"}';
const CANONICAL_ARGS_PARSED = {
  location: "Beijing",
  unit: "celsius",
};

const TOOL_DEFINITION = {
  type: "function" as const,
  function: {
    name: TOOL_NAME,
    description: "Get the current weather for a location",
    parameters: {
      type: "object",
      properties: {
        location: { type: "string" },
        unit: { type: "string", enum: ["celsius", "fahrenheit"] },
      },
      required: ["location"],
    },
  },
};

// Second tool — exercised on the request-side wire-shape gate so a
// regression that kept only `tools[0]` (e.g. `tools.slice(0, 1)`)
// is caught. The model isn't expected to invoke this one in the
// canned response below, but it MUST reach upstream so the model
// sees the full tool catalogue.
const SECOND_TOOL_DEFINITION = {
  type: "function" as const,
  function: {
    name: "get_time",
    description: "Get the current time in a timezone",
    parameters: {
      type: "object",
      properties: {
        timezone: { type: "string" },
      },
      required: ["timezone"],
    },
  },
};

const NON_STREAM_BODY = {
  id: "chatcmpl-tool-nonstream-1",
  object: "chat.completion",
  created: Math.floor(Date.now() / 1000),
  model: "gpt-4o-mini",
  choices: [
    {
      index: 0,
      message: {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: TOOL_CALL_ID,
            type: "function",
            function: {
              name: TOOL_NAME,
              arguments: CANONICAL_ARGS_STRING,
            },
          },
        ],
      },
      finish_reason: "tool_calls",
    },
  ],
  usage: {
    prompt_tokens: 7,
    completion_tokens: 9,
    total_tokens: 16,
  },
};

describe("tool_calls non-streaming e2e: round-trip via OpenAI SDK", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      nonStreamBody: NON_STREAM_BODY,
    });
    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "tool-nonstream-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "tool-nonstream-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["tool-nonstream-model"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("tools array round-trips both directions verbatim", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "tool-nonstream-model",
          messages: [{ role: "user", content: "ready-probe" }],
        });
        return probe.choices.length > 0;
      } catch {
        return false;
      }
    });

    const baseline = upstream.receivedRequests.length;

    const resp = await client.chat.completions.create({
      model: "tool-nonstream-model",
      messages: [
        {
          role: "user",
          content: "What's the weather in Beijing?",
        },
      ],
      // Two tools so a regression that kept only the first
      // (e.g. tools.slice(0, 1) or "first-tool wins" dispatch)
      // surfaces against the per-position equality below.
      tools: [TOOL_DEFINITION, SECOND_TOOL_DEFINITION],
      tool_choice: "auto",
    });

    // ── Response side ────────────────────────────────────────────
    // (1) finish_reason == "tool_calls" reaches the caller.
    expect(resp.choices[0]?.finish_reason).toBe("tool_calls");

    // (2) message.tool_calls is the single canonical array the
    // upstream emitted. Catches a regression that swallowed the
    // tool_calls field (the same family of bug that #202 covers
    // for the streaming path — this test confirms the bug is
    // streaming-specific and the non-streaming path is clean).
    const toolCalls = resp.choices[0]?.message?.tool_calls;
    expect(toolCalls).toBeDefined();
    expect(toolCalls).toHaveLength(1);
    const tc = toolCalls?.[0];
    expect(tc?.id).toBe(TOOL_CALL_ID);
    expect(tc?.type).toBe("function");
    expect(tc?.function?.name).toBe(TOOL_NAME);

    // (3) function.arguments arrives byte-equal AND parses to the
    // canonical object. A regression that re-serialized (key
    // reorder, whitespace insertion) would fail the byte-equal
    // gate; one that mutated values would fail the parse-equal
    // gate.
    expect(tc?.function?.arguments).toBe(CANONICAL_ARGS_STRING);
    expect(JSON.parse(tc?.function?.arguments ?? "")).toEqual(
      CANONICAL_ARGS_PARSED,
    );

    // ── Request side (wire-shape blind-spot guard) ───────────────
    // The mock replays its canned response regardless of request
    // body, so the response-side gates above could pass even if
    // the gateway stripped the `tools` array from the upstream
    // request. Pin the upstream-side wire shape too. Closes the
    // same blind spot CLAUDE.md §8 calls out.
    const sent = upstream.receivedRequests
      .slice(baseline)
      .filter((r) => r.path === "/v1/chat/completions");
    expect(sent).toHaveLength(1);
    const sentReq = sent[0]!;
    expect(sentReq.method).toBe("POST");
    expect(sentReq.headers.authorization).toBe("Bearer sk-mock");
    const sentBody = JSON.parse(sentReq.body);
    expect(sentBody.model).toBe("gpt-4o-mini");
    // (4) tools array preserved in full — type, function name +
    // description, AND the JSON Schema parameters block (which
    // upstream needs verbatim to dispatch the right tool).
    // toEqual is structural; that is the right contract on the
    // request side because upstreams consume `tools` as a JSON
    // tree, not a byte stream. The function.arguments byte-equal
    // gate above is where strict serialization matters (downstream
    // tool handlers DO consume the literal string).
    expect(sentBody.tools).toHaveLength(2);
    expect(sentBody.tools).toEqual([
      TOOL_DEFINITION,
      SECOND_TOOL_DEFINITION,
    ]);
    // (5) tool_choice preserved. A regression that defaulted it
    // to a different value (e.g. "none" by accident) would change
    // upstream behavior even with the tools array intact.
    expect(sentBody.tool_choice).toBe("auto");
  }, 60_000);
});
