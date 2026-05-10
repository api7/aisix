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

// E2E: streaming + tool_calls delta assembly. Per OpenAI's chat
// completions wire shape
// (https://platform.openai.com/docs/api-reference/chat/streaming),
// when the model invokes a tool the upstream streams the
// `tool_calls[*].function.arguments` field across multiple SSE
// chunks — the first chunk carries the tool_call `id`, `type`,
// and `function.name` plus an arguments fragment; subsequent chunks
// carry only an `arguments` fragment under the same `index`. The
// caller is expected to concatenate the fragments back into a
// valid JSON document.
//
// One contract pinned here:
//
//   - The gateway preserves the OpenAI tool_calls delta wire shape
//     end-to-end. Concretely, the chunks the caller observes can be
//     assembled into:
//       * a single tool_call with stable `id` and `function.name`
//         (carried only on the first delta);
//       * an `arguments` string equal to the byte-concatenation of
//         every per-chunk `arguments` fragment;
//       * a final chunk whose `finish_reason === "tool_calls"`.
//
// A regression that lost or reordered fragments would either fail
// JSON parsing or surface as an arguments string different from
// the canonical concatenation. A regression that swallowed the
// trailing `finish_reason` chunk would fail the last-chunk assertion.
//
// Reference: OpenAI streaming spec linked above + the OpenAI Node
// SDK's `chat.completions.create({stream: true})` chunk type.

const CALLER_PLAINTEXT = "sk-stream-tools-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

// Canonical tool_call identity carried only on the first delta.
const TOOL_CALL_ID = "call_stream_tool_e2e_1";
const TOOL_NAME = "get_weather";

// Arguments fragments — concatenation must equal CANONICAL_ARGS.
// Split mid-key, mid-value, and mid-quote to exercise non-aligned
// boundaries the assembler cannot trivially short-circuit on.
const ARGS_FRAGMENTS = ['{"', "loc", 'ation":"Bei', 'jing"}'];
const CANONICAL_ARGS = ARGS_FRAGMENTS.join("");
const CANONICAL_PARSED = { location: "Beijing" };

const SSE_EVENTS = [
  // Role delta + first tool_call fragment (carries id, type, name,
  // first slice of arguments).
  `{"id":"chatcmpl-stream-1","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"${TOOL_CALL_ID}","type":"function","function":{"name":"${TOOL_NAME}","arguments":"${ARGS_FRAGMENTS[0]}"}}]},"finish_reason":null}]}`,
  // Subsequent fragments carry only arguments under index 0.
  `{"id":"chatcmpl-stream-1","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"${ARGS_FRAGMENTS[1]}"}}]},"finish_reason":null}]}`,
  `{"id":"chatcmpl-stream-1","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"${ARGS_FRAGMENTS[2]}"}}]},"finish_reason":null}]}`,
  `{"id":"chatcmpl-stream-1","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"${ARGS_FRAGMENTS[3]}"}}]},"finish_reason":null}]}`,
  // Terminal chunk: empty delta, finish_reason = tool_calls.
  '{"id":"chatcmpl-stream-1","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
  "[DONE]",
];

describe("streaming tool_calls e2e: arguments fragments concatenate to a valid JSON tool call", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({ streamEvents: SSE_EVENTS });

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "stream-tools-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "stream-tools-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["stream-tools-model"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("OpenAI SDK stream — fragments concatenate, id stable, finish on last chunk", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // Snapshot propagation — drive the streaming path itself so the
    // probe asserts on the same code path the test uses. A non-
    // streaming probe could pass while the streaming dispatcher
    // is still missing the Model.
    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "stream-tools-model",
          messages: [{ role: "user", content: "ready-probe" }],
          stream: true,
        });
        for await (const _chunk of probe) {
          break;
        }
        return true;
      } catch {
        return false;
      }
    });

    const stream = await client.chat.completions.create({
      model: "stream-tools-model",
      messages: [
        { role: "user", content: "What's the weather in Beijing?" },
      ],
      tools: [
        {
          type: "function",
          function: {
            name: TOOL_NAME,
            description: "Get the current weather for a location",
            parameters: {
              type: "object",
              properties: {
                location: { type: "string" },
              },
              required: ["location"],
            },
          },
        },
      ],
      stream: true,
    });

    // Capture every per-chunk view of choices[0]. Capturing in arrival
    // order, not summed, so a regression that reordered chunks (e.g.
    // emitted finish_reason mid-stream) is visible.
    type CapturedChunk = {
      idDelta: string | undefined;
      nameDelta: string | undefined;
      argsDelta: string | undefined;
      finish: string | null;
    };
    const captured: CapturedChunk[] = [];
    for await (const chunk of stream) {
      const choice = chunk.choices[0];
      const tc = choice?.delta?.tool_calls?.[0];
      captured.push({
        idDelta: tc?.id,
        nameDelta: tc?.function?.name,
        argsDelta: tc?.function?.arguments,
        finish: choice?.finish_reason ?? null,
      });
    }

    // (1) The first chunk that carries any tool_calls delta must
    // also carry the stable id and function name. Subsequent chunks
    // must NOT redeclare them — they only carry argument fragments
    // under the same index. A regression that re-emitted id/name on
    // every chunk would still produce a valid concatenated args but
    // would break consumers that rely on "id is set exactly once".
    const firstWithTool = captured.find(
      (c) =>
        c.idDelta !== undefined ||
        c.nameDelta !== undefined ||
        c.argsDelta !== undefined,
    );
    expect(firstWithTool).toBeDefined();
    expect(firstWithTool?.idDelta).toBe(TOOL_CALL_ID);
    expect(firstWithTool?.nameDelta).toBe(TOOL_NAME);

    const subsequentWithTool = captured
      .slice(captured.indexOf(firstWithTool!) + 1)
      .filter((c) => c.argsDelta !== undefined);
    for (const c of subsequentWithTool) {
      expect(c.idDelta).toBeUndefined();
      expect(c.nameDelta).toBeUndefined();
    }

    // (2) Concatenating every per-chunk arguments fragment must equal
    // the upstream's byte-by-byte concatenation, AND must parse to
    // the canonical object. A regression that dropped a fragment
    // (e.g. flushed an empty SSE event) would fail JSON.parse here.
    const assembled = captured
      .map((c) => c.argsDelta ?? "")
      .join("");
    expect(assembled).toBe(CANONICAL_ARGS);
    expect(JSON.parse(assembled)).toEqual(CANONICAL_PARSED);

    // (3) finish_reason arrives exactly once, on a chunk whose delta
    // is empty (no tool_calls fragment), and the value is
    // "tool_calls" — matching the upstream's terminal SSE event.
    const finishChunks = captured.filter((c) => c.finish !== null);
    expect(finishChunks).toHaveLength(1);
    expect(finishChunks[0]?.finish).toBe("tool_calls");
    expect(finishChunks[0]?.argsDelta).toBeUndefined();

    // (4) The terminal finish chunk is the last one captured; no
    // tool_calls fragments arrive after finish_reason. Catches a
    // regression that flushed a stray fragment after the terminator.
    expect(captured[captured.length - 1]?.finish).toBe("tool_calls");
  }, 60_000);
});
