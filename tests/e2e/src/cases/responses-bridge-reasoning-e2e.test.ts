import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
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

// E2E for the RESPONSE direction of the `/v1/responses` → chat-completions
// bridge.
//
// Two contracts are pinned here.
//
// 1. A chat upstream's chain-of-thought reaches the caller. Reasoning models
//    served over the OpenAI-compatible chat wire report their thinking as
//    `delta.reasoning_content` (streaming) / `message.reasoning_content`
//    (non-streaming). The bridge used to drop it on the floor, so a Codex-CLI
//    user talking to such a model through `/v1/responses` saw the answer with
//    no reasoning at all, while the same model over `/v1/chat/completions`
//    showed it. It now renders as a `reasoning` output item carrying a
//    `summary_text` part, ahead of the message item, with the Responses-API
//    reasoning event sequence in the streaming case.
//
// 2. The usage a client reads is the usage the gateway records. When a
//    bridged stream ends with no upstream usage frame, the gateway counts the
//    tokens locally; it used to hand the client an all-zero `usage` block
//    while billing the estimate, which is irreconcilable from the caller's
//    side.
//
// The bridge is reached by declaring an EMPTY `apis` map on the provider key:
// that is the operator saying "this OpenAI-compatible endpoint has no
// `/v1/responses`", which is exactly the deployment (a vLLM / SGLang / vendor
// relay serving chat only) these contracts matter for.

const CALLER_PLAINTEXT = "sk-responses-bridge-reasoning";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const CREDENTIAL_REF = "mock";
const MOCK_AK_ID = "mock-akid";
const MOCK_AK_SECRET = "mock-secret";
const SLS_PROJECT = "aisix-e2e-obs";
const META_LOGSTORE = "bridge-reasoning-events";

const REASONING_A = "Let me work through it. ";
const REASONING_B = "Seven sixes are forty-two.";
const ANSWER = "42";

/** Chat SSE: reasoning deltas, then content, then finish + usage. */
const CHAT_STREAM_REASONING = [
  JSON.stringify({
    id: "chatcmpl-reason-1",
    object: "chat.completion.chunk",
    model: "relay-reasoner",
    choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
  }),
  JSON.stringify({
    id: "chatcmpl-reason-1",
    object: "chat.completion.chunk",
    model: "relay-reasoner",
    choices: [
      { index: 0, delta: { reasoning_content: REASONING_A }, finish_reason: null },
    ],
  }),
  JSON.stringify({
    id: "chatcmpl-reason-1",
    object: "chat.completion.chunk",
    model: "relay-reasoner",
    choices: [
      { index: 0, delta: { reasoning_content: REASONING_B }, finish_reason: null },
    ],
  }),
  JSON.stringify({
    id: "chatcmpl-reason-1",
    object: "chat.completion.chunk",
    model: "relay-reasoner",
    choices: [{ index: 0, delta: { content: ANSWER }, finish_reason: null }],
  }),
  JSON.stringify({
    id: "chatcmpl-reason-1",
    object: "chat.completion.chunk",
    model: "relay-reasoner",
    choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
  }),
  JSON.stringify({
    id: "chatcmpl-reason-1",
    object: "chat.completion.chunk",
    model: "relay-reasoner",
    choices: [],
    usage: {
      prompt_tokens: 9,
      completion_tokens: 21,
      total_tokens: 30,
      completion_tokens_details: { reasoning_tokens: 15 },
    },
  }),
  "[DONE]",
];

/** Chat non-streaming 200 carrying `message.reasoning_content`. */
const CHAT_NON_STREAM_REASONING = {
  id: "chatcmpl-reason-2",
  object: "chat.completion",
  created: Math.floor(Date.now() / 1000),
  model: "relay-reasoner",
  choices: [
    {
      index: 0,
      message: {
        role: "assistant",
        content: ANSWER,
        reasoning_content: REASONING_A + REASONING_B,
      },
      finish_reason: "stop",
    },
  ],
  usage: {
    prompt_tokens: 9,
    completion_tokens: 21,
    total_tokens: 30,
    completion_tokens_details: { reasoning_tokens: 15 },
  },
};

/**
 * Chat SSE from a relay that ignores `stream_options.include_usage`: content,
 * a finish chunk, and no usage frame ever.
 *
 * The expected counts are ground truth from the de-facto OpenAI counting
 * scheme (https://github.com/openai/openai-cookbook — "How to count tokens"),
 * NOT read off the gateway: per message 3 tokens + its text, +3 reply
 * priming, plain-text counting on the completion side. The upstream model
 * name is deliberately non-OpenAI so the fallback `cl100k_base` encoding
 * applies. One user message "hi" ⇒ 3 + 1 ("user") + 1 ("hi") + 3 = 8 prompt
 * tokens; "Hello world" ⇒ 2 completion tokens.
 */
const CHAT_STREAM_NO_USAGE = [
  JSON.stringify({
    id: "chatcmpl-nousage-1",
    object: "chat.completion.chunk",
    model: "relay-mini",
    choices: [{ index: 0, delta: { content: "Hello" }, finish_reason: null }],
  }),
  JSON.stringify({
    id: "chatcmpl-nousage-1",
    object: "chat.completion.chunk",
    model: "relay-mini",
    choices: [{ index: 0, delta: { content: " world" }, finish_reason: null }],
  }),
  JSON.stringify({
    id: "chatcmpl-nousage-1",
    object: "chat.completion.chunk",
    model: "relay-mini",
    choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
  }),
  "[DONE]",
];
const EXPECTED_PROMPT_TOKENS = 8;
const EXPECTED_COMPLETION_TOKENS = 2;

interface SseEvent {
  type: string;
  data: Record<string, unknown>;
}

/** Parse a Responses-API SSE body into its ordered event list. */
function parseSse(raw: string): SseEvent[] {
  const events: SseEvent[] = [];
  for (const frame of raw.split("\n\n")) {
    const line = frame
      .split("\n")
      .find((l) => l.startsWith("data: "));
    if (!line) continue;
    const payload = line.slice("data: ".length);
    if (payload === "[DONE]") continue;
    const data = JSON.parse(payload) as Record<string, unknown>;
    events.push({ type: String(data.type), data });
  }
  return events;
}

describe("/v1/responses bridged onto a chat upstream: reasoning + client-visible usage", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let streamUpstream: OpenAiUpstream | undefined;
  let nonStreamUpstream: OpenAiUpstream | undefined;
  let noUsageUpstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    streamUpstream = await startOpenAiUpstream({
      streamEvents: CHAT_STREAM_REASONING,
    });
    nonStreamUpstream = await startOpenAiUpstream({
      nonStreamBody: CHAT_NON_STREAM_REASONING,
    });
    noUsageUpstream = await startOpenAiUpstream({
      streamEvents: CHAT_STREAM_NO_USAGE,
    });

    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: MOCK_AK_ID,
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: MOCK_AK_SECRET,
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "bridge-reasoning-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: META_LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });

    // `apis: {}` — an OpenAI-compatible endpoint with no `/v1/responses`, so
    // `/v1/responses` traffic is bridged onto `/v1/chat/completions`.
    const seedModel = async (display: string, upstream: OpenAiUpstream) => {
      const pk = await seed.createProviderKey({
        display_name: `${display}-pk`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
        apis: {},
      });
      await seed.createModel({
        display_name: display,
        provider: "openai",
        model_name: "relay-compat-x",
        provider_key_id: pk.id,
      });
    };
    await seedModel("bridge-reason-stream", streamUpstream);
    await seedModel("bridge-reason-nonstream", nonStreamUpstream);
    await seedModel("bridge-no-usage-stream", noUsageUpstream);
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [
        "bridge-reason-stream",
        "bridge-reason-nonstream",
        "bridge-no-usage-stream",
      ],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await sls?.close();
    await streamUpstream?.close();
    await nonStreamUpstream?.close();
    await noUsageUpstream?.close();
  });

  function post(body: unknown): Promise<Response> {
    return fetch(`${app!.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    });
  }

  test("streaming: reasoning_content deltas become a reasoning item ahead of the message item", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    await waitConfigPropagation(async () => {
      try {
        const probe = await post({
          model: "bridge-reason-stream",
          input: "ready",
          stream: true,
        });
        return probe.status === 200 && (await probe.text()).includes("response.completed");
      } catch {
        return false;
      }
    });

    const res = await post({
      model: "bridge-reason-stream",
      input: "what is 6 times 7",
      stream: true,
    });
    expect(res.status).toBe(200);
    const events = parseSse(await res.text());

    // The reasoning item opens, is summarised, and is closed BEFORE the
    // message item opens — the order a Responses client renders in.
    expect(events.map((e) => e.type)).toEqual([
      "response.created",
      "response.in_progress",
      "response.output_item.added",
      "response.reasoning_summary_part.added",
      "response.reasoning_summary_text.delta",
      "response.reasoning_summary_text.delta",
      "response.reasoning_summary_text.done",
      "response.reasoning_summary_part.done",
      "response.output_item.done",
      "response.output_item.added",
      "response.content_part.added",
      "response.output_text.delta",
      "response.output_text.done",
      "response.content_part.done",
      "response.output_item.done",
      "response.completed",
    ]);

    const reasoningAdded = events[2];
    expect((reasoningAdded.data.item as Record<string, unknown>).type).toBe("reasoning");
    expect(reasoningAdded.data.output_index).toBe(0);
    const reasoningId = String((reasoningAdded.data.item as Record<string, string>).id);
    expect(reasoningId.startsWith("rs_")).toBe(true);

    expect(events[4].data.item_id).toBe(reasoningId);
    expect(events[4].data.delta).toBe(REASONING_A);
    expect(events[5].data.delta).toBe(REASONING_B);
    expect(events[6].data.text).toBe(REASONING_A + REASONING_B);

    // The message item takes the NEXT output_index.
    expect(events[9].data.output_index).toBe(1);
    expect((events[9].data.item as Record<string, unknown>).type).toBe("message");

    // Sequence numbers keep counting across the whole stream, reasoning
    // events included.
    expect(events.map((e) => e.data.sequence_number)).toEqual(
      events.map((_, i) => i),
    );

    const completed = events.at(-1)!.data.response as Record<string, unknown>;
    const output = completed.output as Array<Record<string, unknown>>;
    expect(output.map((i) => i.type)).toEqual(["reasoning", "message"]);
    expect(
      (output[0].summary as Array<Record<string, string>>)[0],
    ).toEqual({ type: "summary_text", text: REASONING_A + REASONING_B });
    expect(
      (output[1].content as Array<Record<string, string>>)[0].text,
    ).toBe(ANSWER);
    const usage = completed.usage as Record<string, Record<string, number> & number>;
    expect(usage.output_tokens_details.reasoning_tokens).toBe(15);

    // The gateway spoke the chat wire upstream.
    expect(streamUpstream!.receivedRequests.at(-1)?.path).toBe("/v1/chat/completions");
  });

  test("non-streaming: message.reasoning_content becomes a leading reasoning item", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    await waitConfigPropagation(async () => {
      try {
        const probe = await post({ model: "bridge-reason-nonstream", input: "ready" });
        return probe.status === 200 && (await probe.json()).object === "response";
      } catch {
        return false;
      }
    });

    const res = await post({
      model: "bridge-reason-nonstream",
      input: "what is 6 times 7",
    });
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.output[0].type).toBe("reasoning");
    expect(String(body.output[0].id).startsWith("rs_")).toBe(true);
    expect(body.output[0].summary).toEqual([
      { type: "summary_text", text: REASONING_A + REASONING_B },
    ]);
    expect(body.output[1].type).toBe("message");
    expect(body.output[1].content[0].text).toBe(ANSWER);
    expect(body.usage.output_tokens_details.reasoning_tokens).toBe(15);
  });

  test("streaming with no upstream usage frame: response.completed reports the counts the usage record gets", async (ctx) => {
    if (!etcdReachable || !app || !sls) {
      ctx.skip();
      return;
    }
    await waitConfigPropagation(async () => {
      try {
        const probe = await post({
          model: "bridge-no-usage-stream",
          input: "ready",
          stream: true,
        });
        return probe.status === 200 && (await probe.text()).includes("response.completed");
      } catch {
        return false;
      }
    });

    const res = await post({
      model: "bridge-no-usage-stream",
      input: "hi",
      stream: true,
    });
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id");
    expect(requestId).toBeTruthy();
    const events = parseSse(await res.text());

    const completed = events.at(-1)!;
    expect(completed.type).toBe("response.completed");
    const usage = (completed.data.response as Record<string, unknown>)
      .usage as Record<string, number>;
    expect(usage.input_tokens).toBe(EXPECTED_PROMPT_TOKENS);
    expect(usage.output_tokens).toBe(EXPECTED_COMPLETION_TOKENS);
    expect(usage.total_tokens).toBe(
      EXPECTED_PROMPT_TOKENS + EXPECTED_COMPLETION_TOKENS,
    );

    // …and the usage record the operator bills from carries the same pair.
    const row = await waitForSlsLog(
      sls,
      META_LOGSTORE,
      (log) => log.get("request_id") === requestId,
      `a usage row for request_id=${requestId}`,
    );
    expect(row.get("prompt_tokens")).toBe(String(usage.input_tokens));
    expect(row.get("completion_tokens")).toBe(String(usage.output_tokens));
  });
});
