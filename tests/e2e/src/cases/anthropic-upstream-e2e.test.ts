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

// E2E: cross-provider translation. The caller speaks OpenAI Chat
// Completions to the gateway; the gateway speaks Anthropic Messages
// to the upstream. The Anthropic-shaped response (`type: "message"`,
// `content: [{type:"text",text}]`, `stop_reason`,
// `usage: {input_tokens, output_tokens}`) must come back to the SDK
// caller as an OpenAI shape (`object: "chat.completion"`,
// `choices[0].message.content`, `choices[0].finish_reason`,
// `usage: {prompt_tokens, completion_tokens, total_tokens}`).
//
// The mock-upstream harness's body is path-agnostic so
// `startOpenAiUpstream` doubles as the Anthropic mock when fed an
// Anthropic-shaped `nonStreamBody`. The Anthropic bridge appends
// `/v1/messages` to the api_base on its own, so we still reach the
// mock — `receivedRequests` confirms it.
//
// Reference:
// - OpenAI Chat Completions API spec:
//   <https://platform.openai.com/docs/api-reference/chat/create>
// - Anthropic Messages API spec:
//   <https://docs.anthropic.com/en/api/messages>

const CALLER_PLAINTEXT = "sk-an-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("anthropic upstream e2e: OpenAI in, Anthropic out, OpenAI back to caller", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "msg_01",
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: "Hello from Claude!" }],
        model: "claude-3-5-haiku-20241022",
        stop_reason: "end_turn",
        usage: { input_tokens: 5, output_tokens: 4 },
      },
    });
    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    // The Anthropic bridge appends `/v1/messages` to the api_base, so
    // we point it at the bare host (no `/v1` suffix) — the OpenAI
    // bridge convention is the opposite (host + `/v1`), and getting
    // these mixed up was the lesson from the unit-level matrix tests.
    const pk = await admin.createProviderKey({
      display_name: "an-e2e-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-mock",
      api_base: upstream.baseUrl,
    });
    await admin.createModel({
      display_name: "an-e2e",
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["an-e2e"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("OpenAI client + Anthropic upstream round-trip translates wire shape both ways", async (ctx) => {
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
        const r = await client.chat.completions.create({
          model: "an-e2e",
          messages: [{ role: "user", content: "ready-probe" }],
        });
        return r.choices[0]?.message.role === "assistant";
      } catch {
        return false;
      }
    });

    // Baseline-isolate the readiness probe so the request-shape
    // assertions below match the test call's request, not the probe's.
    // The probe also targets `an-e2e` and so also lands on
    // `/v1/messages`; without slicing from baseline, `find()` over
    // `receivedRequests` would return the probe and the assertions
    // would verify the wrong request — a regression that broke ONLY
    // the test call's translation would slip through.
    const baseline = upstream.receivedRequests.length;

    const completion = await client.chat.completions.create({
      model: "an-e2e",
      messages: [{ role: "user", content: "hi" }],
    });

    // Response wire shape: must be OpenAI-shaped on the way out.
    expect(completion.object).toBe("chat.completion");
    expect(completion.choices[0]?.message.role).toBe("assistant");
    expect(completion.choices[0]?.message.content).toBe("Hello from Claude!");
    // Anthropic stop_reason "end_turn" must translate to OpenAI
    // finish_reason "stop". A regression that left "end_turn" through
    // would break every OpenAI-compatible client downstream.
    expect(completion.choices[0]?.finish_reason).toBe("stop");
    // Anthropic input_tokens / output_tokens → OpenAI prompt_tokens /
    // completion_tokens / total_tokens.
    expect(completion.usage?.prompt_tokens).toBe(5);
    expect(completion.usage?.completion_tokens).toBe(4);
    expect(completion.usage?.total_tokens).toBe(9);

    // Confirm the gateway hit /v1/messages on the test call (slice
    // from baseline so the probe's earlier hit cannot stand in).
    const messagesReq = upstream.receivedRequests
      .slice(baseline)
      .find((r) => r.path.startsWith("/v1/messages"));
    expect(messagesReq).toBeDefined();

    // Wire-shape on the request side: the gateway must speak Anthropic
    // Messages, not OpenAI Chat Completions. Without these assertions
    // the mock would happily 200 even on an OpenAI-shaped body, so
    // the response-side translation could pass while the request-side
    // translation was completely broken (the regression that prompted
    // this audit pass).
    const sentBody = JSON.parse(messagesReq!.body) as {
      model?: string;
      max_tokens?: unknown;
      messages?: Array<{ role?: string; content?: unknown }>;
    };
    // model_name from the Model resource (Anthropic's id), not the
    // gateway's display_name "an-e2e".
    expect(sentBody.model).toBe("claude-3-5-haiku-20241022");
    // Anthropic's API requires `max_tokens`; OpenAI's doesn't. The
    // gateway must inject a value when crossing the boundary.
    expect(typeof sentBody.max_tokens).toBe("number");
    // Caller's user message must reach the upstream, role + content
    // intact — pin to the test call's content (`"hi"`), not the probe's
    // (`"ready-probe"`). This doubles as a baseline-isolation check.
    expect(sentBody.messages?.[0]?.role).toBe("user");
    // Anthropic API serialises content as an array of typed blocks
    // (`[{type:"text", text:"hi"}]`), not a bare string — gateway
    // converts on the way out. The OpenAI-side input was a string,
    // so a regression that dropped the content (or sent the wrong
    // text) would no longer match.
    expect(sentBody.messages?.[0]?.content).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: "text", text: "hi" }),
      ]),
    );

    // Auth shape: Anthropic uses `x-api-key` + `anthropic-version`,
    // not `Authorization: Bearer`. A regression that forwarded the
    // OpenAI auth shape to the Anthropic upstream would 401 in
    // production but pass against the permissive mock here without
    // these explicit header assertions.
    expect(messagesReq!.headers["x-api-key"]).toBe("sk-ant-mock");
    expect(messagesReq!.headers["anthropic-version"]).toBeDefined();
  });
});

// E2E (#395): an Anthropic upstream that returns ONLY a `tool_use` block
// (no text block) must surface `choices[0].message.content === null` on
// the OpenAI shape — not `""`. The text-block join produced `""` before
// the fix; we now map empty/absent text to `null`, per the documented
// `string | null` content shape.
const TOOL_CALLER_PLAINTEXT = "sk-an-e2e-toolnull-caller";
const TOOL_CALLER_KEY_HASH = createHash("sha256")
  .update(TOOL_CALLER_PLAINTEXT)
  .digest("hex");

describe("anthropic upstream e2e: tool_use-only response surfaces content:null (#395)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "msg_toolonly_01",
        type: "message",
        role: "assistant",
        // Only a tool_use block, no text block — the bug-class case.
        content: [
          {
            type: "tool_use",
            id: "toolu_abc",
            name: "get_weather",
            input: { location: "SF" },
          },
        ],
        model: "claude-3-5-haiku-20241022",
        stop_reason: "tool_use",
        usage: { input_tokens: 7, output_tokens: 3 },
      },
    });
    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "an-e2e-toolnull-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-mock",
      api_base: upstream.baseUrl,
    });
    await admin.createModel({
      display_name: "an-e2e-toolnull",
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: TOOL_CALLER_KEY_HASH,
      allowed_models: ["an-e2e-toolnull"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("tool_use-only Anthropic response → content is exactly null on OpenAI shape", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: TOOL_CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    await waitConfigPropagation(async () => {
      try {
        const r = await client.chat.completions.create({
          model: "an-e2e-toolnull",
          messages: [{ role: "user", content: "ready-probe" }],
        });
        return r.choices[0]?.message.role === "assistant";
      } catch {
        return false;
      }
    });

    const completion = await client.chat.completions.create({
      model: "an-e2e-toolnull",
      messages: [{ role: "user", content: "weather in SF?" }],
    });

    expect(completion.choices[0]?.finish_reason).toBe("tool_calls");
    // The tool_use block translates to OpenAI tool_calls.
    expect(completion.choices[0]?.message.tool_calls?.[0]?.function.name).toBe(
      "get_weather",
    );
    // The fix: content must be exactly null, not "".
    expect(completion.choices[0]?.message.content).toBeNull();
  });
});
