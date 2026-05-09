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
// The unit-level matrix_openai_in_anthropic_upstream_non_streaming
// covers this in process; this case proves the wire contract holds
// against a real binary, etcd watch, and the official OpenAI SDK
// client. The mock-upstream harness's body is path-agnostic so
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
// - In-process counterpart:
//   `crates/aisix-proxy/src/lib.rs::matrix_openai_in_anthropic_upstream_non_streaming`

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

    // Confirm the gateway actually hit the Anthropic Messages endpoint
    // (not /v1/chat/completions). A regression that mis-routed an
    // anthropic-provider Model through the OpenAI bridge would never
    // land on /v1/messages.
    expect(
      upstream.receivedRequests.some((r) =>
        r.path.startsWith("/v1/messages"),
      ),
    ).toBe(true);
  });
});
