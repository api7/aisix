import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: DeepSeek-reasoner `reasoning_content` pass-through on the
// non-streaming /v1/chat/completions path (#466).
//
// Pre-fix the DP's non-streaming response parser had no field for the
// upstream `choices[0].message.reasoning_content`, so DeepSeek's
// chain-of-thought text was silently dropped — the final answer came
// back but the reasoning trace did not. (The streaming path already
// surfaced it via the reasoning-field extractor; only non-streaming
// was affected.)
//
// We assert against the raw response JSON because `reasoning_content`
// is a DeepSeek extension not on the typed OpenAI SDK surface — the
// wire body is what reasoning-aware clients and observability consume.
//
// References:
// - DeepSeek reasoning API: https://api-docs.deepseek.com/guides/reasoning_model
// - Issue: api7/AISIX-Cloud#466

const CALLER_PLAINTEXT = "sk-ds-reasoning-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const REASONING_TEXT =
  "Let me work through this. 6 times 7. 6*7 = 42. So the answer is 42.";
const FINAL_ANSWER = "The answer is 42.";

describe("deepseek reasoning_content passthrough on /v1/chat/completions (#466)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // DeepSeek-reasoner non-streaming response: message carries both
    // `content` (final answer) and `reasoning_content` (chain-of-thought).
    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-ds-reasoner",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "deepseek-reasoner",
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: FINAL_ANSWER,
              reasoning_content: REASONING_TEXT,
            },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 12, completion_tokens: 40, total_tokens: 52 },
      },
    });

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    // DeepSeek dispatches through the OpenAI-compat family bridge.
    const pk = await seed.createProviderKey({
      display_name: "ds-reasoning-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
      provider: "deepseek",
      adapter: "openai",
    });
    await seed.createModel({
      display_name: "ds-reasoner",
      provider: "deepseek",
      model_name: "deepseek-reasoner",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["ds-reasoner"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("non-streaming response surfaces choices[0].message.reasoning_content (#466)", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    await waitConfigPropagation(async () => {
      try {
        const r = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            authorization: `Bearer ${CALLER_PLAINTEXT}`,
          },
          body: JSON.stringify({
            model: "ds-reasoner",
            messages: [{ role: "user", content: "probe" }],
          }),
        });
        return r.ok;
      } catch {
        return false;
      }
    });

    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
      },
      body: JSON.stringify({
        model: "ds-reasoner",
        messages: [{ role: "user", content: "What is 6 times 7?" }],
      }),
    });
    expect(res.status).toBe(200);

    const body = (await res.json()) as {
      choices: { message: Record<string, unknown> }[];
    };
    const message = body.choices[0]?.message;
    expect(message, JSON.stringify(body)).toBeDefined();
    // Final answer still byte-for-byte.
    expect(message.content).toBe(FINAL_ANSWER);
    // Reasoning trace now surfaced (pre-#466 this was dropped).
    expect(
      message.reasoning_content,
      `reasoning_content missing from message: ${JSON.stringify(message)}`,
    ).toBe(REASONING_TEXT);
  });
});
