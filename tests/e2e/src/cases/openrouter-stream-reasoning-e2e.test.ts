import { createHash } from "node:crypto";
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

// E2E: OpenRouter `delta.reasoning` normalization on the *streaming*
// /v1/chat/completions path (#502, streaming parity with non-stream #648).
//
// OpenRouter (and some OpenAI-compatible aggregators) stream a reasoning
// model's chain-of-thought at `delta.reasoning`, NOT the DeepSeek-canonical
// `delta.reasoning_content`. The non-stream half (#648/#501) already
// normalizes `message.reasoning` → `reasoning_content`; before #502 the
// streaming delta type had no `reasoning` field, so with no per-key
// override an OpenRouter reasoning model dropped its whole reasoning trace
// on the streaming path. We assert the gateway surfaces it in the canonical
// `delta.reasoning_content` slot with no operator configuration.
//
// References:
// - Issue: api7/ai-gateway#502 (follow-up to #501 / api7/AISIX-Cloud#648)

const CALLER_PLAINTEXT = "sk-or-stream-reasoning-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const REASONING_A = "Let me think. ";
const REASONING_B = "6 * 7 = 42.";
const FINAL_ANSWER = "The answer is 42.";

function chunk(delta: Record<string, unknown>, finish: string | null = null) {
  return JSON.stringify({
    id: "cmpl-or-stream",
    object: "chat.completion.chunk",
    created: 1,
    model: "openrouter/some-reasoner",
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
}

describe("openrouter delta.reasoning normalization on streaming /v1/chat/completions (#502)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    // OpenRouter-style streaming: reasoning text arrives on `delta.reasoning`,
    // the final answer on `delta.content`. No `reasoning_content` anywhere.
    upstream = await startOpenAiUpstream({
      streamEvents: [
        chunk({ role: "assistant" }),
        chunk({ reasoning: REASONING_A }),
        chunk({ reasoning: REASONING_B }),
        chunk({ content: FINAL_ANSWER }),
        chunk({}, "stop"),
        "[DONE]",
      ],
    });

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    // OpenRouter dispatches through the OpenAI-compat family bridge.
    const pk = await admin.createProviderKey({
      display_name: "or-stream-reasoning-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
      provider: "openrouter",
      adapter: "openai",
    });
    await admin.createModel({
      display_name: "or-reasoner",
      provider: "openrouter",
      model_name: "some-reasoner",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["or-reasoner"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("streaming deltas surface delta.reasoning_content from upstream delta.reasoning (#502)", async (ctx) => {
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
            model: "or-reasoner",
            stream: true,
            messages: [{ role: "user", content: "probe" }],
          }),
        });
        // Drain so the next request isn't blocked on a dangling stream.
        await r.text();
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
        model: "or-reasoner",
        stream: true,
        messages: [{ role: "user", content: "What is 6 times 7?" }],
      }),
    });
    expect(res.status).toBe(200);

    const text = await res.text();
    const reasoningParts: string[] = [];
    const contentParts: string[] = [];
    for (const line of text.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const payload = trimmed.slice("data:".length).trim();
      if (payload === "[DONE]" || payload === "") continue;
      const evt = JSON.parse(payload) as {
        choices?: { delta?: Record<string, unknown> }[];
      };
      const delta = evt.choices?.[0]?.delta ?? {};
      if (typeof delta.reasoning_content === "string") {
        reasoningParts.push(delta.reasoning_content);
      }
      // The upstream's raw `reasoning` field must NOT leak through verbatim;
      // it should be normalized to `reasoning_content`.
      expect(
        delta.reasoning,
        `raw delta.reasoning leaked: ${JSON.stringify(delta)}`,
      ).toBeUndefined();
      if (typeof delta.content === "string") contentParts.push(delta.content);
    }

    expect(reasoningParts.join(""), `stream:\n${text}`).toBe(
      REASONING_A + REASONING_B,
    );
    expect(contentParts.join("")).toBe(FINAL_ANSWER);
  });
});
