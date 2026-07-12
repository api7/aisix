import { createHash } from "node:crypto";
import OpenAI from "openai";
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

// E2E for #911 finding [22]: the audio endpoints dispatched directly to the
// upstream WITHOUT applying the model's `timeout` (request_timeout) — the
// #554 per-model E2E timeout that every other non-streaming path already
// wires. A slow/blackholed audio provider could therefore pin a
// transcription request open past the model's configured deadline.
//
// Setup: an audio model whose provider upstream stalls for SLOW_MS before
// responding, with the model's `timeout` set to TIMEOUT_MS. The transcription
// request must be abandoned at ~TIMEOUT_MS. Before the fix it waited the full
// SLOW_MS (no timeout was applied); after, it fails fast.

const CALLER_PLAINTEXT = "sk-audio-timeout-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const SLOW_MS = 3000;
const TIMEOUT_MS = 400;

function chatReply(content: string): unknown {
  return {
    id: `cmpl-${content}`,
    object: "chat.completion",
    created: 0,
    model: "gpt-4o-mini",
    choices: [
      { index: 0, message: { role: "assistant", content }, finish_reason: "stop" },
    ],
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
  };
}

describe("audio request timeout (#911 [22])", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let slow: OpenAiUpstream | undefined;
  let fast: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // Slow upstream (delays status + headers) behind the audio model, and a
    // fast chat upstream used only to gate on config propagation.
    slow = await startOpenAiUpstream({
      responseDelayMs: SLOW_MS,
      nonStreamBody: { text: "slow transcription" },
    });
    fast = await startOpenAiUpstream({ nonStreamBody: chatReply("ready") });

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const slowPk = (
      await seed.createProviderKey({
        display_name: "audio-slow-pk",
        secret: "sk-mock",
        api_base: `${slow.baseUrl}/v1`,
      })
    ).id;
    const fastPk = (
      await seed.createProviderKey({
        display_name: "audio-gate-pk",
        secret: "sk-mock",
        api_base: `${fast.baseUrl}/v1`,
      })
    ).id;

    await seed.createModel({
      display_name: "audio-slow",
      provider: "openai",
      model_name: "whisper-1",
      provider_key_id: slowPk,
      timeout: TIMEOUT_MS,
      // Disable cooldown so the slow primary isn't taken out of rotation
      // between the propagation probe and the test call.
      cooldown: { enabled: false },
    });
    await seed.createModel({
      display_name: "gate-fast",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: fastPk,
    });

    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["audio-slow", "gate-fast"],
    });

    // Gate on the fast chat model resolving — all config above is written
    // first, so once this loads the audio model is loaded too.
    const gate = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });
    await waitConfigPropagation(async () => {
      try {
        const probe = await gate.chat.completions.create({
          model: "gate-fast",
          messages: [{ role: "user", content: "ready" }],
        });
        return probe.choices[0]?.message.content === "ready";
      } catch {
        return false;
      }
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all([slow?.close(), fast?.close()]);
  });

  test("transcription against a slow upstream is abandoned at the per-model timeout", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const form = new FormData();
    form.set("model", "audio-slow");
    form.set(
      "file",
      new Blob([new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7])], { type: "audio/wav" }),
      "clip.wav",
    );

    const started = Date.now();
    const res = await fetch(`${app.proxyUrl}/v1/audio/transcriptions`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      body: form,
    });
    const elapsed = Date.now() - started;

    // The upstream stalls for SLOW_MS; the model timeout must abandon it well
    // before that. Before the fix no timeout was applied and this waited the
    // full SLOW_MS, so the elapsed-time bound is what fails pre-fix.
    expect(res.ok).toBe(false);
    expect(elapsed).toBeLessThan(SLOW_MS - 800);
  }, 30_000);
});
