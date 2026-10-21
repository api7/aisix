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

// E2E: a chat completion carries the upstream's audio-token breakdown.
//
// An audio-capable chat model reports `prompt_tokens_details.audio_tokens`
// and `completion_tokens_details.audio_tokens` beside the cache and reasoning
// counters. The gateway kept them for its usage record but rendered the
// caller's `usage` without them, so a client pricing audio separately saw a
// response with no audio in it. They are passed on exactly when the upstream
// reported them, under OpenAI's own names — on the non-streamed body and on
// the stream's final usage chunk — and never invented for an upstream that
// did not report them.

const CALLER = "sk-chat-usage-audio-e2e";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

const AUDIO_USAGE = {
  prompt_tokens: 120,
  completion_tokens: 80,
  total_tokens: 200,
  prompt_tokens_details: { cached_tokens: 0, audio_tokens: 100 },
  completion_tokens_details: { reasoning_tokens: 0, audio_tokens: 64 },
};
const PLAIN_USAGE = { prompt_tokens: 7, completion_tokens: 3, total_tokens: 10 };
/** What OpenAI sends for a text-only request: every counter, all zero. */
const ZERO_USAGE = {
  prompt_tokens: 7,
  completion_tokens: 3,
  total_tokens: 10,
  prompt_tokens_details: { cached_tokens: 0, audio_tokens: 0 },
  completion_tokens_details: {
    reasoning_tokens: 0,
    audio_tokens: 0,
    accepted_prediction_tokens: 0,
    rejected_prediction_tokens: 0,
  },
};
type Kind = "audio" | "plain" | "zero";
const modelName = (kind: Kind, stream: boolean) => `usage-${kind}${stream ? "-stream" : ""}`;

/** The mock streams whenever it has stream events, so each mode gets its own. */
function upstreamReporting(usage: Record<string, unknown>, stream: boolean) {
  if (!stream) {
    return startOpenAiUpstream({
      nonStreamBody: {
        id: "chatcmpl-audio",
        object: "chat.completion",
        model: "gpt-4o-audio-preview",
        choices: [{ index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "stop" }],
        usage,
      },
    });
  }
  return startOpenAiUpstream({
    nonStreamBody: {
      id: "chatcmpl-audio",
      object: "chat.completion",
      model: "gpt-4o-audio-preview",
      choices: [{ index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "stop" }],
      usage,
    },
    streamEvents: [
      JSON.stringify({
        id: "chatcmpl-audio",
        object: "chat.completion.chunk",
        model: "gpt-4o-audio-preview",
        choices: [{ index: 0, delta: { role: "assistant", content: "hi" }, finish_reason: null }],
      }),
      JSON.stringify({
        id: "chatcmpl-audio",
        object: "chat.completion.chunk",
        model: "gpt-4o-audio-preview",
        choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
      }),
      JSON.stringify({
        id: "chatcmpl-audio",
        object: "chat.completion.chunk",
        model: "gpt-4o-audio-preview",
        choices: [],
        usage,
      }),
      "[DONE]",
    ],
  });
}

describe("chat completion usage carries the upstream's audio tokens", () => {
  let app: SpawnedApp | undefined;
  const upstreams: OpenAiUpstream[] = [];
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const models: string[] = [];
    for (const [kind, usage] of [
      ["audio", AUDIO_USAGE],
      ["plain", PLAIN_USAGE],
      ["zero", ZERO_USAGE],
    ] as const) {
      for (const stream of [false, true]) {
        const name = modelName(kind, stream);
        const up = await upstreamReporting(usage, stream);
        upstreams.push(up);
        models.push(name);
        const pk = await seed.createProviderKey({ display_name: `${name}-pk`, secret: "sk-mock", api_base: `${up.baseUrl}/v1` });
        await seed.createModel({ display_name: name, provider: "openai", model_name: "gpt-4o-audio-preview", provider_key_id: pk.id });
      }
    }
    await seed.createApiKey({ key_hash: hash(CALLER), allowed_models: models });
    await waitConfigPropagation(async () => {
      const r = await fetch(`${app!.proxyUrl}/v1/models`, { headers: { authorization: `Bearer ${CALLER}` } });
      const body = await r.text();
      return r.status === 200 && models.every((m) => body.includes(m));
    });
  });

  afterAll(async () => {
    await app?.exit();
    for (const up of upstreams) await up.close();
  });

  async function usageOf(model: string, stream: boolean): Promise<Record<string, any>> {
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: "hi" }],
        ...(stream ? { stream: true, stream_options: { include_usage: true } } : {}),
      }),
    });
    const text = await res.text();
    expect(res.status, text).toBe(200);
    if (!stream) return JSON.parse(text).usage;
    const usages = text
      .split("\n")
      .filter((l) => l.startsWith("data: ") && l !== "data: [DONE]")
      .map((l) => JSON.parse(l.slice(6)).usage)
      .filter((u) => u);
    expect(usages.length, text).toBeGreaterThan(0);
    return usages.at(-1);
  }

  for (const stream of [false, true]) {
    const mode = stream ? "streamed" : "non-streamed";

    test(`${mode}: reported audio tokens reach the caller`, async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const usage = await usageOf(modelName("audio", stream), stream);
      expect(usage.prompt_tokens).toBe(120);
      expect(usage.completion_tokens).toBe(80);
      // The objects keep the counters they always carry, as OpenAI sends them.
      expect(usage.prompt_tokens_details).toEqual({ cached_tokens: 0, audio_tokens: 100 });
      expect(usage.completion_tokens_details).toEqual({ reasoning_tokens: 0, audio_tokens: 64 });
    });

    test(`${mode}: an all-zero breakdown reaches the caller as reported`, async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const usage = await usageOf(modelName("zero", stream), stream);
      expect(usage.prompt_tokens).toBe(7);
      expect(usage.prompt_tokens_details).toEqual(ZERO_USAGE.prompt_tokens_details);
      expect(usage.completion_tokens_details).toEqual(ZERO_USAGE.completion_tokens_details);
    });

    test(`${mode}: no audio count is invented when the upstream reported none`, async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const usage = await usageOf(modelName("plain", stream), stream);
      expect(usage.prompt_tokens).toBe(7);
      expect(usage.prompt_tokens_details?.audio_tokens).toBeUndefined();
      expect(usage.completion_tokens_details?.audio_tokens).toBeUndefined();
    });
  }
});
