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

// E2E: a streamed chat completion from an audio-output model. The
// upstream below replays the sequence OpenAI sends for `gpt-audio-mini`
// with `modalities: ["text","audio"]` and `include_usage`: `delta.audio`
// chunks, a finish chunk, a chunk with no `choices` key at all, the usage
// chunk, and a trailing `delta.audio.expires_at` chunk. The stream must
// reach the caller whole — the audio in OpenAI's own shape, the final
// usage with its audio tokens, and `[DONE]` — instead of losing the audio
// and ending in an `upstream_decode_error`.
//
// Reference: <https://platform.openai.com/docs/guides/audio>

const CALLER = "sk-chat-stream-audio-e2e";
const MODEL = "stream-audio";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

const chunk = (rest: Record<string, unknown>) =>
  JSON.stringify({
    id: "chatcmpl-audio",
    object: "chat.completion.chunk",
    created: 1,
    model: "gpt-audio-mini-2025-10-06",
    ...rest,
  });

const EXPIRES_AT = 1760003600;
const OPENAI_SEQUENCE = [
  chunk({
    choices: [
      {
        index: 0,
        delta: { role: "assistant", content: null, audio: { id: "audio_1", transcript: "Hel" } },
        finish_reason: null,
      },
    ],
  }),
  chunk({ choices: [{ index: 0, delta: { audio: { data: "AAAA" } }, finish_reason: null }] }),
  chunk({ choices: [{ index: 0, delta: { audio: { transcript: "lo" } }, finish_reason: null }] }),
  chunk({ choices: [{ index: 0, delta: { audio: { data: "BBBB" } }, finish_reason: null }] }),
  chunk({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }),
  // No `choices` key at all.
  chunk({ usage: null, obfuscation: "x1y2" }),
  chunk({
    choices: [],
    usage: {
      prompt_tokens: 40,
      completion_tokens: 50,
      total_tokens: 90,
      prompt_tokens_details: { cached_tokens: 0, audio_tokens: 24, text_tokens: 16 },
      completion_tokens_details: { reasoning_tokens: 0, audio_tokens: 30, text_tokens: 20 },
    },
  }),
  chunk({ choices: [{ index: 0, delta: { audio: { expires_at: EXPIRES_AT } }, finish_reason: null }] }),
  "[DONE]",
];

describe("a streamed audio chat completion reaches the caller whole", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    if (!(await etcd.ping())) return;
    upstream = await startOpenAiUpstream({ streamEvents: OPENAI_SEQUENCE });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "stream-audio-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "openai",
      model_name: "gpt-audio-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({ key_hash: hash(CALLER), allowed_models: [MODEL] });
    await waitConfigPropagation(async () => {
      const r = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER}` },
      });
      const body = await r.text();
      return r.status === 200 && body.includes(MODEL);
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("audio deltas, the final usage and [DONE] all arrive", async (ctx) => {
    if (!app || !upstream) return ctx.skip();
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify({
        model: MODEL,
        modalities: ["text", "audio"],
        audio: { voice: "alloy", format: "pcm16" },
        stream: true,
        stream_options: { include_usage: true },
        messages: [{ role: "user", content: "say hello" }],
      }),
    });
    const text = await res.text();
    expect(res.status, text).toBe(200);
    expect(text).not.toContain("event: error");

    const data = text
      .split("\n")
      .filter((l) => l.startsWith("data: "))
      .map((l) => l.slice(6));
    expect(data.at(-1), text).toBe("[DONE]");
    const chunks = data.slice(0, -1).map((d) => JSON.parse(d));

    const audio = chunks.flatMap((c) => (c.choices ?? []).map((ch: any) => ch.delta?.audio).filter(Boolean));
    expect(audio[0]?.id, text).toBe("audio_1");
    expect(audio.map((a: any) => a.transcript ?? "").join("")).toBe("Hello");
    expect(audio.map((a: any) => a.data ?? "").join("")).toBe("AAAABBBB");
    expect(audio.some((a: any) => a.expires_at === EXPIRES_AT), text).toBe(true);

    // The chunk with no `choices` carries nothing and is not relayed.
    const empty = chunks.filter(
      (c) =>
        !c.usage &&
        !c.choices?.[0]?.finish_reason &&
        Object.keys(c.choices?.[0]?.delta ?? {}).length === 0,
    );
    expect(empty, text).toEqual([]);

    const usage = chunks.map((c) => c.usage).filter(Boolean).at(-1);
    expect(usage?.prompt_tokens_details?.audio_tokens, text).toBe(24);
    expect(usage?.completion_tokens_details?.audio_tokens, text).toBe(30);
  });
});
