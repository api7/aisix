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

// E2E regression for #998: the audio endpoints must relay a response as
// it arrives, not collect it and hand the caller one write at the end.
//
// The functional assertions on these routes all passed before the fix —
// the caller got the right transcript and the right audio — because a
// buffered relay differs from a streaming one only in DELIVERY SHAPE.
// Both legs below therefore measure time and chunk count, not content:
// the mock upstream trickles its response over ~1s and the spec asserts
// the caller sees that same spread. Pre-fix each leg read exactly one
// chunk with a zero spread.

const CALLER_PLAINTEXT = "sk-issue-998-audio-stream-relay";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

/** ~1s of upstream deltas: 8 frames, 120ms apart. */
const CHUNK_DELAY_MS = 120;
const TRANSCRIPT_FRAMES = [
  ...["Streaming", " recognition", " emits", " partial", " results"].map(
    (delta) =>
      `data: ${JSON.stringify({ type: "transcript.text.delta", delta })}\n\n`,
  ),
  `data: ${JSON.stringify({
    type: "transcript.text.done",
    text: "Streaming recognition emits partial results",
    usage: {
      type: "tokens",
      input_tokens: 26,
      output_tokens: 12,
      total_tokens: 38,
    },
  })}\n\n`,
  "data: [DONE]\n\n",
];
const SPEECH_CHUNKS = ["ID3", "audio-part-1", "audio-part-2", "audio-part-3"];

/** One measured read loop over a response body. */
interface Delivery {
  chunks: number;
  firstByteMs: number;
  lastByteMs: number;
  body: string;
}

async function measure(res: Response): Promise<Delivery> {
  const started = Date.now();
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  const out: Delivery = {
    chunks: 0,
    firstByteMs: 0,
    lastByteMs: 0,
    body: "",
  };
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (out.chunks === 0) out.firstByteMs = Date.now() - started;
    out.chunks += 1;
    out.lastByteMs = Date.now() - started;
    out.body += decoder.decode(value, { stream: true });
  }
  return out;
}

/** A whole-file speech answer, served with a Content-Length. */
const WHOLE_SPEECH_BODY = "ID3whole-file-audio-body";

describe("the audio endpoints relay their response as it arrives (#998)", () => {
  let app: SpawnedApp | undefined;
  let transcribeUpstream: OpenAiUpstream | undefined;
  let speechUpstream: OpenAiUpstream | undefined;
  let wholeSpeechUpstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // The upload is multipart, so the mock can't be driven off a
    // `stream: true` JSON field — it serves the SSE frames as the raw
    // streaming 200 body instead.
    transcribeUpstream = await startOpenAiUpstream({
      rawStreamFrames: TRANSCRIPT_FRAMES,
      eventDelayMs: CHUNK_DELAY_MS,
    });
    speechUpstream = await startOpenAiUpstream({
      rawBodyChunks: SPEECH_CHUNKS,
      rawContentType: "audio/mpeg",
      eventDelayMs: CHUNK_DELAY_MS,
    });

    wholeSpeechUpstream = await startOpenAiUpstream({
      rawBody: WHOLE_SPEECH_BODY,
      rawContentType: "audio/mpeg",
    });

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const transcribePk = await seed.createProviderKey({
      display_name: "issue998-transcribe-pk",
      secret: "sk-openai-mock",
      api_base: `${transcribeUpstream.baseUrl}/v1`,
    });
    const speechPk = await seed.createProviderKey({
      display_name: "issue998-speech-pk",
      secret: "sk-openai-mock",
      api_base: `${speechUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "relay-transcribe",
      provider: "openai",
      model_name: "gpt-4o-transcribe",
      provider_key_id: transcribePk.id,
    });
    await seed.createModel({
      display_name: "relay-tts",
      provider: "openai",
      model_name: "tts-1",
      provider_key_id: speechPk.id,
    });
    const wholeSpeechPk = await seed.createProviderKey({
      display_name: "issue998-whole-speech-pk",
      secret: "sk-openai-mock",
      api_base: `${wholeSpeechUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "whole-tts",
      provider: "openai",
      model_name: "tts-1",
      provider_key_id: wholeSpeechPk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["relay-transcribe", "relay-tts", "whole-tts"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await transcribeUpstream?.close();
    await speechUpstream?.close();
    await wholeSpeechUpstream?.close();
  });

  test("a streamed transcription arrives frame by frame", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const call = () => {
      const form = new FormData();
      form.set("model", "relay-transcribe");
      form.set("stream", "true");
      form.set(
        "file",
        new Blob([new Uint8Array([0x49, 0x44, 0x33])], { type: "audio/mpeg" }),
        "a.mp3",
      );
      return fetch(`${app!.proxyUrl}/v1/audio/transcriptions`, {
        method: "POST",
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
        body: form,
      });
    };

    await waitConfigPropagation(async () => {
      try {
        const probe = await call();
        const ok = probe.ok;
        await probe.text();
        return ok;
      } catch {
        return false;
      }
    });

    const res = await call();
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("text/event-stream");
    const delivery = await measure(res);

    // Content first: relaying must not rewrite the stream.
    expect(delivery.body).toBe(TRANSCRIPT_FRAMES.join(""));

    // Delivery shape: the upstream spread its frames over ~840ms, so a
    // relay that forwards them keeps that spread. A buffered one reads
    // the whole body first and answers in a single write.
    expect(
      delivery.chunks,
      "a relayed stream reaches the caller in several reads",
    ).toBeGreaterThan(1);
    expect(
      delivery.lastByteMs - delivery.firstByteMs,
      "the caller must see the upstream's own delivery spread",
    ).toBeGreaterThan(2 * CHUNK_DELAY_MS);
  });

  test("synthesized speech arrives chunk by chunk", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const call = () =>
      fetch(`${app!.proxyUrl}/v1/audio/speech`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: "relay-tts",
          input: "Streaming recognition emits partial results.",
          voice: "alloy",
        }),
      });

    await waitConfigPropagation(async () => {
      try {
        const probe = await call();
        const ok = probe.ok;
        await probe.text();
        return ok;
      } catch {
        return false;
      }
    });

    const res = await call();
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("audio/mpeg");
    const delivery = await measure(res);

    expect(delivery.body).toBe(SPEECH_CHUNKS.join(""));
    expect(
      delivery.chunks,
      "the audio must reach the caller while the upstream is still writing it",
    ).toBeGreaterThan(1);
    expect(
      delivery.lastByteMs - delivery.firstByteMs,
      "a player must be able to start before the whole file exists",
    ).toBeGreaterThan(CHUNK_DELAY_MS);
  });

  // Relaying the body must not cost the caller the size of the download:
  // an upstream that declares a Content-Length still has it declared
  // downstream, which is what a client needs to show progress.
  test("an upstream Content-Length survives the relay", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const call = () =>
      fetch(`${app!.proxyUrl}/v1/audio/speech`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: "whole-tts",
          input: "Hello from AISIX.",
          voice: "alloy",
        }),
      });

    await waitConfigPropagation(async () => {
      try {
        const probe = await call();
        const ok = probe.ok;
        await probe.text();
        return ok;
      } catch {
        return false;
      }
    });

    const res = await call();
    expect(res.status).toBe(200);
    expect(res.headers.get("content-length")).toBe(
      String(WHOLE_SPEECH_BODY.length),
    );
    expect(res.headers.get("transfer-encoding")).toBeNull();
    expect(await res.text()).toBe(WHOLE_SPEECH_BODY);
  });
});
