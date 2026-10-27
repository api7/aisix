import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  pickFreePort,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// The audio length a streamed transcription is priced on.
//
// Two ways it went missing on `stream=true`:
//
// - The upstream reports the length itself — a duration-billed model puts
//   `usage: {type: "duration", seconds: N}` on its terminal
//   `transcript.text.done` event — and the gateway ignored it, pricing the
//   request on the uploaded file instead. What the upstream reports wins,
//   as it already did for a non-streamed transcription.
// - The upload is a WAV written by a streaming encoder, whose RIFF and
//   `data` sizes are the `0xFFFFFFFF` placeholder. That is exactly what
//   OpenAI's own `/v1/audio/speech` returns for `response_format=wav`, and
//   the gateway measured it as no audio at all.
//
// Observed through the per-env OTLP/HTTP fan-out, like the other audio
// usage specs.

const CALLER_PLAINTEXT = "sk-audio-stream-duration";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const REPORTED_SECONDS = 11;
const UPLOADED_SECONDS = 3;

/**
 * `seconds` of 24 kHz 16-bit mono PCM in a RIFF/WAVE container whose sizes
 * are written as a streaming encoder writes them: `sized=false` puts the
 * `0xFFFFFFFF` placeholder in both the RIFF and the `data` size.
 */
function wavBytes(seconds: number, sized: boolean): Uint8Array {
  const sampleRate = 24_000;
  const dataLen = sampleRate * 2 * seconds;
  const buf = Buffer.alloc(44 + dataLen);
  buf.write("RIFF", 0, "ascii");
  buf.writeUInt32LE(sized ? 36 + dataLen : 0xffffffff, 4);
  buf.write("WAVE", 8, "ascii");
  buf.write("fmt ", 12, "ascii");
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20); // PCM
  buf.writeUInt16LE(1, 22); // mono
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * 2, 28);
  buf.writeUInt16LE(2, 32);
  buf.writeUInt16LE(16, 34);
  buf.write("data", 36, "ascii");
  buf.writeUInt32LE(sized ? dataLen : 0xffffffff, 40);
  return new Uint8Array(buf);
}

/**
 * A FLAC header declaring 2^36-1 samples at 1 Hz — about 2,000 years of
 * audio, far beyond what the control plane's duration column can hold.
 */
function absurdFlacBytes(): Uint8Array {
  return new Uint8Array([
    ...Buffer.from("fLaC"),
    0x80, 0, 0, 34, // last metadata block: STREAMINFO
    0x10, 0x00, 0x10, 0x00, // block size 4096..4096
    0, 0, 0, 0, 0, 0, // frame sizes unknown
    0x00, 0x00, 0x10, 0xff, 0xff, 0xff, 0xff, 0xff, // 1 Hz, mono, 16-bit, 2^36-1 samples
    ...new Array(16).fill(0), // MD5
    0xff, 0xf8, 0x69, 0x08, 0, 0, 0, 0,
  ]);
}

// OpenAI frames this stream with CRLF.
const FRAME_END = "\r\n\r\n";
function transcriptSse(usage: Record<string, unknown>): string {
  return [
    `data: ${JSON.stringify({ type: "transcript.text.delta", delta: "hello" })}`,
    `data: ${JSON.stringify({ type: "transcript.text.done", text: "hello", usage })}`,
  ]
    .join(FRAME_END)
    .concat(FRAME_END);
}

interface OtlpReceiver {
  url: string;
  spanAttrs: Array<Record<string, string>>;
  /** Exports that did not parse — answered 400 and surfaced by `waitSpan`. */
  errors: string[];
  close(): Promise<void>;
}

async function startOtlpReceiver(): Promise<OtlpReceiver> {
  const spanAttrs: Array<Record<string, string>> = [];
  const errors: string[] = [];
  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      try {
        const body = JSON.parse(raw);
        for (const rs of body.resourceSpans ?? []) {
          for (const ss of rs.scopeSpans ?? []) {
            for (const span of ss.spans ?? []) {
              const attrs: Record<string, string> = {};
              for (const a of span.attributes ?? []) {
                const v = a.value ?? {};
                attrs[a.key] =
                  v.stringValue ??
                  String(v.intValue ?? v.doubleValue ?? v.boolValue ?? "");
              }
              spanAttrs.push(attrs);
            }
          }
        }
      } catch (err) {
        errors.push(String(err));
        res.statusCode = 400;
        res.end("{}");
        return;
      }
      res.statusCode = 200;
      res.end("{}");
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    url: `http://127.0.0.1:${port}/v1/traces`,
    spanAttrs,
    errors,
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

async function waitSpan(
  recv: OtlpReceiver,
  requestId: string,
  timeoutMs = 10_000,
): Promise<Record<string, string>> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (recv.errors.length > 0) {
      throw new Error(`malformed OTLP export: ${recv.errors[0]}`);
    }
    const hit = recv.spanAttrs.find(
      (a) =>
        a["aisix.request_id"] === requestId &&
        a["aisix.attempt_index"] !== undefined,
    );
    if (hit) return hit;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(`no usage span for request_id=${requestId}`);
}

describe("streamed transcription duration cost basis", () => {
  let app: SpawnedApp | undefined;
  let tokensUp: OpenAiUpstream | undefined;
  let durationUp: OpenAiUpstream | undefined;
  let absurdUp: OpenAiUpstream | undefined;
  let otlp: OtlpReceiver | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    tokensUp = await startOpenAiUpstream({
      rawBody: transcriptSse({
        type: "tokens",
        input_tokens: 20,
        output_tokens: 2,
        total_tokens: 22,
      }),
      rawContentType: "text/event-stream",
    });
    durationUp = await startOpenAiUpstream({
      rawBody: transcriptSse({ type: "duration", seconds: REPORTED_SECONDS }),
      rawContentType: "text/event-stream",
    });
    absurdUp = await startOpenAiUpstream({
      nonStreamBody: { text: "hello", usage: { type: "duration", seconds: 4e9 } },
    });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    otlp = await startOtlpReceiver();
    await seed.createObservabilityExporter({
      name: "stream-duration-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });
    const model = async (name: string, up: OpenAiUpstream, upstreamModel: string) => {
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        secret: "sk-openai-mock",
        api_base: `${up.baseUrl}/v1`,
      });
      await seed.createModel({
        display_name: name,
        provider: "openai",
        model_name: upstreamModel,
        provider_key_id: pk.id,
      });
    };
    await model("stream-tokens", tokensUp, "gpt-4o-transcribe");
    await model("stream-duration", durationUp, "whisper-1");
    await model("absurd-duration", absurdUp, "whisper-1");
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  }, 90_000);

  afterAll(async () => {
    await app?.exit();
    await tokensUp?.close();
    await durationUp?.close();
    await absurdUp?.close();
    await otlp?.close();
  });

  const transcribe = async (
    model: string,
    audio: Uint8Array,
    stream: boolean,
    expectBody = "transcript.text.done",
  ) => {
    const form = new FormData();
    form.set("model", model);
    if (stream) form.set("stream", "true");
    form.set("file", new Blob([audio], { type: "audio/wav" }), "a.wav");
    const resp = await fetch(`${app!.proxyUrl}/v1/audio/transcriptions`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      body: form,
    });
    expect(resp.status).toBe(200);
    expect(await resp.text()).toContain(expectBody);
    const span = await waitSpan(otlp!, resp.headers.get("x-aisix-request-id") ?? "");
    return Number(span["aisix.audio.duration_seconds"] ?? 0);
  };

  test("a streaming-encoder WAV upload is measured to its end", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    // The terminal event carries tokens only, so the upload is the basis.
    expect(
      await transcribe("stream-tokens", wavBytes(UPLOADED_SECONDS, false), true),
      "a WAV whose sizes are the streaming placeholder still has a length",
    ).toBeCloseTo(UPLOADED_SECONDS, 3);
  });

  test("the terminal event's reported duration is the cost basis", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const upload = wavBytes(UPLOADED_SECONDS, true);
    expect(
      await transcribe("stream-duration", upload, true),
      "a relayed stream prices what the upstream reported, not the upload",
    ).toBe(REPORTED_SECONDS);
    // The same stream read whole (the caller did not ask to stream).
    expect(
      await transcribe("stream-duration", upload, false),
      "a buffered stream prices what the upstream reported, not the upload",
    ).toBe(REPORTED_SECONDS);
  });

  // The control plane stores the duration as numeric(12,3); one it cannot
  // hold would fail the whole usage batch, so it is reported as unmeasured.
  test("a duration the control plane cannot store is not measured", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    expect(
      await transcribe("stream-tokens", absurdFlacBytes(), true),
      "a header declaring an impossible length",
    ).toBe(0);
    expect(
      await transcribe("absurd-duration", wavBytes(UPLOADED_SECONDS, true), false, "hello"),
      "an upstream reporting an impossible length",
    ).toBe(0);
  });
});
