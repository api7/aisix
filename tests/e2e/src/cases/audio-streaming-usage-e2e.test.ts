import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  pickFreePort,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E regression for AISIX-Cloud#1138: a `stream=true` transcription must
// be billed for the tokens the upstream reports.
//
// The transcribe models answer a streaming transcription with
// `text/event-stream`: `transcript.text.delta` events and a terminal
// `transcript.text.done` carrying the same `usage` block the
// non-streaming response would have returned. Pre-fix the audio handler
// only looked for a JSON object, so the whole streaming surface emitted
// zero tokens — unbilled spend that also never moved TPM/TPD, while the
// identical non-streaming request billed normally.
//
// Usage telemetry has no cp-api receiver in DP e2e, so — like the
// /v1/responses streaming test (#808) — the emitted values are observed
// through the per-env OTLP/HTTP fan-out.

const CALLER_PLAINTEXT = "sk-issue-1138-audio-stream";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const INPUT_TOKENS = 26;
const OUTPUT_TOKENS = 12;

// Real streaming-transcription wire shape.
// CRLF, not LF: that is how OpenAI frames this stream, and a decoder
// that only knows `\n\n` unlocks no event at all on it — which is how a
// streamed transcription came to bill zero tokens against the real
// provider while this spec passed against an LF mock (#998 follow-up).
const FRAME_END = "\r\n\r\n";
const TRANSCRIPT_SSE = [
  `data: ${JSON.stringify({ type: "transcript.text.delta", delta: "hello" })}`,
  `data: ${JSON.stringify({ type: "transcript.text.delta", delta: " world" })}`,
  `data: ${JSON.stringify({
    type: "transcript.text.done",
    text: "hello world",
    usage: {
      type: "tokens",
      input_tokens: INPUT_TOKENS,
      output_tokens: OUTPUT_TOKENS,
      total_tokens: INPUT_TOKENS + OUTPUT_TOKENS,
      input_token_details: { text_tokens: 0, audio_tokens: INPUT_TOKENS },
    },
  })}`,
  "data: [DONE]",
].join(FRAME_END).concat(FRAME_END);

interface OtlpReceiver {
  url: string;
  spanAttrs: Array<Record<string, string>>;
  close(): Promise<void>;
}

async function startOtlpReceiver(): Promise<OtlpReceiver> {
  const spanAttrs: Array<Record<string, string>> = [];
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
                  v.stringValue ?? String(v.intValue ?? v.boolValue ?? "");
              }
              spanAttrs.push(attrs);
            }
          }
        }
      } catch {
        // ignore malformed bodies — assertions fail on missing spans
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
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

async function waitUsageSpan(
  recv: OtlpReceiver,
  requestId: string,
  timeoutMs = 10_000,
): Promise<Record<string, string>> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const hit = recv.spanAttrs.find(
      (a) =>
        a["aisix.request_id"] === requestId &&
        a["gen_ai.usage.input_tokens"] !== undefined,
    );
    if (hit) return hit;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(`no usage span for request_id=${requestId}`);
}

describe("streaming transcription usage emission (#1138)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let otlp: OtlpReceiver | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // The upload is multipart, so the upstream can't be asked to stream
    // off a `stream: true` JSON field — serve the SSE bytes as the raw
    // 200 body with the streaming content type instead.
    upstream = await startOpenAiUpstream({
      rawBody: TRANSCRIPT_SSE,
      rawContentType: "text/event-stream",
    });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    otlp = await startOtlpReceiver();
    await seed.createObservabilityExporter({
      name: "issue1138-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });

    const pk = await seed.createProviderKey({
      display_name: "issue1138-pk",
      secret: "sk-openai-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "stream-transcribe",
      provider: "openai",
      model_name: "gpt-4o-transcribe",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["stream-transcribe"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await otlp?.close();
  });

  test("a streamed transcription bills the terminal event's tokens", async (ctx) => {
    if (!etcdReachable || !app || !upstream || !otlp) {
      ctx.skip();
      return;
    }
    const proxyUrl = app.proxyUrl;

    const call = () => {
      const form = new FormData();
      form.set("model", "stream-transcribe");
      form.set("stream", "true");
      form.set(
        "file",
        new Blob([new Uint8Array([0x49, 0x44, 0x33])], { type: "audio/mpeg" }),
        "a.mp3",
      );
      return fetch(`${proxyUrl}/v1/audio/transcriptions`, {
        method: "POST",
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
        body: form,
      });
    };

    await waitConfigPropagation(async () => {
      try {
        return (await call()).ok;
      } catch {
        return false;
      }
    });

    const resp = await call();
    expect(resp.status).toBe(200);
    expect(resp.headers.get("content-type")).toContain("text/event-stream");
    const requestId = resp.headers.get("x-aisix-request-id") ?? "";
    expect(requestId, "the DP must stamp a request id").not.toBe("");

    // The caller still gets the upstream's stream verbatim.
    const body = await resp.text();
    expect(body).toContain("transcript.text.delta");
    expect(body).toContain("transcript.text.done");

    const span = await waitUsageSpan(otlp, requestId);
    expect(
      Number(span["gen_ai.usage.input_tokens"]),
      "the terminal transcript.text.done usage must be billed",
    ).toBe(INPUT_TOKENS);
    expect(Number(span["gen_ai.usage.output_tokens"])).toBe(OUTPUT_TOKENS);
  });
});
