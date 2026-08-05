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

// E2E for AISIX-Cloud#1138 / api7/aisix#457: a transcription billed by
// audio length must carry that length off the gateway.
//
// whisper-class models report `usage: {type: "duration", seconds: N}` and
// no token counts at all, so a token-only usage event leaves cp-api with
// nothing to price the request with. And because `response_format=text`
// answers with a body that carries no usage whatsoever, the cost basis
// cannot come from the response alone — otherwise the caller decides
// whether the request is metered by choosing a response format.
//
// Usage telemetry has no cp-api receiver in DP e2e, so the emitted value
// is observed through the per-env OTLP/HTTP fan-out.

const CALLER_PLAINTEXT = "sk-issue-1138-duration";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const REPORTED_SECONDS = 11;
const UPLOADED_SECONDS = 3;

/** `seconds` of 8 kHz 16-bit mono PCM in a minimal RIFF/WAVE container. */
function wavBytes(seconds: number): Uint8Array {
  const sampleRate = 8000;
  const dataLen = sampleRate * 2 * seconds;
  const buf = Buffer.alloc(44 + dataLen);
  buf.write("RIFF", 0, "ascii");
  buf.writeUInt32LE(36 + dataLen, 4);
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
  buf.writeUInt32LE(dataLen, 40);
  return new Uint8Array(buf);
}

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
                  v.stringValue ??
                  String(v.intValue ?? v.doubleValue ?? v.boolValue ?? "");
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

async function waitSpan(
  recv: OtlpReceiver,
  requestId: string,
  timeoutMs = 10_000,
): Promise<Record<string, string>> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const hit = recv.spanAttrs.find((a) => a["aisix.request_id"] === requestId);
    if (hit) return hit;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(`no usage span for request_id=${requestId}`);
}

describe("audio duration cost basis (#1138)", () => {
  let app: SpawnedApp | undefined;
  let jsonUpstream: OpenAiUpstream | undefined;
  let textUpstream: OpenAiUpstream | undefined;
  let otlp: OtlpReceiver | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // One upstream per response shape rather than a scripted sequence:
    // the readiness probe would otherwise consume the first scripted
    // reply and shift every later assertion onto the wrong one.
    jsonUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        text: "hello world",
        usage: { type: "duration", seconds: REPORTED_SECONDS },
      },
    });
    textUpstream = await startOpenAiUpstream({
      rawBody: "hello world",
      rawContentType: "text/plain",
    });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    otlp = await startOtlpReceiver();
    await seed.createObservabilityExporter({
      name: "issue1138-duration-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });

    const jsonPK = await seed.createProviderKey({
      display_name: "issue1138-duration-pk",
      secret: "sk-openai-mock",
      api_base: `${jsonUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "duration-transcribe",
      provider: "openai",
      model_name: "whisper-1",
      provider_key_id: jsonPK.id,
    });
    const textPK = await seed.createProviderKey({
      display_name: "issue1138-text-pk",
      secret: "sk-openai-mock",
      api_base: `${textUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "text-transcribe",
      provider: "openai",
      model_name: "whisper-1",
      provider_key_id: textPK.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["duration-transcribe", "text-transcribe"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await jsonUpstream?.close();
    await textUpstream?.close();
    await otlp?.close();
  });

  test("a duration-billed transcription reports its audio length", async (ctx) => {
    if (!etcdReachable || !app || !otlp) {
      ctx.skip();
      return;
    }
    const proxyUrl = app.proxyUrl;

    const call = (
      model: string,
      audio: Uint8Array,
      extra: Record<string, string> = {},
    ) => {
      const form = new FormData();
      form.set("model", model);
      for (const [k, v] of Object.entries(extra)) form.set(k, v);
      form.set("file", new Blob([audio], { type: "audio/wav" }), "a.wav");
      return fetch(`${proxyUrl}/v1/audio/transcriptions`, {
        method: "POST",
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
        body: form,
      });
    };

    const probe = wavBytes(UPLOADED_SECONDS);
    await waitConfigPropagation(async () => {
      try {
        const [a, b] = await Promise.all([
          call("duration-transcribe", probe),
          call("text-transcribe", probe),
        ]);
        return a.ok && b.ok;
      } catch {
        return false;
      }
    });

    // 1. The upstream reported a duration — that figure wins.
    const reported = await call("duration-transcribe", probe);
    expect(reported.status).toBe(200);
    const reportedSpan = await waitSpan(
      otlp,
      reported.headers.get("x-aisix-request-id") ?? "",
    );
    expect(
      Number(reportedSpan["aisix.audio.duration_seconds"]),
      "the upstream-reported duration is the cost basis",
    ).toBe(REPORTED_SECONDS);

    // 2. `response_format=text` carries no usage at all, so the cost
    //    basis comes off the uploaded audio instead of vanishing.
    const plain = await call("text-transcribe", probe, {
      response_format: "text",
    });
    expect(plain.status).toBe(200);
    const plainSpan = await waitSpan(
      otlp,
      plain.headers.get("x-aisix-request-id") ?? "",
    );
    expect(
      Number(plainSpan["aisix.audio.duration_seconds"]),
      "a response carrying no usage must still be priceable",
    ).toBeCloseTo(UPLOADED_SECONDS, 1);
  });
});
