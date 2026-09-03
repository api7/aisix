import { createHash, randomUUID } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startMockSls,
  waitConfigPropagation,
  waitForSlsLog,
  type MockSls,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: a transcription response the gateway cannot decode is scanned as a
// lossy copy, so the bytes `from_utf8_lossy` replaced reach the caller
// having been read by nothing.
//
// Two legs, one per failure policy on the response side, because the whole
// point is that the policy decides:
//
//   fail-closed → the transcript is refused, and the caller is told the
//                 guardrail could not evaluate it rather than that it
//                 objected to the content;
//   fail-open   → the transcript is relayed byte-for-byte, and the usage
//                 row says part of it went unread.
//
// The two rows are attached per MODEL rather than to the environment: a
// chain refuses whenever ANY member both reads the response and fails
// closed on it, so an env-wide fail-closed row would decide the fail-open
// leg too.

const KEY = "sk-audio-unscannable-e2e";
const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

const CREDENTIAL_REF = "mock";
const SLS_PROJECT = "aisix-e2e-obs";
const LOGSTORE = "audio-unscannable";

/** The tag the proxy raises for a body it could not read. */
const UNSCANNABLE_TAG = "unscannable_body";

/** The readable head of the transcript, so a relayed body is recognisable
 *  and a refused one is provably absent. */
const TRANSCRIPT_HEAD = "the transcript ends abruptly here";

/** `response_format=text` answers with a bare transcript. `0xC4` opens a
 *  two-byte sequence and `0xE3` is not a continuation byte, so what the
 *  provider sent is not valid UTF-8 — the tail no scan can cover. */
const TRANSCRIPT = Buffer.concat([
  Buffer.from(TRANSCRIPT_HEAD, "utf8"),
  Buffer.from([0xc4, 0xe3, 0xba, 0xc3]),
]);

interface AudioUpstream {
  url: string;
  close(): Promise<void>;
}

async function startAudioUpstream(): Promise<AudioUpstream> {
  const server: Server = createServer((req, res) => {
    req.resume();
    req.on("end", () => {
      if (req.url?.startsWith("/v1/audio/transcriptions")) {
        res.writeHead(200, { "content-type": "text/plain" });
        res.end(TRANSCRIPT);
        return;
      }
      res.writeHead(404, { "content-type": "application/json" });
      res.end("{}");
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = (server.address() as { port: number }).port;
  return {
    url: `http://127.0.0.1:${port}/v1`,
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
}

/** `model` + `response_format=text` + a token audio file. */
function transcriptionForm(model: string): { contentType: string; body: string } {
  return {
    contentType: "multipart/form-data; boundary=b",
    body:
      `--b\r\nContent-Disposition: form-data; name="model"\r\n\r\n${model}\r\n` +
      `--b\r\nContent-Disposition: form-data; name="response_format"\r\n\r\ntext\r\n` +
      `--b\r\nContent-Disposition: form-data; name="file"; filename="a.mp3"\r\n` +
      `Content-Type: audio/mpeg\r\n\r\nID3fakeaudio\r\n--b--\r\n`,
  };
}

describe("an undecodable transcription response follows the response-side failure policy", () => {
  let app: SpawnedApp | undefined;
  let upstream: AudioUpstream | undefined;
  let sls: MockSls | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startAudioUpstream();
    sls = await startMockSls();
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });

    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "sls-audio-unscannable",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
    });

    const pk = await seed.createProviderKey({
      display_name: "audio-unscannable-up",
      secret: "sk-mock-upstream",
      api_base: upstream.url,
    });

    // One model per failure policy, each carrying its own row. The literal
    // is absent from the transcript on purpose: what these rows decide is
    // the undecodable body, not a content match.
    for (const [display, failOpen] of [
      ["unscannable-closed", false],
      ["unscannable-open", true],
    ] as const) {
      const model = await seed.createModel({
        display_name: display,
        provider: "openai",
        model_name: "whisper-1",
        provider_key_id: pk.id,
      });
      const guardrail = await seed.createGuardrail(
        {
          name: `gr-${display}`,
          enabled: true,
          hook_point: "output",
          fail_open: failOpen,
          kind: "keyword",
          patterns: [{ kind: "literal", value: "zzz-never-present-zzz" }],
        },
        { attach: false },
      );
      await etcd.put(
        `${app.etcdPrefix}/guardrail_attachments/${randomUUID()}`,
        JSON.stringify({
          guardrail_id: guardrail.id,
          env_id: randomUUID(),
          scope_type: "model",
          scope_id: model.id,
          priority: 0,
          enabled: true,
        }),
      );
    }

    // Caller key LAST: a key that authenticates implies every row above is
    // already in the snapshot. The gate must not be the behaviour under
    // test, or a regression would read as a propagation timeout.
    await seed.createApiKey({ key_hash: sha256(KEY), allowed_models: ["*"] });
    const proxy = new ProxyClient(app.proxyUrl, KEY);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  }, 90_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await sls?.close();
  });

  const transcribe = (model: string): Promise<Response> => {
    const form = transcriptionForm(model);
    return fetch(`${app!.proxyUrl}/v1/audio/transcriptions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${KEY}`,
        "content-type": form.contentType,
      },
      body: form.body,
    });
  };

  test("a fail-closed row refuses it, naming the evaluation failure", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    const res = await transcribe("unscannable-closed");
    expect(res.status).toBe(422);
    const body = (await res.json()) as {
      error: { type: string; code?: string; message: string };
    };
    expect(body.error.type).toBe("content_filter");
    // Not a content refusal: the caller must be able to tell a working
    // policy from a body the gateway could not put in front of one.
    expect(body.error.code).toBe("guardrail_unavailable");
    expect(body.error.message).toContain(UNSCANNABLE_TAG);
    expect(body.error.message).not.toContain(TRANSCRIPT_HEAD);
  });

  test("a fail-open row relays it verbatim and records the bypass", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();

    const res = await transcribe("unscannable-open");
    expect(res.status).toBe(200);
    const relayed = Buffer.from(await res.arrayBuffer());
    expect(relayed.equals(TRANSCRIPT)).toBe(true);

    // Gated on the requested model, which rides every event this route
    // emits and is independent of the field under test.
    const log = await waitForSlsLog(
      sls,
      LOGSTORE,
      (l) => l.get("requested_model") === "unscannable-open",
      "a usage row for the relayed transcription",
    );
    expect(
      log.get("guardrail_bypassed_reason") ?? "",
      "part of the transcript reached the caller unread and the usage row does not say so",
    ).toBe(UNSCANNABLE_TAG);
  });
});
