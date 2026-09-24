import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForSlsLog,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// AISIX-Cloud#1461: a customer auditing traffic in SLS asked how to tell a
// text request from an image or a video one. Nothing on the event answered
// it — every OpenAI-shaped route reports `inbound_protocol=openai`, and the
// only remaining discriminator was a regex over a captured prompt, which the
// exporter this suite configures (`content_mode = metadata_only`) does not
// have and which a video submission (zero tokens, no captured content) would
// not answer anyway.
//
// So the whole suite runs against a metadata_only exporter deliberately: that
// is the customer's configuration, and it is the one where the field has to
// stand on its own.
//
// The assertions that matter are the ones a coarser label cannot make:
//   - /v1/chat/completions vs /v1/responses vs /v1/messages — three routes
//     that used to be one undifferentiated `openai` stream;
//   - /v1/images/generations vs /v1/images/edits — one handler, one metric
//     label, two operations;
//   - POST /v1/videos — the zero-token, no-content event the customer could
//     not tell apart from any other zero-token call;
//   - a REFUSED request — the operation says what was asked for, so it is on
//     the row whether or not anything came back.

const CALLER_PLAINTEXT = "sk-usage-operation-1461-PLAINTEXT";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const PROVIDER_SECRET = "sk-mock-usage-operation";

const CREDENTIAL_REF = "mock";
const SLS_PROJECT = "aisix-e2e-obs";
const META_LOGSTORE = "operation-events";

/** Each request carries its own model alias, which is what joins its SLS row. */
const CHAT_MODEL = "op1461-chat";
const RESPONSES_MODEL = "op1461-responses";
const IMAGES_MODEL = "op1461-images";
const SPEECH_MODEL = "op1461-speech";
/** Serves BOTH audio-upload routes, so their rows differ only by the field under test. */
const AUDIO_IN_MODEL = "op1461-audio-in";
const VIDEO_MODEL = "op1461-video";

function chatBody(text: string) {
  return {
    id: "chatcmpl-op1461",
    object: "chat.completion",
    created: 1_700_000_000,
    model: "mock-model",
    choices: [
      { index: 0, message: { role: "assistant", content: text }, finish_reason: "stop" },
    ],
    usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
  };
}

function responsesBody(text: string) {
  return {
    id: "resp_op1461",
    object: "response",
    status: "completed",
    model: "mock-model",
    output: [
      {
        type: "message",
        id: "msg_op1461",
        role: "assistant",
        content: [{ type: "output_text", text }],
      },
    ],
    usage: { input_tokens: 5, output_tokens: 3, total_tokens: 8 },
  };
}

async function postJson(app: SpawnedApp, path: string, body: unknown): Promise<Response> {
  const res = await fetch(`${app.proxyUrl}${path}`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    },
    body: JSON.stringify(body),
  });
  // Drain so the DP sees the response consumed before the next request.
  await res.arrayBuffer();
  return res;
}

/**
 * The row this request produced, found by its model alias.
 *
 * Keyed on `requested_model` rather than on the operation being asserted —
 * looking the row up by the value under test would let a run where every row
 * carries the same wrong operation still find "a" row.
 */
async function rowFor(sls: MockSls, model: string): Promise<Map<string, string>> {
  return waitForSlsLog(
    sls,
    META_LOGSTORE,
    (log) => log.get("requested_model") === model,
    `a row for requested_model=${model}`,
  );
}

/**
 * The row one specific call produced, found by the id the gateway echoed in
 * `x-aisix-request-id`.
 *
 * The paired routes below — the two image routes, the two audio-upload
 * routes — are driven against ONE model each, so `requested_model` cannot
 * separate them and a set assertion over both rows cannot either: swapping
 * the two constants at their emit sites leaves the same set. Reading each
 * call's own row is what makes the pair meaningful, and the shared model
 * still guarantees that nothing but the field under test distinguishes them.
 */
async function rowForRequest(sls: MockSls, requestId: string): Promise<Map<string, string>> {
  expect(requestId).not.toBe("");
  return waitForSlsLog(
    sls,
    META_LOGSTORE,
    (log) => log.get("request_id") === requestId,
    `a row for request_id=${requestId}`,
  );
}

/** POST a multipart form and hand back the gateway's request id for the call. */
async function postForm(
  app: SpawnedApp,
  path: string,
  parts: Record<string, string | Blob>,
): Promise<string> {
  const form = new FormData();
  for (const [name, value] of Object.entries(parts)) {
    if (value instanceof Blob) form.set(name, value, "a.bin");
    else form.set(name, value);
  }
  const res = await fetch(`${app.proxyUrl}${path}`, {
    method: "POST",
    headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
    body: form,
  });
  await res.arrayBuffer();
  expect(res.status).toBe(200);
  return res.headers.get("x-aisix-request-id") ?? "";
}

describe("usage operation e2e (AISIX-Cloud#1461)", () => {
  let etcdReachable = false;
  let chatUpstream: OpenAiUpstream | undefined;
  let responsesUpstream: OpenAiUpstream | undefined;
  let imagesUpstream: OpenAiUpstream | undefined;
  let speechUpstream: OpenAiUpstream | undefined;
  let audioInUpstream: OpenAiUpstream | undefined;
  let videoUpstream: OpenAiUpstream | undefined;
  let sls: MockSls | undefined;
  const apps: SpawnedApp[] = [];

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    chatUpstream = await startOpenAiUpstream({ nonStreamBody: chatBody("hello") });
    responsesUpstream = await startOpenAiUpstream({ nonStreamBody: responsesBody("hello") });
    imagesUpstream = await startOpenAiUpstream({
      nonStreamBody: { created: 1_700_000_000, data: [{ url: "https://img.example/a.png" }] },
    });
    speechUpstream = await startOpenAiUpstream({
      nonStreamBody: { fake: "binary-audio-placeholder" },
    });
    audioInUpstream = await startOpenAiUpstream({
      nonStreamBody: { text: "the speaker said something" },
    });
    videoUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        output: { task_id: "task-op1461", task_status: "PENDING" },
        request_id: "req-op1461",
      },
    });
    sls = await startMockSls();
  });

  afterAll(async () => {
    await Promise.all(apps.map((a) => a.exit()));
    await chatUpstream?.close();
    await responsesUpstream?.close();
    await imagesUpstream?.close();
    await speechUpstream?.close();
    await audioInUpstream?.close();
    await videoUpstream?.close();
    await sls?.close();
  });

  test(
    "every endpoint family reports its own operation, with no content captured",
    async (ctx) => {
      if (
        !etcdReachable ||
        !chatUpstream ||
        !responsesUpstream ||
        !imagesUpstream ||
        !speechUpstream ||
        !audioInUpstream ||
        !videoUpstream ||
        !sls
      ) {
        ctx.skip();
        return;
      }
      const app = await spawnApp({
        extraEnv: {
          [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
          [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
        },
      });
      apps.push(app);
      const seed = new SeedClient(new EtcdClient(), app.etcdPrefix);
      await seed.createObservabilityExporter({
        name: "sls-operation",
        enabled: true,
        kind: "aliyun_sls",
        endpoint: sls.url,
        project: SLS_PROJECT,
        logstore: META_LOGSTORE,
        credential_ref: CREDENTIAL_REF,
        content_mode: "metadata_only",
      });

      const seedModel = async (
        name: string,
        modelName: string,
        upstream: OpenAiUpstream,
        provider = "openai",
        apiBase = `${upstream.baseUrl}/v1`,
      ) => {
        const pk = await seed.createProviderKey({
          display_name: `${name}-pk`,
          secret: PROVIDER_SECRET,
          api_base: apiBase,
          provider,
        });
        await seed.createModel({
          display_name: name,
          provider,
          model_name: modelName,
          provider_key_id: pk.id,
        });
      };
      await seedModel(CHAT_MODEL, "gpt-4o-mini", chatUpstream);
      await seedModel(RESPONSES_MODEL, "gpt-4o-mini", responsesUpstream);
      await seedModel(IMAGES_MODEL, "dall-e-3", imagesUpstream);
      await seedModel(SPEECH_MODEL, "tts-1", speechUpstream);
      await seedModel(AUDIO_IN_MODEL, "whisper-1", audioInUpstream);
      // Video submission speaks the Alibaba task API, whose base URL carries
      // no `/v1` suffix (see videos-e2e).
      await seedModel(
        VIDEO_MODEL,
        "wan-mock",
        videoUpstream,
        "alibaba",
        videoUpstream.baseUrl,
      );
      await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });

      // The caller key is seeded last, so it authenticating implies every
      // model and the exporter above are already in the snapshot. Per
      // tests/e2e/AGENTS.md the gate must not drive the behavior under test
      // (a broken operation would then surface as a 30s propagation timeout
      // instead of an assertion) and must not swallow errors — `listModels`
      // returns a status rather than throwing, so a transport failure cannot
      // be mistaken for "not ready yet".
      const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
      await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);

      // --- the three routes that used to be one `openai` stream ----------
      expect(
        (
          await postJson(app, "/v1/chat/completions", {
            model: CHAT_MODEL,
            messages: [{ role: "user", content: "hi" }],
          })
        ).status,
      ).toBe(200);

      expect(
        (await postJson(app, "/v1/responses", { model: RESPONSES_MODEL, input: "hi" })).status,
      ).toBe(200);

      // --- one handler label, two operations ------------------------------
      const generationsRes = await postJson(app, "/v1/images/generations", {
        model: IMAGES_MODEL,
        prompt: "a cat",
      });
      expect(generationsRes.status).toBe(200);
      const generationsId = generationsRes.headers.get("x-aisix-request-id") ?? "";

      const editsId = await postForm(app, "/v1/images/edits", {
        model: IMAGES_MODEL,
        prompt: "make it blue",
        image: new Blob(["fake-png-bytes"], { type: "image/png" }),
      });

      // --- audio out, and the zero-token video submission -----------------
      expect(
        (
          await postJson(app, "/v1/audio/speech", {
            model: SPEECH_MODEL,
            input: "say something",
            voice: "alloy",
          })
        ).status,
      ).toBe(200);

      expect(
        (await postJson(app, "/v1/videos", { model: VIDEO_MODEL, prompt: "a river" })).status,
      ).toBe(200);

      // Transcription and translation share a handler, an emitter and — here —
      // a model. Both are driven because the in-crate census only reaches
      // their REFUSED path, which is a different call site from the success
      // one: swapping the two constants where they succeed is invisible to
      // every other test.
      const audioFile = () => new Blob(["ID3fake-audio"], { type: "audio/mpeg" });
      const transcriptionId = await postForm(app, "/v1/audio/transcriptions", {
        model: AUDIO_IN_MODEL,
        file: audioFile(),
      });
      const translationId = await postForm(app, "/v1/audio/translations", {
        model: AUDIO_IN_MODEL,
        file: audioFile(),
      });

      const chatRow = await rowFor(sls, CHAT_MODEL);
      const responsesRow = await rowFor(sls, RESPONSES_MODEL);
      const speechRow = await rowFor(sls, SPEECH_MODEL);
      const videoRow = await rowFor(sls, VIDEO_MODEL);

      expect(chatRow.get("operation")).toBe("chat");
      expect(responsesRow.get("operation")).toBe("responses");
      expect(speechRow.get("operation")).toBe("speech");
      expect(videoRow.get("operation")).toBe("video_generation");

      // The video row is the one the customer could not identify at all: no
      // tokens and no captured content, so the operation is its only
      // description.
      expect(videoRow.get("prompt_tokens") ?? "0").toBe("0");
      expect(videoRow.has("prompt")).toBe(false);

      // The paired routes, each read by its own call's id — see rowForRequest.
      for (const [requestId, expected] of [
        [generationsId, "image_generation"],
        [editsId, "image_edit"],
        [transcriptionId, "transcription"],
        [translationId, "translation"],
      ] as const) {
        expect((await rowForRequest(sls, requestId)).get("operation")).toBe(expected);
      }

      // `inbound_protocol` cannot make any of these distinctions: it is the
      // same value on all of them, which is the reason the field exists.
      for (const row of [chatRow, responsesRow, speechRow]) {
        expect(row.get("inbound_protocol")).toBe("openai");
      }

      // metadata_only means metadata only — the classification rode along
      // without turning the exporter into a content sink.
      for (const row of [chatRow, responsesRow, speechRow, videoRow]) {
        expect(row.has("prompt")).toBe(false);
        expect(row.has("response")).toBe(false);
      }

      // --- a refusal reports what was ASKED for --------------------------
      // An unknown model is refused before dispatch, so this row has no
      // tokens, no provider and no response. The operation is the only thing
      // that says which endpoint was refused — and it is on the row because
      // it describes the request, not the outcome.
      const denied = "op1461-no-such-model";
      const deniedRes = await postJson(app, "/v1/embeddings", {
        model: denied,
        input: "x",
      });
      expect(deniedRes.status).toBeGreaterThanOrEqual(400);
      const deniedRow = await rowFor(sls, denied);
      expect(deniedRow.get("operation")).toBe("embeddings");
      expect(Number(deniedRow.get("status_code"))).toBeGreaterThanOrEqual(400);
    },
    180_000,
  );

});
