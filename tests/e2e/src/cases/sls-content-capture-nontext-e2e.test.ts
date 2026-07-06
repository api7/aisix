import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  decodedTextFor,
  EtcdClient,
  spawnApp,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForToken,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// ai-gateway#700 (follow-up to AISIX-Cloud#947): content capture for the
// non-text-generation endpoints, LiteLLM-parity scope — embeddings / rerank /
// images capture the post-redaction request JSON as the prompt and the full
// response JSON (vectors / scores / image url|b64) as the response; audio
// speech captures the input text only (binary response not logged); audio
// transcription represents the file by its sha256 and captures the transcript.

const CALLER_PLAINTEXT = "sk-content-capture-nontext-PLAINTEXT";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const PROVIDER_SECRET = "sk-mock-nontext";

const CREDENTIAL_REF = "mock";
const SLS_PROJECT = "aisix-e2e-obs";
const FULL_LOGSTORE = "full-events-nontext";
const META_LOGSTORE = "meta-events-nontext";

const EMBED_PROMPT_TOKEN = "embed-prompt-tok-11aa22";
const EMBED_RESPONSE_MARK = 0.987654; // survives into the captured response JSON
const RERANK_PROMPT_TOKEN = "rerank-prompt-tok-33bb44";
const RERANK_DOC_TOKEN = "rerank-doc-tok-55cc66";
const RERANK_RESPONSE_MARK = 0.918273;
const IMAGES_PROMPT_TOKEN = "images-prompt-tok-77dd88";
const IMAGES_RESPONSE_URL = "https://img.example/images-response-tok-99ee00.png";
const SPEECH_PROMPT_TOKEN = "speech-prompt-tok-aa11bb";
const TRANSCRIPT_PROMPT_TOKEN = "transcribe-prompt-tok-cc22dd";
const TRANSCRIPT_RESPONSE_TOKEN = "transcribe-response-tok-ee33ff";
const AUDIO_FILE_BYTES = "ID3fakeaudio-nontext";

async function postJson(app: SpawnedApp, path: string, body: unknown): Promise<Response> {
  return fetch(`${app.proxyUrl}${path}`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    },
    body: JSON.stringify(body),
  });
}

describe("sls content capture e2e (#700): embeddings / rerank / images / audio", () => {
  let etcdReachable = false;
  let embedUpstream: OpenAiUpstream | undefined;
  let rerankUpstream: OpenAiUpstream | undefined;
  let imagesUpstream: OpenAiUpstream | undefined;
  let speechUpstream: OpenAiUpstream | undefined;
  let transcribeUpstream: OpenAiUpstream | undefined;
  let sls: MockSls | undefined;
  const apps: SpawnedApp[] = [];

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    embedUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        object: "list",
        data: [{ object: "embedding", index: 0, embedding: [EMBED_RESPONSE_MARK, 0.1] }],
        model: "text-embedding-3-small",
        usage: { prompt_tokens: 4, total_tokens: 4 },
      },
    });
    rerankUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        results: [{ index: 0, relevance_score: RERANK_RESPONSE_MARK }],
        usage: { total_tokens: 6 },
      },
    });
    imagesUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        created: 1_700_000_000,
        data: [{ url: IMAGES_RESPONSE_URL }],
      },
    });
    speechUpstream = await startOpenAiUpstream({
      nonStreamBody: { fake: "binary-audio-placeholder" },
    });
    transcribeUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        text: `the speaker said ${TRANSCRIPT_RESPONSE_TOKEN}`,
        usage: { type: "tokens", input_tokens: 12, output_tokens: 5, total_tokens: 17 },
      },
    });
    sls = await startMockSls();
  });

  afterAll(async () => {
    await Promise.all(apps.map((a) => a.exit()));
    await embedUpstream?.close();
    await rerankUpstream?.close();
    await imagesUpstream?.close();
    await speechUpstream?.close();
    await transcribeUpstream?.close();
    await sls?.close();
  });

  test(
    "full-capture exporter logs request + response content on all four endpoint families",
    async (ctx) => {
      if (
        !etcdReachable ||
        !embedUpstream ||
        !rerankUpstream ||
        !imagesUpstream ||
        !speechUpstream ||
        !transcribeUpstream ||
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
      const admin = new AdminClient(app.adminUrl, app.adminKey);
      await admin.createObservabilityExporter({
        name: "sls-full-nontext",
        enabled: true,
        kind: "aliyun_sls",
        endpoint: sls.url,
        project: SLS_PROJECT,
        logstore: FULL_LOGSTORE,
        credential_ref: CREDENTIAL_REF,
        content_mode: "full",
      });
      await admin.createObservabilityExporter({
        name: "sls-meta-nontext",
        enabled: true,
        kind: "aliyun_sls",
        endpoint: sls.url,
        project: SLS_PROJECT,
        logstore: META_LOGSTORE,
        credential_ref: CREDENTIAL_REF,
        content_mode: "metadata_only",
      });
      const seed = async (name: string, modelName: string, upstream: OpenAiUpstream) => {
        const pk = await admin.createProviderKey({
          display_name: `${name}-pk`,
          secret: PROVIDER_SECRET,
          api_base: `${upstream.baseUrl}/v1`,
        });
        await admin.createModel({
          display_name: name,
          provider: "openai",
          model_name: modelName,
          provider_key_id: pk.id,
        });
      };
      await seed("cc700-embed", "text-embedding-3-small", embedUpstream);
      await seed("cc700-rerank", "gpt-rerank", rerankUpstream);
      await seed("cc700-images", "dall-e-3", imagesUpstream);
      await seed("cc700-speech", "tts-1", speechUpstream);
      await seed("cc700-transcribe", "whisper-1", transcribeUpstream);
      await admin.createApiKey({
        key_hash: CALLER_KEY_HASH,
        allowed_models: [
          "cc700-embed",
          "cc700-rerank",
          "cc700-images",
          "cc700-speech",
          "cc700-transcribe",
        ],
      });

      await waitConfigPropagation(async () => {
        try {
          const r = await postJson(app, "/v1/embeddings", {
            model: "cc700-embed",
            input: "warmup",
          });
          await r.text();
          return r.status === 200;
        } catch {
          return false;
        }
      });

      // -- embeddings: prompt = request JSON, response = vectors JSON --
      const embedRes = await postJson(app, "/v1/embeddings", {
        model: "cc700-embed",
        input: `embed the token ${EMBED_PROMPT_TOKEN}`,
      });
      expect(embedRes.status).toBe(200);
      await embedRes.text();
      await waitForToken(sls, FULL_LOGSTORE, EMBED_PROMPT_TOKEN);
      let fullText = decodedTextFor(sls, FULL_LOGSTORE);
      expect(fullText).toContain(EMBED_PROMPT_TOKEN);
      expect(fullText).toContain(String(EMBED_RESPONSE_MARK)); // vector captured

      // -- rerank: prompt = query + documents, response = scores JSON --
      const rerankRes = await postJson(app, "/v1/rerank", {
        model: "cc700-rerank",
        query: `find ${RERANK_PROMPT_TOKEN}`,
        documents: [`doc about ${RERANK_DOC_TOKEN}`],
      });
      expect(rerankRes.status).toBe(200);
      await rerankRes.text();
      await waitForToken(sls, FULL_LOGSTORE, RERANK_PROMPT_TOKEN);
      fullText = decodedTextFor(sls, FULL_LOGSTORE);
      expect(fullText).toContain(RERANK_PROMPT_TOKEN);
      expect(fullText).toContain(RERANK_DOC_TOKEN);
      expect(fullText).toContain(String(RERANK_RESPONSE_MARK));

      // -- images: prompt captured, response url captured --
      const imagesRes = await postJson(app, "/v1/images/generations", {
        model: "cc700-images",
        prompt: `paint ${IMAGES_PROMPT_TOKEN}`,
      });
      expect(imagesRes.status).toBe(200);
      await imagesRes.text();
      await waitForToken(sls, FULL_LOGSTORE, IMAGES_PROMPT_TOKEN);
      fullText = decodedTextFor(sls, FULL_LOGSTORE);
      expect(fullText).toContain(IMAGES_PROMPT_TOKEN);
      expect(fullText).toContain(IMAGES_RESPONSE_URL);

      // -- audio speech: input text captured; binary response not logged --
      const speechRes = await postJson(app, "/v1/audio/speech", {
        model: "cc700-speech",
        input: `say ${SPEECH_PROMPT_TOKEN}`,
        voice: "alloy",
      });
      expect(speechRes.status).toBe(200);
      await speechRes.arrayBuffer();
      await waitForToken(sls, FULL_LOGSTORE, SPEECH_PROMPT_TOKEN);

      // -- audio transcription: file → sha256, prompt field + transcript --
      const form = new FormData();
      form.set("model", "cc700-transcribe");
      form.set("prompt", `context: ${TRANSCRIPT_PROMPT_TOKEN}`);
      form.set("file", new Blob([AUDIO_FILE_BYTES], { type: "audio/mpeg" }), "a.mp3");
      const trRes = await fetch(`${app.proxyUrl}/v1/audio/transcriptions`, {
        method: "POST",
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
        body: form,
      });
      expect(trRes.status).toBe(200);
      await trRes.text();
      await waitForToken(sls, FULL_LOGSTORE, TRANSCRIPT_PROMPT_TOKEN);
      fullText = decodedTextFor(sls, FULL_LOGSTORE);
      expect(fullText).toContain(TRANSCRIPT_PROMPT_TOKEN);
      expect(fullText).toContain(TRANSCRIPT_RESPONSE_TOKEN);
      const expectedSha = createHash("sha256").update(AUDIO_FILE_BYTES).digest("hex");
      expect(fullText).toContain(expectedSha); // file represented by checksum
      expect(fullText).not.toContain(AUDIO_FILE_BYTES); // never the raw bytes

      // -- metadata_only exporter received none of the content --
      // Wait until the LAST request's metadata (its requested_model, which
      // both content modes carry) has landed in the meta logstore, so the
      // negative assertions below can't pass trivially against an empty or
      // lagging store.
      await waitForToken(sls, META_LOGSTORE, "cc700-transcribe");
      const metaText = decodedTextFor(sls, META_LOGSTORE);
      for (const token of [
        EMBED_PROMPT_TOKEN,
        RERANK_PROMPT_TOKEN,
        RERANK_DOC_TOKEN,
        IMAGES_PROMPT_TOKEN,
        SPEECH_PROMPT_TOKEN,
        TRANSCRIPT_PROMPT_TOKEN,
        TRANSCRIPT_RESPONSE_TOKEN,
      ]) {
        expect(metaText).not.toContain(token);
      }
    },
    120_000,
  );
});
