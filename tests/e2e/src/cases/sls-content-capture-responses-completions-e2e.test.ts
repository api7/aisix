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
  waitForLogstore,
  waitForToken,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// AISIX-Cloud#947: a `content_mode = full` exporter must capture the request
// prompt + response for /v1/responses and /v1/completions too — pre-fix those
// handlers fanned out with `content: None`, so SLS received metadata only even
// with "full (prompt + response)" selected in the console. This suite pins all
// four /v1/responses paths that carry content (verbatim non-streaming,
// verbatim streaming, cross-provider bridge non-streaming + streaming) plus
// /v1/completions, and that a `metadata_only` exporter still receives none.

const CALLER_PLAINTEXT = "sk-content-capture-947-PLAINTEXT";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const PROVIDER_SECRET = "sk-mock-content-capture-947";

const CREDENTIAL_REF = "mock";
const MOCK_AK_ID = "mock-akid";
const MOCK_AK_SECRET = "mock-secret";
const SLS_PROJECT = "aisix-e2e-obs";
const FULL_LOGSTORE = "full-events-947";
const META_LOGSTORE = "meta-events-947";

// Unique tokens per scenario, planted in the request input and the mock
// upstream's output text.
const RESP_PROMPT_TOKEN = "responses-prompt-tok-1a2b3c";
const RESP_RESPONSE_TOKEN = "responses-response-tok-4d5e6f";
const RESP_STREAM_PROMPT_TOKEN = "responses-stream-prompt-tok-7a8b9c";
const RESP_STREAM_RESPONSE_TOKEN = "responses-stream-response-tok-0d1e2f";
const BRIDGE_PROMPT_TOKEN = "responses-bridge-prompt-tok-3a4b5c";
const BRIDGE_RESPONSE_TOKEN = "responses-bridge-response-tok-6d7e8f";
const BRIDGE_STREAM_PROMPT_TOKEN = "responses-bridge-stream-prompt-tok-9a0b1c";
const BRIDGE_STREAM_RESPONSE_TOKEN = "responses-bridge-stream-response-tok-2d3e4f";
const COMPLETIONS_PROMPT_TOKEN = "completions-prompt-tok-5a6b7c";
const COMPLETIONS_RESPONSE_TOKEN = "completions-response-tok-8d9e0f";

/** Responses-API non-streaming body carrying the response token. */
function responsesBody(text: string) {
  return {
    id: "resp_mock947",
    object: "response",
    status: "completed",
    model: "mock-model",
    output: [
      {
        type: "message",
        id: "msg_mock947",
        role: "assistant",
        content: [{ type: "output_text", text }],
      },
    ],
    usage: { input_tokens: 5, output_tokens: 3, total_tokens: 8 },
  };
}

/** Responses-API SSE events: the response token split across two deltas plus
 * the terminal `response.completed` carrying the full output + usage. */
function responsesStreamEvents(): string[] {
  const half = Math.floor(RESP_STREAM_RESPONSE_TOKEN.length / 2);
  return [
    JSON.stringify({
      type: "response.output_text.delta",
      delta: `streamed ${RESP_STREAM_RESPONSE_TOKEN.slice(0, half)}`,
    }),
    JSON.stringify({
      type: "response.output_text.delta",
      delta: RESP_STREAM_RESPONSE_TOKEN.slice(half),
    }),
    JSON.stringify({
      type: "response.completed",
      response: responsesBody(`streamed ${RESP_STREAM_RESPONSE_TOKEN}`),
    }),
    "[DONE]",
  ];
}

/** OpenAI chat-completions chunks for the cross-provider bridge stream. */
function bridgeStreamEvents(): string[] {
  return [
    JSON.stringify({
      id: "mock-bridge-1",
      object: "chat.completion.chunk",
      model: "mock-model",
      choices: [{ index: 0, delta: { role: "assistant" } }],
    }),
    JSON.stringify({
      id: "mock-bridge-1",
      object: "chat.completion.chunk",
      model: "mock-model",
      choices: [{ index: 0, delta: { content: `bridged ${BRIDGE_STREAM_RESPONSE_TOKEN}` } }],
    }),
    JSON.stringify({
      id: "mock-bridge-1",
      object: "chat.completion.chunk",
      model: "mock-model",
      choices: [{ index: 0, delta: { content: " done" }, finish_reason: "stop" }],
      usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
    }),
    "[DONE]",
  ];
}

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

describe("sls content capture e2e (AISIX-Cloud#947): /v1/responses + /v1/completions", () => {
  let etcdReachable = false;
  let responsesUpstream: OpenAiUpstream | undefined;
  let responsesStreamUpstream: OpenAiUpstream | undefined;
  let bridgeUpstream: OpenAiUpstream | undefined;
  let bridgeStreamUpstream: OpenAiUpstream | undefined;
  let completionsUpstream: OpenAiUpstream | undefined;
  let sls: MockSls | undefined;
  const apps: SpawnedApp[] = [];

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    responsesUpstream = await startOpenAiUpstream({
      nonStreamBody: responsesBody(`sure, ${RESP_RESPONSE_TOKEN} noted`),
    });
    responsesStreamUpstream = await startOpenAiUpstream({
      streamEvents: responsesStreamEvents(),
    });
    // Cross-provider bridge (non-openai provider): the DP translates the
    // Responses request into ChatFormat and calls the OpenAI-compatible
    // family bridge, so the mock serves chat-completions wire shapes.
    bridgeUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "mock-bridge-nonstream",
        object: "chat.completion",
        created: 1_700_000_000,
        model: "mock-model",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: `bridged ${BRIDGE_RESPONSE_TOKEN}` },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
      },
    });
    bridgeStreamUpstream = await startOpenAiUpstream({
      streamEvents: bridgeStreamEvents(),
    });
    completionsUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-mock947",
        object: "text_completion",
        created: 1_700_000_000,
        model: "mock-model",
        choices: [
          { text: `ok ${COMPLETIONS_RESPONSE_TOKEN}`, index: 0, finish_reason: "stop" },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
      },
    });
    sls = await startMockSls();
  });

  afterAll(async () => {
    await Promise.all(apps.map((a) => a.exit()));
    await responsesUpstream?.close();
    await responsesStreamUpstream?.close();
    await bridgeUpstream?.close();
    await bridgeStreamUpstream?.close();
    await completionsUpstream?.close();
    await sls?.close();
  });

  test(
    "full-capture exporter logs prompt + response on every /v1/responses path and /v1/completions; metadata_only logs neither",
    async (ctx) => {
      if (
        !etcdReachable ||
        !responsesUpstream ||
        !responsesStreamUpstream ||
        !bridgeUpstream ||
        !bridgeStreamUpstream ||
        !completionsUpstream ||
        !sls
      ) {
        ctx.skip();
        return;
      }
      const app = await spawnApp({
        extraEnv: {
          [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: MOCK_AK_ID,
          [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: MOCK_AK_SECRET,
        },
      });
      apps.push(app);
      const admin = new AdminClient(app.adminUrl, app.adminKey);
      await admin.createObservabilityExporter({
        name: "sls-full-947",
        enabled: true,
        kind: "aliyun_sls",
        endpoint: sls.url,
        project: SLS_PROJECT,
        logstore: FULL_LOGSTORE,
        credential_ref: CREDENTIAL_REF,
        content_mode: "full",
      });
      await admin.createObservabilityExporter({
        name: "sls-meta-947",
        enabled: true,
        kind: "aliyun_sls",
        endpoint: sls.url,
        project: SLS_PROJECT,
        logstore: META_LOGSTORE,
        credential_ref: CREDENTIAL_REF,
        content_mode: "metadata_only",
      });

      const seedModel = async (name: string, provider: string, upstream: OpenAiUpstream) => {
        const pk = await admin.createProviderKey({
          display_name: `${name}-pk`,
          secret: PROVIDER_SECRET,
          api_base: `${upstream.baseUrl}/v1`,
          provider,
          adapter: "openai",
        });
        await admin.createModel({
          display_name: name,
          provider,
          model_name: "gpt-4o-mini",
          provider_key_id: pk.id,
        });
      };
      await seedModel("cc947-responses", "openai", responsesUpstream);
      await seedModel("cc947-responses-stream", "openai", responsesStreamUpstream);
      await seedModel("cc947-bridge", "deepseek", bridgeUpstream);
      await seedModel("cc947-bridge-stream", "deepseek", bridgeStreamUpstream);
      await seedModel("cc947-completions", "openai", completionsUpstream);
      await admin.createApiKey({
        key_hash: CALLER_KEY_HASH,
        allowed_models: [
          "cc947-responses",
          "cc947-responses-stream",
          "cc947-bridge",
          "cc947-bridge-stream",
          "cc947-completions",
        ],
      });

      await waitConfigPropagation(async () => {
        try {
          const r = await postJson(app, "/v1/responses", {
            model: "cc947-responses",
            input: "warmup",
          });
          await r.text();
          return r.status === 200;
        } catch {
          return false;
        }
      });

      // -- /v1/responses, verbatim non-streaming --
      const res = await postJson(app, "/v1/responses", {
        model: "cc947-responses",
        input: `note the token ${RESP_PROMPT_TOKEN}`,
      });
      expect(res.status).toBe(200);
      await res.text();
      await waitForToken(sls, FULL_LOGSTORE, RESP_PROMPT_TOKEN);
      let fullText = decodedTextFor(sls, FULL_LOGSTORE);
      expect(fullText).toContain(RESP_PROMPT_TOKEN);
      expect(fullText).toContain(RESP_RESPONSE_TOKEN);

      // -- /v1/responses, verbatim streaming (live-forward Drop-guard emit) --
      const streamRes = await postJson(app, "/v1/responses", {
        model: "cc947-responses-stream",
        stream: true,
        input: `recall the token ${RESP_STREAM_PROMPT_TOKEN}`,
      });
      expect(streamRes.status).toBe(200);
      const streamBody = await streamRes.text();
      expect(streamBody).toContain(RESP_STREAM_RESPONSE_TOKEN); // client received it
      await waitForToken(sls, FULL_LOGSTORE, RESP_STREAM_PROMPT_TOKEN);
      fullText = decodedTextFor(sls, FULL_LOGSTORE);
      expect(fullText).toContain(RESP_STREAM_PROMPT_TOKEN);
      expect(fullText).toContain(RESP_STREAM_RESPONSE_TOKEN);

      // -- /v1/responses, cross-provider bridge non-streaming --
      const bridgeRes = await postJson(app, "/v1/responses", {
        model: "cc947-bridge",
        input: `remember the token ${BRIDGE_PROMPT_TOKEN}`,
      });
      expect(bridgeRes.status).toBe(200);
      await bridgeRes.text();
      await waitForToken(sls, FULL_LOGSTORE, BRIDGE_PROMPT_TOKEN);
      fullText = decodedTextFor(sls, FULL_LOGSTORE);
      expect(fullText).toContain(BRIDGE_PROMPT_TOKEN);
      expect(fullText).toContain(BRIDGE_RESPONSE_TOKEN);

      // -- /v1/responses, cross-provider bridge streaming --
      const bridgeStreamRes = await postJson(app, "/v1/responses", {
        model: "cc947-bridge-stream",
        stream: true,
        input: `keep the token ${BRIDGE_STREAM_PROMPT_TOKEN}`,
      });
      expect(bridgeStreamRes.status).toBe(200);
      const bridgeStreamBody = await bridgeStreamRes.text();
      expect(bridgeStreamBody).toContain(BRIDGE_STREAM_RESPONSE_TOKEN);
      await waitForToken(sls, FULL_LOGSTORE, BRIDGE_STREAM_PROMPT_TOKEN);
      fullText = decodedTextFor(sls, FULL_LOGSTORE);
      expect(fullText).toContain(BRIDGE_STREAM_PROMPT_TOKEN);
      expect(fullText).toContain(BRIDGE_STREAM_RESPONSE_TOKEN);

      // -- /v1/completions --
      const cmplRes = await postJson(app, "/v1/completions", {
        model: "cc947-completions",
        prompt: `say the token ${COMPLETIONS_PROMPT_TOKEN}`,
      });
      expect(cmplRes.status).toBe(200);
      await cmplRes.text();
      await waitForToken(sls, FULL_LOGSTORE, COMPLETIONS_PROMPT_TOKEN);
      fullText = decodedTextFor(sls, FULL_LOGSTORE);
      expect(fullText).toContain(COMPLETIONS_PROMPT_TOKEN);
      expect(fullText).toContain(COMPLETIONS_RESPONSE_TOKEN);

      // -- metadata_only exporter received events but never content --
      await waitForLogstore(sls, META_LOGSTORE);
      const metaText = decodedTextFor(sls, META_LOGSTORE);
      for (const token of [
        RESP_PROMPT_TOKEN,
        RESP_RESPONSE_TOKEN,
        RESP_STREAM_PROMPT_TOKEN,
        RESP_STREAM_RESPONSE_TOKEN,
        BRIDGE_PROMPT_TOKEN,
        BRIDGE_RESPONSE_TOKEN,
        BRIDGE_STREAM_PROMPT_TOKEN,
        BRIDGE_STREAM_RESPONSE_TOKEN,
        COMPLETIONS_PROMPT_TOKEN,
        COMPLETIONS_RESPONSE_TOKEN,
      ]) {
        expect(metaText).not.toContain(token);
      }
    },
    120_000,
  );
});
