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

// #932 × AISIX-Cloud#947: on a streaming response with a mask-action PII
// guardrail, the wire chunks released to the client are masked — and the
// content handed to a `content_mode = full` exporter must be the SAME masked
// text. Pre-fix, the capture accumulator collected raw deltas and only the
// wire bytes were rewritten, so SLS received PII the client never saw. The
// email value is split across two SSE deltas so only the assembled (capture)
// text ever contains it whole — exactly the shape that leaked.

const CALLER_PLAINTEXT = "sk-content-capture-masked-PLAINTEXT";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const PROVIDER_SECRET = "sk-mock-content-capture-masked";

const CREDENTIAL_REF = "mock";
const SLS_PROJECT = "aisix-e2e-obs";
const FULL_LOGSTORE = "full-events-masked";
const META_LOGSTORE = "meta-events-masked";

const EMAIL = "alice@example.com";
const MASK = "[EMAIL_REDACTED]";
const CHAT_PROMPT_TOKEN = "masked-chat-prompt-tok-1f2e3d";
const BRIDGE_PROMPT_TOKEN = "masked-bridge-prompt-tok-4c5b6a";

/** OpenAI chat chunks with the email split across two deltas (channel
 * reassembly): neither wire chunk carries the whole value; only the
 * assembled text does. */
function chatStreamEvents(marker: string): string[] {
  const split = EMAIL.indexOf("@") + 2; // "alice@e" | "xample.com"
  return [
    JSON.stringify({
      id: "mock-masked-1",
      object: "chat.completion.chunk",
      model: "mock-model",
      choices: [{ index: 0, delta: { role: "assistant" } }],
    }),
    JSON.stringify({
      id: "mock-masked-1",
      object: "chat.completion.chunk",
      model: "mock-model",
      choices: [{ index: 0, delta: { content: `${marker} reach me at ${EMAIL.slice(0, split)}` } }],
    }),
    JSON.stringify({
      id: "mock-masked-1",
      object: "chat.completion.chunk",
      model: "mock-model",
      choices: [
        { index: 0, delta: { content: `${EMAIL.slice(split)} thanks` }, finish_reason: "stop" },
      ],
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

describe("sls content capture e2e: streaming capture is post-mask (#932 × AISIX-Cloud#947)", () => {
  let etcdReachable = false;
  let chatUpstream: OpenAiUpstream | undefined;
  let bridgeUpstream: OpenAiUpstream | undefined;
  let sls: MockSls | undefined;
  const apps: SpawnedApp[] = [];

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    chatUpstream = await startOpenAiUpstream({ streamEvents: chatStreamEvents("chat-reply") });
    // The /v1/responses cross-provider bridge consumes OpenAI chat chunks too.
    bridgeUpstream = await startOpenAiUpstream({ streamEvents: chatStreamEvents("bridge-reply") });
    sls = await startMockSls();
  });

  afterAll(async () => {
    await Promise.all(apps.map((a) => a.exit()));
    await chatUpstream?.close();
    await bridgeUpstream?.close();
    await sls?.close();
  });

  test(
    "full-capture exporter receives the masked stream text, never the raw PII",
    async (ctx) => {
      if (!etcdReachable || !chatUpstream || !bridgeUpstream || !sls) {
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
        name: "sls-full-masked",
        enabled: true,
        kind: "aliyun_sls",
        endpoint: sls.url,
        project: SLS_PROJECT,
        logstore: FULL_LOGSTORE,
        credential_ref: CREDENTIAL_REF,
        content_mode: "full",
      });
      await admin.createObservabilityExporter({
        name: "sls-meta-masked",
        enabled: true,
        kind: "aliyun_sls",
        endpoint: sls.url,
        project: SLS_PROJECT,
        logstore: META_LOGSTORE,
        credential_ref: CREDENTIAL_REF,
        content_mode: "metadata_only",
      });
      const chatPk = await admin.createProviderKey({
        display_name: "masked-chat-pk",
        secret: PROVIDER_SECRET,
        api_base: `${chatUpstream.baseUrl}/v1`,
      });
      await admin.createModel({
        display_name: "masked-chat-model",
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: chatPk.id,
      });
      const bridgePk = await admin.createProviderKey({
        display_name: "masked-bridge-pk",
        secret: PROVIDER_SECRET,
        api_base: `${bridgeUpstream.baseUrl}/v1`,
        provider: "deepseek",
        adapter: "openai",
      });
      await admin.createModel({
        display_name: "masked-bridge-model",
        provider: "deepseek",
        model_name: "gpt-4o-mini",
        provider_key_id: bridgePk.id,
      });
      await admin.createApiKey({
        key_hash: CALLER_KEY_HASH,
        allowed_models: ["masked-chat-model", "masked-bridge-model"],
      });
      // Env-wide output-hook PII guardrail: email → mask. Its presence also
      // forces the streaming hold-back (BufferFull) path — the path where
      // the capture/wire divergence lived.
      await admin.json("POST", "/admin/v1/guardrails", {
        name: "masked-capture-guard",
        enabled: true,
        hook_point: "output",
        kind: "pii",
        detectors: [{ type: "email", action: "mask" }],
      });

      await waitConfigPropagation(async () => {
        try {
          const r = await postJson(app, "/v1/chat/completions", {
            model: "masked-chat-model",
            stream: true,
            messages: [{ role: "user", content: "warmup" }],
          });
          const body = await r.text();
          // The guardrail must be live too, not just the model: wait until
          // the masked form appears in the wire response.
          return r.status === 200 && body.includes(MASK);
        } catch {
          return false;
        }
      });

      // -- streaming chat (hold-back release path) --
      const chatRes = await postJson(app, "/v1/chat/completions", {
        model: "masked-chat-model",
        stream: true,
        messages: [{ role: "user", content: `note ${CHAT_PROMPT_TOKEN}` }],
      });
      expect(chatRes.status).toBe(200);
      const chatBody = await chatRes.text();
      expect(chatBody).toContain(MASK); // client got masked text
      expect(chatBody).not.toContain(EMAIL);

      await waitForToken(sls, FULL_LOGSTORE, CHAT_PROMPT_TOKEN);
      let fullText = decodedTextFor(sls, FULL_LOGSTORE);
      expect(fullText).toContain(CHAT_PROMPT_TOKEN);
      expect(fullText).toContain(MASK); // capture carries the masked reply
      expect(fullText).not.toContain(EMAIL); // and never the raw PII

      // -- streaming /v1/responses via the cross-provider bridge --
      const bridgeRes = await postJson(app, "/v1/responses", {
        model: "masked-bridge-model",
        stream: true,
        input: `note ${BRIDGE_PROMPT_TOKEN}`,
      });
      expect(bridgeRes.status).toBe(200);
      const bridgeBody = await bridgeRes.text();
      expect(bridgeBody).toContain(MASK);
      expect(bridgeBody).not.toContain(EMAIL);

      await waitForToken(sls, FULL_LOGSTORE, BRIDGE_PROMPT_TOKEN);
      fullText = decodedTextFor(sls, FULL_LOGSTORE);
      expect(fullText).toContain(BRIDGE_PROMPT_TOKEN);
      expect(fullText).not.toContain(EMAIL);

      // metadata_only exporter never sees content at all.
      const metaText = decodedTextFor(sls, META_LOGSTORE);
      expect(metaText).not.toContain(EMAIL);
      expect(metaText).not.toContain(MASK);
      expect(metaText).not.toContain(CHAT_PROMPT_TOKEN);
    },
    120_000,
  );
});
