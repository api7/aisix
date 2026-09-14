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

// E2E for the REQUEST direction of the `/v1/responses` → chat-completions
// bridge, plus the sibling usage gap on the bridged `/v1/messages` path.
//
// Three contracts are pinned here.
//
// 1. Multimodal input reaches the upstream. The bridge used to extract only
//    `.text` from each Responses content slot, so an `input_image` part
//    vanished and the model answered a question about an image it had never
//    been sent — no error, just a wrong answer.
//
// 2. `text.format` reaches the upstream as `response_format`. The whole
//    `text` object used to be dropped, so a caller that asked for a JSON
//    schema got prose.
//
// 3. The usage a `/v1/messages` client reads is the usage the gateway
//    records. When a bridged reply carries no upstream usage, the gateway
//    counts the tokens locally; it used to hand the client zeros while
//    billing the estimate, which is irreconcilable from the caller's side.
//    Same gap the bridged `/v1/responses` path was fixed for.
//
// The bridge is reached by declaring an EMPTY `apis` map on the provider
// key: the operator saying "this OpenAI-compatible endpoint has no
// `/v1/responses`", which is the deployment (a vLLM / SGLang / vendor relay
// serving chat only) these contracts matter for. `/v1/messages` bridges
// whenever the provider is not Anthropic.

const CALLER_PLAINTEXT = "sk-responses-bridge-request-parity";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const CREDENTIAL_REF = "mock";
const MOCK_AK_ID = "mock-akid";
const MOCK_AK_SECRET = "mock-secret";
const SLS_PROJECT = "aisix-e2e-obs";
const META_LOGSTORE = "bridge-request-parity-events";

const IMAGE_URL = "https://example.com/cat.png";
const QUESTION = "what is in this image?";

const CHAT_NON_STREAM = {
  id: "chatcmpl-request-parity",
  object: "chat.completion",
  created: 1765000000,
  model: "relay-mini",
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: '{"animal":"cat"}' },
      finish_reason: "stop",
    },
  ],
  usage: { prompt_tokens: 11, completion_tokens: 5, total_tokens: 16 },
};

/** A chat reply with NO usage block at all — the estimate path. */
const CHAT_NON_STREAM_NO_USAGE = {
  id: "chatcmpl-msg-nousage",
  object: "chat.completion",
  created: 1765000000,
  model: "relay-mini",
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: "Hello world" },
      finish_reason: "stop",
    },
  ],
};

/** A chat SSE relay that ignores `stream_options.include_usage`. */
const CHAT_STREAM_NO_USAGE = [
  JSON.stringify({
    id: "chatcmpl-msg-stream-nousage",
    object: "chat.completion.chunk",
    model: "relay-mini",
    choices: [{ index: 0, delta: { content: "Hello" }, finish_reason: null }],
  }),
  JSON.stringify({
    id: "chatcmpl-msg-stream-nousage",
    object: "chat.completion.chunk",
    model: "relay-mini",
    choices: [{ index: 0, delta: { content: " world" }, finish_reason: null }],
  }),
  JSON.stringify({
    id: "chatcmpl-msg-stream-nousage",
    object: "chat.completion.chunk",
    model: "relay-mini",
    choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
  }),
  "[DONE]",
];

function sseData(raw: string): Record<string, any>[] {
  return raw
    .split("\n")
    .filter((l) => l.startsWith("data: "))
    .map((l) => l.slice("data: ".length))
    .filter((p) => p !== "[DONE]")
    .map((p) => JSON.parse(p));
}

describe("bridged request direction: multimodal input, text.format, /v1/messages usage", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let chatUpstream: OpenAiUpstream | undefined;
  let msgUpstream: OpenAiUpstream | undefined;
  let msgStreamUpstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    chatUpstream = await startOpenAiUpstream({ nonStreamBody: CHAT_NON_STREAM });
    msgUpstream = await startOpenAiUpstream({
      nonStreamBody: CHAT_NON_STREAM_NO_USAGE,
    });
    msgStreamUpstream = await startOpenAiUpstream({
      streamEvents: CHAT_STREAM_NO_USAGE,
    });

    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: MOCK_AK_ID,
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: MOCK_AK_SECRET,
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "bridge-request-parity-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: META_LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });

    const seedModel = async (display: string, upstream: OpenAiUpstream) => {
      const pk = await seed.createProviderKey({
        display_name: `${display}-pk`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
        apis: {},
      });
      await seed.createModel({
        display_name: display,
        provider: "openai",
        model_name: "relay-compat-x",
        provider_key_id: pk.id,
      });
    };
    await seedModel("bridge-request-parity", chatUpstream);
    await seedModel("bridge-msg-nousage", msgUpstream);
    await seedModel("bridge-msg-stream-nousage", msgStreamUpstream);
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [
        "bridge-request-parity",
        "bridge-msg-nousage",
        "bridge-msg-stream-nousage",
      ],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await sls?.close();
    await chatUpstream?.close();
    await msgUpstream?.close();
    await msgStreamUpstream?.close();
  });

  /**
   * One gate for the whole file: the caller key is seeded last, so it
   * authenticating implies every resource before it is in the snapshot. It
   * also spends no token budget and exercises none of the behavior under
   * test, so a failure here reads as "config never propagated" rather than
   * as a silent timeout in place of an assertion.
   */
  async function waitReady() {
    const probe = new ProxyClient(app!.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      if (res.status !== 200) return false;
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return ["bridge-request-parity", "bridge-msg-nousage", "bridge-msg-stream-nousage"].every(
        (m) => data.some((entry) => entry.id === m),
      );
    });
  }

  function postResponses(body: unknown): Promise<Response> {
    return fetch(`${app!.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    });
  }

  function postMessages(model: string, stream: boolean): Promise<Response> {
    return fetch(`${app!.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: {
        "x-api-key": CALLER_PLAINTEXT,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model,
        max_tokens: 200,
        stream,
        messages: [{ role: "user", content: "hi" }],
      }),
    });
  }

  test("an input_image part and text.format reach the chat upstream", async (ctx) => {
    if (!etcdReachable || !app || !chatUpstream) {
      ctx.skip();
      return;
    }
    await waitReady();

    const res = await postResponses({
      model: "bridge-request-parity",
      input: [
        {
          role: "user",
          content: [
            { type: "input_text", text: QUESTION },
            { type: "input_image", image_url: IMAGE_URL, detail: "high" },
          ],
        },
      ],
      text: {
        format: {
          type: "json_schema",
          name: "animal",
          schema: {
            type: "object",
            properties: { animal: { type: "string" } },
            required: ["animal"],
            additionalProperties: false,
          },
          strict: true,
        },
      },
    });
    expect(res.status).toBe(200);

    const lastReq = chatUpstream.receivedRequests.at(-1);
    expect(lastReq?.path).toContain("/chat/completions");
    const outbound = JSON.parse(lastReq!.body) as Record<string, any>;

    // The image survived the translation, as a typed content-block array —
    // pre-fix the upstream received only the question text.
    expect(outbound.messages).toEqual([
      {
        role: "user",
        content: [
          { type: "text", text: QUESTION },
          { type: "image_url", image_url: { url: IMAGE_URL, detail: "high" } },
        ],
      },
    ]);

    // …and the structured-output request survived as `response_format` —
    // pre-fix the whole `text` object was dropped and the model was free to
    // answer in prose.
    expect(outbound.response_format).toEqual({
      type: "json_schema",
      json_schema: {
        name: "animal",
        schema: {
          type: "object",
          properties: { animal: { type: "string" } },
          required: ["animal"],
          additionalProperties: false,
        },
        strict: true,
      },
    });
  });

  test("/v1/messages non-streaming with no upstream usage reports the counts the usage record gets", async (ctx) => {
    if (!etcdReachable || !app || !sls) {
      ctx.skip();
      return;
    }
    await waitReady();

    const res = await postMessages("bridge-msg-nousage", false);
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id");
    expect(requestId).toBeTruthy();
    const body = (await res.json()) as Record<string, any>;

    // Pre-fix: both of these were 0 while the usage record carried the
    // estimate the request was billed on.
    expect(body.usage.input_tokens).toBeGreaterThan(0);
    expect(body.usage.output_tokens).toBeGreaterThan(0);

    const row = await waitForSlsLog(
      sls,
      META_LOGSTORE,
      (log) => log.get("request_id") === requestId,
      `a usage row for request_id=${requestId}`,
    );
    expect(row.get("prompt_tokens")).toBe(String(body.usage.input_tokens));
    expect(row.get("completion_tokens")).toBe(String(body.usage.output_tokens));
  });

  test("/v1/messages streaming with no upstream usage frame closes on the counts the usage record gets", async (ctx) => {
    if (!etcdReachable || !app || !sls) {
      ctx.skip();
      return;
    }
    await waitReady();

    const res = await postMessages("bridge-msg-stream-nousage", true);
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id");
    expect(requestId).toBeTruthy();
    const events = sseData(await res.text());

    const messageDelta = events.find((d) => d.type === "message_delta");
    expect(messageDelta?.delta?.stop_reason).toBe("end_turn");
    // Pre-fix: the forced closing pair reported output_tokens 0 (and no
    // input_tokens at all) while the record carried the estimate.
    expect(messageDelta?.usage?.input_tokens).toBeGreaterThan(0);
    expect(messageDelta?.usage?.output_tokens).toBeGreaterThan(0);

    const row = await waitForSlsLog(
      sls,
      META_LOGSTORE,
      (log) => log.get("request_id") === requestId,
      `a usage row for request_id=${requestId}`,
    );
    expect(row.get("prompt_tokens")).toBe(
      String(messageDelta!.usage.input_tokens),
    );
    expect(row.get("completion_tokens")).toBe(
      String(messageDelta!.usage.output_tokens),
    );
  });
});
