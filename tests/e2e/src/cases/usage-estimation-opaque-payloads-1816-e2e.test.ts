import { createHash, randomBytes } from "node:crypto";
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
import { startMockSls, waitForSlsLog, type MockSls } from "../harness/sls-mock.js";

// E2E for AISIX-Cloud#1816: when the upstream reports no usage, the
// gateway's local prompt estimate must not tokenize images or opaque
// payloads as text. A coding agent's request routinely carries a
// screenshot inside a tool output and an encrypted reasoning item; counted
// as text, a few hundred KiB of base64 becomes hundreds of thousands of
// "prompt tokens" in the usage record, the cost and the rate-limit
// counters.
//
// What the request really costs: a handful of short text fields plus one
// image. An image is at most a couple of thousand tokens under OpenAI's
// vision token formula (85 + 170 per 512-px tile of the fitted image), so
// the whole prompt stays far below the bound asserted here, while the
// base64 alone — ~700 KiB — would be well over 100k tokens as text.
const PROMPT_TOKEN_CEILING = 5_000;

const CALLER_PLAINTEXT = "sk-usage-estimation-1816-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const CREDENTIAL_REF = "mock";
const SLS_PROJECT = "aisix-e2e-obs";
const LOGSTORE = "est-1816-events";

/** A PNG of the given size: a valid signature + IHDR, then incompressible
 * filler standing in for the pixel data. */
function pngBase64(width: number, height: number, fillerBytes: number): string {
  const header = Buffer.alloc(33);
  Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]).copy(header, 0);
  header.writeUInt32BE(13, 8);
  header.write("IHDR", 12, "ascii");
  header.writeUInt32BE(width, 16);
  header.writeUInt32BE(height, 20);
  header.set([8, 6, 0, 0, 0], 24);
  return Buffer.concat([header, randomBytes(fillerBytes)]).toString("base64");
}

const IMAGE_BASE64 = pngBase64(1024, 2048, 384 * 1024);
const ENCRYPTED_REASONING = randomBytes(256 * 1024).toString("base64");

// Streams that end without any usage (a relay that drops it, or a client
// that went away before the terminal event).
const RESPONSES_STREAM_NO_USAGE = [
  '{"type":"response.created","response":{"id":"resp_1816"}}',
  '{"type":"response.output_text.delta","delta":"Done"}',
  "[DONE]",
];
const MESSAGES_STREAM_NO_USAGE = [
  JSON.stringify({
    type: "message_start",
    message: { id: "msg_1816", role: "assistant", content: [], model: "relay-claude", stop_reason: null },
  }),
  JSON.stringify({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
  JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Done" } }),
  JSON.stringify({ type: "content_block_stop", index: 0 }),
  JSON.stringify({ type: "message_stop" }),
];

describe("usage estimation (AISIX-Cloud#1816): images and opaque payloads are not counted as text", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let responsesUpstream: OpenAiUpstream | undefined;
  let messagesUpstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    responsesUpstream = await startOpenAiUpstream({ streamEvents: RESPONSES_STREAM_NO_USAGE });
    messagesUpstream = await startOpenAiUpstream({ streamEvents: MESSAGES_STREAM_NO_USAGE });

    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "est-1816-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });
    const seedModel = async (display: string, upstream: OpenAiUpstream, provider: string) => {
      const pk = await seed.createProviderKey({
        display_name: `${display}-pk`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      await seed.createModel({
        display_name: display,
        provider,
        model_name: "relay-compat-x",
        provider_key_id: pk.id,
      });
    };
    await seedModel("est-1816-responses", responsesUpstream, "openai");
    await seedModel("est-1816-messages", messagesUpstream, "anthropic");
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["est-1816-responses", "est-1816-messages"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await sls?.close();
    await responsesUpstream?.close();
    await messagesUpstream?.close();
  });

  async function post(path: string, body: unknown, headers: Record<string, string>): Promise<Response> {
    return fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json", ...headers },
      body: JSON.stringify(body),
    });
  }

  /** The caller key is seeded last, so its authenticating implies every
   * other seeded resource is in the snapshot. Not a request to the models
   * under test: each model's only usage record is the asserted request's. */
  async function keyAuthenticates(): Promise<void> {
    await waitConfigPropagation(async () => {
      const r = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      await r.text();
      return r.status === 200;
    });
  }

  /** The usage record for `model`, once it carries an estimated prompt. */
  async function estimatedRecord(model: string): Promise<Map<string, string>> {
    return waitForSlsLog(
      sls!,
      LOGSTORE,
      (log) => log.get("requested_model") === model && log.get("usage_estimated") === "true",
      `usage_estimated record for ${model}`,
    );
  }

  test("/v1/responses: an image inside a tool output and an encrypted reasoning item", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const auth = { authorization: `Bearer ${CALLER_PLAINTEXT}` };
    await keyAuthenticates();

    const res = await post(
      "/v1/responses",
      {
        model: "est-1816-responses",
        stream: true,
        input: [
          { role: "user", content: [{ type: "input_text", text: "Take a screenshot of the page." }] },
          { type: "reasoning", id: "rs_1816", summary: [], encrypted_content: ENCRYPTED_REASONING },
          { type: "custom_tool_call", call_id: "call_1816", name: "screenshot", input: "{}" },
          {
            type: "custom_tool_call_output",
            call_id: "call_1816",
            output: [
              { type: "input_text", text: "Here is the screenshot." },
              { type: "input_image", image_url: `data:image/png;base64,${IMAGE_BASE64}`, detail: "high" },
            ],
          },
        ],
      },
      auth,
    );
    expect(res.status).toBe(200);
    expect(await res.text()).toContain("Done");

    const record = await estimatedRecord("est-1816-responses");
    const prompt = Number(record.get("prompt_tokens") ?? "0");
    expect(prompt).toBeGreaterThan(0);
    expect(prompt).toBeLessThan(PROMPT_TOKEN_CEILING);
  }, 60_000);

  test("/v1/messages: a base64 image inside a tool_result", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const auth = { "x-api-key": CALLER_PLAINTEXT };
    await keyAuthenticates();

    const res = await post(
      "/v1/messages",
      {
        model: "est-1816-messages",
        max_tokens: 16,
        stream: true,
        messages: [
          { role: "user", content: "Take a screenshot of the page." },
          {
            role: "assistant",
            content: [{ type: "tool_use", id: "toolu_1816", name: "screenshot", input: {} }],
          },
          {
            role: "user",
            content: [
              {
                type: "tool_result",
                tool_use_id: "toolu_1816",
                content: [
                  { type: "text", text: "Here is the screenshot." },
                  { type: "image", source: { type: "base64", media_type: "image/png", data: IMAGE_BASE64 } },
                ],
              },
            ],
          },
        ],
      },
      auth,
    );
    expect(res.status).toBe(200);
    expect(await res.text()).toContain("Done");

    const record = await estimatedRecord("est-1816-messages");
    const prompt = Number(record.get("prompt_tokens") ?? "0");
    expect(prompt).toBeGreaterThan(0);
    expect(prompt).toBeLessThan(PROMPT_TOKEN_CEILING);
  }, 60_000);
});
