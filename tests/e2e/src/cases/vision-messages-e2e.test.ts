import { createHash } from "node:crypto";
import OpenAI from "openai";
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

// E2E: vision messages (image_url content blocks).
//
// Per OpenAI's chat completions spec — when a message's `content`
// is an array of typed blocks (`{type:"text",...}`,
// `{type:"image_url", image_url: {url:"data:image/png;base64,..."}}`),
// the gateway must forward the structured content array to the
// upstream byte-for-byte. The base64 payload, the wrapping object
// shape, and the order of `content[]` entries are all part of the
// caller's contract.
//
// Reference:
//   - OpenAI vision API:
//     <https://platform.openai.com/docs/guides/vision>
//   - chat.completions.create(messages: ChatCompletionMessageParam[])
//     `content` parameter: string | ContentPart[]
//
// One contract pinned here:
//
//   - The gateway preserves vision-shape `content` arrays end-to-end
//     when proxying chat completions. Concretely, the upstream
//     receives a `messages[0].content` whose JSON serialization is
//     deeply equal to what the OpenAI Node SDK constructed from the
//     caller's input (text part + image_url part with the exact
//     base64 data URL).
//
// Why this matters: vision is a shipped feature surface today
// (per docs/api-proxy.md §4.2) but has zero e2e coverage. A
// regression that stripped image_url blocks (treating them as
// non-text), rewrote the data URL, or reordered content parts
// would silently degrade vision callers without surfacing in
// any existing test.

const CALLER_PLAINTEXT = "sk-vision-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

// 1×1 transparent PNG, base64-encoded. Small enough that the test
// fixture is human-inspectable; not so small that an accidental
// substring match could pass off as a full-payload preserve.
const PNG_BASE64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";
const DATA_URL = `data:image/png;base64,${PNG_BASE64}`;
const TEXT_PROMPT = "What's in this image?";

const NON_STREAM_BODY = {
  id: "chatcmpl-vision-1",
  object: "chat.completion",
  created: Math.floor(Date.now() / 1000),
  model: "gpt-4o-mini",
  choices: [
    {
      index: 0,
      message: {
        role: "assistant",
        content: "A 1×1 transparent pixel.",
      },
      finish_reason: "stop",
    },
  ],
  usage: {
    prompt_tokens: 8,
    completion_tokens: 6,
    total_tokens: 14,
  },
};

describe("vision messages e2e: image_url content array forwarded byte-for-byte", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      nonStreamBody: NON_STREAM_BODY,
    });
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "vision-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "vision-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["vision-model"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("text + image_url content blocks reach upstream verbatim and response passes back", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // Snapshot propagation through the same code path the test uses
    // (non-streaming chat). A simple text probe is enough — readiness
    // is about Model + ProviderKey + ApiKey visibility, not vision-
    // specific dispatcher state.
    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "vision-model",
          messages: [{ role: "user", content: "ready-probe" }],
        });
        return probe.choices.length > 0;
      } catch {
        return false;
      }
    });

    // Baseline-isolate the readiness probe so the wire-shape
    // assertion measures only the actual test call.
    const baseline = upstream.receivedRequests.length;

    const visionContent = [
      { type: "text" as const, text: TEXT_PROMPT },
      {
        type: "image_url" as const,
        image_url: { url: DATA_URL },
      },
    ];
    const resp = await client.chat.completions.create({
      model: "vision-model",
      messages: [{ role: "user", content: visionContent }],
    });

    // (1) Response passes back unmodified — the canned upstream body
    // surfaces to the caller. Catches a regression that swallowed
    // the upstream response on vision requests.
    expect(resp.choices[0]?.message?.content).toBe(
      "A 1×1 transparent pixel.",
    );
    expect(resp.usage?.total_tokens).toBe(14);

    // (2) Upstream wire-shape assertion. Exactly one POST to
    // /v1/chat/completions, with the right model, the right auth,
    // and a `messages[0].content` array that deep-equals what the
    // SDK constructed from `visionContent`. Closes the same blind
    // spot CLAUDE.md §8 calls out — a regression that stripped
    // image_url blocks (or swapped their positions, or rewrote the
    // base64) would still pass (1) because the mock replays its
    // canned response regardless of the request body.
    const sent = upstream.receivedRequests
      .slice(baseline)
      .filter((r) => r.path === "/v1/chat/completions");
    expect(sent).toHaveLength(1);
    const sentReq = sent[0]!;
    expect(sentReq.method).toBe("POST");
    expect(sentReq.headers.authorization).toBe("Bearer sk-mock");
    const sentBody = JSON.parse(sentReq.body);
    expect(sentBody.model).toBe("gpt-4o-mini");
    expect(Array.isArray(sentBody.messages)).toBe(true);
    expect(sentBody.messages).toHaveLength(1);
    expect(sentBody.messages[0]?.role).toBe("user");

    // (3) Content array deep-equality. The forwarded `content` must
    // be exactly the two-element array (text part, then image_url
    // part) — same shape, same order, same base64 payload. A
    // regression that re-coerced the array to a string, dropped the
    // image_url part, or mutated the `data:image/png;base64,...`
    // URL would fail here.
    expect(sentBody.messages[0]?.content).toEqual(visionContent);

    // (4) Belt-and-suspenders: explicitly assert the image_url
    // payload is byte-equal to the canonical data URL, and the
    // text part is byte-equal to the prompt. Catches a regression
    // that ECHOed-but-mutated either field (e.g. URL-encoded the
    // base64) which would still serialize to a valid OpenAI shape
    // but would fail any consumer that compared payloads.
    const content = sentBody.messages[0]?.content;
    expect(Array.isArray(content)).toBe(true);
    expect(content[0]?.type).toBe("text");
    expect(content[0]?.text).toBe(TEXT_PROMPT);
    expect(content[1]?.type).toBe("image_url");
    expect(content[1]?.image_url?.url).toBe(DATA_URL);
  }, 60_000);
});
