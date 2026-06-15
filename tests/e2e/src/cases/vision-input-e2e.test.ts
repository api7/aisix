import { createHash } from "node:crypto";
import OpenAI from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: vision input pass-through on the OpenAI-native path.
// Multimodal callers attach images to chat completions via the
// OpenAI `messages[].content` array of typed content blocks per
// <https://platform.openai.com/docs/guides/vision>. When the
// resolved Model's provider is OpenAI (native, no translation),
// the gateway must forward the content array byte-for-byte to the
// upstream — including both `image_url` blocks (URL form) and
// `text` blocks interleaved.
//
// Two shapes pinned:
//
//   1. Image-URL form — `image_url: { url: "https://..." }`
//   2. Base64 data-URL form — `image_url: { url: "data:image/png;base64,..." }`
//
// Both are documented in OpenAI's vision guide as accepted forms.
// A regression that re-encoded, re-shaped, or dropped the
// content-block array would break every multimodal caller.
//
// Cross-provider vision translation (caller's image_url →
// Anthropic image content block) is documented as a known gap in
// `docs/api-proxy.md` §4.5 ("image blocks ... skip non-text blocks
// silently on the inbound parse") and is out of scope for this
// PR — the OpenAI-native path is the most-used vision path and a
// regression there breaks the largest callers.
//
// References:
// - OpenAI Vision guide
//   <https://platform.openai.com/docs/guides/vision>
// - OpenAI Chat Completions content-block spec
//   <https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages>

const CALLER_PLAINTEXT = "sk-vision-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

// Tiny 1×1 transparent PNG, base64-encoded. Works as a real
// data-URL the SDK will accept; small enough to keep the test
// fixture compact.
const TINY_PNG_BASE64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkAAIAAAoAAv/lxKUAAAAASUVORK5CYII=";
const TINY_PNG_DATA_URL = `data:image/png;base64,${TINY_PNG_BASE64}`;
const REMOTE_IMAGE_URL = "https://example.com/test-image.jpg";

describe("vision input e2e: OpenAI-native image content blocks pass through to upstream", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-vision",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: "I see a 1x1 transparent pixel.",
            },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 12, completion_tokens: 7, total_tokens: 19 },
      },
    });
    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "vision-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "vision-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["vision-e2e"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("remote image_url + text blocks reach upstream byte-for-byte", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    await waitConfigPropagation(async () => {
      try {
        const r = await client.chat.completions.create({
          model: "vision-e2e",
          messages: [{ role: "user", content: "ready-probe" }],
        });
        return r.choices[0]?.message.role === "assistant";
      } catch {
        return false;
      }
    });

    const baseline = upstream.receivedRequests.length;

    // OpenAI vision request shape: content is an array of typed
    // blocks instead of a bare string. Per
    // <https://platform.openai.com/docs/guides/vision>.
    const completion = await client.chat.completions.create({
      model: "vision-e2e",
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "What's in this image?" },
            { type: "image_url", image_url: { url: REMOTE_IMAGE_URL } },
          ],
        },
      ],
    });

    expect(completion.choices[0]?.message.role).toBe("assistant");
    expect(completion.choices[0]?.message.content).toBe(
      "I see a 1x1 transparent pixel.",
    );

    const testCalls = upstream.receivedRequests
      .slice(baseline)
      .filter((r) => r.path === "/v1/chat/completions");
    expect(testCalls).toHaveLength(1);

    // Wire-shape contract: content array reaches upstream with
    // both blocks intact. A regression that flattened to a bare
    // string (e.g. concatenating text and discarding the image)
    // would break vision entirely.
    const sentBody = JSON.parse(testCalls[0]!.body) as {
      messages?: Array<{
        role?: string;
        content?: Array<{ type?: string; text?: string; image_url?: { url?: string } }>;
      }>;
    };
    const userMessage = sentBody.messages?.[0];
    expect(userMessage?.role).toBe("user");
    expect(Array.isArray(userMessage?.content)).toBe(true);
    expect(userMessage?.content).toHaveLength(2);
    expect(userMessage?.content?.[0]?.type).toBe("text");
    expect(userMessage?.content?.[0]?.text).toBe("What's in this image?");
    expect(userMessage?.content?.[1]?.type).toBe("image_url");
    // The remote URL must reach upstream verbatim — a regression
    // that re-encoded, signed, or proxied the URL would break
    // upstream's ability to fetch the image.
    expect(userMessage?.content?.[1]?.image_url?.url).toBe(REMOTE_IMAGE_URL);
  });

  test("base64 data-URL image reaches upstream byte-for-byte", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    const baseline = upstream.receivedRequests.length;

    // Base64 data-URL form. Common path for callers who don't
    // want to host images on a public URL.
    await client.chat.completions.create({
      model: "vision-e2e",
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "Describe the embedded image." },
            { type: "image_url", image_url: { url: TINY_PNG_DATA_URL } },
          ],
        },
      ],
    });

    const testCalls = upstream.receivedRequests
      .slice(baseline)
      .filter((r) => r.path === "/v1/chat/completions");
    expect(testCalls).toHaveLength(1);

    // The base64 payload must reach upstream byte-for-byte. A
    // regression that re-encoded the base64 (e.g. unwrapped + re-
    // wrapped through internal buffers) could corrupt the image.
    const sentBody = JSON.parse(testCalls[0]!.body) as {
      messages?: Array<{
        content?: Array<{ type?: string; image_url?: { url?: string } }>;
      }>;
    };
    const imageBlock = sentBody.messages?.[0]?.content?.[1];
    expect(imageBlock?.type).toBe("image_url");
    expect(imageBlock?.image_url?.url).toBe(TINY_PNG_DATA_URL);
  });
});
