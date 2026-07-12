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

// E2E: vision message edge cases beyond the single-image happy path.
//
// Companion to vision-messages-e2e (PR #192) which pinned
// {text, image_url(base64)} as a two-element content array. This
// file covers three orthogonal user journeys real callers actually
// take with the OpenAI vision API:
//
//   1. Multi-image message — `content: [text, image_url, image_url]`
//      with two distinct images. Order and per-image payloads must
//      survive end-to-end. A regression that deduplicated content
//      blocks (e.g. by hash) or coalesced same-type entries would
//      ship a wrong upstream request.
//
//   2. `detail` parameter forwarded — OpenAI vision exposes a
//      `detail: "low" | "high" | "auto"` field per image, which
//      controls the upstream's resize policy and feeds directly
//      into token budgeting. A regression that stripped it would
//      silently push every image to "auto" — billing surprise +
//      worse OCR.
//
//   3. Mixed remote URL + base64 in one message. Same caller, same
//      request — one image hosted, one inlined. The remote URL
//      must reach the upstream verbatim (still as `http(s)://...`,
//      not pre-fetched and inlined as `data:...`); the inlined
//      base64 must reach upstream byte-intact.
//
//      Note: this test only verifies the upstream-side request
//      shape. It does not (and the harness cannot) observe an
//      out-of-band fetch the gateway might issue against the
//      remote URL host — a regression that pre-fetched purely
//      for caching/telemetry, without modifying the upstream
//      payload, would slip through. Treat as a wire-shape pin,
//      not a full SSRF guard. A full SSRF assertion needs a
//      sentinel host that 404s loudly when hit and is observed
//      via egress-network instrumentation; tracked as a
//      follow-up.
//
// Reference:
//   - OpenAI vision API
//     <https://platform.openai.com/docs/guides/vision>
//   - OpenAI Node SDK
//     `ChatCompletionContentPart` / `ChatCompletionContentPartImage`

const CALLER_PLAINTEXT = "sk-vision-edges-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

// Two distinct 1×1 PNG payloads so test (1) catches a regression
// that deduped on hash. Image A is solid red; Image B is solid blue.
// Bytes hand-written, not generated, so a substring match in either
// direction can't accidentally pass.
const PNG_RED_BASE64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAACklEQVR42mNkAAIAAAoAAv/lxKUAAAAASUVORK5CYII=";
const PNG_BLUE_BASE64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAACklEQVR4AWNg+A8AAQQBAGFm3WgAAAAASUVORK5CYII=";
const DATA_URL_RED = `data:image/png;base64,${PNG_RED_BASE64}`;
const DATA_URL_BLUE = `data:image/png;base64,${PNG_BLUE_BASE64}`;
const REMOTE_URL =
  "https://example.com/fixture/vision-test-image.png";

const NON_STREAM_BODY = {
  id: "chatcmpl-vision-edges-1",
  object: "chat.completion",
  created: Math.floor(Date.now() / 1000),
  model: "gpt-4o-mini",
  choices: [
    {
      index: 0,
      message: {
        role: "assistant",
        content: "vision edges ok",
      },
      finish_reason: "stop",
    },
  ],
  usage: {
    prompt_tokens: 4,
    completion_tokens: 4,
    total_tokens: 8,
  },
};

describe("vision messages edges e2e: multi-image, detail param, mixed-source", () => {
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
      display_name: "vision-edges-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "vision-edges-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["vision-edges-model"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("(1) multi-image content array preserves order and per-image bytes", async (ctx) => {
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
        const probe = await client.chat.completions.create({
          model: "vision-edges-model",
          messages: [{ role: "user", content: "ready-probe-1" }],
        });
        return probe.choices.length > 0;
      } catch {
        return false;
      }
    });

    const baseline = upstream.receivedRequests.length;

    const content = [
      { type: "text" as const, text: "Compare these two images." },
      { type: "image_url" as const, image_url: { url: DATA_URL_RED } },
      { type: "image_url" as const, image_url: { url: DATA_URL_BLUE } },
    ];
    await client.chat.completions.create({
      model: "vision-edges-model",
      messages: [{ role: "user", content }],
    });

    const sent = upstream.receivedRequests
      .slice(baseline)
      .filter((r) => r.path === "/v1/chat/completions");
    expect(sent).toHaveLength(1);
    const body = JSON.parse(sent[0]!.body);

    // (1.a) Three content parts in the exact order the caller sent.
    // A regression that reordered (e.g. text-last) or deduped
    // (e.g. on type) would fail here.
    expect(body.messages[0]?.content).toEqual(content);

    // (1.b) Belt-and-suspenders: red and blue payloads are distinct
    // byte strings on the wire. A regression that "deduped" by
    // image-block hash and reused the first payload twice would
    // pass (1.a) only if it accidentally also reordered the array;
    // this check pins the per-position payload independently.
    const sentParts = body.messages[0]?.content;
    expect(sentParts[1]?.image_url?.url).toBe(DATA_URL_RED);
    expect(sentParts[2]?.image_url?.url).toBe(DATA_URL_BLUE);
    expect(sentParts[1]?.image_url?.url).not.toBe(
      sentParts[2]?.image_url?.url,
    );
  }, 60_000);

  test("(2) `detail` parameter forwarded verbatim per image_url", async (ctx) => {
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
        const probe = await client.chat.completions.create({
          model: "vision-edges-model",
          messages: [{ role: "user", content: "ready-probe-2" }],
        });
        return probe.choices.length > 0;
      } catch {
        return false;
      }
    });

    const baseline = upstream.receivedRequests.length;

    const content = [
      { type: "text" as const, text: "Read this label exactly." },
      {
        type: "image_url" as const,
        // `detail: "high"` is the value that costs the caller more
        // tokens — a regression that dropped the field would
        // silently fall back to "auto" and undercount tokens
        // upstream-side. Pinning the exact string is the contract.
        image_url: { url: DATA_URL_RED, detail: "high" as const },
      },
    ];
    await client.chat.completions.create({
      model: "vision-edges-model",
      messages: [{ role: "user", content }],
    });

    const sent = upstream.receivedRequests
      .slice(baseline)
      .filter((r) => r.path === "/v1/chat/completions");
    expect(sent).toHaveLength(1);
    const body = JSON.parse(sent[0]!.body);

    // (2) `detail` field arrives byte-equal to "high".
    expect(body.messages[0]?.content[1]?.image_url?.detail).toBe(
      "high",
    );
    // Also pin url so a regression that swapped image_url payload
    // for a synthetic placeholder doesn't pass.
    expect(body.messages[0]?.content[1]?.image_url?.url).toBe(
      DATA_URL_RED,
    );
  }, 60_000);

  test("(3) mixed remote URL + base64 in same content array", async (ctx) => {
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
        const probe = await client.chat.completions.create({
          model: "vision-edges-model",
          messages: [{ role: "user", content: "ready-probe-3" }],
        });
        return probe.choices.length > 0;
      } catch {
        return false;
      }
    });

    const baseline = upstream.receivedRequests.length;

    const content = [
      { type: "text" as const, text: "What's the difference?" },
      // Remote http(s) URL: upstream resolves it itself; the
      // gateway must forward the URL as-is (not pre-fetch and
      // re-inline as data:). This pin catches the in-payload
      // mutation form of that regression — a side-channel pre-fetch
      // that leaves the upstream payload intact would still slip
      // through; see file header note.
      { type: "image_url" as const, image_url: { url: REMOTE_URL } },
      // Inlined base64: upstream receives bytes directly.
      { type: "image_url" as const, image_url: { url: DATA_URL_BLUE } },
    ];
    await client.chat.completions.create({
      model: "vision-edges-model",
      messages: [{ role: "user", content }],
    });

    const sent = upstream.receivedRequests
      .slice(baseline)
      .filter((r) => r.path === "/v1/chat/completions");
    expect(sent).toHaveLength(1);
    const body = JSON.parse(sent[0]!.body);

    // (3.a) Both URL types reach upstream verbatim under their
    // original positions.
    expect(body.messages[0]?.content[1]?.image_url?.url).toBe(
      REMOTE_URL,
    );
    expect(body.messages[0]?.content[2]?.image_url?.url).toBe(
      DATA_URL_BLUE,
    );

    // (3.b) Remote URL was NOT inlined into a data: URI in the
    // upstream payload — would happen if the gateway pre-fetched
    // and re-encoded. This is an in-payload check only; a
    // side-channel pre-fetch that left the payload alone would
    // not be caught here (see file header note).
    const remoteSent = body.messages[0]?.content[1]?.image_url?.url;
    expect(remoteSent.startsWith("http")).toBe(true);
    expect(remoteSent.startsWith("data:")).toBe(false);
  }, 60_000);
});
