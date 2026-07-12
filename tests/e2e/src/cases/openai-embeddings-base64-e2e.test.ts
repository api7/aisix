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

// E2E: OpenAI client → OpenAI upstream — `/v1/embeddings` accepts both
// `encoding_format: "float"` AND `encoding_format: "base64"`.
//
// Pins issue #393: the gateway-internal wire type for the upstream
// embedding response was previously `Vec<f32>` (strict-typed). When
// the upstream returns base64 (which the OpenAI SDK requests BY
// DEFAULT — `client.embeddings.create({ model, input })` without an
// explicit `encoding_format` sends `encoding_format: "base64"`),
// the deserializer rejected the JSON string shape and the gateway
// surfaced `502 upstream_decode_error`. This made the default SDK
// embeddings path unusable.
//
// What this spec proves:
//
//   1. `encoding_format: "float"` round-trip — vectors come back as a
//      JSON array of numbers. (Pre-fix this worked; pin it so a
//      future refactor of the enum can't regress the float path.)
//   2. `encoding_format: "base64"` round-trip — base64 string from
//      the upstream reaches the SDK verbatim. (The #393 bug.)
//   3. SDK default (no `encoding_format` argument) — the OpenAI
//      client library defaults to `base64`. A vanilla
//      `embeddings.create({ model, input })` MUST return 200 with
//      the base64-string variant. This is the customer-facing
//      contract the bug broke.
//
// References:
// - OpenAI embeddings request/response shape:
//   <https://platform.openai.com/docs/api-reference/embeddings>
// - OpenAI Python SDK default encoding_format (base64 since v1.0):
//   <https://github.com/openai/openai-python/blob/main/src/openai/resources/embeddings.py>
//
// Source-blind discipline: this spec asserts on the SDK-observed
// shape only. The gateway's internal enum representation is an
// implementation detail; the SDK contract is "embedding can be
// either array-of-numbers OR base64 string per OpenAI's documented
// `string | array` union".

const CALLER_PLAINTEXT = "sk-openai-embed-b64-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

// A short fixed-content embedding response is fine — the test
// doesn't validate numerical accuracy, just the shape that reaches
// the SDK.
const FLOAT_VECTOR = [0.1, -0.2, 0.3, 0.4, -0.5];

// Base64 encoding of a small Float32Array — what OpenAI would
// emit on a base64 request. Hand-pinned: a 5-float vector encoded
// as little-endian Float32 occupies 20 bytes; base64 of 20 bytes
// is 28 chars (incl. padding). We don't need a "correct" base64
// for any particular floats — just a valid base64-shaped string
// that the gateway must pass through verbatim.
const BASE64_VECTOR = "zczMPc3MzL3NzEw+zcxMPgAAAL8=";

const FLOAT_RESPONSE = {
  object: "list",
  model: "text-embedding-3-small",
  data: [
    {
      index: 0,
      object: "embedding",
      embedding: FLOAT_VECTOR,
    },
  ],
  usage: { prompt_tokens: 2, total_tokens: 2 },
};

const BASE64_RESPONSE = {
  object: "list",
  model: "text-embedding-3-small",
  data: [
    {
      index: 0,
      object: "embedding",
      embedding: BASE64_VECTOR,
    },
  ],
  usage: { prompt_tokens: 2, total_tokens: 2 },
};

describe("OpenAI /v1/embeddings encoding_format (#393)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // Scripted upstream — readiness probe (chat) consumes step 0
    // with a plain content response. The three test calls consume
    // steps 1-3, each with the encoding-format-matched response.
    upstream = await startOpenAiUpstream({
      scriptedResponses: [
        // Readiness probe via chat (the harness's waitConfigPropagation
        // sends a /v1/chat/completions request — see existing specs).
        {
          nonStreamBody: {
            id: "chatcmpl-ready",
            object: "chat.completion",
            created: 0,
            model: "text-embedding-3-small",
            choices: [
              {
                index: 0,
                message: { role: "assistant", content: "ready" },
                finish_reason: "stop",
              },
            ],
            usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
          },
        },
        // Test 1: explicit encoding_format: "float" → float array.
        { nonStreamBody: FLOAT_RESPONSE },
        // Test 2: explicit encoding_format: "base64" → base64 string.
        { nonStreamBody: BASE64_RESPONSE },
        // Test 3: SDK default (no encoding_format) → base64 string.
        { nonStreamBody: BASE64_RESPONSE },
      ],
      nonStreamBody: FLOAT_RESPONSE,
    });
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "openai-embed-b64-pk",
      secret: "sk-openai-mock",
      api_base: upstream.baseUrl,
    });
    await seed.createModel({
      display_name: "openai-embed-b64",
      provider: "openai",
      model_name: "text-embedding-3-small",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["openai-embed-b64"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("encoding_format=float → vectors come back as a JSON array", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // Readiness probe — consume scripted step 0.
    await waitConfigPropagation(async () => {
      try {
        const r = await client.chat.completions.create({
          model: "openai-embed-b64",
          messages: [{ role: "user", content: "ready-probe" }],
        });
        return r.choices[0]?.message.role === "assistant";
      } catch {
        return false;
      }
    });

    // Float request — consume scripted step 1.
    const resp = await client.embeddings.create({
      model: "openai-embed-b64",
      input: "hello",
      encoding_format: "float",
    });

    expect(resp.data).toHaveLength(1);
    expect(resp.data[0]?.object).toBe("embedding");
    // Float request → SDK observes a number[] on `embedding`.
    expect(Array.isArray(resp.data[0]?.embedding)).toBe(true);
    const v = resp.data[0]?.embedding as unknown as number[];
    expect(v).toHaveLength(FLOAT_VECTOR.length);
    expect(v[0]).toBeCloseTo(FLOAT_VECTOR[0]!, 5);
    expect(v[4]).toBeCloseTo(FLOAT_VECTOR[4]!, 5);
  });

  test("encoding_format=base64 → base64 string preserved end-to-end (#393)", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // Explicit base64 request — consume scripted step 2.
    const resp = await client.embeddings.create({
      model: "openai-embed-b64",
      input: "hello",
      encoding_format: "base64",
    });

    expect(resp.data).toHaveLength(1);
    // Base64 request → SDK observes a string on `embedding`. Pre-#393
    // this would never reach the SDK — the gateway 502'd at
    // deserialize time inside the OpenAI bridge.
    expect(typeof resp.data[0]?.embedding).toBe("string");
    expect(resp.data[0]?.embedding).toBe(BASE64_VECTOR);
  });

  test("SDK default (no encoding_format) → base64 round-trip works (#393 customer path)", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    // The OpenAI SDK defaults `encoding_format` to "base64" since
    // v1.0. A vanilla call without specifying the format MUST
    // succeed — this is the path 99% of customer code hits, and
    // the path #393 broke for every customer.
    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    const resp = await client.embeddings.create({
      model: "openai-embed-b64",
      input: "hello",
      // NO encoding_format argument — SDK default.
    });

    expect(resp.data).toHaveLength(1);
    // SDK default is base64 → gateway must surface a string here.
    expect(typeof resp.data[0]?.embedding).toBe("string");
    expect(resp.data[0]?.embedding).toBe(BASE64_VECTOR);
  });
});
