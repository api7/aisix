import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
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

// AISIX-Cloud#1630: a customer relays DashScope's native services (rerank,
// multimodal embedding) and Cohere-style rerank APIs through passthrough
// routes, and every call was recorded with zero tokens and no model. None of
// those request bodies is an LLM envelope, so the route treated them as
// opaque and never read the response's usage.
//
// Each upstream below answers with the usage block the real provider
// returns for that API, and the recorded UsageEvent must carry the tokens
// the provider reported and the model the caller asked for:
//
//   - Jina rerank: `usage.total_tokens`
//   - Cohere rerank: `meta.billed_units.input_tokens`
//   - DashScope native text rerank: only `usage.total_tokens`
//   - DashScope native multimodal embedding: a flat `image_tokens` counted
//     beside `input_tokens` (44 + 64 = 108 = total_tokens)
//   - DashScope tongyi-embedding-vision: `input_tokens` already includes the
//     nested `input_tokens_details.image_tokens` breakdown (903, not 1799)
//   - DashScope native multimodal generation: the flat `image_tokens` is
//     already inside `input_tokens` (79, not 145), with its cache-hit and
//     reasoning counts carried through
//   - DashScope embedding with no `total_tokens`: images beside the text
//     (5 + 128 = 133)
//
// A streamed response to a newly recognised body is metered exactly as any
// opaque stream: DashScope native frames carry cumulative usage, and the
// total-less-completion prompt rule is a unary-response rule.

const CALLER_PLAINTEXT = "sk-ptr-rerank-ds-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const CREDENTIAL_REF = "ptrrerankds";
const LOGSTORE = "ptr-rerank-ds";

interface Case {
  route: string;
  prefix: string;
  upstream: () => Promise<OpenAiUpstream>;
}

const CASES: Case[] = [
  {
    route: "ptr-jina-rerank",
    prefix: "/jina",
    upstream: () =>
      startOpenAiUpstream({
        nonStreamBody: {
          model: "jina-reranker-v2-base-multilingual",
          usage: { total_tokens: 37 },
          results: [{ index: 0, relevance_score: 0.9 }],
        },
      }),
  },
  {
    route: "ptr-cohere-rerank",
    prefix: "/cohere",
    upstream: () =>
      startOpenAiUpstream({
        nonStreamBody: {
          id: "cohere-rr-1",
          results: [{ index: 0, relevance_score: 0.8 }],
          meta: { api_version: { version: "2" }, billed_units: { search_units: 1, input_tokens: 51 } },
        },
      }),
  },
  {
    route: "ptr-ds-text-rerank",
    prefix: "/ds-rerank",
    upstream: () =>
      startOpenAiUpstream({
        nonStreamBody: {
          output: { results: [{ index: 0, relevance_score: 0.7 }] },
          usage: { total_tokens: 29 },
          request_id: "ds-rr-1",
        },
      }),
  },
  {
    route: "ptr-ds-mm-embed",
    prefix: "/ds-mm-embed",
    upstream: () =>
      startOpenAiUpstream({
        nonStreamBody: {
          output: { embeddings: [{ index: 0, embedding: [0.1, 0.2], type: "text" }] },
          usage: { input_tokens: 44, image_tokens: 64, total_tokens: 108 },
          request_id: "ds-emb-1",
        },
      }),
  },
  {
    route: "ptr-ds-vision-embed",
    prefix: "/ds-vision-embed",
    upstream: () =>
      startOpenAiUpstream({
        nonStreamBody: {
          output: { embeddings: [{ index: 0, embedding: [0.3], type: "image" }] },
          usage: {
            input_tokens: 903,
            input_tokens_details: { image_tokens: 896, text_tokens: 7 },
            output_tokens: 3,
            total_tokens: 906,
          },
          request_id: "ds-emb-2",
        },
      }),
  },
  {
    route: "ptr-ds-mm-gen",
    prefix: "/ds-mm-gen",
    upstream: () =>
      startOpenAiUpstream({
        nonStreamBody: {
          output: {
            choices: [
              { finish_reason: "stop", message: { role: "assistant", content: [{ text: "a dog" }] } },
            ],
          },
          usage: {
            input_tokens: 79,
            image_tokens: 66,
            input_tokens_details: { image_tokens: 66, text_tokens: 13 },
            prompt_tokens_details: { cached_tokens: 40 },
            output_tokens: 14,
            output_tokens_details: { text_tokens: 8, reasoning_tokens: 6 },
            total_tokens: 93,
          },
          request_id: "ds-gen-1",
        },
      }),
  },
  {
    route: "ptr-ds-no-total",
    prefix: "/ds-no-total",
    upstream: () =>
      startOpenAiUpstream({
        nonStreamBody: {
          output: { embeddings: [{ index: 0, embedding: [0.5], type: "image" }] },
          usage: { duration: 0, image_count: 1, image_tokens: 128, input_tokens: 5 },
          request_id: "ds-emb-3",
        },
      }),
  },
  {
    route: "ptr-ds-stream",
    prefix: "/ds-stream",
    upstream: () =>
      startOpenAiUpstream({
        rawStreamFrames: [1, 2, 3].map(
          (n) =>
            `data:${JSON.stringify({
              output: { text: "x".repeat(n) },
              usage: { input_tokens: 44, image_tokens: 64, output_tokens: n, total_tokens: 108 + n },
              request_id: "ds-stream-1",
            })}\n\n`,
        ),
      }),
  },
];

describe("passthrough route meters rerank and DashScope native bodies", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "sls-ptr-rerank-ds",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "aisix-e2e",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });
    const pk = await seed.createProviderKey({
      display_name: "ptr-rerank-ds-pk",
      secret: "sk-mock",
      api_base: "http://unused-on-routes",
    });
    for (const c of CASES) {
      const upstream = await c.upstream();
      upstreams.push(upstream);
      await seed.createPassthroughRoute({
        name: c.route,
        path_prefix: c.prefix,
        target_url: upstream.baseUrl,
        provider_key_id: pk.id,
      });
    }
    // Seeded last: its authenticating implies every route is loaded.
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
      allowed_routes: ["*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
    await sls?.close();
  });

  async function call(path: string, body: unknown): Promise<Map<string, string>> {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    });
    await res.arrayBuffer();
    expect(res.status).toBe(200);
    const route = CASES.find((c) => path.startsWith(`${c.prefix}/`))!.route;
    return waitForSlsLog(
      sls!,
      LOGSTORE,
      (log) => log.get("passthrough_route_name") === route,
      `a usage row for route ${route}`,
    );
  }

  const tokens = (log: Map<string, string>) => ({
    prompt: Number(log.get("prompt_tokens")),
    completion: Number(log.get("completion_tokens")),
    model: log.get("requested_model"),
    operation: log.get("operation"),
  });

  test("each recognised body records the provider's tokens and the caller's model", async (ctx) => {
    if (!etcdReachable || !app || !sls) {
      ctx.skip();
      return;
    }
    await waitConfigPropagation(async () => {
      const r = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      await r.text();
      return r.status === 200;
    });

    const jina = await call("/jina/v1/rerank", {
      model: "jina-reranker-v2-base-multilingual",
      query: "what is a gateway",
      documents: ["an API gateway", "a garden gate"],
      top_n: 1,
    });
    expect(tokens(jina)).toEqual({
      prompt: 37,
      completion: 0,
      model: "jina-reranker-v2-base-multilingual",
      operation: "passthrough",
    });

    const cohere = await call("/cohere/v2/rerank", {
      model: "rerank-v3.5",
      query: "what is a gateway",
      documents: ["an API gateway", "a garden gate"],
    });
    expect(tokens(cohere)).toEqual({
      prompt: 51,
      completion: 0,
      model: "rerank-v3.5",
      operation: "passthrough",
    });

    const dsRerank = await call("/ds-rerank/api/v1/services/rerank/text-rerank/text-rerank", {
      model: "gte-rerank-v2",
      input: { query: "what is a gateway", documents: ["an API gateway", "a garden gate"] },
      parameters: { return_documents: false, top_n: 1 },
    });
    expect(tokens(dsRerank)).toEqual({
      prompt: 29,
      completion: 0,
      model: "gte-rerank-v2",
      operation: "passthrough",
    });

    const mmEmbed = await call(
      "/ds-mm-embed/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding",
      {
        model: "qwen3-vl-embedding",
        input: {
          contents: [{ text: "hello world" }, { image: "https://example.com/a.png" }],
        },
      },
    );
    expect(tokens(mmEmbed)).toEqual({
      prompt: 108,
      completion: 0,
      model: "qwen3-vl-embedding",
      operation: "passthrough",
    });

    const visionEmbed = await call(
      "/ds-vision-embed/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding",
      {
        model: "tongyi-embedding-vision-plus",
        input: { contents: [{ image: "https://example.com/b.png" }] },
      },
    );
    expect(tokens(visionEmbed)).toEqual({
      prompt: 903,
      completion: 3,
      model: "tongyi-embedding-vision-plus",
      operation: "passthrough",
    });

    // Multimodal generation counts its flat image tokens INSIDE
    // input_tokens (79 = 66 image + 13 text), and its cache hit and
    // reasoning breakdowns reach the event like any other usage report.
    const generation = await call(
      "/ds-mm-gen/api/v1/services/aigc/multimodal-generation/generation",
      {
        model: "qwen3-vl-plus",
        input: {
          messages: [
            {
              role: "user",
              content: [{ image: "https://example.com/c.png" }, { text: "what is this" }],
            },
          ],
        },
      },
    );
    expect(tokens(generation)).toEqual({
      prompt: 79,
      completion: 14,
      model: "qwen3-vl-plus",
      operation: "passthrough",
    });
    expect(Number(generation.get("cached_prompt_tokens"))).toBe(40);
    expect(Number(generation.get("reasoning_tokens"))).toBe(6);

    // An embedding model that reports no total counts its image tokens
    // beside the text ones.
    const noTotal = await call(
      "/ds-no-total/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding",
      {
        model: "multimodal-embedding-v1",
        input: { contents: [{ text: "hello" }, { image: "https://example.com/d.png" }] },
      },
    );
    expect(tokens(noTotal)).toEqual({
      prompt: 133,
      completion: 0,
      model: "multimodal-embedding-v1",
      operation: "passthrough",
    });
  });

  test("a streamed response to a recognised body is metered as an opaque stream", async (ctx) => {
    if (!etcdReachable || !app || !sls) {
      ctx.skip();
      return;
    }
    await waitConfigPropagation(async () => {
      const r = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      await r.text();
      return r.status === 200;
    });

    const streamed = await call(
      "/ds-stream/api/v1/services/aigc/multimodal-generation/generation",
      {
        model: "qwen3-vl-plus",
        input: { messages_ref: "opaque" },
        parameters: { incremental_output: false },
      },
    );
    // The caller's model attributes the row; the tokens are what an opaque
    // stream reads from its `usage` objects (`input_tokens`) — the unary
    // total-less-completion rule (108) is not applied to stream frames.
    expect(tokens(streamed)).toEqual({
      prompt: 44,
      completion: 3,
      model: "qwen3-vl-plus",
      operation: "passthrough",
    });
  });
});
