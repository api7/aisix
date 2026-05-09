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

// E2E: /v1/embeddings end-to-end. Embeddings is one of the two
// most-used LLM API surfaces (alongside chat completions) — every
// RAG / semantic-search application in production hits this
// endpoint thousands of times per request batch. Prior to this
// file, the gateway had **zero** e2e coverage on /v1/embeddings.
//
// One user journey pinned:
//
//   - Array input — `client.embeddings.create({input: [s1, s2, s3]})`
//     returns N embeddings in the SAME order as the input array.
//     A regression that re-ordered, deduplicated, or truncated the
//     array would break every batched embedding caller.
//
// (The "single-string input" case is held back pending a product
// fix — the gateway currently normalises single-string `input` into
// a single-element array on the upstream wire, contradicting the
// gateway's own published contract in `docs/api-proxy.md` §4.4
// ("both pass through"). See follow-up issue.)
//
// The case pins the upstream-side wire shape: the gateway hits
// `/v1/embeddings` (not `/v1/chat/completions`), the body is
// OpenAI-shape with the configured upstream model id (display name
// → upstream model_name translation), and the caller's input
// reaches the upstream verbatim.
//
// References:
// - OpenAI Embeddings API spec
//   <https://platform.openai.com/docs/api-reference/embeddings/create>
// - Gateway's own /v1/embeddings contract: `docs/api-proxy.md` §4.4
// - OpenAI Node SDK embeddings client
//   <https://github.com/openai/openai-node/blob/master/src/resources/embeddings.ts>

const CALLER_PLAINTEXT = "sk-emb-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

// Pinning the upstream's exact embedding values (not the harness
// default) so a regression that synthesized different vectors or
// re-normalized them would surface here.
const VEC_HELLO = [0.11, 0.22, 0.33, 0.44, 0.55];
const VEC_WORLD = [-0.5, -0.4, -0.3, -0.2, -0.1];
const VEC_FOO = [0.01, 0.02, 0.03, 0.04, 0.05];

describe("embeddings e2e: /v1/embeddings dispatch + response passthrough", () => {
  let app: SpawnedApp | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  test("array input: N embeddings returned in the SAME ORDER as input array", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    // Three distinct vectors, one per input string. The order MUST
    // be preserved on the way out — callers index into `data[i]`
    // assuming `data[i]` corresponds to `input[i]`. A regression
    // that re-sorted by hash, deduped, or batched-out-of-order
    // would silently break every consumer doing per-input lookups
    // (e.g. RAG callers building a `{document: vector}` map).
    const upstream = await startOpenAiUpstream({
      nonStreamBody: {
        object: "list",
        data: [
          { object: "embedding", index: 0, embedding: VEC_HELLO },
          { object: "embedding", index: 1, embedding: VEC_WORLD },
          { object: "embedding", index: 2, embedding: VEC_FOO },
        ],
        model: "text-embedding-3-small",
        usage: { prompt_tokens: 3, total_tokens: 3 },
      },
    });
    upstreams.push(upstream);

    const pk = await admin.createProviderKey({
      display_name: "emb-array-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "emb-array",
      provider: "openai",
      model_name: "text-embedding-3-small",
      provider_key_id: pk.id,
    });

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    await waitConfigPropagation(async () => {
      try {
        const r = await client.embeddings.create({
          model: "emb-array",
          input: ["ready-probe"],
        });
        // Shape-check the probe response so a half-propagated
        // snapshot (e.g. 200 OK with a malformed body) doesn't
        // falsely report ready.
        return r.object === "list" && Array.isArray(r.data) && r.data.length > 0;
      } catch {
        return false;
      }
    });

    const inputs = ["hello", "world", "foo"];
    const baseline = upstream.receivedRequests.length;
    const response = await client.embeddings.create({
      model: "emb-array",
      input: inputs,
    });

    expect(response.data).toHaveLength(inputs.length);
    // Order preservation: data[i].index === i AND each vector
    // matches what the upstream emitted at the same index. Pin
    // `object: "embedding"` on every element per OpenAI Embeddings
    // spec — a regression that emitted the field on only some
    // elements (or substituted the wrong literal) would slip past
    // a single-element check.
    expect(response.data[0]?.object).toBe("embedding");
    expect(response.data[0]?.index).toBe(0);
    expect(response.data[0]?.embedding).toEqual(VEC_HELLO);
    expect(response.data[1]?.object).toBe("embedding");
    expect(response.data[1]?.index).toBe(1);
    expect(response.data[1]?.embedding).toEqual(VEC_WORLD);
    expect(response.data[2]?.object).toBe("embedding");
    expect(response.data[2]?.index).toBe(2);
    expect(response.data[2]?.embedding).toEqual(VEC_FOO);
    expect(response.usage?.prompt_tokens).toBe(3);
    expect(response.usage?.total_tokens).toBe(3);

    const testCalls = upstream.receivedRequests
      .slice(baseline)
      .filter((r) => r.path === "/v1/embeddings");
    expect(testCalls).toHaveLength(1);

    // The full input array reached the upstream in the original
    // order. A regression that re-ordered the array on the way out
    // (e.g. sort-for-cache-stability) would corrupt the index→input
    // mapping the caller assumes.
    const sentBody = JSON.parse(testCalls[0]!.body) as {
      model?: string;
      input?: string[];
    };
    expect(sentBody.model).toBe("text-embedding-3-small");
    expect(sentBody.input).toEqual(inputs);
  });
});
