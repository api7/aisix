import { createHash } from "node:crypto";
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

// E2E: /v1/rerank end-to-end. Per gateway docs `docs/api-proxy.md`
// §4.7:
//
//   > Cohere-style rerank. Routed to `{base}/v1/rerank`. The Model's
//   > provider supplies the API key; the request body is forwarded
//   > verbatim after rewriting the `model` field.
//
// Prior to this file, the gateway had **zero** e2e coverage on
// /v1/rerank. Rerank is the standard relevance-scoring step in
// modern RAG pipelines: a retriever returns N candidate documents,
// rerank scores them against the query, the top-K go to the LLM.
// Without e2e coverage, regressions on this path would silently
// corrupt RAG quality.
//
// One user journey pinned:
//
//   - Caller POSTs Cohere-shape rerank request to /v1/rerank.
//     Gateway forwards verbatim to upstream's /v1/rerank with only
//     the `model` field rewritten to the upstream model_name.
//     Caller receives upstream's response back unchanged.
//
// References:
// - Gateway's own /v1/rerank contract: `docs/api-proxy.md` §4.7
// - Cohere Rerank API spec: <https://docs.cohere.com/reference/rerank>

const CALLER_PLAINTEXT = "sk-rerank-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("rerank e2e: /v1/rerank verbatim forward + model translation", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    // Mock upstream returns a canned Cohere-shape rerank response
    // so a regression that synthesised different scores or shuffled
    // results would surface here.
    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "rerank-resp-01",
        results: [
          { index: 2, relevance_score: 0.92 },
          { index: 0, relevance_score: 0.78 },
          { index: 1, relevance_score: 0.31 },
        ],
        meta: {
          api_version: { version: "1" },
          billed_units: { search_units: 1 },
        },
      },
    });
    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "rerank-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "rerank-e2e",
      provider: "openai",
      model_name: "rerank-english-v3.0",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["rerank-e2e"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("Cohere-shape rerank: caller's body verbatim + model translated, response byte-for-byte", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const headers = {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    };

    // Readiness gate: poll /v1/rerank until 200 with the canned
    // body. A 200 with a different shape would be the gateway
    // interfering (which §4.7 says it must not — "forwarded
    // verbatim").
    await waitConfigPropagation(async () => {
      try {
        const r = await fetch(`${app!.proxyUrl}/v1/rerank`, {
          method: "POST",
          headers,
          body: JSON.stringify({
            model: "rerank-e2e",
            query: "ready-probe",
            documents: ["doc"],
          }),
        });
        if (r.status !== 200) {
          await r.text();
          return false;
        }
        const j = (await r.json()) as { id?: unknown };
        return j.id === "rerank-resp-01";
      } catch {
        return false;
      }
    });

    const baseline = upstream.receivedRequests.length;
    const requestPayload = {
      model: "rerank-e2e",
      query: "What is the capital of France?",
      documents: [
        "Berlin is the capital of Germany.",
        "London is the capital of the United Kingdom.",
        "Paris is the capital of France.",
      ],
      top_n: 3,
    };
    const res = await fetch(`${app.proxyUrl}/v1/rerank`, {
      method: "POST",
      headers,
      body: JSON.stringify(requestPayload),
    });

    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      id?: unknown;
      results?: Array<{ index?: unknown; relevance_score?: unknown }>;
      meta?: unknown;
    };
    // Caller-side: response byte-for-byte from upstream. Per docs
    // §4.7 the gateway is a pass-through for the response body;
    // any normalisation here would silently change relevance
    // scores and break RAG ranking.
    expect(body.id).toBe("rerank-resp-01");
    expect(body.results).toHaveLength(3);
    expect(body.results?.[0]?.index).toBe(2);
    expect(body.results?.[0]?.relevance_score).toBe(0.92);
    expect(body.results?.[1]?.index).toBe(0);
    expect(body.results?.[1]?.relevance_score).toBe(0.78);
    expect(body.results?.[2]?.index).toBe(1);
    expect(body.results?.[2]?.relevance_score).toBe(0.31);
    expect(body.meta).toBeDefined();

    // Dispatch contract: gateway hit `/v1/rerank` exactly once,
    // not /v1/chat/completions or /v1/embeddings. A regression
    // that mis-routed would change the path here.
    const testCalls = upstream.receivedRequests
      .slice(baseline)
      .filter((r) => r.path === "/v1/rerank");
    expect(testCalls).toHaveLength(1);
    expect(testCalls[0]?.method).toBe("POST");
    expect(testCalls[0]?.headers["authorization"]).toBe("Bearer sk-mock");

    // Body contract per docs §4.7: forwarded verbatim after
    // rewriting the `model` field. Verify:
    //   - `model` rewritten to upstream model_name
    //   - everything else byte-for-byte (query, documents, top_n)
    const sentBody = JSON.parse(testCalls[0]!.body) as {
      model?: string;
      query?: string;
      documents?: string[];
      top_n?: number;
    };
    expect(sentBody.model).toBe("rerank-english-v3.0");
    expect(sentBody.query).toBe(requestPayload.query);
    expect(sentBody.documents).toEqual(requestPayload.documents);
    expect(sentBody.top_n).toBe(requestPayload.top_n);
  });
});
