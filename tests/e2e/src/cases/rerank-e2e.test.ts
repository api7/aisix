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
// User journeys pinned:
//
//   1. OpenAI provider — caller POSTs Cohere-shape rerank request
//      to /v1/rerank against an OpenAI-provider Model. Gateway
//      forwards verbatim to upstream's /v1/rerank with only the
//      `model` field rewritten.
//   2. Cohere provider (#213 Phase 1) — caller POSTs against a
//      Cohere-provider Model. Gateway dispatches to Cohere's
//      `/v1/rerank` with Bearer auth + the rewritten model.
//   3. Jina provider (#213 Phase 2) — caller POSTs against a
//      Jina-provider Model. Gateway dispatches to Jina's
//      `/v1/rerank` (https://api.jina.ai/v1/rerank). Jina's wire
//      shape is identity-mapped to Cohere's (same body fields,
//      same Bearer auth, same `results: [...]` response), so the
//      gateway forwards verbatim with only the `model` field
//      rewritten.
//   4. Non-{OpenAI, Cohere, Jina} provider — gateway rejects at
//      the boundary with 400 + `error.type: "invalid_request_error"`,
//      upstream NOT hit (#212 / #168).
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

  test("Cohere provider: gateway dispatches with Bearer auth + rewritten model (#213 Phase 1)", async (ctx) => {
    if (!etcdReachable || !app || !admin || !upstream) {
      ctx.skip();
      return;
    }

    // Per #213 Phase 1: a Model with `provider: "cohere"` is now a
    // valid configuration on /v1/rerank (parallel to OpenAI). Cohere
    // natively implements the same body shape (`{model, query,
    // documents, top_n}`) at `…/v1/rerank` with `Authorization:
    // Bearer <key>` auth, so the gateway forwards verbatim with the
    // `model` field rewritten — no transform required.
    const cohereSecret = "sk-cohere-mock-e2e";
    const coherePk = await admin.createProviderKey({
      display_name: "rerank-cohere-pk",
      secret: cohereSecret,
      // Operator can also set this to `https://api.cohere.com`; the
      // gateway's `build_v1_url` produces `…/v1/rerank` from either
      // form.
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "rerank-cohere",
      provider: "cohere",
      model_name: "rerank-english-v3.0",
      provider_key_id: coherePk.id,
    });
    const cohereCallerPlaintext = `${CALLER_PLAINTEXT}-cohere`;
    await admin.createApiKey({
      key_hash: createHash("sha256")
        .update(cohereCallerPlaintext)
        .digest("hex"),
      allowed_models: ["rerank-cohere"],
    });

    const headers = {
      authorization: `Bearer ${cohereCallerPlaintext}`,
      "content-type": "application/json",
    };

    // Readiness gate: poll /v1/rerank with the Cohere Model until
    // the gateway returns 200 (Cohere is now allowed). A 400 here
    // would mean the gate is still rejecting Cohere (regression);
    // a 404 would be snapshot-lag.
    await waitConfigPropagation(async () => {
      try {
        const r = await fetch(`${app!.proxyUrl}/v1/rerank`, {
          method: "POST",
          headers,
          body: JSON.stringify({
            model: "rerank-cohere",
            query: "ready-probe",
            documents: ["doc"],
          }),
        });
        if (r.status !== 200) {
          await r.text();
          return false;
        }
        await r.json();
        return true;
      } catch {
        return false;
      }
    });

    const baseline = upstream.receivedRequests.length;
    const requestPayload = {
      model: "rerank-cohere",
      query: "search query",
      documents: ["doc one", "doc two", "doc three"],
      top_n: 2,
    };
    const res = await fetch(`${app.proxyUrl}/v1/rerank`, {
      method: "POST",
      headers,
      body: JSON.stringify(requestPayload),
    });

    expect(res.status).toBe(200);
    await res.json();

    // Upstream wire-shape contract:
    //   - path is `/v1/rerank`
    //   - `Authorization: Bearer <Cohere secret>` (NOT the caller's
    //     plaintext bearer — that would leak the gateway's caller
    //     credential to the upstream provider)
    //   - body: `model` rewritten to upstream model_name; everything
    //     else byte-for-byte
    const testCalls = upstream.receivedRequests
      .slice(baseline)
      .filter((r) => r.path === "/v1/rerank");
    expect(testCalls).toHaveLength(1);
    expect(testCalls[0]?.method).toBe("POST");
    expect(testCalls[0]?.headers["authorization"]).toBe(`Bearer ${cohereSecret}`);
    expect(testCalls[0]?.headers["authorization"]).not.toContain(cohereCallerPlaintext);

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

    // Per #213 audit MEDIUM-2: pin the EXACT field set sent to
    // Cohere. Cohere's `/v1/rerank` documents
    // `{model, query, documents, top_n, return_documents, ...}`
    // — https://docs.cohere.com/reference/rerank. The gateway
    // forwards verbatim, so the upstream body MUST contain ONLY
    // the keys the caller sent — no gateway-injected extras.
    // A future regression that injects an OpenAI-only field
    // (e.g. `dimensions` from embeddings, `stream` from chat)
    // would break Cohere without failing a "happy-path 200"
    // assertion alone.
    const sentKeys = Object.keys(
      sentBody as Record<string, unknown>,
    ).sort();
    expect(sentKeys).toEqual(["documents", "model", "query", "top_n"]);
  });

  test("Jina provider: gateway dispatches with Bearer auth + rewritten model (#213 Phase 2)", async (ctx) => {
    if (!etcdReachable || !app || !admin || !upstream) {
      ctx.skip();
      return;
    }

    // Per #213 Phase 2: a Model with `provider: "jina"` is now a
    // valid configuration on /v1/rerank. Jina's rerank
    // (https://api.jina.ai/v1/rerank) implements the same body
    // shape (`{model, query, documents, top_n, ...}`) at
    // `…/v1/rerank` with Bearer auth — identity-mapped to Cohere /
    // OpenAI-compat. The gateway forwards verbatim with only the
    // `model` field rewritten.
    const jinaSecret = "jina_mock_secret_e2e";
    const jinaPk = await admin.createProviderKey({
      display_name: "rerank-jina-pk",
      secret: jinaSecret,
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "rerank-jina",
      provider: "jina",
      model_name: "jina-reranker-v2-base-multilingual",
      provider_key_id: jinaPk.id,
    });
    const jinaCallerPlaintext = `${CALLER_PLAINTEXT}-jina`;
    await admin.createApiKey({
      key_hash: createHash("sha256")
        .update(jinaCallerPlaintext)
        .digest("hex"),
      allowed_models: ["rerank-jina"],
    });

    const headers = {
      authorization: `Bearer ${jinaCallerPlaintext}`,
      "content-type": "application/json",
    };

    // Readiness gate: poll until 200 from the gateway. A 400 here
    // would mean the gate is rejecting Jina (regression); 404 is
    // snapshot-lag.
    await waitConfigPropagation(async () => {
      try {
        const r = await fetch(`${app!.proxyUrl}/v1/rerank`, {
          method: "POST",
          headers,
          body: JSON.stringify({
            model: "rerank-jina",
            query: "ready-probe",
            documents: ["doc"],
          }),
        });
        if (r.status !== 200) {
          await r.text();
          return false;
        }
        await r.json();
        return true;
      } catch {
        return false;
      }
    });

    const baseline = upstream.receivedRequests.length;
    const requestPayload = {
      model: "rerank-jina",
      query: "search query",
      documents: ["doc one", "doc two", "doc three"],
      top_n: 2,
    };
    const res = await fetch(`${app.proxyUrl}/v1/rerank`, {
      method: "POST",
      headers,
      body: JSON.stringify(requestPayload),
    });

    expect(res.status).toBe(200);
    await res.json();

    // Upstream wire-shape contract:
    //   - path is `/v1/rerank`
    //   - `Authorization: Bearer <Jina secret>` (NOT caller's bearer)
    //   - body: `model` rewritten to upstream model_name; everything
    //     else byte-for-byte
    const testCalls = upstream.receivedRequests
      .slice(baseline)
      .filter((r) => r.path === "/v1/rerank");
    expect(testCalls).toHaveLength(1);
    expect(testCalls[0]?.method).toBe("POST");
    expect(testCalls[0]?.headers["authorization"]).toBe(`Bearer ${jinaSecret}`);
    expect(testCalls[0]?.headers["authorization"]).not.toContain(jinaCallerPlaintext);

    const sentBody = JSON.parse(testCalls[0]!.body) as {
      model?: string;
      query?: string;
      documents?: string[];
      top_n?: number;
    };
    expect(sentBody.model).toBe("jina-reranker-v2-base-multilingual");
    expect(sentBody.query).toBe(requestPayload.query);
    expect(sentBody.documents).toEqual(requestPayload.documents);
    expect(sentBody.top_n).toBe(requestPayload.top_n);

    // Same exact-field-set pin as the Cohere case (#213 audit
    // MEDIUM-2 contract): no gateway-injected extras.
    const sentKeys = Object.keys(
      sentBody as Record<string, unknown>,
    ).sort();
    expect(sentKeys).toEqual(["documents", "model", "query", "top_n"]);
  });

  test("non-OpenAI provider returns 400 invalid_request_error, upstream untouched (#212 / docs §4.7)", async (ctx) => {
    if (!etcdReachable || !app || !admin || !upstream) {
      ctx.skip();
      return;
    }

    // Per docs §4.7 + #168 (closed by #211): /v1/rerank only works
    // with OpenAI providers; non-OpenAI Models reject at the gateway
    // boundary with 400. Pre-#211 the gateway dispatched silently
    // and the upstream returned 404 (Anthropic / DeepSeek have no
    // rerank API; Gemini's OpenAI-compat surface doesn't expose
    // /v1/rerank). #212 covers the e2e gap on this contract.
    const nonOaPk = await admin.createProviderKey({
      display_name: "rerank-anthropic-pk",
      secret: "sk-ant-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "rerank-anthropic",
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: nonOaPk.id,
    });
    const nonOaCaller = `${CALLER_PLAINTEXT}-non-oa`;
    await admin.createApiKey({
      key_hash: createHash("sha256").update(nonOaCaller).digest("hex"),
      allowed_models: ["rerank-anthropic"],
    });

    const headers = {
      authorization: `Bearer ${nonOaCaller}`,
      "content-type": "application/json",
    };

    // Readiness gate: poll until the gateway returns the documented
    // 400 with `error.type: "invalid_request_error"` per docs §2
    // status→type table. A 404 here would be the snapshot-lag
    // "model_not_found" case (which is mapped to 404, NOT 400);
    // probing on 400 + invalid_request_error gates on the model
    // resolving AND the gateway refusing per §4.7.
    await waitConfigPropagation(async () => {
      try {
        const r = await fetch(`${app!.proxyUrl}/v1/rerank`, {
          method: "POST",
          headers,
          body: JSON.stringify({
            model: "rerank-anthropic",
            query: "ready-probe",
            documents: ["doc"],
          }),
        });
        if (r.status !== 400) {
          await r.text();
          return false;
        }
        const j = (await r.json()) as { error?: { type?: unknown } };
        return j.error?.type === "invalid_request_error";
      } catch {
        return false;
      }
    });

    const upstreamHitsBefore = upstream.receivedRequests.length;
    const res = await fetch(`${app.proxyUrl}/v1/rerank`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model: "rerank-anthropic",
        query: "search",
        documents: ["doc1", "doc2"],
      }),
    });

    // Per docs §4.7: non-OpenAI providers return 400. 5xx would mean
    // the gateway crashed; 404 would mean snapshot-lag (caught by
    // the readiness gate above).
    expect(res.status).toBe(400);
    const body = (await res.json()) as {
      error?: { type?: unknown; message?: unknown };
    };
    expect(body.error?.type).toBe("invalid_request_error");
    // Per #168/#211 wording: the rejection message names the
    // OpenAI-only restriction. The "requires OpenAI" substring is
    // a stable marker (vs the OpenAI taxonomy enum string, which
    // a generic invalid-request failure would also carry).
    expect(typeof body.error?.message).toBe("string");
    expect(body.error?.message as string).toMatch(/requires OpenAI/i);

    // Hard contract: upstream must NEVER be hit when the gateway
    // refuses for provider mismatch — otherwise the gateway is
    // billing the caller's quota on a request it claims to reject.
    expect(upstream.receivedRequests.length).toBe(upstreamHitsBefore);
  });
});
