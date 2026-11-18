import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  slsLogsFor,
  spawnApp,
  startMockSls,
  waitConfigPropagation,
  waitForSlsLog,
  type MockSls,
  type SpawnedApp,
} from "../harness/index.js";

// This file deliberately does not use Vitest's ordinary *.test.ts naming:
// normal CI must not discover a credentialed provider smoke by accident.
// .github/workflows/semantic-embedding-real-provider.yml runs it only after
// the protected real-provider-e2e environment has released its secret.
//
// The upstream embedding and chat calls are real OpenAI API requests. The
// local SLS receiver only observes the gateway's exported usage events — the
// request's own and the child event of each gateway embedding
// (AISIX-Cloud#2075); it never stands in for either upstream provider.

const OPENAI_API_KEY = process.env.AISIX_E2E_OPENAI_API_KEY;
const OPENAI_EMBEDDING_MODEL =
  process.env.AISIX_E2E_OPENAI_EMBEDDING_MODEL ?? "text-embedding-3-small";
const OPENAI_CHAT_MODEL = process.env.AISIX_E2E_OPENAI_CHAT_MODEL ?? "gpt-4o-mini";
const OPENAI_EMBEDDING_DIMENSIONS = Number(
  process.env.AISIX_E2E_OPENAI_EMBEDDING_DIMENSIONS ?? "1536",
);

if (!OPENAI_API_KEY) {
  throw new Error(
    "AISIX_E2E_OPENAI_API_KEY is required by the protected real-provider smoke workflow",
  );
}
if (!Number.isSafeInteger(OPENAI_EMBEDDING_DIMENSIONS) || OPENAI_EMBEDDING_DIMENSIONS < 1) {
  throw new Error("AISIX_E2E_OPENAI_EMBEDDING_DIMENSIONS must be a positive integer");
}

const CALLER_PLAINTEXT = "sk-semantic-embedding-real-provider-1747";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const CREDENTIAL_REF = "real_provider_capture";
const SLS_PROJECT = "aisix-e2e-obs";
const LOGSTORE = "semantic-embedding-real-provider-1747";
const EMBEDDING_MODEL = "semantic-embedding-real-provider-1747-embedding";
const ROUTER_MODEL = "semantic-embedding-real-provider-1747-router";
const TARGET_MODEL = "semantic-embedding-real-provider-1747-target";
const ROUTE_NAME = "semantic-embedding-real-provider-route";
const ROUTE_PROMPT = "semantic embedding telemetry real provider smoke";
const GUARDRAIL_MODEL = "semantic-embedding-real-provider-1747-guardrail";
const GUARDRAIL_EXAMPLE = "semantic embedding telemetry guardrail prototype";
const GUARDRAIL_PROMPT = "semantic embedding telemetry guardrail candidate";
const CACHE_MODEL = "semantic-embedding-real-provider-1747-cache";
const CACHE_PROMPT = "semantic embedding telemetry cache exact hit";
const BARRIER_PROMPT = "semantic embedding telemetry export barrier";

type Row = Map<string, string>;

const isChild = (row: Row) => row.has("internal_call_purpose");

/** A gateway embedding's child usage event, checked against what a real provider reports. */
function expectEmbeddingChild(child: Row, parent: Row, purpose: string, modelID: string): void {
  expect(child.get("internal_call_purpose")).toBe(purpose);
  expect(child.get("request_id")).toBe(parent.get("request_id"));
  expect(child.get("api_key_id")).toBe(parent.get("api_key_id"));
  expect(child.get("is_final_attempt")).toBe("false");
  expect(child.get("operation")).toBe("embeddings");
  expect(child.get("model_id")).toBe(modelID);
  expect(child.get("requested_model_id")).toBe(modelID);
  expect(child.get("status_code")).toBe("200");
  // OpenAI's embeddings response reports usage. Requiring it catches a real
  // upstream wire or parser regression rather than accepting a locally
  // manufactured child record.
  expect(Number(child.get("prompt_tokens"))).toBeGreaterThan(0);
  expect(Number(child.get("upstream_latency_ms"))).toBeGreaterThan(0);
}

describe("semantic embedding usage with a real OpenAI provider (#1747)", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let embeddingModelID = "";
  let targetModelID = "";
  let guardrailModelID = "";
  let cacheModelID = "";

  async function chat(model: string, prompt: string): Promise<{
    status: number;
    requestId: string;
    servedBy: string | null;
    cache: string | null;
    cacheLayer: string | null;
  }> {
    const response = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
      },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: prompt }],
        max_tokens: 8,
      }),
    });
    await response.arrayBuffer();
    return {
      status: response.status,
      requestId: response.headers.get("x-aisix-request-id") ?? "",
      servedBy: response.headers.get("x-aisix-served-by"),
      cache: response.headers.get("x-aisix-cache"),
      cacheLayer: response.headers.get("x-aisix-cache-layer"),
    };
  }

  /** The child usage events of `requestId`; read after [`usageRow`]. */
  function childrenOf(requestId: string): Row[] {
    return slsLogsFor(sls!, LOGSTORE).filter(
      (candidate) => candidate.get("request_id") === requestId && isChild(candidate),
    );
  }

  async function usageRow(requestId: string): Promise<Map<string, string>> {
    expect(requestId, "the gateway stamps the real-provider parent request id").not.toBe("");
    const row = await waitForSlsLog(
      sls!,
      LOGSTORE,
      (candidate) => candidate.get("request_id") === requestId && !isChild(candidate),
      `the real-provider parent UsageEvent for ${requestId}`,
      30_000,
    );

    // The exporter queue is FIFO. A later direct request makes the exact-one
    // assertion below a stable statement about the routed request rather than
    // a snapshot while its parent event is still being delivered.
    const barrier = await chat(TARGET_MODEL, BARRIER_PROMPT);
    expect(barrier.status, "the real-provider SLS barrier must complete").toBe(200);
    expect(barrier.requestId, "the real-provider SLS barrier has a request id").not.toBe("");
    await waitForSlsLog(
      sls!,
      LOGSTORE,
      (candidate) => candidate.get("request_id") === barrier.requestId,
      `the real-provider barrier UsageEvent for ${barrier.requestId}`,
      30_000,
    );
    return row;
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    expect(await etcd.ping(), "the real-provider smoke requires etcd").toBe(true);

    sls = await startMockSls();
    app = await spawnApp({
      exposeDebugHeaders: true,
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "real-provider-capture-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "real-provider-capture-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);

    await seed.createObservabilityExporter({
      name: "semantic-embedding-real-provider-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });
    const providerKey = await seed.createProviderKey({
      display_name: "semantic-embedding-real-provider-openai",
      secret: OPENAI_API_KEY,
      api_base: "https://api.openai.com/v1",
    });
    const embedding = await seed.createModel({
      display_name: EMBEDDING_MODEL,
      provider: "openai",
      model_name: OPENAI_EMBEDDING_MODEL,
      provider_key_id: providerKey.id,
      embedding: { dimensions: OPENAI_EMBEDDING_DIMENSIONS, normalize: true },
    });
    embeddingModelID = embedding.id;
    const target = await seed.createModel({
      display_name: TARGET_MODEL,
      provider: "openai",
      model_name: OPENAI_CHAT_MODEL,
      provider_key_id: providerKey.id,
    });
    targetModelID = target.id;
    const guardrailTarget = await seed.createModel({
      display_name: GUARDRAIL_MODEL,
      provider: "openai",
      model_name: OPENAI_CHAT_MODEL,
      provider_key_id: providerKey.id,
    });
    guardrailModelID = guardrailTarget.id;
    const cacheTarget = await seed.createModel({
      display_name: CACHE_MODEL,
      provider: "openai",
      model_name: OPENAI_CHAT_MODEL,
      provider_key_id: providerKey.id,
    });
    cacheModelID = cacheTarget.id;
    await seed.createModel({
      display_name: ROUTER_MODEL,
      semantic: {
        embedding_model: EMBEDDING_MODEL,
        routes: [
          {
            name: ROUTE_NAME,
            target: TARGET_MODEL,
            examples: [ROUTE_PROMPT],
            threshold: 0.999,
          },
        ],
        default: TARGET_MODEL,
        match: { threshold: 0.999 },
      },
    });
    const guardrail = await seed.createGuardrail(
      {
        enabled: true,
        name: "semantic-embedding-real-provider-usage-guardrail",
        hook_point: "input",
        // The smoke needs the target request to complete regardless of the
        // provider's exact cosine result; its assertion is the recorded child
        // work, not a particular classification verdict.
        enforcement_mode: "monitor",
        kind: "semantic",
        embedding_model: EMBEDDING_MODEL,
        deny_examples: [GUARDRAIL_EXAMPLE],
        deny_threshold: 0.999,
      },
      { attach: false },
    );
    await seed.attachGuardrailToModel(guardrail.id, guardrailModelID);
    await seed.createCachePolicy({
      name: "semantic-embedding-real-provider-usage-cache",
      backend: "memory",
      applies_to: `model:${CACHE_MODEL}`,
      ttl_seconds: 600,
      semantic: { embedding_model: EMBEDDING_MODEL, threshold: 0.999 },
    });

    // Seed this last: successful authentication proves every preceding
    // watched resource has reached the running real gateway.
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
  }, 90_000);

  afterAll(async () => {
    await app?.exit();
    await sls?.close();
  });

  test(
    "records the real semantic embedding as child work without taking the chat parent's attribution",
    async () => {
      const response = await chat(ROUTER_MODEL, ROUTE_PROMPT);
      expect(response.status).toBe(200);
      expect(response.servedBy).toBe(TARGET_MODEL);

      const row = await usageRow(response.requestId);
      const rows = slsLogsFor(sls!, LOGSTORE).filter(
        (candidate) => candidate.get("request_id") === response.requestId && !isChild(candidate),
      );
      expect(rows).toHaveLength(1);
      const children = childrenOf(response.requestId);
      expect(children).toHaveLength(1);
      expectEmbeddingChild(children[0]!, row, "semantic_route_embedding", embeddingModelID);
      expect(row.get("route_target_model_id")).toBe(targetModelID);

      // The caller received the target's chat response. Its parent event
      // remains priced and attributed as that target, never as the internal
      // embedding model that selected it.
      expect(Number(row.get("prompt_tokens"))).toBeGreaterThan(0);
      expect(Number(row.get("total_tokens"))).toBeGreaterThanOrEqual(
        Number(row.get("prompt_tokens")),
      );
      expect(row.has("cost_usd")).toBe(false);
      expect(row.get("model_id")).toBe(targetModelID);
      expect(row.get("attempt_model")).toBe(TARGET_MODEL);
      expect(row.get("model_id")).not.toBe(embeddingModelID);
      expect(row.get("attempt_model")).not.toBe(EMBEDDING_MODEL);
    },
    90_000,
  );

  test(
    "records real semantic guardrail work on the direct target parent",
    async () => {
      const response = await chat(GUARDRAIL_MODEL, GUARDRAIL_PROMPT);
      expect(response.status).toBe(200);
      expect(response.servedBy).toBe(GUARDRAIL_MODEL);

      const row = await usageRow(response.requestId);
      // One prototype batch plus the screened candidate are separate real
      // OpenAI embeddings. Their token counts need not be hard-coded: OpenAI
      // tokenization may legitimately change between model revisions.
      const children = childrenOf(response.requestId);
      expect(children).toHaveLength(2);
      for (const child of children) {
        expectEmbeddingChild(child, row, "guardrail_embedding", embeddingModelID);
      }
      expect(row.has("cost_usd")).toBe(false);
      expect(row.get("model_id")).toBe(guardrailModelID);
      expect(row.get("attempt_model")).toBeUndefined();
      expect(row.get("model_id")).not.toBe(embeddingModelID);
    },
    90_000,
  );

  test(
    "records a real semantic-cache miss and no child call on the exact hit",
    async () => {
      const miss = await chat(CACHE_MODEL, CACHE_PROMPT);
      expect(miss.status).toBe(200);
      expect(miss.servedBy).toBe(CACHE_MODEL);
      expect(miss.cache).toBe("miss");

      const missRow = await usageRow(miss.requestId);
      const missChildren = childrenOf(miss.requestId);
      expect(missChildren).toHaveLength(1);
      expectEmbeddingChild(missChildren[0]!, missRow, "semantic_cache_embedding", embeddingModelID);
      expect(missRow.has("cost_usd")).toBe(false);
      expect(missRow.get("model_id")).toBe(cacheModelID);
      expect(missRow.get("attempt_model")).toBeUndefined();
      expect(missRow.get("model_id")).not.toBe(embeddingModelID);

      const hit = await chat(CACHE_MODEL, CACHE_PROMPT);
      expect(hit.status).toBe(200);
      expect(hit.cache).toBe("hit");
      expect(hit.cacheLayer).toBe("exact");
      await usageRow(hit.requestId);
      // An exact hit is answered before any embedding is made.
      expect(childrenOf(hit.requestId)).toHaveLength(0);
    },
    90_000,
  );
});
