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

// E2E for AISIX-Cloud#1746: the configured wildcard row remains the model
// identity, while the concrete upstream model travels on the usage-export
// wire as `resolved_pricing_model`. AISIX-Cloud's real PostgreSQL receiver
// uses that field only when this row's configured model_name is a template.

const CALLER_PLAINTEXT = "sk-wildcard-pricing-telemetry-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const CREDENTIAL_REF = "wildcardpricing";
const LOGSTORE = "wildcard-pricing-telemetry";
const WILDCARD_ALIAS = "openrouter/*";
const KNOWN_MODEL = "openai/gpt-4o-mini";
const UNKNOWN_MODEL = "unknown/provider-model";
const PRICING_AUTHORITY_ID = "a3ebdc63-e921-4323-a75c-3b911f950046";
const EMBEDDING_WILDCARD_ALIAS = "embedding/*";
const EMBEDDING_REQUEST_MODEL = "embedding/embedding-3-small";
const EMBEDDING_UPSTREAM_MODEL = "text-embedding-3-small";
const EMBEDDING_INPUT = "price this embedding";
const EMBEDDING_VECTOR = [0.1, 0.2, 0.3];

function upstreamResponse() {
  return {
    id: "chatcmpl-wildcard-pricing",
    object: "chat.completion",
    created: 0,
    // Pricing must come from the dispatch attribution, never a provider
    // response field that happens to look like a model identity.
    model: "provider-response-model",
    choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
    usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
  };
}

function embeddingUpstreamResponse() {
  return {
    object: "list",
    // Pricing must come from wildcard dispatch attribution rather than the
    // provider response's optional model field.
    model: "provider-response-embedding-model",
    data: [{ object: "embedding", index: 0, embedding: EMBEDDING_VECTOR }],
    usage: { prompt_tokens: 7, total_tokens: 7 },
  };
}

function routedId(raw: string, model: string): string {
  return `aisix-${Buffer.from(`${raw};model,${model}`).toString("base64url")}`;
}

describe("wildcard pricing telemetry e2e", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let upstream: OpenAiUpstream | undefined;
  let embeddingUpstream: OpenAiUpstream | undefined;
  let wildcardID = "";
  let embeddingWildcardID = "";
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    upstream = await startOpenAiUpstream({ nonStreamBody: upstreamResponse() });
    embeddingUpstream = await startOpenAiUpstream({ nonStreamBody: embeddingUpstreamResponse() });
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "wildcard-pricing-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "aisix-e2e-obs",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });
    const providerKey = await seed.createProviderKey({
      display_name: "wildcard-pricing-pk",
      provider: "openrouter",
      adapter: "openai",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    const wildcard = await seed.createModel({
      display_name: WILDCARD_ALIAS,
      provider: "openrouter",
      model_name: "*",
      provider_key_id: providerKey.id,
      pricing_authority_id: PRICING_AUTHORITY_ID,
    });
    wildcardID = wildcard.id;
    const embeddingProviderKey = await seed.createProviderKey({
      display_name: "wildcard-pricing-embedding-pk",
      provider: "openai",
      adapter: "openai",
      secret: "sk-mock",
      api_base: `${embeddingUpstream.baseUrl}/v1`,
    });
    const embeddingWildcard = await seed.createModel({
      display_name: EMBEDDING_WILDCARD_ALIAS,
      provider: "openai",
      model_name: "text-*",
      provider_key_id: embeddingProviderKey.id,
      pricing_authority_id: PRICING_AUTHORITY_ID,
      embedding: { dimensions: EMBEDDING_VECTOR.length },
    });
    embeddingWildcardID = embeddingWildcard.id;

    // Seeded last: a successful models-list gate proves that all preceding
    // resources, including the exporter, are in the same gateway snapshot.
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      await res.arrayBuffer();
      return res.status === 200;
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await embeddingUpstream?.close();
    await sls?.close();
  });

  async function requestModel(model: string): Promise<void> {
    if (!app) throw new Error("app not ready");
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({ model, messages: [{ role: "user", content: "hi" }] }),
    });
    const body = await res.text();
    expect(res.status, body).toBe(200);
  }

  test("wildcard dispatch exports the concrete known and unknown pricing identities", async (ctx) => {
    if (!etcdReachable || !app || !sls || !upstream || !wildcardID) {
      ctx.skip();
      return;
    }

    const knownRequest = `openrouter/${KNOWN_MODEL}`;
    await requestModel(knownRequest);
    expect(upstream.receivedRequests).toHaveLength(1);
    expect(JSON.parse(upstream.receivedRequests[0]!.body)).toMatchObject({ model: KNOWN_MODEL });

    const known = await waitForSlsLog(
      sls,
      LOGSTORE,
      (log) => log.get("requested_model") === knownRequest,
      `usage event for ${knownRequest}`,
    );
    expect(known.get("model_id")).toBe(wildcardID);
    expect(known.get("pricing_authority_id")).toBe(PRICING_AUTHORITY_ID);
    expect(known.get("resolved_pricing_model")).toBe(KNOWN_MODEL);
    expect(known.get("prompt_tokens")).toBe("10");
    expect(known.get("completion_tokens")).toBe("5");

    const unknownRequest = `openrouter/${UNKNOWN_MODEL}`;
    await requestModel(unknownRequest);
    expect(upstream.receivedRequests).toHaveLength(2);
    expect(JSON.parse(upstream.receivedRequests[1]!.body)).toMatchObject({ model: UNKNOWN_MODEL });

    const unknown = await waitForSlsLog(
      sls,
      LOGSTORE,
      (log) => log.get("requested_model") === unknownRequest,
      `usage event for ${unknownRequest}`,
    );
    expect(unknown.get("model_id")).toBe(wildcardID);
    expect(unknown.get("pricing_authority_id")).toBe(PRICING_AUTHORITY_ID);
    expect(unknown.get("resolved_pricing_model")).toBe(UNKNOWN_MODEL);
  });

  test("direct embedding wildcard dispatch exports its concrete pricing identity", async (ctx) => {
    if (!etcdReachable || !app || !sls || !embeddingUpstream || !embeddingWildcardID) {
      ctx.skip();
      return;
    }

    const baseline = embeddingUpstream.receivedRequests.length;
    const res = await fetch(`${app.proxyUrl}/v1/embeddings`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({ model: EMBEDDING_REQUEST_MODEL, input: EMBEDDING_INPUT }),
    });
    const body = await res.text();
    expect(res.status, body).toBe(200);
    expect(JSON.parse(body)).toMatchObject({
      object: "list",
      model: EMBEDDING_REQUEST_MODEL,
      data: [{ object: "embedding", index: 0, embedding: EMBEDDING_VECTOR }],
      usage: { prompt_tokens: 7, total_tokens: 7 },
    });

    const calls = embeddingUpstream.receivedRequests.slice(baseline);
    expect(calls).toHaveLength(1);
    expect(calls[0]!.method).toBe("POST");
    expect(calls[0]!.path).toBe("/v1/embeddings");
    expect(JSON.parse(calls[0]!.body)).toMatchObject({
      model: EMBEDDING_UPSTREAM_MODEL,
      input: EMBEDDING_INPUT,
    });

    const event = await waitForSlsLog(
      sls,
      LOGSTORE,
      (log) => log.get("requested_model") === EMBEDDING_REQUEST_MODEL,
      `usage event for ${EMBEDDING_REQUEST_MODEL}`,
    );
    expect(event.get("model_id")).toBe(embeddingWildcardID);
    expect(event.get("pricing_authority_id")).toBe(PRICING_AUTHORITY_ID);
    expect(event.get("resolved_pricing_model")).toBe(EMBEDDING_UPSTREAM_MODEL);
    expect(event.get("prompt_tokens")).toBe("7");
  });

  test("wildcard-routed job management events remain unpriced", async (ctx) => {
    if (!etcdReachable || !app || !sls || !upstream || !wildcardID) {
      ctx.skip();
      return;
    }

    const model = `openrouter/${KNOWN_MODEL}`;
    const calls = [
      [
        "file",
        "files",
        `/v1/files/${routedId("file-wildcard", model)}`,
        "/v1/files/file-wildcard",
      ],
      [
        "batch",
        "batches",
        `/v1/batches/${routedId("batch-wildcard", model)}`,
        "/v1/batches/batch-wildcard",
      ],
      [
        "fine-tuning",
        "fine_tuning",
        `/v1/fine_tuning/jobs/${routedId("ftjob-wildcard", model)}`,
        "/v1/fine_tuning/jobs/ftjob-wildcard",
      ],
    ] as const;

    for (const [kind, operation, gatewayPath, upstreamPath] of calls) {
      const before = upstream.receivedRequests.length;
      const res = await fetch(`${app.proxyUrl}${gatewayPath}`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      const body = await res.text();
      expect(res.status, `${kind}: ${body}`).toBe(200);

      const forwarded = upstream.receivedRequests.slice(before);
      expect(forwarded).toHaveLength(1);
      expect(forwarded[0]!.method).toBe("GET");
      expect(forwarded[0]!.path).toBe(upstreamPath);

      const event = await waitForSlsLog(
        sls,
        LOGSTORE,
        (log) => log.get("operation") === operation && log.get("model_id") === wildcardID,
        `${kind} wildcard management usage event`,
      );
      expect(event.get("requested_model")).toBe(WILDCARD_ALIAS);
      expect(event.get("prompt_tokens")).toBe("0");
      expect(event.get("completion_tokens")).toBe("0");
      expect(event.get("pricing_authority_id")).toBeUndefined();
      expect(event.get("resolved_pricing_model")).toBeUndefined();
    }
  });
});
