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

describe("wildcard pricing telemetry e2e", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let upstream: OpenAiUpstream | undefined;
  let wildcardID = "";
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    upstream = await startOpenAiUpstream({ nonStreamBody: upstreamResponse() });
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
    });
    wildcardID = wildcard.id;

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
    expect(unknown.get("resolved_pricing_model")).toBe(UNKNOWN_MODEL);
  });
});
