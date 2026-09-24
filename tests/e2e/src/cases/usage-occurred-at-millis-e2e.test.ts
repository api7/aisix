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

// E2E for AISIX-Cloud#1368: a usage row's `occurred_at` carries
// milliseconds. With whole seconds every row a request writes — each
// attempt of a failover, and any other request finished in the same
// second — tied on the one column the Logs view orders by, so their order
// was arbitrary. The stamp is RFC 3339 in UTC with exactly three
// fractional digits, on every endpoint family; two are driven here and
// read back off a real Aliyun-SLS export, which ships the field verbatim.

const CALLER_PLAINTEXT = "sk-occurred-at-millis-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const CREDENTIAL_REF = "mock";
const LOGSTORE = "occurred-at-millis";
const MODEL = "oam-model";

const MILLIS_RFC3339_UTC = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/;

describe("usage occurred_at has millisecond precision (AISIX-Cloud#1368)", () => {
  let upstream: OpenAiUpstream | undefined;
  let sls: MockSls | undefined;
  let app: SpawnedApp | undefined;
  let etcdReachable = false;
  const auth = { authorization: `Bearer ${CALLER_PLAINTEXT}`, "content-type": "application/json" };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    // One body both endpoints' parsers accept.
    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-oam",
        object: "chat.completion",
        created: 1,
        model: "gpt-4o-mini",
        choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
        data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2] }],
        usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
      },
    });

    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "LTAI_mock_ak",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock_ak_secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "oam-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "aisix-e2e-obs",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
    });
    const pk = await seed.createProviderKey({
      display_name: "oam-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    // Seeded last: once it authenticates, everything above is in the snapshot.
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, { headers: auth });
      await res.arrayBuffer();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await sls?.close();
  });

  const cases: Array<[string, Record<string, unknown>]> = [
    ["/v1/chat/completions", { model: MODEL, messages: [{ role: "user", content: "hi" }] }],
    ["/v1/embeddings", { model: MODEL, input: "hi" }],
  ];

  test.for(cases)("%s stamps occurred_at to the millisecond", async ([path, body], ctx) => {
    if (!etcdReachable || !app || !sls) {
      ctx.skip();
      return;
    }
    const res = await fetch(`${app.proxyUrl}${path}`, {
      method: "POST",
      headers: auth,
      body: JSON.stringify(body),
    });
    const text = await res.text();
    expect(res.status, text).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id");
    expect(requestId).toBeTruthy();

    const row = await waitForSlsLog(
      sls,
      LOGSTORE,
      (l) => l.get("request_id") === requestId,
      `${path}: the usage row of ${requestId}`,
    );
    const occurredAt = row.get("occurred_at") ?? "";
    expect(occurredAt).toMatch(MILLIS_RFC3339_UTC);
    // And it is the time of THIS request, not a fixed or truncated value.
    expect(Math.abs(Date.parse(occurredAt) - Date.now())).toBeLessThan(60_000);
  });
});
