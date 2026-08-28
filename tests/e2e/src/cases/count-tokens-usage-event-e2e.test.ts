import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  slsLogsFor,
  spawnApp,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForSlsLog,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E for AISIX-Cloud#1435: `/v1/messages/count_tokens` records the
// requests it serves and the ones it refuses.
//
// The route shipped deliberately unmetered — it generates nothing, so
// there is nothing to bill — and that was read as "emits nothing", which
// is a different statement. It forwards the caller's entire `system` +
// `messages` + `tools` payload to a real provider, and #1064 made it
// refusable by an input guardrail. So a refusal was a 422 the caller
// definitely saw and the Logs "Guardrail blocks" view could not find:
// nine handler families gained the flag in #1065, this one had no event
// to put it on.
//
// Both halves are asserted here because only the pair pins the design.
// The refusal must be findable; the SUCCESS must be findable too and must
// still bill nothing — the `{"input_tokens": N}` a caller gets back is a
// measurement of a prompt, not tokens an upstream consumed, and copying it
// into `prompt_tokens` would charge for a free call and double-count the
// prompt once the caller issues the real `/v1/messages`.
//
// Read back off a real Aliyun-SLS export from a real `aisix` binary, so
// what is asserted is the row a consumer receives.

const CALLER_PLAINTEXT = "sk-ct-usage-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const CREDENTIAL_REF = "mock";
const MOCK_AK_ID = "LTAI_mock_ak";
const MOCK_AK_SECRET = "mock_ak_secret";
const SLS_PROJECT = "aisix-e2e-obs";
const LOGSTORE = "count-tokens-usage";

const FORBIDDEN_WORD = "counttokensentinel";
const MODEL_ALIAS = "ctu-e2e";
const UPSTREAM_MODEL_ID = "claude-haiku-4-5-20251001";

// The route emits no captured content, so a row cannot be found by a marker
// planted in the prompt the way the sibling specs do. `x-aisix-request-id`
// is the anchor instead: `ensure_request_id` is the outermost layer and
// stamps it onto every response, short-circuited 4xx included, and it IS
// the telemetry `request_id` — so each assertion pins the row its own
// request produced rather than any row that happens to share a status.
const rowsForRequest = (sls: MockSls, requestId: string) =>
  slsLogsFor(sls, LOGSTORE).filter((l) => l.get("request_id") === requestId);

describe("count_tokens usage e2e: served and refused requests both leave a row (#1435)", () => {
  let upstream: OpenAiUpstream | undefined;
  let sls: MockSls | undefined;
  let app: SpawnedApp | undefined;
  let etcdReachable = false;

  async function countTokens(
    text: string,
  ): Promise<{ status: number; requestId: string }> {
    const res = await fetch(`${app!.proxyUrl}/v1/messages/count_tokens`, {
      method: "POST",
      headers: {
        "x-api-key": CALLER_PLAINTEXT,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: MODEL_ALIAS,
        messages: [{ role: "user", content: text }],
      }),
    });
    await res.text();
    return {
      status: res.status,
      requestId: res.headers.get("x-aisix-request-id") ?? "",
    };
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    // Anthropic's documented count_tokens response shape. The mock is
    // path-agnostic, so it stands in for the upstream's own sub-route.
    upstream = await startOpenAiUpstream({ nonStreamBody: { input_tokens: 42 } });

    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: MOCK_AK_ID,
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: MOCK_AK_SECRET,
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);

    await seed.createObservabilityExporter({
      name: "ctu-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
    });

    // Anthropic bridge appends the path to the bare host (no `/v1`), and
    // count_tokens only dispatches to Anthropic-protocol targets.
    const pk = await seed.createProviderKey({
      display_name: "ctu-pk",
      secret: "sk-ant-mock",
      api_base: upstream.baseUrl,
    });
    await seed.createModel({
      display_name: MODEL_ALIAS,
      provider: "anthropic",
      model_name: UPSTREAM_MODEL_ID,
      provider_key_id: pk.id,
    });
    await seed.createGuardrail({
      name: "ctu-guard",
      enabled: true,
      hook_point: "input",
      kind: "keyword",
      patterns: [{ kind: "literal", value: FORBIDDEN_WORD }],
    });
    // Written last: one etcd watch applies events in revision order, so
    // the moment this key authenticates everything above is in the
    // snapshot.
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [MODEL_ALIAS],
    });

    // Independent of the route under test (tests/e2e/AGENTS.md). Gating on
    // count_tokens itself would make a regression in it surface as a
    // propagation timeout in `beforeAll` instead of a failed assertion, and
    // every gate attempt would plant a served row the first test could then
    // match instead of its own.
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await sls?.close();
  });

  test("a served count_tokens leaves a row, and bills nothing", async (ctx) => {
    if (!etcdReachable || !app || !sls) {
      ctx.skip();
      return;
    }
    const res = await countTokens("how long is this prompt");
    expect(res.status).toBe(200);
    expect(res.requestId).not.toBe("");

    const log = await waitForSlsLog(
      sls,
      LOGSTORE,
      (l) => l.get("request_id") === res.requestId,
      "served count_tokens usage row",
      15_000,
    );
    expect(log.get("status_code")).toBe("200");
    expect(log.get("requested_model")).toBe(MODEL_ALIAS);
    expect(log.get("inbound_protocol")).toBe("anthropic");
    expect(log.get("guardrail_blocked")).not.toBe("true");
    // The upstream answered `input_tokens: 42`; that is a measurement of
    // the prompt, not consumption, so it must not become spend.
    expect(log.get("prompt_tokens") ?? "0").toBe("0");
    expect(log.get("completion_tokens") ?? "0").toBe("0");
    // One request, one row.
    expect(rowsForRequest(sls, res.requestId)).toHaveLength(1);
  });

  test("a refused count_tokens reaches the Blocked view", async (ctx) => {
    if (!etcdReachable || !app || !sls) {
      ctx.skip();
      return;
    }
    const res = await countTokens(`please ${FORBIDDEN_WORD} now`);
    expect(res.status).toBe(422);
    expect(res.requestId).not.toBe("");

    const log = await waitForSlsLog(
      sls,
      LOGSTORE,
      (l) => l.get("request_id") === res.requestId,
      "refused count_tokens usage row",
      15_000,
    );
    // The predicate the dashboard's "Guardrail blocks" view filters on.
    expect(log.get("guardrail_blocked")).toBe("true");
    expect(log.get("status_code")).toBe("422");
    expect(log.get("requested_model")).toBe(MODEL_ALIAS);
    expect(log.get("inbound_protocol")).toBe("anthropic");
    // Refused before dispatch, so nothing was sent and nothing is owed.
    expect(log.get("prompt_tokens") ?? "0").toBe("0");
    expect(log.get("completion_tokens") ?? "0").toBe("0");
    // One refusal, one row — not one per resolved target.
    expect(rowsForRequest(sls, res.requestId)).toHaveLength(1);
  });
});
