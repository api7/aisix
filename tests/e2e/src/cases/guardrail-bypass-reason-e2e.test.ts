import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  metricDelta,
  scrapeMetrics,
  spawnApp,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForSlsLog,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: `UsageEvent.guardrail_bypassed_reason` across the handler family
// (api7/aisix, follow-up to #1115), against a real DP + etcd + a real SOC
// export target.
//
// A fail-open bypass is the one guardrail outcome nobody sees. The caller
// gets a normal 200, the response looks screened, and the usage row is
// byte-identical to a request the chain actually decided. The field exists
// so an operator can tell those apart — and until this change it was
// written on `/v1/chat/completions` and nowhere else, so on every other
// route an outage passed traffic unscreened and left no trace.
//
// Two causes reach the field and both are driven here:
//   - the guardrail could not evaluate (a `kind: custom` script that
//     throws is the provider-outage shape without a provider), which
//     reports the kind's own bounded tag;
//   - the body could not be scanned at all, which reports the same tag the
//     fail-CLOSED direction puts in its refusal envelope.
//
// One `fail_open: true` row governs the whole family, which is the point:
// the reason has to reach every handler's terminal event, not just the one
// whose test was written first.

const KEY = "sk-bypass-reason-e2e";
const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

const CREDENTIAL_REF = "mock";
const SLS_PROJECT = "aisix-e2e-obs";
const LOGSTORE = "bypass-reason";

const ROW = "bypass-reason-open";
/** `ScriptFailure::Threw`'s bounded tag — the operator's filter value. */
const SCRIPT_TAG = "custom_script_error";
/** The tag the proxy raises for a body it could not scan at all. */
const UNSCANNABLE_TAG = "unscannable_body";

/**
 * Faults instead of deciding. Paired with `fail_open: true` this is the
 * outage shape: the chain runs, decides nothing, and the request goes
 * upstream with nothing screening it.
 *
 * Input hook only, deliberately. `kind: custom` reads its OUTPUT policy
 * from `output_fail_open`, which defaults to fail-CLOSED, so a row that
 * also faulted on the response would refuse every one of these requests
 * and the cases below would be measuring a refusal instead of a bypass.
 */
const FAULTING_SCRIPT = `
export function checkInput() {
  throw new Error("bypass-reason-e2e");
}
`;

describe("guardrail bypass reason across the handler family", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  const upstreams: OpenAiUpstream[] = [];
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    const chatUp = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-bypass-1",
        object: "chat.completion",
        model: "gpt-4o-2024-08-06",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "ok" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 4, total_tokens: 9 },
      },
    });
    const messagesUp = await startOpenAiUpstream({
      nonStreamBody: {
        id: "msg_bypass_2",
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: "ok" }],
        model: "claude-3-5-haiku-20241022",
        stop_reason: "end_turn",
        usage: { input_tokens: 6, output_tokens: 4 },
      },
    });
    const embeddingsUp = await startOpenAiUpstream({
      nonStreamBody: {
        object: "list",
        data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2, 0.3] }],
        model: "text-embedding-3-small",
        usage: { prompt_tokens: 3, total_tokens: 3 },
      },
    });
    upstreams.push(chatUp, messagesUp, embeddingsUp);

    sls = await startMockSls();
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });

    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "sls-bypass-reason",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
    });
    await seed.createGuardrail({
      name: ROW,
      enabled: true,
      hook_point: "input",
      fail_open: true,
      kind: "custom",
      script: FAULTING_SCRIPT,
      timeout_ms: 5000,
    });

    const mk = async (
      display: string,
      provider: string,
      modelName: string,
      up: OpenAiUpstream,
      extra: Record<string, unknown> = {},
    ): Promise<void> => {
      const pk = await seed.createProviderKey({
        display_name: `${display}-pk`,
        secret: "sk-mock-upstream",
        api_base: up.baseUrl,
        provider,
        adapter: provider,
      });
      await seed.createModel({
        display_name: display,
        provider,
        model_name: modelName,
        provider_key_id: pk.id,
        ...extra,
      });
    };
    await mk("bypass-chat", "openai", "gpt-4o", chatUp);
    await mk("bypass-messages", "anthropic", "claude-3-5-haiku-20241022", messagesUp);
    // Its own model so the unscannable case can be gated on `requested_model`
    // — a field independent of the one under test — without racing the
    // `/v1/messages` case above for the same name.
    await mk("bypass-unscannable", "anthropic", "claude-3-5-haiku-20241022", messagesUp);
    await mk("bypass-embeddings", "openai", "text-embedding-3-small", embeddingsUp, {
      kind: "embedding",
    });

    // Caller key LAST: a key that authenticates implies every row above is
    // already in the snapshot.
    await seed.createApiKey({ key_hash: sha256(KEY), allowed_models: ["*"] });
    const proxy = new ProxyClient(app.proxyUrl, KEY);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  }, 90_000);

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
    await sls?.close();
  });

  const post = (path: string, body: unknown, anthropic = false): Promise<Response> =>
    fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: anthropic
        ? {
            "content-type": "application/json",
            "x-api-key": KEY,
            "anthropic-version": "2023-06-01",
          }
        : { "content-type": "application/json", authorization: `Bearer ${KEY}` },
      body: JSON.stringify(body),
    });

  /**
   * Gated on the requested MODEL, which rides every event the route emits
   * and is therefore independent of the field under test — waiting on the
   * reason itself would turn a lost value into a poll timeout instead of an
   * assertion (tests/e2e/AGENTS.md).
   */
  const expectBypassReason = async (model: string, want: string): Promise<void> => {
    const log = await waitForSlsLog(
      sls!,
      LOGSTORE,
      (entry) => entry.get("requested_model") === model,
      `${model} usage event`,
    );
    expect(
      log.get("guardrail_bypassed_reason") ?? "",
      `${model}: the request went upstream with nothing screening it and its usage row does not say so`,
    ).toBe(want);
  };

  test("/v1/chat/completions reports the kind's failure tag", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();

    const res = await post("/v1/chat/completions", {
      model: "bypass-chat",
      messages: [{ role: "user", content: "go" }],
    });
    // Premise: a fail-open row must NOT refuse. Without this, an empty
    // reason and a refused request would be the same observation.
    expect(res.status).toBe(200);

    await expectBypassReason("bypass-chat", SCRIPT_TAG);
  });

  test("/v1/messages reports it too (Claude-Code path)", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();

    const res = await post(
      "/v1/messages",
      { model: "bypass-messages", max_tokens: 64, messages: [{ role: "user", content: "go" }] },
      true,
    );
    expect(res.status).toBe(200);

    await expectBypassReason("bypass-messages", SCRIPT_TAG);
  });

  test("/v1/embeddings reports it on its own terminal event", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();

    const res = await post("/v1/embeddings", { model: "bypass-embeddings", input: "go" });
    expect(res.status).toBe(200);

    await expectBypassReason("bypass-embeddings", SCRIPT_TAG);
  });

  test("a body the scanner cannot read reports the unscannable tag", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();

    // No `messages`: the Anthropic scan parser rejects it, so the chain is
    // never offered the body at all. Nothing in scope refuses an
    // unevaluable request, so it is forwarded — as a bypass, which is what
    // the tag has to say. A different tag from the case above on purpose:
    // "the guardrail broke" and "we could not give it the body" are
    // different outages with different fixes.
    const before = await scrapeMetrics(app.metricsUrl);
    const res = await post("/v1/messages", { model: "bypass-unscannable", max_tokens: 64 }, true);
    expect(res.status).toBe(200);

    await expectBypassReason("bypass-unscannable", UNSCANNABLE_TAG);

    // The same pass-through has to show up in the scrape, not only in the
    // usage row. No member executed, so the per-execution path that feeds
    // this counter for a provider outage never fires here — an operator
    // sizing "how much unscreened traffic got through" from
    // `aisix_guardrail_bypasses_total` would otherwise read zero for the
    // one cause the counter exists to expose. Scraped after the terminal
    // event so the request is provably finished.
    //
    // Only the positive direction is driven here: which pass-throughs are
    // bypasses at all is decided by the chain's member set, and that
    // discrimination is pinned in aisix-guardrails' chain unit test, where
    // an output-only chain can be built without a second seeded route.
    const after = await scrapeMetrics(app.metricsUrl);
    expect(
      metricDelta(before, after, "aisix_guardrail_bypasses_total", {
        reason: UNSCANNABLE_TAG,
      }),
      "an unscannable body a fail-open chain let through did not reach the bypass counter",
    ).toBe(1);
  });
});
