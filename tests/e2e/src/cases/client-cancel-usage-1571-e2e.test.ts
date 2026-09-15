import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
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

// AISIX-Cloud#1571: a caller that gives up before the response head is
// written left NO row in the usage log. Every endpoint emits from the tail of
// its own handler, and axum drops that future when the client disconnects —
// so a request that reached a provider and kept it busy was invisible to the
// control plane, which is the case an operator most needs to see (the usual
// reason a caller gives up is a long time to first token).
//
// The scenario is deliberately a ROUTING group: the group is what the caller
// addressed and the target is what the gateway was waiting on, and a single
// `direct` model — which every earlier cancel test used — makes those two
// identities the same value, so it cannot tell a correct row from one that
// reports the group where the target belongs.
const CALLER_PLAINTEXT = "sk-cancel-1571-PLAINTEXT";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const PROVIDER_SECRET = "sk-mock-cancel-1571";

const CREDENTIAL_REF = "mock";
const SLS_PROJECT = "aisix-e2e-obs";
const LOGSTORE = "cancel-events";

const GROUP = "c1571-group";
const TARGET = "c1571-target";
const UPSTREAM_MODEL = "gpt-4o-mini";
/** A direct model on a fast upstream, for the success-path line below. */
const FAST_MODEL = "c1571-fast";
const FAST_UPSTREAM_MODEL = "gpt-4o-fast";

describe("client cancel before the response head (AISIX-Cloud#1571)", () => {
  let etcdReachable = false;
  let slow: OpenAiUpstream | undefined;
  let fast: OpenAiUpstream | undefined;
  let sls: MockSls | undefined;
  let app: SpawnedApp | undefined;
  let targetModelId = "";
  let providerKeyId = "";
  let fastProviderKeyId = "";

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    // Long enough that the caller is certainly still waiting when it aborts,
    // and that the gateway never gets a head to forward.
    slow = await startOpenAiUpstream({
      responseDelayMs: 30_000,
      streamEvents: ["[DONE]"],
    });
    fast = await startOpenAiUpstream({
      nonStreamBody: {
        id: "chatcmpl-c1571",
        object: "chat.completion",
        created: 1_700_000_000,
        model: FAST_UPSTREAM_MODEL,
        choices: [
          { index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    });
    sls = await startMockSls();

    app = await spawnApp({
      // The access-log line is `tracing::info!`; the harness default of
      // `warn` would leave it unwritten and the assertion below vacuous.
      logLevel: "info",
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    const seed = new SeedClient(new EtcdClient(), app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "sls-cancel",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });
    const pk = await seed.createProviderKey({
      display_name: "c1571-pk",
      secret: PROVIDER_SECRET,
      api_base: `${slow.baseUrl}/v1`,
    });
    providerKeyId = pk.id;
    await seed.createModel({
      display_name: GROUP,
      routing: { strategy: "failover", targets: [{ model: TARGET }] },
    });
    const target = await seed.createModel({
      display_name: TARGET,
      provider: "openai",
      model_name: UPSTREAM_MODEL,
      provider_key_id: pk.id,
    });
    targetModelId = target.id;
    const fastPk = await seed.createProviderKey({
      display_name: "c1571-fast-pk",
      secret: PROVIDER_SECRET,
      api_base: `${fast.baseUrl}/v1`,
    });
    fastProviderKeyId = fastPk.id;
    await seed.createModel({
      display_name: FAST_MODEL,
      provider: "openai",
      model_name: FAST_UPSTREAM_MODEL,
      provider_key_id: fastPk.id,
    });
    // Seeded last, so it authenticating implies everything above is in the
    // snapshot (tests/e2e/AGENTS.md).
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await slow?.close();
    await fast?.close();
    await sls?.close();
  });

  test(
    "an abandoned request is metered, and names the target it was waiting on",
    async (ctx) => {
      if (!etcdReachable || !app || !slow || !fast || !sls) {
        ctx.skip();
        return;
      }
      const controller = new AbortController();
      const inflight = fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: GROUP,
          messages: [{ role: "user", content: "hang up on me" }],
          stream: true,
        }),
        signal: controller.signal,
      });

      // Abort once the upstream has actually received the call: by then the
      // gateway has authenticated, resolved the group, picked the target and
      // dispatched, and the upstream is still 30s from answering — so the
      // response head is unwritten when the caller goes away. A fixed sleep
      // could fire before dispatch on a loaded machine and silently assert
      // the pre-dispatch shape instead.
      for (let i = 0; i < 200 && slow.receivedRequests.length === 0; i++) {
        await new Promise((r) => setTimeout(r, 25));
      }
      expect(
        slow.receivedRequests.length,
        "the gateway never dispatched to the slow upstream",
      ).toBeGreaterThan(0);
      controller.abort();
      await expect(inflight).rejects.toThrow();

      const row = await waitForSlsLog(
        sls,
        LOGSTORE,
        (log) => log.get("requested_model") === GROUP,
        `a usage row for the cancelled request to ${GROUP}`,
        20_000,
      );

      expect(row.get("status_code")).toBe("499");
      expect(row.get("error_class")).toBe("client_disconnected");
      expect(row.get("error_message")).toContain("before the response head");
      // The two identities the row has to keep apart: the caller addressed
      // the group, the gateway was waiting on the target. `model_id` is what
      // the control plane prices against, and a group has no pricing row.
      expect(row.get("requested_model")).toBe(GROUP);
      expect(row.get("model_id")).toBe(targetModelId);
      expect(row.get("attempt_model")).toBe(TARGET);
      expect(row.get("operation")).toBe("chat");
      // Abandoned before a single token existed — the row must cost nothing.
      expect(row.get("prompt_tokens") ?? "0").toBe("0");
      expect(row.get("completion_tokens") ?? "0").toBe("0");

      // The access-log line for the SAME request names the target too.
      // Before this change `model=` (the group) was the only identity on the
      // line and the target was unreachable by request id anywhere. The row
      // above carries the id to join on — the aborted fetch never got to
      // read the `x-aisix-request-id` header.
      const requestId = row.get("request_id") ?? "";
      expect(requestId).not.toBe("");
      const line = app
        .output()
        .split("\n")
        .find((l) => l.includes(`request_id="${requestId}"`) && l.includes("status=499"));
      expect(line, `no 499 access-log line for ${requestId} in:\n${app.output()}`).toBeTruthy();
      expect(line).toContain(`model="${GROUP}"`);
      expect(line).toContain(`upstream_model="${UPSTREAM_MODEL}"`);
      expect(line).toContain(`provider_key_id="${providerKeyId}"`);
    },
    60_000,
  );

  // The pair is not a 499-only field. Asserting it only on the cancel line
  // would leave a version that fills it from the guard and nowhere else
  // looking entirely correct.
  test("an ordinary completed request names its target on the line too", async (ctx) => {
    if (!etcdReachable || !app || !fast) {
      ctx.skip();
      return;
    }
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: FAST_MODEL,
        messages: [{ role: "user", content: "hi" }],
      }),
    });
    await res.arrayBuffer();
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    expect(requestId).not.toBe("");

    const line = app
      .output()
      .split("\n")
      .find((l) => l.includes(`request_id="${requestId}"`) && l.includes("status=200"));
    expect(line, `no 200 access-log line for ${requestId} in:\n${app.output()}`).toBeTruthy();
    expect(line).toContain(`upstream_model="${FAST_UPSTREAM_MODEL}"`);
    expect(line).toContain(`provider_key_id="${fastProviderKeyId}"`);
  });
});
