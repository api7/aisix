import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  decodedTextFor,
  EtcdClient,
  spawnApp,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// AISIX-Cloud#562: an `enforcement_mode: monitor` guardrail must surface
// what it WOULD have done on the request's usage event
// (`guardrail_monitor_hits`), not just in gateway logs — that record is
// what lets an operator stage a policy, audit its hit rate in the
// dashboard, and only then flip it to `block`. We attach a monitor-mode
// kind=pii guardrail (email → mask, china_id_card → block), drive both
// detector classes, and read the emitted usage events through a real
// `aliyun_sls` exporter (metadata_only) against a mock SLS endpoint:
// - the email prompt reaches the upstream UNMASKED (monitor never
//   rewrites) but the event carries a `would_mask` hit with the email
//   detector count;
// - the ID-card prompt returns 200 (monitor never blocks) but the event
//   carries a `would_block` hit naming the detector;
// - the matched values themselves never appear in the event (#153).

const CALLER_PLAINTEXT = "sk-monitor-telemetry-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const CREDENTIAL_REF = "mock";
const SLS_PROJECT = "aisix-e2e-obs";
const META_LOGSTORE = "monitor-telemetry-events";

const EMAIL = "carol@example.com";
const CN_ID = "11010519491231002X"; // valid ISO 7064 MOD 11-2 check digit
const GUARD_NAME = "monitor-telemetry-guard";

describe("guardrail monitor-hit telemetry: would_block/would_mask on usage events", () => {
  let upstream: OpenAiUpstream | undefined;
  let sls: MockSls | undefined;
  let app: SpawnedApp | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-monitor",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "a plain reply" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
      },
    });

    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    const admin = new AdminClient(app.adminUrl, app.adminKey);

    await admin.createObservabilityExporter({
      name: "sls-monitor-telemetry",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: META_LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });

    const pk = await admin.createProviderKey({
      display_name: "monitor-telemetry-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "monitor-telemetry-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["monitor-telemetry-model"],
    });

    await admin.json("POST", "/admin/v1/guardrails", {
      name: GUARD_NAME,
      enabled: true,
      hook_point: "input",
      enforcement_mode: "monitor",
      kind: "pii",
      detectors: [
        { type: "email", action: "mask" },
        { type: "china_id_card", action: "block" },
      ],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await sls?.close();
  });

  const chat = async (content: string) => {
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "monitor-telemetry-model",
        messages: [{ role: "user", content }],
      }),
    });
    return res;
  };

  test("would_mask and would_block hits land on exported usage events; traffic unaffected", async (ctx) => {
    if (!etcdReachable || !app || !upstream || !sls) {
      ctx.skip();
      return;
    }

    // Monitor mode is invisible on the wire, so gate propagation on the
    // telemetry itself: keep sending an email-bearing probe until an
    // exported event carries a monitor hit.
    await waitConfigPropagation(async () => {
      const res = await chat(`probe mail ${EMAIL} ok`);
      expect(res.status).toBe(200);
      return decodedTextFor(sls!, META_LOGSTORE).includes("guardrail_monitor_hits");
    });

    // The prompt reached the upstream UNMASKED — monitor observes, never
    // rewrites.
    const lastReq = upstream.receivedRequests.at(-1);
    expect(lastReq).toBeDefined();
    expect(lastReq!.body).toContain(EMAIL);

    // The exported event carries the would_mask hit: action, detector
    // name, and the guardrail row name.
    let text = decodedTextFor(sls, META_LOGSTORE);
    expect(text).toContain("would_mask");
    expect(text).toContain('"email"');
    expect(text).toContain(GUARD_NAME);

    // A block-action detector in monitor mode: 200 to the caller, upstream
    // called, would_block hit with the detector name on the event.
    const upstreamBefore = upstream.receivedRequests.length;
    const res = await chat(`my id is ${CN_ID} thanks`);
    expect(res.status).toBe(200);
    expect(upstream.receivedRequests.length).toBe(upstreamBefore + 1);

    // Poll until the second event is exported.
    await waitConfigPropagation(async () =>
      decodedTextFor(sls!, META_LOGSTORE).includes("would_block"),
    );
    text = decodedTextFor(sls, META_LOGSTORE);
    expect(text).toContain("would_block");
    expect(text).toContain("china_id_card");

    // No-leak (#153): the matched values never appear in the events.
    expect(text).not.toContain(EMAIL);
    expect(text).not.toContain(CN_ID);
  });
});
