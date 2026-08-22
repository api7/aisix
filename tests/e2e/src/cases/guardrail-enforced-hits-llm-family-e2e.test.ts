import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  decodedTextFor,
  EtcdClient,
  pickFreePort,
  ProxyClient,
  SeedClient,
  spawnApp,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForToken,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: `UsageEvent.guardrail_enforced_hits` across the LLM handler family
// (api7/aisix#1024), against a real DP + etcd + a real SOC export target.
//
// AISIX-Cloud#1330 shipped the audit chain on `/mcp` only. The gap on the
// rest of the family was SILENT: an ENFORCING mask on `/v1/messages` or
// `/v1/responses` showed up in Prometheus and rewrote the response, while
// the /logs row for that same request looked exactly like "no guardrail
// acted". `/v1/messages` carries Claude-Code traffic and `/v1/responses`
// carries Codex traffic, so a chat-only test would have stayed green
// forever while the two families that matter misbehaved — which is why
// both are driven here by hand, streaming and not.
//
// One guardrail row governs every endpoint. Each endpoint gets its OWN
// detector inside that row, matching a marker only that endpoint's
// upstream emits, so the exported entry's `counts` key says which handler
// produced it — no ordering assumptions, no cross-talk between cases.
//
// Pinned contract, per endpoint:
//   - the exported usage event carries an enforced-hit entry naming the
//     configured ROW (not the detector, not the kind), the hook it fired
//     on, and `action: "masked"`;
//   - the per-detector count is the endpoint's own detector;
//   - the masked VALUE never reaches the SOC target (#153 no-leak).

const KEY = "sk-enforced-hits-llm-e2e";
const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

const CREDENTIAL_REF = "mock";
const SLS_PROJECT = "aisix-e2e-obs";
const LOGSTORE = "enforced-hits-llm";

const ROW = "llm-family-mask";

/** Per-endpoint marker + the detector name that masks it. */
const CASES = {
  chat: { marker: "chatmark-7f3a01", detector: "chat_marker" },
  chatStream: { marker: "streammark-7f3a02", detector: "chat_stream_marker" },
  messages: { marker: "msgmark-7f3a03", detector: "messages_marker" },
  responses: { marker: "respmark-7f3a04", detector: "responses_marker" },
} as const;

interface EnforcedHit {
  guardrail_name: string;
  hook: string;
  action: string;
  error_type?: string;
  counts?: Record<string, number>;
  duration_us?: number;
}

/**
 * Pull every `guardrail_enforced_hits` array the exporter rendered.
 *
 * The SLS encoder writes each field with `serde_json::to_value`, whose
 * objects are BTreeMap-ordered, so an entry's first key is `action` rather
 * than the struct's declaration order. Both spellings are accepted so a
 * later encoder change does not silently turn this into a no-op match.
 *
 * `guardrail_monitor_hits` renders with the same leading keys, so its
 * entries are dropped by their `would_*` actions — otherwise adding a
 * monitor-mode member to one of these rows would make the per-endpoint
 * assertions below compare a staged hit against an enforced contract.
 */
const hitsIn = (decoded: string): EnforcedHit[] =>
  [...decoded.matchAll(/\[\{"(?:action|guardrail_name)":.*?\}\]/g)]
    .map((m) => {
      try {
        return JSON.parse(m[0]) as EnforcedHit[];
      } catch {
        return null;
      }
    })
    .filter((a): a is EnforcedHit[] => Array.isArray(a))
    .flat()
    .filter((h) => !h.action?.startsWith("would_"));

describe("guardrail enforced hits on the LLM handler family", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  const upstreams: OpenAiUpstream[] = [];
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // One upstream per endpoint: the mock serves a single canned reply on
    // every path, and the four endpoints need four different wire shapes
    // (plus one of them streams). Separate servers keep the cases
    // independent of the order vitest runs them in.
    const chatUp = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-enforced-1",
        object: "chat.completion",
        model: "gpt-4o-2024-08-06",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: `build ${CASES.chat.marker} done` },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 4, total_tokens: 9 },
      },
    });
    const chatStreamUp = await startOpenAiUpstream({
      streamEvents: [
        JSON.stringify({
          id: "cmpl-enforced-2",
          model: "gpt-4o-2024-08-06",
          choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
        }),
        JSON.stringify({
          id: "cmpl-enforced-2",
          model: "gpt-4o-2024-08-06",
          choices: [
            {
              index: 0,
              delta: { content: `build ${CASES.chatStream.marker} done` },
              finish_reason: null,
            },
          ],
        }),
        JSON.stringify({
          id: "cmpl-enforced-2",
          model: "gpt-4o-2024-08-06",
          choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
        }),
        "[DONE]",
      ],
      eventDelayMs: 2,
    });
    const messagesUp = await startOpenAiUpstream({
      nonStreamBody: {
        id: "msg_enforced_3",
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: `build ${CASES.messages.marker} done` }],
        model: "claude-3-5-haiku-20241022",
        stop_reason: "end_turn",
        usage: { input_tokens: 6, output_tokens: 4 },
      },
    });
    const responsesUp = await startOpenAiUpstream({
      nonStreamBody: {
        id: "resp-enforced-4",
        object: "response",
        model: "gpt-4o-2024-08-06",
        output: [
          {
            type: "message",
            id: "msg-1",
            role: "assistant",
            content: [{ type: "output_text", text: `build ${CASES.responses.marker} done` }],
          },
        ],
        usage: { input_tokens: 6, output_tokens: 4, total_tokens: 10 },
      },
    });
    upstreams.push(chatUp, chatStreamUp, messagesUp, responsesUp);

    sls = await startMockSls();
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });

    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "sls-enforced-hits",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
    });
    // ONE row governs the whole family — which is the point: the drain
    // must be present at every handler's terminal event, not just at the
    // one whose test was written first.
    await seed.createGuardrail({
      name: ROW,
      enabled: true,
      hook_point: "both",
      kind: "pii",
      custom_patterns: Object.values(CASES).map((c) => ({
        name: c.detector,
        regex: c.marker,
        action: "mask",
        replacement: "***",
      })),
    });

    const mk = async (
      display: string,
      provider: string,
      modelName: string,
      up: OpenAiUpstream,
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
      });
    };
    await mk("enforced-chat", "openai", "gpt-4o", chatUp);
    await mk("enforced-chat-stream", "openai", "gpt-4o", chatStreamUp);
    await mk("enforced-messages", "anthropic", "claude-3-5-haiku-20241022", messagesUp);
    await mk("enforced-responses", "openai", "gpt-4o", responsesUp);

    // Caller key LAST (AGENTS.md gate rule): a key that authenticates
    // implies every row above is already in the snapshot.
    await seed.createApiKey({ key_hash: sha256(KEY), allowed_models: ["*"] });
    // Gate on the key authenticating, not on a masked request: a gate that
    // exercises the behavior under test fails by timeout instead of by an
    // assertion, and its own traffic would be exported alongside the cases'.
    const proxy = new ProxyClient(app.proxyUrl, KEY);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  }, 90_000);

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
    await sls?.close();
  });

  /**
   * Assert the row that masked is named on the exported event, with the
   * endpoint's own detector count — and that the value it masked is not
   * anywhere in the SOC feed.
   *
   * Gated on the requested MODEL, which rides every event this endpoint
   * emits and is therefore independent of the array under test. Waiting on
   * the detector name would look equivalent — it also rides
   * `redacted_entity_counts` — but that couples the gate to a second field
   * that could itself regress, and then a lost drain fails by burning the
   * 10s poll instead of by the assertion below (tests/e2e/AGENTS.md).
   */
  const expectAuditedMask = async (
    model: string,
    detector: string,
    marker: string,
  ): Promise<void> => {
    await waitForToken(sls!, LOGSTORE, model);
    const decoded = decodedTextFor(sls!, LOGSTORE);

    const hits = hitsIn(decoded).filter(
      (h) => h.guardrail_name === ROW && Object.keys(h.counts ?? {}).includes(detector),
    );
    expect(
      hits.length,
      `no enforced-hit entry for detector '${detector}' — the handler masked the response but its usage event reported nothing`,
    ).toBeGreaterThan(0);
    for (const hit of hits) {
      expect(hit.action).toBe("masked");
      expect(hit.hook).toBe("output");
      expect(hit.counts?.[detector]).toBeGreaterThan(0);
      // A mask is not a refusal, so no cause rides along
      // (AISIX-Cloud#1365).
      expect(hit.error_type ?? "").toBe("");
    }
    // The masked value is caller content, never audit data (#153).
    expect(decoded).not.toContain(marker);
  };

  const post = (path: string, body: unknown): Promise<Response> =>
    fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${KEY}` },
      body: JSON.stringify(body),
    });

  test("/v1/chat/completions (non-streaming) names the row that masked", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();

    const res = await post("/v1/chat/completions", {
      model: "enforced-chat",
      messages: [{ role: "user", content: "go" }],
    });
    expect(res.status).toBe(200);
    const body = await res.text();
    expect(body).toContain("***");
    expect(body).not.toContain(CASES.chat.marker);

    await expectAuditedMask("enforced-chat", CASES.chat.detector, CASES.chat.marker);
  });

  test("/v1/chat/completions (streaming) names the row that masked", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();

    // The streaming terminal event is emitted from the response body's
    // Drop guard, long after the handler frame is gone — the case the
    // per-handler drain has to reach through a cloned audit handle.
    const res = await post("/v1/chat/completions", {
      model: "enforced-chat-stream",
      messages: [{ role: "user", content: "go" }],
      stream: true,
    });
    expect(res.status).toBe(200);
    const body = await res.text();
    expect(body).toContain("***");
    expect(body).not.toContain(CASES.chatStream.marker);

    await expectAuditedMask("enforced-chat-stream", CASES.chatStream.detector, CASES.chatStream.marker);
  });

  test("/v1/messages names the row that masked (Claude-Code path)", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();

    const res = await post("/v1/messages", {
      model: "enforced-messages",
      max_tokens: 64,
      messages: [{ role: "user", content: "go" }],
    });
    expect(res.status).toBe(200);
    const body = await res.text();
    expect(body).toContain("***");
    expect(body).not.toContain(CASES.messages.marker);

    await expectAuditedMask("enforced-messages", CASES.messages.detector, CASES.messages.marker);
  });

  test("/v1/responses names the row that masked (Codex path)", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();

    const res = await post("/v1/responses", {
      model: "enforced-responses",
      input: "go",
    });
    expect(res.status).toBe(200);
    const body = await res.text();
    expect(body).toContain("***");
    expect(body).not.toContain(CASES.responses.marker);

    await expectAuditedMask("enforced-responses", CASES.responses.detector, CASES.responses.marker);
  });
});

// AISIX-Cloud#1365: the third enforcement state.
//
// A guardrail with `fail_open: false` returns a REFUSAL when its upstream
// is unreachable — the same verdict a content violation produces. Reported
// as plain `blocked`, a 30-second provider outage stamps every request in
// that window as a policy violation, and an operator reading /logs (or the
// CSV an auditor pulls) concludes a burst of customer prompts broke policy
// when the provider was simply down. That is a wrong answer, not a missing
// one, which is why it gets its own action and carries the cause.
//
// Its own app + env: the guardrail is env-scoped and refuses every request,
// so it cannot share a snapshot with the masking cases above.
describe("a fail-closed guardrail outage is audited apart from a policy block", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  const DEAD_LOGSTORE = "enforced-hits-unavailable";

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({});
    sls = await startMockSls();
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });

    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "sls-enforced-unavailable",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: DEAD_LOGSTORE,
      credential_ref: CREDENTIAL_REF,
    });
    // A real remote guardrail pointed at a port nothing listens on. Not a
    // stub of a failure — the actual connect error the DP classifies.
    const deadPort = await pickFreePort();
    await seed.createGuardrail({
      name: "presidio-prod",
      enabled: true,
      hook_point: "input",
      fail_open: false,
      kind: "presidio",
      analyzer_url: `http://127.0.0.1:${deadPort}`,
      anonymizer_url: `http://127.0.0.1:${deadPort}`,
      timeout_ms: 500,
    });
    const pk = await seed.createProviderKey({
      display_name: "unavailable-pk",
      secret: "sk-mock-upstream",
      api_base: upstream.baseUrl,
      provider: "openai",
      adapter: "openai",
    });
    await seed.createModel({
      display_name: "enforced-unavailable",
      provider: "openai",
      model_name: "gpt-4o",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({ key_hash: sha256(KEY), allowed_models: ["*"] });
    const proxy = new ProxyClient(app.proxyUrl, KEY);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  }, 90_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await sls?.close();
  });

  test("the refusal records blocked_unavailable with its cause, not blocked", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();

    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${KEY}` },
      body: JSON.stringify({
        model: "enforced-unavailable",
        messages: [{ role: "user", content: "anything at all" }],
      }),
    });
    // Fail-closed still means closed — separating the two attributions
    // must not weaken the guarantee itself.
    expect(res.status).toBe(422);

    // Waited on the requested MODEL, not the guardrail row: the row name
    // rides only the array under test, so waiting on it would turn a lost
    // drain into a 10s timeout instead of the assertion below — the
    // failure mode tests/e2e/AGENTS.md calls out.
    await waitForToken(sls, DEAD_LOGSTORE, "enforced-unavailable");
    const decoded = decodedTextFor(sls, DEAD_LOGSTORE);
    const hits = hitsIn(decoded).filter((h) => h.guardrail_name === "presidio-prod");
    expect(hits.length).toBeGreaterThan(0);
    for (const hit of hits) {
      expect(
        hit.action,
        "an unreachable guardrail upstream must not be reported as a policy decision",
      ).toBe("blocked_unavailable");
      expect(hit.hook).toBe("input");
      // A bounded per-kind cause, never free text and never the request.
      expect(hit.error_type ?? "").toMatch(/^presidio_/);
      expect(hit.counts ?? {}).toEqual({});
    }
    // ...and no entry on this request claims a policy decision, which is
    // the whole failure mode: one row saying `blocked` here is one row an
    // auditor counts as a customer violation.
    expect(hits.filter((h) => h.action === "blocked")).toHaveLength(0);
  });
});
