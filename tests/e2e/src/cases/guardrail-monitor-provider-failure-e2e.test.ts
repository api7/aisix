import { createServer, type Server } from "node:http";
import { createHash } from "node:crypto";
import OpenAI, { APIError } from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  pickFreePort,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// AISIX-Cloud#1010: a guardrail set to `enforcement_mode: "monitor"`
// ("monitor only, never intercept") must not block traffic under ANY
// circumstance a dashboard user can configure — including when the
// remote moderation provider itself fails (HTTP 5xx / timeout), not
// just when content is flagged. Before ai-gateway#640 the DP ignored
// `enforcement_mode` entirely, so a fail-closed provider outage
// produced intermittent 422s exactly as reported. This suite pins the
// post-#640 contract end-to-end with a real `aisix` binary + etcd:
//
//   1. block mode + fail_open=false + broken provider → 422
//      (the only guardrail-path mechanism that can yield the reported
//      symptom; doubles as the config-propagation gate)
//   2. same guardrail flipped to monitor mode → traffic flows again,
//      upstream is reached (the #1010 regression assertion)
//   3. monitor mode + a provider that flags everything → traffic still
//      flows (remote-provider flag downgrade, complementing the
//      keyword-based guardrail-monitor-mode-e2e)
//
// The moderation endpoint mocks green-cip TextModerationPlus
// (<https://help.aliyun.com/zh/document_detail/2671445.html>).

const CALLER_PLAINTEXT = "sk-gr-monitor-failure-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

interface ModerationMock {
  baseUrl: string;
  requests: number;
  close(): Promise<void>;
}

// Mock green-cip endpoint. `mode: "http500"` simulates a provider
// outage; `mode: "flagAll"` grades every text as RiskLevel "high".
async function startModerationMock(
  mode: "http500" | "flagAll",
): Promise<ModerationMock> {
  const mock = {
    requests: 0,
  } as ModerationMock;
  const server: Server = createServer((req, res) => {
    req.on("data", () => {});
    req.on("end", () => {
      mock.requests += 1;
      if (mode === "http500") {
        res.statusCode = 500;
        res.end("mock moderation outage");
        return;
      }
      res.statusCode = 200;
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          Code: 200,
          Data: {
            RiskLevel: "high",
            Result: [{ Label: "violent_content" }],
          },
          RequestId: "mock-req-flag-all",
        }),
      );
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  mock.baseUrl = `http://127.0.0.1:${port}`;
  mock.close = async () => {
    await new Promise<void>((resolve, reject) => {
      server.close((err) => (err ? reject(err) : resolve()));
    });
  };
  return mock;
}

function guardrailBody(
  enforcementMode: "block" | "monitor",
  endpoint: string,
) {
  return {
    name: "gr-monitor-failure-e2e",
    enabled: true,
    hook_point: "input",
    enforcement_mode: enforcementMode,
    // Fail closed on paper: without monitor mode a provider outage must
    // 422; with monitor mode it must not.
    fail_open: false,
    kind: "aliyun_text_moderation",
    region: "cn-shanghai",
    endpoint,
    access_key_id: "LTAI_E2E",
    access_key_secret: "e2e-secret",
    risk_level_threshold: "high",
  };
}

describe("guardrail e2e: monitor mode never blocks, even on provider failure (#1010)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let broken: ModerationMock | undefined;
  let flagAll: ModerationMock | undefined;
  let admin: AdminClient | undefined;
  let guardrailId = "";
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    broken = await startModerationMock("http500");
    flagAll = await startModerationMock("flagAll");

    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-monitor-failure",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "a clean reply" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
      },
    });

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "gr-monitor-failure-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "gr-monitor-failure-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["gr-monitor-failure-model"],
    });

    const g = (await admin.json(
      "POST",
      "/admin/v1/guardrails",
      guardrailBody("block", broken.baseUrl),
    )) as { id: string };
    guardrailId = g.id;
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await broken?.close();
    await flagAll?.close();
  });

  function client(): OpenAI {
    return new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app!.proxyUrl}/v1`,
      maxRetries: 0,
    });
  }

  async function probe(): Promise<{ status: number } | { ok: true }> {
    try {
      await client().chat.completions.create({
        model: "gr-monitor-failure-model",
        messages: [{ role: "user", content: "an ordinary question" }],
      });
      return { ok: true };
    } catch (e) {
      if (e instanceof APIError) {
        // status is undefined for connection-class errors; report a
        // sentinel so gates keep polling instead of failing the test.
        return { status: typeof e.status === "number" ? e.status : 0 };
      }
      throw e;
    }
  }

  test("block mode + fail_open=false: provider outage fails the request closed (422)", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    // Gate: poll until the guardrail is live and failing closed. This is
    // the one guardrail-path mechanism that can produce the symptom in
    // AISIX-Cloud#1010 (0 tokens, 422, latency = moderation call time).
    await waitConfigPropagation(async () => {
      const r = await probe();
      return "status" in r && r.status === 422;
    });

    let caught: unknown;
    try {
      await client().chat.completions.create({
        model: "gr-monitor-failure-model",
        messages: [{ role: "user", content: "an ordinary question" }],
      });
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(APIError);
    if (!(caught instanceof APIError)) throw new Error("unreachable");
    expect(caught.status).toBe(422);
    expect((caught.error as { type?: unknown })?.type).toBe("content_filter");
  });

  test("monitor mode: the same provider outage no longer blocks traffic", async (ctx) => {
    if (!etcdReachable || !app || !upstream || !broken) {
      ctx.skip();
      return;
    }
    await admin!.json(
      "PUT",
      `/admin/v1/guardrails/${guardrailId}`,
      guardrailBody("monitor", broken.baseUrl),
    );

    // The flip from 422 to 200 can only mean monitor mode took effect —
    // the moderation endpoint is still hard-down.
    await waitConfigPropagation(async () => {
      const r = await probe();
      return "ok" in r;
    });

    const upstreamBefore = upstream.receivedRequests.length;
    const brokenBefore = broken.requests;
    const res = await client().chat.completions.create({
      model: "gr-monitor-failure-model",
      messages: [{ role: "user", content: "an ordinary question" }],
    });
    expect(res.choices[0]?.message.role).toBe("assistant");
    // The request reached the upstream, and the (failing) moderation
    // provider was still consulted — monitored, not skipped.
    expect(upstream.receivedRequests.length).toBe(upstreamBefore + 1);
    expect(broken.requests).toBeGreaterThan(brokenBefore);
  });

  // The "observed" half (the would_block monitor hit) is pinned at the
  // unit layer — build.rs monitor_mode tests + guardrail-monitor-telemetry
  // e2e; this case pins the traffic half for a remote provider.
  test("monitor mode: flagged content is not blocked", async (ctx) => {
    if (!etcdReachable || !app || !upstream || !flagAll) {
      ctx.skip();
      return;
    }
    await admin!.json(
      "PUT",
      `/admin/v1/guardrails/${guardrailId}`,
      guardrailBody("monitor", flagAll.baseUrl),
    );

    // Gate on the endpoint switch: the flag-all mock starts receiving
    // the moderation calls.
    const flagBefore = flagAll.requests;
    await waitConfigPropagation(async () => {
      const r = await probe();
      if (!("ok" in r)) return false;
      return flagAll!.requests > flagBefore;
    });

    const upstreamBefore = upstream.receivedRequests.length;
    const res = await client().chat.completions.create({
      model: "gr-monitor-failure-model",
      messages: [{ role: "user", content: "content the provider grades as high risk" }],
    });
    expect(res.choices[0]?.message.role).toBe("assistant");
    expect(upstream.receivedRequests.length).toBe(upstreamBefore + 1);
  });
});
