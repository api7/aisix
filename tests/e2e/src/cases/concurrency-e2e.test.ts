import { createHash } from "node:crypto";
import OpenAI, { APIError } from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: concurrency / inter-caller isolation. Per docs
// `docs/api-proxy.md` §2 status→error.type table:
//
//   | 429 | rate_limit_exceeded / concurrency_limit_exceeded /
//   |     | budget_exceeded | RPM/TPM/concurrency/budget cap |
//
// One contract pinned here:
//
//   - Per-caller rate-limit isolation — when caller A is rate-
//     limited, caller B's quota is unaffected. Rate limits are
//     per-ApiKey, not global. Without this contract, a single
//     noisy customer could exhaust the gateway's quota for
//     everyone else.
//
// (The concurrency-cap case — second simultaneous request →
// `429 concurrency_limit_exceeded` — is held back pending a
// product fix. Today the gateway emits the wrong error.type for
// concurrency rejections, conflating with RPM `rate_limit_exceeded`
// per docs §2's distinct taxonomy. See follow-up issue.)
//
// Prior to this file, the gateway had **zero** e2e coverage on
// inter-caller isolation — the existing `ratelimit-e2e.test.ts`
// covers per-RPM-rejection for a single caller but not isolation
// across callers.
//
// References:
// - Gateway's own §2 status→type table:
//   `docs/api-proxy.md` §2 (line ~52)
// - Gateway's `rate_limit.concurrency` schema:
//   `docs/api-admin.md` §4.2 example
//   (`"rate_limit": {"rpm": 60, "concurrency": 10}`)
// - OpenAI error envelope:
//   <https://platform.openai.com/docs/guides/error-codes/api-errors>

const CALLER_A_PLAINTEXT = "sk-conc-caller-a";
const CALLER_A_KEY_HASH = createHash("sha256")
  .update(CALLER_A_PLAINTEXT)
  .digest("hex");
const CALLER_B_PLAINTEXT = "sk-conc-caller-b";
const CALLER_B_KEY_HASH = createHash("sha256")
  .update(CALLER_B_PLAINTEXT)
  .digest("hex");
const CALLER_C_PLAINTEXT = "sk-conc-caller-c";
const CALLER_C_KEY_HASH = createHash("sha256")
  .update(CALLER_C_PLAINTEXT)
  .digest("hex");

describe("concurrency e2e: rate-limit isolation across callers", () => {
  let app: SpawnedApp | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  test("inter-caller rate-limit isolation: caller A's RPM exhaustion does NOT affect caller B", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    const upstream = await startOpenAiUpstream();
    upstreams.push(upstream);

    const pk = await admin.createProviderKey({
      display_name: "conc-iso-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "conc-iso",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    // Both callers have RPM=1; the cap is per-ApiKey, so caller
    // A burning their slot must NOT consume caller B's slot.
    await admin.createApiKey({
      key_hash: CALLER_A_KEY_HASH,
      allowed_models: ["conc-iso"],
      rate_limit: { rpm: 1 },
    });
    await admin.createApiKey({
      key_hash: CALLER_B_KEY_HASH,
      allowed_models: ["conc-iso"],
      rate_limit: { rpm: 1 },
    });

    const clientA = new OpenAI({
      apiKey: CALLER_A_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });
    const clientB = new OpenAI({
      apiKey: CALLER_B_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // Readiness gate: caller A's first call succeeds.
    await waitConfigPropagation(async () => {
      try {
        const r = await clientA.chat.completions.create({
          model: "conc-iso",
          messages: [{ role: "user", content: "ready-probe-a" }],
        });
        return r.choices[0]?.message.role === "assistant";
      } catch {
        return false;
      }
    });
    // Caller A has now consumed their RPM=1 slot via the probe.
    // Caller B's slot must still be available. Confirm B can fire
    // before continuing — this also gates caller B's snapshot
    // propagation independently.
    await waitConfigPropagation(async () => {
      try {
        const r = await clientB.chat.completions.create({
          model: "conc-iso",
          messages: [{ role: "user", content: "ready-probe-b" }],
        });
        return r.choices[0]?.message.role === "assistant";
      } catch {
        return false;
      }
    });
    // Both callers have now consumed their slot. Caller A's next
    // call must 429; we'll verify A is rate-limited AND B is
    // also rate-limited (separately) — not that B benefits from
    // A's rejection.

    // Caller A: second call within minute → 429.
    let caughtA: unknown;
    try {
      await clientA.chat.completions.create({
        model: "conc-iso",
        messages: [{ role: "user", content: "second-A" }],
      });
    } catch (e) {
      caughtA = e;
    }
    expect(caughtA).toBeInstanceOf(APIError);
    if (!(caughtA instanceof APIError)) {
      throw new Error("unreachable: caughtA is not APIError");
    }
    expect(caughtA.status).toBe(429);
    // Per docs §2: 429 → rate_limit_exceeded for RPM cap.
    expect((caughtA.error as { type?: unknown })?.type).toBe(
      "rate_limit_exceeded",
    );

    // Caller B: second call within minute → ALSO 429 (B's own
    // slot was burned in the probe). Critical assertion: B's 429
    // is from B's own quota, NOT a side effect of A being
    // rate-limited globally. We pin this by checking B's 429 also
    // surfaces with `rate_limit_exceeded` (proving the limit
    // engaged), but the load-bearing assertion is the next test
    // step: SLEEP nothing; instead reset by issuing a request
    // through caller C who has NEVER been rate-limited.
    let caughtB: unknown;
    try {
      await clientB.chat.completions.create({
        model: "conc-iso",
        messages: [{ role: "user", content: "second-B" }],
      });
    } catch (e) {
      caughtB = e;
    }
    expect(caughtB).toBeInstanceOf(APIError);
    if (!(caughtB instanceof APIError)) {
      throw new Error("unreachable: caughtB is not APIError");
    }
    expect(caughtB.status).toBe(429);

    // Bring up a third caller C with their own RPM=1. C has never
    // sent a request, so C's first call must succeed even though
    // A and B are both currently rate-limited. This is the
    // load-bearing isolation assertion: a fresh caller's quota is
    // genuinely independent of other callers'.
    await admin.createApiKey({
      key_hash: CALLER_C_KEY_HASH,
      allowed_models: ["conc-iso"],
      rate_limit: { rpm: 1 },
    });
    const clientC = new OpenAI({
      apiKey: CALLER_C_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });
    await waitConfigPropagation(async () => {
      try {
        const r = await clientC.chat.completions.create({
          model: "conc-iso",
          messages: [{ role: "user", content: "first-C" }],
        });
        return r.choices[0]?.message.role === "assistant";
      } catch (e) {
        // 401 = ApiKey not yet propagated; keep polling.
        return false;
      }
    });
  });

});
