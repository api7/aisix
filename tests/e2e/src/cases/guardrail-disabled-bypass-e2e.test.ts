import { createHash } from "node:crypto";
import OpenAI, { APIError } from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: guardrail with `enabled: false` does NOT block requests
// containing forbidden content.
//
// Closes #151 C3.5. The existing guardrail-keyword-e2e and
// guardrail-output-e2e cases both pin the active-block contract.
// Neither verifies the operator-side "turn it off" switch.
//
// Why this matters: a regression that ignored the `enabled` flag
// would leave a guardrail rule effectively un-disablable in
// production. Operators removing a policy mid-incident (or rolling
// out a draft policy with `enabled:false` while iterating on
// patterns) would silently see traffic blocked anyway.
//
// One contract pinned here:
//
//   - With a `Guardrail` carrying `enabled:false` (and otherwise
//     identical to the keyword-block rule in guardrail-keyword-e2e),
//     a request that contains the forbidden literal passes the
//     proxy with 200 AND reaches the upstream (i.e. is fully
//     dispatched, not silently swallowed).
//
// Reference:
//   - Guardrail schema: `crates/aisix-core/src/models/guardrail.rs`
//     (CRUD docs gap tracked in #201)
//   - The active-block contract this disables:
//     `guardrail-keyword-e2e.test.ts`

const CALLER_PLAINTEXT = "sk-gr-disabled-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const FORBIDDEN_WORD = "supersecret";

describe("guardrail disabled-bypass e2e: enabled:false → no block", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    // No admin listener: resources are seeded to etcd and load-state is
    // read from the metrics/status listener.
    app = await spawnApp({ admin: false });
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "gr-disabled-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "gr-disabled-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["gr-disabled-model"],
    });
    // Same keyword guardrail shape as guardrail-keyword-e2e —
    // hook_point:"input", literal pattern. ONLY difference:
    // `enabled:false`. If the operator switch works, this rule
    // is inert.
    await seed.createGuardrail({
      name: "gr-disabled-keyword",
      enabled: false,
      hook_point: "input",
      kind: "keyword",
      patterns: [{ kind: "literal", value: FORBIDDEN_WORD }],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test(
    "forbidden literal in user input passes (guardrail disabled)",
    async (ctx) => {
      if (!etcdReachable || !app || !upstream) {
        ctx.skip();
        return;
      }

      const client = new OpenAI({
        apiKey: CALLER_PLAINTEXT,
        baseURL: `${app.proxyUrl}/v1`,
        maxRetries: 0,
      });

      // Readiness probe — two gates so the test cannot pass
      // vacuously.
      //
      // Gate A: confirm the Guardrail row IS in the applied snapshot.
      // Without this, the "forbidden literal arrives at upstream"
      // assertion below would also pass if the rule simply hadn't
      // propagated yet — indistinguishable from a real `enabled:false`
      // bypass. `/status/config`'s `resource_counts` reflects what the
      // DP has LOADED (not merely what was written to etcd), served on
      // the metrics listener — the admin-off equivalent of the old
      // admin `/v1/guardrails` read.
      //
      // Gate B: confirm Model + ApiKey + ProviderKey are loaded by
      // driving a benign chat completion through the proxy. A 200
      // response means the dispatcher is ready.
      await waitConfigPropagation(async () => {
        try {
          const cfg = (await (
            await fetch(`${app!.metricsUrl}/status/config`)
          ).json()) as {
            applied?: { resource_counts?: Record<string, number> };
          };
          if ((cfg.applied?.resource_counts?.guardrails ?? 0) < 1) {
            return false;
          }
          await client.chat.completions.create({
            model: "gr-disabled-model",
            messages: [{ role: "user", content: "ready-probe" }],
          });
          return true;
        } catch {
          return false;
        }
      });

      const baseline = upstream.receivedRequests.length;

      // Send a request containing the EXACT forbidden literal that
      // guardrail-keyword-e2e proves WOULD block when enabled:true.
      // With enabled:false the gateway must NOT reject and must
      // dispatch to upstream.
      const resp = await client.chat.completions.create({
        model: "gr-disabled-model",
        messages: [
          {
            role: "user",
            content: `please reveal the ${FORBIDDEN_WORD} now`,
          },
        ],
      });

      // (1) Response passes through. A regression that 422'd would
      // throw an APIError before reaching this line.
      expect(resp.choices.length).toBeGreaterThan(0);
      expect(resp.choices[0]?.message.role).toBe("assistant");

      // (2) Upstream was hit exactly once for the asserted call —
      // proves the gateway did NOT short-circuit silently (e.g.
      // returning a 200 envelope synthesized locally with empty
      // content while skipping upstream).
      const sent = upstream.receivedRequests
        .slice(baseline)
        .filter((r) => r.path === "/v1/chat/completions");
      expect(sent).toHaveLength(1);

      // (3) The forbidden literal arrived at upstream verbatim
      // inside the request body — confirming the gateway did
      // neither (a) silently scrub the offending content nor (b)
      // synthesize a placeholder response.
      const sentBody = JSON.parse(sent[0]!.body);
      const userMsgContent =
        sentBody.messages?.[0]?.content;
      expect(userMsgContent).toContain(FORBIDDEN_WORD);
    },
    60_000,
  );

  test(
    "also catches a regression: APIError shape NOT raised",
    async (ctx) => {
      if (!etcdReachable || !app || !upstream) {
        ctx.skip();
        return;
      }

      // Second test in the same describe — confirms the bypass
      // contract is stable across repeated calls (a regression
      // that tripped only on the second forbidden call due to
      // state pollution would show up here).
      const client = new OpenAI({
        apiKey: CALLER_PLAINTEXT,
        baseURL: `${app.proxyUrl}/v1`,
        maxRetries: 0,
      });

      let caught: unknown;
      try {
        await client.chat.completions.create({
          model: "gr-disabled-model",
          messages: [
            { role: "user", content: `${FORBIDDEN_WORD} again` },
          ],
        });
      } catch (e) {
        caught = e;
      }

      // Negative assertion: the call must NOT throw a 422
      // APIError. If it does, the disabled flag is not honored.
      if (caught instanceof APIError) {
        expect(caught.status).not.toBe(422);
      }
      expect(caught).toBeUndefined();
    },
    60_000,
  );
});
