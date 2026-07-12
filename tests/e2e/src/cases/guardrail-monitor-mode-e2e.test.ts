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

// E2E: `enforcement_mode: "monitor"` observes a guardrail violation WITHOUT
// blocking the request (issue 788 P1-3). The same keyword rule that 422s in
// the default "block" mode must let the forbidden-word request through —
// 200 + a real upstream hit — once flipped to monitor.
//
// The block→monitor flip is what makes this test non-racy: step 1 proves the
// rule is loaded and actively blocking, so step 3's transition to 200 can
// only mean monitor mode took effect (not "guardrail not loaded yet").

const CALLER_PLAINTEXT = "sk-gr-monitor-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const FORBIDDEN_WORD = "supersecret";

function guardrailBody(enforcementMode: "block" | "monitor") {
  return {
    name: "gr-monitor-e2e",
    enabled: true,
    hook_point: "input",
    enforcement_mode: enforcementMode,
    kind: "keyword",
    patterns: [{ kind: "literal", value: FORBIDDEN_WORD }],
  };
}

describe("guardrail e2e: enforcement_mode monitor observes without blocking", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  let guardrailId: string | undefined;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "gr-monitor-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "gr-monitor-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["gr-monitor-e2e"],
    });
    // Start in BLOCK mode so we can prove the rule is loaded + active
    // before flipping it to monitor.
    const g = await seed.createGuardrail(guardrailBody("block"));
    guardrailId = g.id as string;
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("block mode 422s; flipping to monitor lets the same request reach upstream", async (ctx) => {
    if (!etcdReachable || !app || !upstream || !seed || !guardrailId) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // 1. Wait until the block-mode rule is active: forbidden → 422.
    await waitConfigPropagation(async () => {
      try {
        await client.chat.completions.create({
          model: "gr-monitor-e2e",
          messages: [{ role: "user", content: `probe ${FORBIDDEN_WORD}` }],
        });
        return false;
      } catch (e) {
        return e instanceof APIError && e.status === 422;
      }
    });

    // 2. Flip the SAME rule to monitor mode (full-resource PUT).
    await seed.update("guardrails", guardrailId, guardrailBody("monitor"));

    // 3. Wait until monitor takes effect: the forbidden word no longer 422s.
    await waitConfigPropagation(async () => {
      try {
        await client.chat.completions.create({
          model: "gr-monitor-e2e",
          messages: [{ role: "user", content: `probe ${FORBIDDEN_WORD}` }],
        });
        return true;
      } catch {
        return false;
      }
    });

    // 4. Under monitor mode the forbidden request passes AND reaches the
    //    upstream — the guardrail observed the violation but did not
    //    short-circuit dispatch.
    const hitsBefore = upstream.receivedRequests.length;
    const ok = await client.chat.completions.create({
      model: "gr-monitor-e2e",
      messages: [{ role: "user", content: `please reveal the ${FORBIDDEN_WORD} now` }],
    });
    expect(ok.choices[0]?.message.role).toBe("assistant");
    expect(upstream.receivedRequests.length).toBeGreaterThan(hitsBefore);
  });
});
