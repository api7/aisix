import { createHash } from "node:crypto";
import OpenAI from "openai";
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

// E2E: a LIVE edit to a routing model's target priorities re-takes
// effect on the dispatch path (#196 L1, ai-gateway #127 L1).
//
// The sibling weighted-routing-distribution-e2e pins that the INITIAL
// weights are honored. The gap this closes: after the model is live
// and serving, an operator rewrites the target priorities on the stored
// model, and the change must propagate through the etcd watch and the
// scheduler must REBUILD — a scheduler that cached its tier partition
// (or WRR wheel) on first dispatch and never rebuilt on config update
// would keep serving the old layout, silently ignoring the operator's
// change.
//
// Design is deterministic (no statistics): the active tier takes ALL
// traffic while it is healthy, the backup tier none (AISIX-Cloud#1206).
//   - Start [wr-edit-a: priority 0, wr-edit-b: priority -1] → all A.
//   - Edit to [wr-edit-a: priority -1, wr-edit-b: priority 0] → all B.
// The propagation signal is unambiguous: a probe through the virtual
// model returning "served by B" is IMPOSSIBLE under the old layout, so
// it proves the edit is live before we count.
//
// Reference: OpenAI Chat Completions shape the caller sees
// (https://platform.openai.com/docs/api-reference/chat).

const CALLER_PLAINTEXT = "sk-wre-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const BATCH = 15;

function upstreamBody(content: string, id: string): Record<string, unknown> {
  return {
    id,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: "gpt-4o-mini",
    choices: [
      { index: 0, message: { role: "assistant", content }, finish_reason: "stop" },
    ],
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
  };
}

describe("routing live-edit: swapping target priorities shifts real traffic (#196 L1)", () => {
  let app: SpawnedApp | undefined;
  let upstreamA: OpenAiUpstream | undefined;
  let upstreamB: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let virtualId = "";
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstreamA = await startOpenAiUpstream({
      nonStreamBody: upstreamBody("served by A", "cmpl-wre-a"),
    });
    upstreamB = await startOpenAiUpstream({
      nonStreamBody: upstreamBody("served by B", "cmpl-wre-b"),
    });

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pkA = await seed.createProviderKey({
      display_name: "wre-a-pk",
      secret: "sk-mock",
      api_base: `${upstreamA.baseUrl}/v1`,
    });
    const pkB = await seed.createProviderKey({
      display_name: "wre-b-pk",
      secret: "sk-mock",
      api_base: `${upstreamB.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "wr-edit-a",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pkA.id,
    });
    await seed.createModel({
      display_name: "wr-edit-b",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pkB.id,
    });
    // Virtual model: round_robin with B parked in a backup tier — ALL
    // traffic to A initially. Capture the generated id for the PUT below.
    const virtual = await seed.createModel({
      display_name: "wr-edit-virtual",
      routing: {
        strategy: "round_robin",
        targets: [
          { model: "wr-edit-a" },
          { model: "wr-edit-b", priority: -1 },
        ],
      },
    });
    virtualId = (virtual as { id: string }).id;

    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["wr-edit-virtual", "wr-edit-a", "wr-edit-b"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstreamA?.close();
    await upstreamB?.close();
  });

  test("swapping tier priorities flips the served upstream", async (ctx) => {
    if (!etcdReachable || !app || !upstreamA || !upstreamB || !seed || !virtualId) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    const callVirtual = async (probe: string): Promise<string | null | undefined> => {
      try {
        const r = await client.chat.completions.create({
          model: "wr-edit-virtual",
          messages: [{ role: "user", content: probe }],
        });
        return r.choices[0]?.message.content;
      } catch {
        return null;
      }
    };

    // Readiness: both leaves registered, then the virtual model serving
    // A under the initial tier layout (A active, B backup).
    await waitConfigPropagation(async () => {
      try {
        const a = await client.chat.completions.create({
          model: "wr-edit-a",
          messages: [{ role: "user", content: "ready-a" }],
        });
        return a.choices[0]?.message.content === "served by A";
      } catch {
        return false;
      }
    });
    await waitConfigPropagation(async () => {
      try {
        const b = await client.chat.completions.create({
          model: "wr-edit-b",
          messages: [{ role: "user", content: "ready-b" }],
        });
        return b.choices[0]?.message.content === "served by B";
      } catch {
        return false;
      }
    });
    await waitConfigPropagation(async () => (await callVirtual("ready-virtual")) === "served by A");

    // --- Phase 1: A is the active tier — every dispatch must hit A. ---
    const aBase1 = upstreamA.receivedRequests.length;
    const bBase1 = upstreamB.receivedRequests.length;
    for (let i = 0; i < BATCH; i++) {
      expect(await callVirtual(`pre-edit-${i}`)).toBe("served by A");
    }
    expect(upstreamA.receivedRequests.length - aBase1).toBe(BATCH);
    expect(upstreamB.receivedRequests.length - bBase1).toBe(0);

    // --- Edit: swap the tiers by rewriting the document. ---
    await seed.update("models", virtualId, {
      display_name: "wr-edit-virtual",
      routing: {
        strategy: "round_robin",
        targets: [
          { model: "wr-edit-a", priority: -1 },
          { model: "wr-edit-b" },
        ],
      },
    });

    // Propagation signal: a virtual dispatch returning "served by B" is
    // impossible under the old tier layout (B was backup behind a healthy
    // A), so it proves the edit is live + the scheduler rebuilt. If the
    // scheduler never rebuilds on a config edit (the regression this test
    // targets), this times out.
    await waitConfigPropagation(async () => (await callVirtual("post-edit-probe")) === "served by B");

    // --- Phase 2: after the swap B is the active tier — every dispatch must hit B. ---
    const aBase2 = upstreamA.receivedRequests.length;
    const bBase2 = upstreamB.receivedRequests.length;
    for (let i = 0; i < BATCH; i++) {
      expect(await callVirtual(`post-edit-${i}`)).toBe("served by B");
    }
    expect(upstreamB.receivedRequests.length - bBase2).toBe(BATCH);
    expect(upstreamA.receivedRequests.length - aBase2).toBe(0);
  }, 90_000);
});
