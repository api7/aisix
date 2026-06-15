import { createHash } from "node:crypto";
import OpenAI from "openai";
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

// E2E: a LIVE edit to a weighted routing model's weights re-takes
// effect on the dispatch path (#196 L1, ai-gateway #127 L1).
//
// The sibling weighted-routing-distribution-e2e pins that the INITIAL
// weights are honored. The gap this closes: after the model is live
// and serving, an operator PATCHes the weights via the admin API, and
// the change must propagate through the etcd watch and the weighted
// scheduler must REBUILD — a scheduler that cached its weight wheel on
// first dispatch and never rebuilt on config update would keep serving
// the old split, silently ignoring the operator's change.
//
// Design is deterministic (no statistics): weight 0 = excluded (see
// routing-strategies-e2e "weighted picks the positive-weight target").
//   - Start [wr-edit-a: 100, wr-edit-b: 0]  → every dispatch hits A.
//   - PATCH to [wr-edit-a: 0, wr-edit-b: 100] → every dispatch hits B.
// The propagation signal is unambiguous: a probe through the virtual
// model returning "served by B" is IMPOSSIBLE under the old [100,0]
// config, so it proves the edit is live before we count.
//
// Reference: OpenAI Chat Completions shape the caller sees
// (https://platform.openai.com/docs/api-reference/chat); admin model
// update is PUT /admin/v1/models/:id (docs api-admin.md).

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

describe("weighted routing live-edit: changing weights shifts real traffic (#196 L1)", () => {
  let app: SpawnedApp | undefined;
  let upstreamA: OpenAiUpstream | undefined;
  let upstreamB: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let virtualId = "";
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    upstreamA = await startOpenAiUpstream({
      nonStreamBody: upstreamBody("served by A", "cmpl-wre-a"),
    });
    upstreamB = await startOpenAiUpstream({
      nonStreamBody: upstreamBody("served by B", "cmpl-wre-b"),
    });

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pkA = await admin.createProviderKey({
      display_name: "wre-a-pk",
      secret: "sk-mock",
      api_base: `${upstreamA.baseUrl}/v1`,
    });
    const pkB = await admin.createProviderKey({
      display_name: "wre-b-pk",
      secret: "sk-mock",
      api_base: `${upstreamB.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "wr-edit-a",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pkA.id,
    });
    await admin.createModel({
      display_name: "wr-edit-b",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pkB.id,
    });
    // Virtual model: weighted, ALL traffic to A initially (B excluded
    // via weight 0). Capture the generated id so we can PUT it below.
    const virtual = await admin.createModel({
      display_name: "wr-edit-virtual",
      routing: {
        strategy: "weighted",
        targets: [
          { model: "wr-edit-a", weight: 100 },
          { model: "wr-edit-b", weight: 0 },
        ],
      },
    });
    virtualId = (virtual as { id: string }).id;

    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["wr-edit-virtual", "wr-edit-a", "wr-edit-b"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstreamA?.close();
    await upstreamB?.close();
  });

  test("editing weights [100,0] → [0,100] flips the served upstream", async (ctx) => {
    if (!etcdReachable || !app || !upstreamA || !upstreamB || !admin || !virtualId) {
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
    // A under the initial [100,0] weights.
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

    // --- Phase 1: under [100,0], every dispatch must hit A. ---
    const aBase1 = upstreamA.receivedRequests.length;
    const bBase1 = upstreamB.receivedRequests.length;
    for (let i = 0; i < BATCH; i++) {
      expect(await callVirtual(`pre-edit-${i}`)).toBe("served by A");
    }
    expect(upstreamA.receivedRequests.length - aBase1).toBe(BATCH);
    expect(upstreamB.receivedRequests.length - bBase1).toBe(0);

    // --- Edit: invert the weights to [0,100] via PUT /admin/v1/models/:id. ---
    await admin.json("PUT", `/admin/v1/models/${virtualId}`, {
      display_name: "wr-edit-virtual",
      routing: {
        strategy: "weighted",
        targets: [
          { model: "wr-edit-a", weight: 0 },
          { model: "wr-edit-b", weight: 100 },
        ],
      },
    });

    // Propagation signal: a virtual dispatch returning "served by B" is
    // impossible under the old [100,0] config, so it proves the edit is
    // live + the scheduler rebuilt. If the scheduler never rebuilds on a
    // config edit (the regression this test targets), this times out.
    await waitConfigPropagation(async () => (await callVirtual("post-edit-probe")) === "served by B");

    // --- Phase 2: under [0,100], every dispatch must hit B. ---
    const aBase2 = upstreamA.receivedRequests.length;
    const bBase2 = upstreamB.receivedRequests.length;
    for (let i = 0; i < BATCH; i++) {
      expect(await callVirtual(`post-edit-${i}`)).toBe("served by B");
    }
    expect(upstreamB.receivedRequests.length - bBase2).toBe(BATCH);
    expect(upstreamA.receivedRequests.length - aBase2).toBe(0);
  }, 90_000);
});
