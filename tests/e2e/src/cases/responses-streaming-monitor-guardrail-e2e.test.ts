import { createHash } from "node:crypto";
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

// E2E regression for AISIX-Cloud#1010: a MONITOR-mode output guardrail must
// never make a streaming /v1/responses request fail closed on the hold-back
// buffer cap.
//
// Pre-fix, ANY output-hook guardrail — enforcement_mode ignored — forced the
// streaming /v1/responses path into the whole-response hold-back branch with
// a 256 KiB cap, and an oversized stream was rejected 422 `content_filter`.
// A monitor rule ("observe, never block") therefore intermittently blocked
// exactly the long generations Codex produces: the customer-visible shape
// was 422 + 0 tokens + tens of seconds of latency on /v1/responses, while
// short responses passed — "monitor mode is flaky".
//
// The test is self-gating against config-propagation races:
//   1. block mode: the oversized stream 422s (proves the rule is loaded AND
//      pins the block-mode fail-closed secure default, which must not
//      change);
//   2. flip the SAME rule to monitor: the same oversized stream returns 200
//      with the full SSE released live — the block→monitor transition can
//      only mean monitor semantics took effect.

const CALLER_PLAINTEXT = "sk-issue-1010-monitor-stream";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const FORBIDDEN_WORD = "BLOCKME";

// > 256 KiB (262 144 bytes) of SSE once the mock wraps each event in
// `data: ...\n\n`: 220 deltas × ~1.3 KB ≈ 295 KB. One delta carries the
// forbidden literal so the monitor scan has a real violation to observe.
const DELTA_TEXT = "z".repeat(1250);
const STREAM_EVENTS = [
  JSON.stringify({ type: "response.created", response: { id: "resp_1010" } }),
  JSON.stringify({
    type: "response.output_text.delta",
    delta: `sure thing ${FORBIDDEN_WORD}: `,
  }),
  ...Array.from({ length: 220 }, () =>
    JSON.stringify({ type: "response.output_text.delta", delta: DELTA_TEXT }),
  ),
  JSON.stringify({
    type: "response.completed",
    response: {
      id: "resp_1010",
      status: "completed",
      usage: { input_tokens: 7, output_tokens: 9 },
    },
  }),
  "[DONE]",
];

function guardrailBody(enforcementMode: "block" | "monitor") {
  return {
    name: "gr-1010-output",
    enabled: true,
    hook_point: "output",
    enforcement_mode: enforcementMode,
    kind: "keyword",
    patterns: [{ kind: "literal", value: FORBIDDEN_WORD }],
  };
}

describe("responses streaming with monitor-mode output guardrail (AISIX-Cloud#1010)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  let guardrailId: string | undefined;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({ streamEvents: STREAM_EVENTS });
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "gr-1010-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "gr-1010-model",
      provider: "openai",
      model_name: "gpt-5-codex",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["gr-1010-model"],
    });
    // Start in BLOCK mode so phase 1 proves the rule is loaded + enforcing
    // before the flip.
    const g = await seed.createGuardrail(guardrailBody("block"));
    guardrailId = g.id as string;
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  async function streamOnce(): Promise<{ status: number; body: string }> {
    const r = await fetch(`${app!.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "gr-1010-model",
        input: "write a very long program",
        stream: true,
      }),
    });
    return { status: r.status, body: await r.text() };
  }

  test("block mode fails the oversized stream closed; monitor mode releases it live", async (ctx) => {
    if (!etcdReachable || !app || !upstream || !seed || !guardrailId) {
      ctx.skip();
      return;
    }

    // 1. Block mode: the >256 KiB stream must fail closed (422
    //    content_filter) — the rule is loaded and the secure default holds.
    await waitConfigPropagation(async () => {
      const { status } = await streamOnce();
      return status === 422;
    });
    const blocked = await streamOnce();
    expect(blocked.status).toBe(422);
    expect(blocked.body).toContain("content_filter");
    expect(blocked.body).not.toContain(DELTA_TEXT);

    // 2. Flip the SAME rule to monitor (full-resource PUT).
    await seed.update("guardrails", guardrailId, guardrailBody("monitor"));

    // 3. Monitor mode: the same oversized stream is released in full — 200,
    //    live SSE, no content_filter rejection. Pre-#1010-fix this stayed
    //    422 forever.
    await waitConfigPropagation(async () => {
      const { status } = await streamOnce();
      return status === 200;
    });
    const released = await streamOnce();
    expect(released.status).toBe(200);
    expect(released.body.length).toBeGreaterThan(262_144);
    expect(released.body).toContain(FORBIDDEN_WORD);
    expect(released.body).toContain("[DONE]");
    expect(released.body).not.toContain('"content_filter"');
  }, 120_000);
});
