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

// E2E: per-kind DEAD knobs (model-kind audit / schema convergence).
// Generic call knobs resolve member → group → deployment default where a
// group slot exists; model-specific knobs are direct-only. Two halves:
//
//   READ (lenient): a stored row carrying such knobs still LOADS —
//   the loader strips the field and reports it through the
//   partially-compatible channel on `GET /status/config`, instead of
//   dropping the whole row (which would take a working group out of
//   service on upgrade). The WRITE half (strict) has no e2e surface
//   here — the DP admin API is read-only for resources — and is pinned
//   at the validator level (model_schema_characterization) and the
//   declarative-file loader (filesource tests).

const CALLER_PLAINTEXT = "sk-dead-knobs-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

function chatBody(content: string) {
  return {
    id: `cmpl-${content}`,
    object: "chat.completion",
    created: 0,
    model: "gpt-4o-mini",
    choices: [
      { index: 0, message: { role: "assistant", content }, finish_reason: "stop" },
    ],
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
  };
}

describe("model kind dead knobs e2e", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });

    const upstream = await startOpenAiUpstream({ nonStreamBody: chatBody("served-dk") });
    upstreams.push(upstream);
    const pk = await seed.createProviderKey({
      display_name: "dk-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "dk-member",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    // A STORED group row carrying dead knobs — written by an older
    // build / directly to etcd. It must keep serving, minus the knobs.
    await seed.createModel({
      display_name: "dk-group",
      routing: { strategy: "failover", targets: [{ model: "dk-member" }] },
      retries: 3,
      cost: { input_per_1k: 0.5, output_per_1k: 1.5 },
    });

    await waitConfigPropagation(async () => {
      try {
        const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
          method: "POST",
          headers: {
            authorization: `Bearer ${CALLER_PLAINTEXT}`,
            "content-type": "application/json",
          },
          body: JSON.stringify({
            model: "dk-group",
            messages: [{ role: "user", content: "hi" }],
          }),
        });
        await res.text();
        return res.status === 200;
      } catch {
        return false;
      }
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  test("a stored group with dead knobs keeps serving and reports them partially compatible", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    // The row served during readiness — the group did NOT get dropped
    // on load. The dead knobs surface on the status report instead.
    const res = await fetch(`${app.metricsUrl}/status/config`);
    expect(res.status).toBe(200);
    const cfg = (await res.json()) as {
      partially_compatible: Array<{ resource_kind: string; field: string; count: number }>;
    };
    expect(cfg.partially_compatible).toContainEqual({
      resource_kind: "models",
      field: "inapplicable:cost",
      count: 1,
    });
    expect(cfg.partially_compatible).toContainEqual({
      resource_kind: "models",
      field: "inapplicable:retries",
      count: 1,
    });
  });

});
