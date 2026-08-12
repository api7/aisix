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

// E2E: provider_key `api_base` re-point takes effect on the next request.
//
// Customer story: an operator migrates an upstream (region move, proxy
// cut-over, vendor endpoint change) by PUT-ing a new `api_base` onto the
// existing provider_key — same id, same secret. Every request after the
// config propagates must dispatch to the NEW host.
//
// What this pins: the gateway resolves the upstream URL from the CURRENT
// provider_key document. Any per-key caching of the resolved/parsed
// upstream URL must revalidate against the live config — a stale cached
// URL would keep routing requests (and the upstream credential) to the
// old host indefinitely, which is the exact regression this test exists
// to catch.
//
// Reference: OpenAI Chat Completions shape
// (https://platform.openai.com/docs/api-reference/chat).

const CALLER_PLAINTEXT = "sk-repoint-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

function chatBody(id: string, content: string) {
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

describe("provider_key api_base re-point reaches the new host", () => {
  let app: SpawnedApp | undefined;
  let hostA: OpenAiUpstream | undefined;
  let hostB: OpenAiUpstream | undefined;
  let etcd: EtcdClient | undefined;
  let seed: SeedClient | undefined;
  let pkId = "";
  let etcdReachable = false;

  beforeAll(async () => {
    etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    hostA = await startOpenAiUpstream({
      nonStreamBody: chatBody("cmpl-host-a", "from-a"),
    });
    hostB = await startOpenAiUpstream({
      nonStreamBody: chatBody("cmpl-host-b", "from-b"),
    });

    app = await spawnApp({ admin: false });
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "repoint-pk",
      secret: "sk-mock",
      api_base: `${hostA.baseUrl}/v1`,
    });
    pkId = pk.id;
    await seed.createModel({
      display_name: "repoint-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["repoint-model"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await hostA?.close();
    await hostB?.close();
  });

  test("requests follow an in-place api_base edit to the new upstream", async (ctx) => {
    if (!etcdReachable || !app || !hostA || !hostB || !pkId) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // Readiness + warm the URL path on host A. Several requests, so any
    // per-key URL cache is definitely populated before the edit.
    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "repoint-model",
          messages: [{ role: "user", content: "ready" }],
        });
        return probe.choices[0]?.message.content === "from-a";
      } catch {
        return false;
      }
    });
    for (let i = 0; i < 3; i++) {
      const r = await client.chat.completions.create({
        model: "repoint-model",
        messages: [{ role: "user", content: `warm-${i}` }],
      });
      expect(r.choices[0]?.message.content).toBe("from-a");
    }

    // In-place re-point: same id + secret, api_base moves to host B.
    await seed!.update("provider_keys", pkId, {
      provider: "openai",
      adapter: "openai",
      display_name: "repoint-pk",
      secret: "sk-mock",
      api_base: `${hostB!.baseUrl}/v1`,
    });

    // After propagation every request must land on host B. A stale
    // cached URL would keep answering "from-a" forever.
    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "repoint-model",
          messages: [{ role: "user", content: "moved?" }],
        });
        return probe.choices[0]?.message.content === "from-b";
      } catch {
        return false;
      }
    });
    for (let i = 0; i < 3; i++) {
      const r = await client.chat.completions.create({
        model: "repoint-model",
        messages: [{ role: "user", content: `post-${i}` }],
      });
      expect(r.choices[0]?.message.content).toBe("from-b");
    }
  });
});
