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

// E2E: precise token-usage pass-through. The mock upstream returns a
// canned `usage` block (`prompt_tokens: 5`, `completion_tokens: 3`,
// `total_tokens: 8`); the OpenAI SDK client must surface those exact
// numbers to the caller, byte-for-byte. Existing sdk-compat case
// only asserts `total_tokens > 0`, which would still pass under a
// regression that synthesized different counts.
//
// This is the wire foundation that downstream usage-tracking and
// billing pipelines build on — caller-visible usage MUST equal what
// the upstream actually billed for.
//
// Reference: OpenAI Chat Completions API spec
// (https://platform.openai.com/docs/api-reference/chat/create) for
// the `usage` object shape; ai-gateway's response renderer at
// `crates/aisix-proxy/src/render.rs` passes prompt_tokens /
// completion_tokens / total_tokens straight through from the
// upstream's ChatResponse.

const CALLER_PLAINTEXT = "sk-tu-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

// These match the mock upstream's default `nonStreamBody` in
// `harness/upstream-openai.ts`. If that default ever changes, this
// test will fail loudly — which is the point.
const EXPECTED_PROMPT_TOKENS = 5;
const EXPECTED_COMPLETION_TOKENS = 3;
const EXPECTED_TOTAL_TOKENS = 8;

describe("token usage e2e: upstream usage block reaches client byte-for-byte", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "tu-e2e-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "tu-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["tu-e2e"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("usage.prompt_tokens / completion_tokens / total_tokens equal upstream's exact counts", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
    });

    await waitConfigPropagation(async () => {
      try {
        const r = await client.chat.completions.create({
          model: "tu-e2e",
          messages: [{ role: "user", content: "ready-probe" }],
        });
        return r.choices[0]?.message.role === "assistant";
      } catch {
        return false;
      }
    });

    const completion = await client.chat.completions.create({
      model: "tu-e2e",
      messages: [{ role: "user", content: "hello" }],
    });

    // Strict equality on all three counters. Upstream said {5, 3, 8};
    // anything else is the gateway tampering with billing-critical
    // data on the way out.
    expect(completion.usage?.prompt_tokens).toBe(EXPECTED_PROMPT_TOKENS);
    expect(completion.usage?.completion_tokens).toBe(EXPECTED_COMPLETION_TOKENS);
    expect(completion.usage?.total_tokens).toBe(EXPECTED_TOTAL_TOKENS);
  });
});
