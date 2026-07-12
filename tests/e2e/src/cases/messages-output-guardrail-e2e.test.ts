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

// E2E: non-streaming /v1/messages runs OUTPUT guardrails (#448 #22).
// Pre-fix the /v1/messages response was returned without any output
// check. Here a cross-provider /v1/messages request (Anthropic protocol →
// OpenAI bridge) gets the mock's canned "mock reply"; an output guardrail
// blocking the literal "reply" must reject the response rather than
// return it.

const CALLER = "sk-msgout-gr-caller";
const HASH = createHash("sha256").update(CALLER).digest("hex");

describe("/v1/messages output guardrail (#448)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "msgout-gr-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    // OpenAI-provider model → /v1/messages takes the cross-provider path
    // (Anthropic protocol translated to the OpenAI bridge).
    await seed.createModel({
      display_name: "msgout-gr",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({ key_hash: HASH, allowed_models: ["msgout-gr"] });
    await seed.createGuardrail({
      name: "msgout-gr-output-keyword",
      enabled: true,
      hook_point: "output",
      kind: "keyword",
      patterns: [{ kind: "literal", value: "reply" }],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  const messages = (content: string) =>
    fetch(`${app!.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: { "content-type": "application/json", "x-api-key": CALLER },
      body: JSON.stringify({
        model: "msgout-gr",
        max_tokens: 64,
        messages: [{ role: "user", content }],
      }),
    });

  test("a forbidden response is blocked on /v1/messages (non-streaming)", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    // The mock always replies "mock reply", so once the output guardrail
    // is live every /v1/messages call is blocked. Gate specifically on the
    // guardrail block (422), not any 4xx, so transient propagation errors
    // don't satisfy the gate.
    await waitConfigPropagation(async () => (await messages("ready-probe")).status === 422);

    const res = await messages("anything at all");
    // ContentFiltered → 422; on the Anthropic /v1/messages envelope a 422
    // maps to error.type "invalid_request_error" (not OpenAI's
    // "content_filter") — see error.rs anthropic_error_type.
    expect(res.status, "output guardrail must block the forbidden reply").toBe(422);
    const json = (await res.json()) as { type?: string; error?: { type?: string; message?: string } };
    expect(json.type).toBe("error");
    expect(json.error?.type).toBe("invalid_request_error");
    expect(json.error?.message ?? "").toContain("content policy");
  });
});
