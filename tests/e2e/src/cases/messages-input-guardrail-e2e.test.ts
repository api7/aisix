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

// E2E: /v1/messages runs input guardrails (#448 #22). Pre-fix the
// Anthropic /v1/messages path dispatched without any guardrail check, so
// prompts reached the upstream unscanned. The handler now translates the
// body to the internal ChatFormat and runs the resolved input guardrail
// chain before dispatch — a blocked prompt must never hit the upstream.

const CALLER = "sk-msg-gr-caller";
const HASH = createHash("sha256").update(CALLER).digest("hex");
const FORBIDDEN = "forbiddenmsgword";

describe("/v1/messages input guardrail (#448)", () => {
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
      display_name: "msg-gr-pk",
      secret: "sk-anth-mock",
      api_base: upstream.baseUrl,
    });
    await seed.createModel({
      display_name: "msg-gr",
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({ key_hash: HASH, allowed_models: ["msg-gr"] });
    await seed.createGuardrail({
      name: "msg-gr-input-keyword",
      enabled: true,
      hook_point: "input",
      kind: "keyword",
      patterns: [{ kind: "literal", value: FORBIDDEN }],
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
        model: "msg-gr",
        max_tokens: 64,
        messages: [{ role: "user", content }],
      }),
    });

  test("a forbidden /v1/messages prompt is blocked before hitting upstream", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    // Gate on guardrail propagation. A bare ">= 400 on forbidden" would
    // also match the transitional 401 while the caller key itself is
    // still propagating — require a benign prompt to pass first, so the
    // rejection can only come from the guardrail.
    await waitConfigPropagation(async () => {
      if ((await messages("propagation probe")).status >= 400) return false;
      return (await messages(`probe ${FORBIDDEN}`)).status >= 400;
    });

    const hitsBefore = upstream.receivedRequests.length;
    const blocked = await messages(`please do ${FORBIDDEN} now`);
    expect(blocked.status, "forbidden prompt must be rejected").toBeGreaterThanOrEqual(400);
    expect(
      upstream.receivedRequests.length,
      "blocked prompt must not reach the upstream",
    ).toBe(hitsBefore);

    // A benign prompt is not blocked by the input guardrail.
    const ok = await messages("hello there");
    expect(ok.status, "benign prompt should not be content-blocked").toBeLessThan(400);
  });
});
