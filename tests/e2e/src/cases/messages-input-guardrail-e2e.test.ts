import { createHash } from "node:crypto";
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
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);
    const pk = await admin.createProviderKey({
      display_name: "msg-gr-pk",
      secret: "sk-anth-mock",
      api_base: upstream.baseUrl,
    });
    await admin.createModel({
      display_name: "msg-gr",
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({ key_hash: HASH, allowed_models: ["msg-gr"] });
    await admin.json("POST", "/admin/v1/guardrails", {
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
    // Gate on guardrail propagation: the forbidden prompt must be rejected.
    await waitConfigPropagation(async () => (await messages(`probe ${FORBIDDEN}`)).status >= 400);

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
