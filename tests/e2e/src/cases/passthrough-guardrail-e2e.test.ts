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

// E2E for #911 finding [6]: the raw /passthrough/:provider/*rest tunnel
// forwarded requests verbatim with NO guardrail scanning, so a tenant that
// configured a content/DLP guardrail could bypass it by routing traffic
// through passthrough. Following LiteLLM's passthrough default, the gateway
// now scans the whole request AND response body as text against the resolved
// chain. This drives both directions:
//   - INPUT: a passthrough request whose body carries a forbidden word is
//     blocked 422 before the upstream is ever called.
//   - OUTPUT: a clean request whose upstream reply carries a forbidden word is
//     blocked 422 and the word never reaches the caller.

const CALLER_PLAINTEXT = "sk-pt-gr-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const FORBIDDEN_INPUT = "forbiddenprompt";
const FORBIDDEN_OUTPUT = "leakedsecret";
const INPUT_GUARDRAIL = "pt-gr-input-keyword";
const OUTPUT_GUARDRAIL = "pt-gr-output-keyword";

describe("passthrough guardrail (#911 [6])", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    // Upstream reply carries the forbidden OUTPUT word; the caller's request
    // body is innocent, so the forbidden content originates from the model.
    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-leak",
        object: "chat.completion",
        created: 0,
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: `here it is: ${FORBIDDEN_OUTPUT}` },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
      },
    });

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "pt-gr-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "pt-gr",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["pt-gr"],
    });
    await admin.json("POST", "/admin/v1/guardrails", {
      name: INPUT_GUARDRAIL,
      enabled: true,
      hook_point: "input",
      kind: "keyword",
      patterns: [{ kind: "literal", value: FORBIDDEN_INPUT }],
    });
    await admin.json("POST", "/admin/v1/guardrails", {
      name: OUTPUT_GUARDRAIL,
      enabled: true,
      hook_point: "output",
      kind: "keyword",
      patterns: [{ kind: "literal", value: FORBIDDEN_OUTPUT }],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  function passthrough(body: unknown): Promise<Response> {
    return fetch(`${app!.proxyUrl}/passthrough/openai/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    });
  }

  test("upstream-emitted forbidden text is blocked by the output guardrail", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    // Output guardrails fire after upstream dispatch, so readiness is signaled
    // by the 422-on-blocked-response itself (a 200 means the chain isn't
    // loaded yet and the leaked content was forwarded). The request body here
    // is clean, so only the OUTPUT guardrail can block it.
    await waitConfigPropagation(async () => {
      const res = await passthrough({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "innocent" }],
      });
      await res.text();
      return res.status === 422;
    });

    const res = await passthrough({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "innocent" }],
    });
    expect(res.status).toBe(422);
    const bodyText = await res.text();
    // The forbidden word MUST NOT reach the caller.
    expect(bodyText).not.toContain(FORBIDDEN_OUTPUT);
    const body = JSON.parse(bodyText) as { error?: { type?: unknown; message?: unknown } };
    expect(body.error?.type).toBe("content_filter");
    expect(String(body.error?.message)).toContain(`guardrail '${OUTPUT_GUARDRAIL}'`);
  });

  test("forbidden request body is blocked by the input guardrail before the upstream is called", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    // Self-synchronize on the INPUT guardrail loading (independent of the
    // output-block test above): a forbidden-input request is blocked 422 only
    // once the chain is live. A blocked request never reaches the upstream, so
    // polling here doesn't perturb the hit-count assertion below.
    await waitConfigPropagation(async () => {
      const probe = await passthrough({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: `probe ${FORBIDDEN_INPUT}` }],
      });
      await probe.text();
      return probe.status === 422;
    });

    const hitsBefore = upstream.receivedRequests.length;

    const res = await passthrough({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: `please ${FORBIDDEN_INPUT} now` }],
    });
    expect(res.status).toBe(422);
    const bodyText = await res.text();
    const body = JSON.parse(bodyText) as { error?: { type?: unknown; message?: unknown } };
    expect(body.error?.type).toBe("content_filter");
    expect(String(body.error?.message)).toContain(`guardrail '${INPUT_GUARDRAIL}'`);

    // Input guardrails run BEFORE the upstream call — a blocked request must
    // not reach the provider.
    expect(upstream.receivedRequests.length - hitsBefore).toBe(0);
  });
});
