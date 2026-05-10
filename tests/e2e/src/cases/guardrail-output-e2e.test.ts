import { createHash } from "node:crypto";
import OpenAI, { APIError } from "openai";
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

// E2E: keyword guardrail blocks the assistant's *output* when it
// contains the forbidden pattern. The existing guardrail-keyword-e2e
// covers `hook_point: "input"`; this case covers the symmetric
// `hook_point: "output"` user journey, which is the more
// interesting safety surface — input filtering only stops users from
// asking for forbidden content; output filtering is what stops the
// model from disclosing it (e.g. trained-in PII, jailbreak responses,
// off-policy text the model produced for an innocent-looking prompt).
//
// Reference:
// - OpenAI Chat Completions API spec
//   <https://platform.openai.com/docs/api-reference/chat/create>
// - OpenAI / Azure content-filter convention for the
//   `error.type: "content_filter"` envelope value
//   <https://learn.microsoft.com/azure/ai-services/openai/concepts/content-filter>

const CALLER_PLAINTEXT = "sk-gr-out-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const FORBIDDEN_WORD = "leakedsecret";

describe("output guardrail e2e: model-emitted forbidden text is blocked before reaching caller", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    // Mock upstream emits an OpenAI-shape completion that CONTAINS
    // the forbidden word. The caller's prompt is innocent; the
    // forbidden content originates from the model's response — that's
    // exactly the case input-only guardrails miss.
    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-leak",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: `Sure, here it is: ${FORBIDDEN_WORD}.`,
            },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 8, total_tokens: 13 },
      },
    });

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "gr-out-e2e-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "gr-out-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["gr-out-e2e"],
    });
    // Output guardrail: runs against the assistant's response after
    // the upstream call returns, before relay to the caller.
    await admin.json("POST", "/admin/v1/guardrails", {
      name: "gr-out-e2e-keyword",
      enabled: true,
      hook_point: "output",
      kind: "keyword",
      patterns: [{ kind: "literal", value: FORBIDDEN_WORD }],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("upstream emits forbidden word → caller sees content_filter 422, NOT the forbidden text", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // Output guardrails fire AFTER upstream dispatch, so propagation
    // readiness is signaled by the same 422-on-blocked-response
    // pattern. A 200 means the guardrail isn't loaded yet
    // (gateway forwarded the leaked content); keep polling.
    await waitConfigPropagation(async () => {
      try {
        await client.chat.completions.create({
          model: "gr-out-e2e",
          messages: [{ role: "user", content: "innocent question" }],
        });
        return false; // 200 means guardrail not ready
      } catch (e) {
        return e instanceof APIError && e.status === 422;
      }
    });

    const upstreamHitsBefore = upstream.receivedRequests.length;

    let caught: unknown;
    try {
      await client.chat.completions.create({
        model: "gr-out-e2e",
        messages: [{ role: "user", content: "tell me something useful" }],
      });
    } catch (e) {
      caught = e;
    }

    expect(caught).toBeInstanceOf(APIError);
    if (!(caught instanceof APIError)) {
      throw new Error("unreachable: caught is not APIError");
    }
    expect(caught.status).toBe(422);
    // Pin envelope to the content_filter type so a regression that
    // 422'd via a different path (e.g. generic schema validation)
    // would fail; the value `content_filter` is the OpenAI / Azure
    // public taxonomy for this exact case.
    expect((caught.error as { type?: unknown })?.type).toBe("content_filter");

    // The leaked word MUST NOT appear anywhere in the caller-visible
    // error envelope. The whole point of an output guardrail is to
    // prevent the forbidden content from reaching the caller — even
    // echoing it back inside an error message would defeat the
    // purpose (and would be a real, reportable security regression).
    const errorBlob = JSON.stringify(caught.error ?? {});
    const messageBlob = caught.message ?? "";
    expect(errorBlob).not.toContain(FORBIDDEN_WORD);
    expect(messageBlob).not.toContain(FORBIDDEN_WORD);

    // Output guardrails run AFTER the upstream call, so the upstream
    // hit count MUST go up by 1 (the guardrail can only inspect what
    // the upstream returned). A regression that short-circuited
    // pre-dispatch would leave the count flat — that's a *safer*
    // failure mode (no upstream call, no token cost), but it would
    // signal the guardrail's hook_point semantics drifted.
    expect(upstream.receivedRequests.length - upstreamHitsBefore).toBe(1);
  });
});
