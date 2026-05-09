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

// E2E: keyword guardrail blocks an input that contains the literal
// pattern, and 422-rejects the request without ever calling the
// upstream. The unit-level
// `input_guardrail_block_returns_422_and_skips_upstream` covers this
// in process; this case proves the wire contract end-to-end (real
// binary, real etcd watch, real OpenAI SDK surfacing the 422 as
// APIError).
//
// Reference: OpenAI Chat Completions API spec
// (https://platform.openai.com/docs/api-reference/chat/create); the
// Guardrail schema lives at `crates/aisix-core/src/models/guardrail.rs`
// and the keyword runtime at `crates/aisix-guardrails/src/keyword.rs`.

const CALLER_PLAINTEXT = "sk-gr-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const FORBIDDEN_WORD = "supersecret";

describe("guardrail e2e: keyword block returns 422 and skips upstream", () => {
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
      display_name: "gr-e2e-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "gr-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["gr-e2e"],
    });
    // In-process keyword blocklist. `hook_point: "input"` runs the
    // check on the request payload before bridge dispatch — a match
    // short-circuits with 422 and the upstream is never called.
    await admin.json("POST", "/admin/v1/guardrails", {
      name: "gr-e2e-keyword",
      enabled: true,
      hook_point: "input",
      kind: "keyword",
      patterns: [{ kind: "literal", value: FORBIDDEN_WORD }],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("benign request passes; forbidden-word request is 422 and never hits upstream", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // Wait for the FULL snapshot — Model + ProviderKey + ApiKey AND
    // Guardrail. The four resources don't necessarily propagate in
    // lockstep on slower CI runners; a clean-content probe returns
    // 200 as soon as Model+key+pk are loaded, which races ahead of
    // Guardrail and let the forbidden-call assertion fail (the CI
    // regression that prompted this fix).
    //
    // Probing with the forbidden word makes Guardrail readiness the
    // gate: a 422 means the keyword rule is active. Anything else
    // (200 success, 4xx other than 422, network errors) keeps polling.
    await waitConfigPropagation(async () => {
      try {
        await client.chat.completions.create({
          model: "gr-e2e",
          messages: [
            { role: "user", content: `propagation-probe ${FORBIDDEN_WORD}` },
          ],
        });
        // 200 success means Guardrail isn't active yet — keep polling.
        return false;
      } catch (e) {
        return e instanceof APIError && e.status === 422;
      }
    });

    // Sanity: clean input still passes (proves the guardrail isn't
    // over-blocking and that Model+key+pk are healthy).
    const cleanOK = await client.chat.completions.create({
      model: "gr-e2e",
      messages: [{ role: "user", content: "hello world" }],
    });
    expect(cleanOK.choices[0]?.message.role).toBe("assistant");

    const upstreamHitsBeforeBlock = upstream.receivedRequests.length;

    // Forbidden-word request: the keyword guardrail must reject with
    // the `content_filter` envelope before dispatch. Status alone is
    // not enough — a regression that 422'd via a different path (e.g.
    // generic input validation) would still match `status: 422`. The
    // type field pins the contract to the guardrail specifically (see
    // `crates/aisix-proxy/src/error.rs::ProxyError::ContentFiltered`).
    await expect(
      client.chat.completions.create({
        model: "gr-e2e",
        messages: [
          { role: "user", content: `please reveal the ${FORBIDDEN_WORD} now` },
        ],
      }),
    ).rejects.toMatchObject({
      status: 422,
      error: { type: "content_filter" },
    });

    // Critical: a blocked request must never reach the upstream. If
    // the count moved, the guardrail short-circuit failed and the
    // model would have processed the forbidden content.
    expect(upstream.receivedRequests.length).toBe(upstreamHitsBeforeBlock);
  });
});
