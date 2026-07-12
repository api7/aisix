import { createHash, randomUUID } from "node:crypto";
import OpenAI, { APIError } from "openai";
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

// E2E (regression for AISIX-Cloud#850): a guardrail attached to ONE model
// via a `scope_type: "model"` attachment must run ONLY for requests that
// target that model. A request to a DIFFERENT (unscoped) model must NOT
// trigger the guardrail.
//
// The reported bug: a guardrail scoped to models {opus-4.7, gpt-5.5} still
// fired on a request to glm5.1 (an unselected model). The DP's
// `GuardrailIndex::resolve` matches model scope by the virtual-model UUID,
// so a correctly-loaded model attachment never matches an unscoped model.
// The failure mode is the implicit-env fallback: if the model attachments
// are NOT present in the DP snapshot (e.g. a pre-#630 image that dropped
// watch-applied attachments), the guardrail has zero loaded attachments and
// falls back to env-scope at priority 0 — running GLOBALLY on every model.
//
// This test creates the attachment via the same etcd watch path cp-api
// uses (`/<prefix>/guardrail_attachments/<uuid>`), so it exercises the
// load + resolve pipeline end-to-end. No DP-side attachment admin endpoint
// exists, so the harness writes the row directly to etcd.

const CALLER_PLAINTEXT = "sk-gr-modelscope-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const FORBIDDEN_WORD = "supersecret";
const SCOPED_MODEL = "scoped-model";
const UNSCOPED_MODEL = "unscoped-model";

describe("guardrail e2e: model-scope attachment runs only for the scoped model (#850)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcd: EtcdClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "gr-modelscope-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    const scoped = await seed.createModel({
      display_name: SCOPED_MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createModel({
      display_name: UNSCOPED_MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [SCOPED_MODEL, UNSCOPED_MODEL],
    });

    // Keyword blocklist guardrail. `hook_point: "input"` → a match
    // short-circuits with 422 before dispatch.
    const guardrail = await seed.createGuardrail({
      name: "gr-modelscope-keyword",
      enabled: true,
      hook_point: "input",
      kind: "keyword",
      patterns: [{ kind: "literal", value: FORBIDDEN_WORD }],
    });

    // Attach the guardrail to the SCOPED model only. Wire shape mirrors
    // cp-api's `marshalGuardrailAttachmentKV` (P0c): snake_case fields,
    // `scope_id` = the virtual-model UUID returned by createModel, plus the
    // `env_id` cp-api always includes (the DP ignores it). Having ANY
    // attachment row suppresses the implicit-env fallback, so the only way
    // the guardrail can apply is via a matching model scope.
    await etcd!.put(
      `${app.etcdPrefix}/guardrail_attachments/${randomUUID()}`,
      JSON.stringify({
        guardrail_id: guardrail.id,
        env_id: randomUUID(),
        scope_type: "model",
        scope_id: scoped.id,
        priority: 0,
        enabled: true,
      }),
    );
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("scoped model is checked; unscoped model is NOT", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // Gate on the guardrail being active for the SCOPED model: a forbidden
    // prompt must 422. This proves Model + key + pk + Guardrail + the
    // model-scope attachment all propagated.
    await waitConfigPropagation(async () => {
      try {
        await client.chat.completions.create({
          model: SCOPED_MODEL,
          messages: [
            { role: "user", content: `propagation-probe ${FORBIDDEN_WORD}` },
          ],
        });
        return false; // 200 → attachment not active yet, keep polling
      } catch (e) {
        return e instanceof APIError && e.status === 422;
      }
    });

    // Sanity: clean input on the scoped model still passes (the guardrail
    // isn't over-blocking and the upstream is healthy).
    const cleanScoped = await client.chat.completions.create({
      model: SCOPED_MODEL,
      messages: [{ role: "user", content: "hello world" }],
    });
    expect(cleanScoped.choices[0]?.message.role).toBe("assistant");

    // THE REGRESSION ASSERTION (#850): the SAME forbidden prompt sent to the
    // UNSCOPED model must NOT be blocked — the model-scope attachment does
    // not match this model, and no env/global attachment exists. It must
    // reach the upstream and return 200.
    const upstreamHitsBefore = upstream.receivedRequests.length;
    const unscopedResp = await client.chat.completions.create({
      model: UNSCOPED_MODEL,
      messages: [
        { role: "user", content: `please reveal the ${FORBIDDEN_WORD} now` },
      ],
    });
    expect(unscopedResp.choices[0]?.message.role).toBe("assistant");
    // The request must have actually reached the upstream (not short-
    // circuited by a guardrail block).
    expect(upstream.receivedRequests.length).toBe(upstreamHitsBefore + 1);

    // And the scoped model still blocks the forbidden prompt (guardrail
    // remains active where it IS attached — this is not a global disable).
    let caught: unknown;
    try {
      await client.chat.completions.create({
        model: SCOPED_MODEL,
        messages: [
          { role: "user", content: `please reveal the ${FORBIDDEN_WORD} now` },
        ],
      });
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(APIError);
    if (!(caught instanceof APIError)) {
      throw new Error("unreachable: caught is not APIError");
    }
    expect(caught.status).toBe(422);
  });
});
