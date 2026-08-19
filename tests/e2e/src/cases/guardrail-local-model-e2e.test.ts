import { createHash } from "node:crypto";
import OpenAI from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: local CPU embedding-model guardrail MVP (AISIX-Cloud#1331).
//
// The one acceptance path of the MVP vertical slice: a real request whose
// user text carries an EDA-software version number in natural-language
// Chinese goes through `/v1/chat/completions`, the in-process ONNX model
// judges the candidate's context window against the category prototype,
// and the version number is rewritten to `***`:
// - request side: the upstream's received body carries the masked text —
//   the version number never left the gateway;
// - response side: the (fixed) upstream reply carrying the same sentence
//   reaches the caller masked.
//
// SCOPE PINS (deliberate, per the MVP brief — not accidental gaps):
// - one happy path only; no negative/threshold/degrade cases;
// - non-streaming only: streamed output rides the guardrail's default
//   BufferFull hold-back + the same segment pass, but is not pinned here;
// - /v1/chat/completions only: the sibling families (/v1/messages,
//   /v1/responses, legacy completions, MCP) are explicitly unwired —
//   tracked on the design issue, not silently missing.
//
// OPT-IN SPEC: skipped unless AISIX_LOCAL_GUARDRAIL_MODEL_DIR points at
// the model directory (model.onnx + tokenizer.json). Setting it implies
// the binary under test was built with `--features local-model-guardrail`
// (a default build would warn, serve unmasked, and fail this spec).
// The opt-in var deliberately carries the harness-stripped AISIX_ prefix
// so it can never leak into OTHER specs' spawned binaries; this spec
// forwards it explicitly as the binary's own GUARDRAIL_LOCAL_MODEL_DIR
// (non-AISIX on purpose — the config loader maps every AISIX_* env var
// onto a config field and strictly rejects unknown ones).
// Model files: https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2
// (onnx/model_quint8_avx2.onnx saved as model.onnx, plus tokenizer.json).

const CALLER = "sk-local-model-e2e-caller";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

const SENSITIVE = "这个 EDA 软件的版本是 12.1";
const MASKED = "这个 EDA 软件的版本是 ***";

const MODEL_DIR = process.env.AISIX_LOCAL_GUARDRAIL_MODEL_DIR;

describe("local-model guardrail e2e: EDA version number masked on request and response", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcd: EtcdClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    if (!MODEL_DIR) return;
    etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // The mock reply carries the SAME sensitive sentence, so one request
    // exercises both moderation hooks: input (what the upstream received)
    // and output (what the caller got back).
    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-local-model",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: SENSITIVE },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 8, total_tokens: 13 },
      },
    });

    app = await spawnApp({
      extraEnv: { GUARDRAIL_LOCAL_MODEL_DIR: MODEL_DIR },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "local-model-e2e-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "local-model-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    // Caller key last: it authenticating implies the whole seed set is in
    // the DP snapshot (per this suite's readiness-gate rule).
    await seed.createApiKey({
      key_hash: hash(CALLER),
      allowed_models: ["local-model-e2e"],
    });
    await waitConfigPropagation(async () => {
      const r = await new ProxyClient(app!.proxyUrl, CALLER).listModels();
      return r.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("version number becomes *** in the reply; the upstream never saw it", async (ctx) => {
    if (!MODEL_DIR || !etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const res = await new OpenAI({
      apiKey: CALLER,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    }).chat.completions.create({
      model: "local-model-e2e",
      messages: [{ role: "user", content: SENSITIVE }],
    });

    // Response side: the reply reaches the caller with the version number
    // rewritten in place and everything else byte-identical.
    expect(res.choices[0]?.message?.content).toBe(MASKED);

    // Request side: the upstream received the masked prompt — the version
    // number never left the gateway.
    const lastReq = upstream.receivedRequests.at(-1);
    expect(lastReq).toBeDefined();
    expect(lastReq!.body).toContain(MASKED);
    expect(lastReq!.body).not.toContain("12.1");
  });
});
