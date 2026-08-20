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

// E2E: local CPU embedding-model guardrail (AISIX-Cloud#1331).
//
// Two acceptance paths through `/v1/chat/completions`, each asserted on
// both sides (request: what the upstream received; response: what the
// caller got back):
// - the MVP acceptance sentence: an EDA-software version number in
//   natural-language Chinese is rewritten to `***`;
// - the layer-② acceptance matrix: one message carrying both hard
//   positives (`升级到 2022.4`, `Virtuoso IC6.1.8` — the shapes the MVP
//   measurably missed) and the hard negatives (compile-log timing,
//   memory size, IPv4, bare number); the positives are rewritten, the
//   negatives come back byte-identical, and the bare number traverses a
//   live layer-③ inference (no lexical evidence → model band).
//
// SCOPE PINS (deliberate, per the MVP brief — not accidental gaps):
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

// The layer-② acceptance matrix in one message: both hard positives, the
// hard negatives, and a bare number with no lexical evidence at all (the
// model band — this candidate pays a real in-process inference).
const MIXED =
  "我们把仿真工具升级到 2022.4 之后,Virtuoso IC6.1.8 反而开始频繁崩溃," +
  "完整的运行日志我贴在下面了,麻烦帮忙看看到底是哪一步出了问题: " +
  "Elapsed: 12.345s, Memory: 4.2 GB, 服务器 IP 是 10.2.255.1, " +
  "另外圆周率约等于 3.14159";
const MIXED_MASKED =
  "我们把仿真工具升级到 *** 之后,Virtuoso IC*** 反而开始频繁崩溃," +
  "完整的运行日志我贴在下面了,麻烦帮忙看看到底是哪一步出了问题: " +
  "Elapsed: 12.345s, Memory: 4.2 GB, 服务器 IP 是 10.2.255.1, " +
  "另外圆周率约等于 3.14159";

const MODEL_DIR = process.env.AISIX_LOCAL_GUARDRAIL_MODEL_DIR;

describe("local-model guardrail e2e: EDA version number masked on request and response", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let mixedUpstream: OpenAiUpstream | undefined;
  let etcd: EtcdClient | undefined;
  let etcdReachable = false;

  const reply = (content: string) => ({
    id: "cmpl-local-model",
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: "gpt-4o-mini",
    choices: [
      {
        index: 0,
        message: { role: "assistant", content },
        finish_reason: "stop",
      },
    ],
    usage: { prompt_tokens: 5, completion_tokens: 8, total_tokens: 13 },
  });

  beforeAll(async () => {
    if (!MODEL_DIR) return;
    etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // Each mock reply carries the SAME text as its test's prompt, so one
    // request exercises both moderation hooks: input (what the upstream
    // received) and output (what the caller got back).
    upstream = await startOpenAiUpstream({ nonStreamBody: reply(SENSITIVE) });
    mixedUpstream = await startOpenAiUpstream({ nonStreamBody: reply(MIXED) });

    app = await spawnApp({
      // 2 lanes so the acceptance path exercises the session POOL
      // dispatch (api7/aisix#1001), not just the single-lane degenerate
      // case; behavior must be identical (lanes are stateless).
      extraEnv: {
        GUARDRAIL_LOCAL_MODEL_DIR: MODEL_DIR,
        GUARDRAIL_LOCAL_MODEL_LANES: "2",
      },
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
    const mixedPk = await seed.createProviderKey({
      display_name: "local-model-e2e-mixed-pk",
      secret: "sk-mock",
      api_base: `${mixedUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "local-model-e2e-mixed",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: mixedPk.id,
    });
    // Caller key last: it authenticating implies the whole seed set is in
    // the DP snapshot (per this suite's readiness-gate rule).
    await seed.createApiKey({
      key_hash: hash(CALLER),
      allowed_models: ["local-model-e2e", "local-model-e2e-mixed"],
    });
    await waitConfigPropagation(async () => {
      const r = await new ProxyClient(app!.proxyUrl, CALLER).listModels();
      return r.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await mixedUpstream?.close();
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

  test("hard positives are rewritten while hard negatives pass byte-identical", async (ctx) => {
    if (!MODEL_DIR || !etcdReachable || !app || !mixedUpstream) {
      ctx.skip();
      return;
    }

    const res = await new OpenAI({
      apiKey: CALLER,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    }).chat.completions.create({
      model: "local-model-e2e-mixed",
      messages: [{ role: "user", content: MIXED }],
    });

    // Response side: versions masked, everything else — the compile-log
    // numbers, the IP, the bare number the model judged — byte-identical.
    expect(res.choices[0]?.message?.content).toBe(MIXED_MASKED);

    // Request side: the upstream saw the same rewrite and neither
    // version value.
    const lastReq = mixedUpstream.receivedRequests.at(-1);
    expect(lastReq).toBeDefined();
    expect(lastReq!.body).toContain(MIXED_MASKED);
    expect(lastReq!.body).not.toContain("2022.4");
    expect(lastReq!.body).not.toContain("6.1.8");
  });
});
