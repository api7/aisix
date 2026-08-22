import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import OpenAI from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  scrapeMetrics,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  sumMetric,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: the semantic guardrail against the REAL embedding model
// (AISIX-Cloud#1331 → #1363). The kind-level contract — wiring, scoping,
// sibling families, /mcp, monitor mode, degrade metrics — is pinned
// model-free in guardrail-semantic-kind-e2e.test.ts; this opt-in suite
// adds the layer-③ acceptance path: a live in-process inference judges
// the model band, and the model-call metric family counts it.
//
// The guardrail is a standard `kind: "semantic"` ROW (the factory EDA
// template) resolved through attachments — the MVP-era env-activated
// global is gone.
//
// OPT-IN SPEC: skipped unless AISIX_LOCAL_GUARDRAIL_MODEL_DIR points at
// the model bundle (model.onnx + tokenizer.json; a manifest.json is
// generated next to them on first run by hashing what is there). The
// opt-in var deliberately carries the harness-stripped AISIX_ prefix so
// it can never leak into OTHER specs' spawned binaries; this spec
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
// hard negatives (including the Chinese-unit and timestamp classes), and
// a bare number with no lexical evidence at all — the model band, which
// on THIS suite pays a real in-process inference and releases (the
// description prototype leans precision).
const MIXED =
  "我们把仿真工具升级到 2022.4 之后,Virtuoso IC6.1.8 反而开始频繁崩溃," +
  "完整的运行日志我贴在下面了,麻烦帮忙看看到底是哪一步出了问题: " +
  "[10:23:45.123] build started, 整个 build 花了 45.5 秒, " +
  "Elapsed: 12.345s, Memory: 4.2 GB, 服务器 IP 是 10.2.255.1, " +
  "另外圆周率约等于 3.14159";
const MIXED_MASKED =
  "我们把仿真工具升级到 *** 之后,Virtuoso *** 反而开始频繁崩溃," +
  "完整的运行日志我贴在下面了,麻烦帮忙看看到底是哪一步出了问题: " +
  "[10:23:45.123] build started, 整个 build 花了 45.5 秒, " +
  "Elapsed: 12.345s, Memory: 4.2 GB, 服务器 IP 是 10.2.255.1, " +
  "另外圆周率约等于 3.14159";

/** The factory EDA-version category (the CP template's exact shape). */
const EDA_CATEGORY = {
  name: "eda_version",
  description: "EDA 软件的版本号",
  candidate_patterns: [
    "[0-9０-９]+(?:[.．][0-9０-９]+)+",
    "[A-Za-z][A-Za-z0-9._-]*[0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?",
    "[0-9][A-Za-z0-9._-]*[A-Za-z](?:[A-Za-z0-9._-]*[A-Za-z0-9])?",
  ],
  negative_patterns: [
    "^\\s*(?:%|％|(?i:ms|us|ns|ps|fs|s|secs?|seconds?|mins?|minutes?|hours?|[kmgt]i?b|um|nm|[kmg]?hz)(?:[^0-9A-Za-z]|$)|纳秒|微秒|毫秒|秒|分钟|个?(?:小时|钟头|星期|月)|天|纳米|微米|毫米|[兆吉太]字节|[千兆吉]?赫兹|[千兆吉]赫|个?百分点|摄氏度|度(?:[^过]|$)|伏特?|瓦特?|安培|毫安)",
    "(?i)[0-9](?:%|ms|us|ns|ps|fs|s|secs?|seconds?|mins?|minutes?|hours?|[kmgt]i?b|um|nm|[kmg]?hz)$",
    "百分之\\s*$",
    "^\\d{1,3}(?:\\.\\d{1,3}){3}$",
    "[\\w.-]+\\.[A-Za-z0-9]+:$",
    "^:\\d",
    "[0-9]{1,2}[:：][0-9]{1,2}[:：]$",
    "(?i)\\.[a-z]{2,4}$",
    "(?:^|[^本])编号[:：]?(?:是|为)?\\s*$",
  ],
  hotword_groups: [
    { terms: ["版本号", "版本", "升级到", "回退到"] },
    { terms: ["version", "release", "build", "upgrade to", "upgraded to"] },
    { terms: ["virtuoso", "calibre", "vcs", "innovus", "icc2", "primetime"] },
  ],
  action: "mask",
  replacement: "***",
};

const MODEL_DIR = process.env.AISIX_LOCAL_GUARDRAIL_MODEL_DIR;

/** Generate `manifest.json` next to the model files when absent — the
 * test-side counterpart of the release bundle's pinned manifest. */
function ensureManifest(dir: string): void {
  const path = join(dir, "manifest.json");
  if (existsSync(path)) return;
  const files: Record<string, string> = {};
  for (const name of ["model.onnx", "tokenizer.json"]) {
    files[name] = `sha256:${createHash("sha256")
      .update(readFileSync(join(dir, name)))
      .digest("hex")}`;
  }
  writeFileSync(
    path,
    JSON.stringify(
      {
        manifest_version: 1,
        model_id: "ibm-granite/granite-embedding-97m-multilingual-r2",
        embedding_dim: 384,
        files,
        calibration: { description: 0.8 },
      },
      null,
      2,
    ),
  );
}

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
    ensureManifest(MODEL_DIR);
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
    // The guardrail is an ordinary semantic ROW, env-scoped through the
    // implicit fallback (no attachment rows — the file-source posture);
    // scoped-attachment behavior is pinned in the kind suite.
    await seed.createGuardrail({
      name: "local-model-semantic",
      enabled: true,
      hook_point: "both",
      kind: "semantic",
      categories: [EDA_CATEGORY],
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

  test("hard positives are rewritten while hard negatives pass byte-identical, through a live inference", async (ctx) => {
    if (!MODEL_DIR || !etcdReachable || !app || !mixedUpstream) {
      ctx.skip();
      return;
    }

    const before = sumMetric(
      await scrapeMetrics(app.metricsUrl),
      "aisix_guardrail_semantic_model_calls_total",
    );
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

    // The bare number traversed a LIVE layer-③ inference, and the
    // model-call metric family counted it (the /metrics shipping rule —
    // this is the real-model half; the degrade half is asserted in the
    // kind suite).
    const after = sumMetric(
      await scrapeMetrics(app.metricsUrl),
      "aisix_guardrail_semantic_model_calls_total",
    );
    expect(after).toBeGreaterThan(before);
  });
});
