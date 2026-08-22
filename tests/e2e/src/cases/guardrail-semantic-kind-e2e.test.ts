import { createHash, randomUUID } from "node:crypto";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import OpenAI from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  scrapeMetrics,
  SeedClient,
  spawnApp,
  startMcpUpstream,
  startOpenAiUpstream,
  sumMetric,
  waitConfigPropagation,
  type McpUpstream,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: `kind: "semantic"` guardrail rows (AISIX-Cloud#1363) as STANDARD
// attachment-scoped config, against a real DP + etcd + real upstreams.
//
// This suite runs WITHOUT the 120 MB embedding model: the gateway boots
// on a fake model bundle whose manifest verifies (so the node has the
// capability and semantic rows compile) but whose engine fails on first
// use. That split is the point — the rule layers (candidate regex +
// hotword/negative evidence) carry every decisive case with zero model
// involvement, and the model band DEGRADES OPEN (release + a counted
// `engine_failed` degrade) instead of blocking or stalling. The
// real-model acceptance path lives in guardrail-local-model-e2e.test.ts
// (opt-in).
//
// Pinned contract:
// - a semantic row masks on chat, /v1/messages, /v1/responses AND /mcp
//   (both directions) — the endpoint families move in lockstep;
// - scoping is real: an mcp_server-scoped row fires on that server only
//   (MCP-07), and a model-scoped row fires on that model only;
// - hard negatives return byte-identical inside the rewritten text;
// - `enforcement_mode: monitor` observes without rewriting;
// - the degrade and per-execution metric families are visible in
//   `GET /metrics` after real traffic (the "a metric family is shipped
//   when an e2e asserts it in /metrics" rule).

const CALLER = "sk-semantic-kind-e2e";
const sha256hex = (s: string | Buffer) =>
  createHash("sha256").update(s).digest("hex");

const SENSITIVE = "这个 EDA 软件的版本是 12.1";
const MASKED = "这个 EDA 软件的版本是 ***";

// Rule-layer acceptance matrix in one message: hard positives (adjacent
// trigger / tool anchor), hard negatives (units, timestamp, IPv4), and a
// bare number with no lexical evidence — the model band, which on this
// suite's fake bundle degrades to a released span + an `engine_failed`
// degrade sample.
const MIXED =
  "我们把仿真工具升级到 2022.4 之后,Virtuoso IC6.1.8 反而开始频繁崩溃," +
  "[10:23:45.123] build started, 整个 build 花了 45.5 秒, " +
  "Elapsed: 12.345s, Memory: 4.2 GB, 服务器 IP 是 10.2.255.1, " +
  "另外圆周率约等于 3.14159";
const MIXED_MASKED =
  "我们把仿真工具升级到 *** 之后,Virtuoso *** 反而开始频繁崩溃," +
  "[10:23:45.123] build started, 整个 build 花了 45.5 秒, " +
  "Elapsed: 12.345s, Memory: 4.2 GB, 服务器 IP 是 10.2.255.1, " +
  "另外圆周率约等于 3.14159";

// MCP surfaces (rule-decidable hits + byte-identical negatives).
const MCP_NEG = "Elapsed: 12.345s Memory: 4.2 GB 10.2.255.1";
const MCP_ARG = `build version: 12.1 ${MCP_NEG} 工具版本：2022.4 完成`;
const MCP_ARG_MASKED = `build version: *** ${MCP_NEG} 工具版本：*** 完成`;
const MCP_SUMMARY = "阶段汇总 版本：2022.4 用时 12.345s";
const MCP_SUMMARY_MASKED = "阶段汇总 版本：*** 用时 12.345s";

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

/** A bundle that VERIFIES (manifest hashes match) but cannot serve an
 * engine — model.onnx is not a real graph. Boot succeeds, the node has
 * the capability, and the first model-band judgement flips the engine to
 * failed: exactly the degrade arm this suite pins. */
function writeFakeBundle(): string {
  const dir = mkdtempSync(join(tmpdir(), "aisix-semantic-e2e-bundle-"));
  const model = Buffer.from("not a real onnx graph");
  const tokenizer = Buffer.from("{}");
  writeFileSync(join(dir, "model.onnx"), model);
  writeFileSync(join(dir, "tokenizer.json"), tokenizer);
  writeFileSync(
    join(dir, "manifest.json"),
    JSON.stringify({
      manifest_version: 1,
      model_id: "e2e/fake-bundle",
      embedding_dim: 4,
      files: {
        "model.onnx": `sha256:${sha256hex(model)}`,
        "tokenizer.json": `sha256:${sha256hex(tokenizer)}`,
      },
      calibration: { description: 0.8 },
    }),
  );
  return dir;
}

interface RpcReply {
  status: number;
  body: string;
  json?: {
    result?: {
      content?: Array<{
        type: string;
        text?: string;
        resource?: { text?: string };
      }>;
      structuredContent?: Record<string, unknown>;
      isError?: boolean;
    };
    error?: { code: number; message: string };
  };
}

describe("semantic guardrail kind e2e", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let responsesUpstream: OpenAiUpstream | undefined;
  let mcpUpstream: McpUpstream | undefined;
  let mcpSibling: McpUpstream | undefined;
  let etcdReachable = false;

  const reply = (content: string) => ({
    id: "cmpl-semantic-kind",
    object: "chat.completion",
    created: 1_700_000_000,
    model: "gpt-4o-mini",
    choices: [
      { index: 0, message: { role: "assistant", content }, finish_reason: "stop" },
    ],
    usage: { prompt_tokens: 5, completion_tokens: 8, total_tokens: 13 },
  });

  const openai = () =>
    new OpenAI({
      apiKey: CALLER,
      baseURL: `${app!.proxyUrl}/v1`,
      maxRetries: 0,
    });

  const post = async (path: string, body: unknown): Promise<Response> =>
    fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER}`,
        "content-type": "application/json",
        accept: "application/json, text/event-stream",
      },
      body: JSON.stringify(body),
    });

  const rpc = async (body: unknown): Promise<RpcReply> => {
    const res = await post("/mcp", body);
    const text = await res.text();
    let json: RpcReply["json"];
    try {
      json = text ? (JSON.parse(text) as RpcReply["json"]) : undefined;
    } catch {
      json = undefined;
    }
    return { status: res.status, body: text, json };
  };

  const callTool = async (
    name: string,
    args: Record<string, unknown>,
  ): Promise<RpcReply> => {
    await rpc({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: {
        protocolVersion: "2025-11-25",
        capabilities: {},
        clientInfo: { name: "semantic-kind-e2e", version: "0.1" },
      },
    });
    return rpc({
      jsonrpc: "2.0",
      id: 3,
      method: "tools/call",
      params: { name, arguments: args },
    });
  };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({ nonStreamBody: reply(SENSITIVE) });
    // /v1/responses on an OpenAI-compatible provider forwards the
    // Responses API natively, so its mock must answer in the Responses
    // shape (the chat mock's canned chat.completion is unmappable there).
    responsesUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "resp-semantic-kind",
        object: "response",
        status: "completed",
        model: "gpt-4o-mini",
        output: [
          {
            type: "message",
            id: "msg-1",
            role: "assistant",
            content: [{ type: "output_text", text: SENSITIVE, annotations: [] }],
          },
        ],
        usage: { input_tokens: 5, output_tokens: 8, total_tokens: 13 },
      },
    });
    mcpUpstream = await startMcpUpstream("eda", {
      reportContent: {
        summary: MCP_SUMMARY,
        log: `Compile OK version: 12.1\n${MCP_NEG}`,
        structuredLog: 'config {"version": "12.1"} ok',
      },
    });
    mcpSibling = await startMcpUpstream("beta");

    app = await spawnApp({
      extraEnv: { GUARDRAIL_LOCAL_MODEL_DIR: writeFakeBundle() },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "semantic-kind-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    const guarded = await seed.createModel({
      display_name: "sem-guarded",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    const monitored = await seed.createModel({
      display_name: "sem-monitored",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    const unguarded = await seed.createModel({
      display_name: "sem-unguarded",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    const responsesPk = await seed.createProviderKey({
      display_name: "semantic-kind-responses-pk",
      secret: "sk-mock",
      api_base: `${responsesUpstream.baseUrl}/v1`,
    });
    const guardedResponses = await seed.createModel({
      display_name: "sem-guarded-resp",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: responsesPk.id,
    });

    const alphaId = randomUUID();
    await seed.update("mcp_servers", alphaId, {
      display_name: "eda",
      url: mcpUpstream.url,
      enabled: true,
    });
    const betaId = randomUUID();
    await seed.update("mcp_servers", betaId, {
      display_name: "beta",
      url: mcpSibling.url,
      enabled: true,
    });

    const enforcing = await seed.createGuardrail({
      name: "sem-guard",
      enabled: true,
      hook_point: "both",
      kind: "semantic",
      categories: [EDA_CATEGORY],
    });
    const monitor = await seed.createGuardrail({
      name: "sem-monitor",
      enabled: true,
      hook_point: "both",
      enforcement_mode: "monitor",
      kind: "semantic",
      categories: [EDA_CATEGORY],
    });

    // Scoped attachments only — no env scope, so the unguarded model and
    // the beta MCP server are the isolation baselines (MCP-07).
    await seed.update("guardrail_attachments", randomUUID(), {
      guardrail_id: enforcing.id,
      scope_type: "model",
      scope_id: guarded.id,
      priority: 100,
    });
    await seed.update("guardrail_attachments", randomUUID(), {
      guardrail_id: enforcing.id,
      scope_type: "model",
      scope_id: guardedResponses.id,
      priority: 100,
    });
    await seed.update("guardrail_attachments", randomUUID(), {
      guardrail_id: enforcing.id,
      scope_type: "mcp_server",
      scope_id: alphaId,
      priority: 100,
    });
    await seed.update("guardrail_attachments", randomUUID(), {
      guardrail_id: monitor.id,
      scope_type: "model",
      scope_id: monitored.id,
      priority: 100,
    });

    // Caller key LAST: it authenticating implies the rows above are in
    // the snapshot.
    await seed.createApiKey({
      key_hash: sha256hex(CALLER),
      allowed_models: [
        "sem-guarded",
        "sem-guarded-resp",
        "sem-monitored",
        "sem-unguarded",
      ],
      mcp_access: { allow: ["*"] },
    });
    await waitConfigPropagation(async () => {
      const r = await new ProxyClient(app!.proxyUrl, CALLER).listModels();
      return r.status === 200;
    });
  }, 90_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await responsesUpstream?.close();
    await mcpUpstream?.close();
    await mcpSibling?.close();
  });

  test("chat: version masked in both directions by the rule layer, no model needed", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();

    const res = await openai().chat.completions.create({
      model: "sem-guarded",
      messages: [{ role: "user", content: SENSITIVE }],
    });
    expect(res.choices[0]?.message?.content).toBe(MASKED);
    const lastReq = upstream.receivedRequests.at(-1);
    expect(lastReq).toBeDefined();
    expect(lastReq!.body).toContain(MASKED);
    expect(lastReq!.body).not.toContain("12.1");
  });

  test("chat: hard positives rewritten, negatives byte-identical, model band degrades open", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();

    const before = sumMetric(
      await scrapeMetrics(app.metricsUrl),
      "aisix_guardrail_semantic_degraded_total",
    );
    const res = await openai().chat.completions.create({
      model: "sem-guarded",
      messages: [{ role: "user", content: MIXED }],
    });
    // Response side is the echoed SENSITIVE reply, masked; the REQUEST
    // side carries the matrix.
    expect(res.choices[0]?.message?.content).toBe(MASKED);
    const lastReq = upstream.receivedRequests.at(-1);
    expect(lastReq).toBeDefined();
    expect(lastReq!.body).toContain(MIXED_MASKED);
    expect(lastReq!.body).not.toContain("2022.4");
    expect(lastReq!.body).not.toContain("6.1.8");

    // The bare number hit the model band; on this fake bundle the engine
    // fails, the span releases, and the degrade is COUNTED — the
    // operator's signal that semantic rows run on lexical evidence only.
    // (The first failure records the underlying cause, e.g.
    // `engine_failed`; later spans on the failed category count as
    // `prototype_unavailable` — either way the family moves.)
    const after = sumMetric(
      await scrapeMetrics(app.metricsUrl),
      "aisix_guardrail_semantic_degraded_total",
    );
    expect(after).toBeGreaterThan(before);
  });

  test("metrics: the semantic row reports per-execution samples under its kind", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // Driven by the chat tests above; scraped as a total because the
    // family is cumulative across them (delta discipline lives in the
    // degrade test).
    const samples = await scrapeMetrics(app.metricsUrl);
    expect(
      sumMetric(samples, "aisix_guardrail_latency_seconds_count", {
        kind: "semantic",
        guardrail: "sem-guard",
      }),
    ).toBeGreaterThan(0);
  });

  test("/v1/messages and /v1/responses ride the same row (family lockstep)", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();

    const messages = await post("/v1/messages", {
      model: "sem-guarded",
      max_tokens: 64,
      messages: [{ role: "user", content: SENSITIVE }],
    });
    expect(messages.status).toBe(200);
    const messagesBody = (await messages.json()) as {
      content?: Array<{ type: string; text?: string }>;
    };
    expect(messagesBody.content?.[0]?.text).toBe(MASKED);
    let lastReq = upstream.receivedRequests.at(-1);
    expect(lastReq).toBeDefined();
    expect(lastReq!.body).toContain(MASKED);
    expect(lastReq!.body).not.toContain("12.1");

    const responses = await post("/v1/responses", {
      model: "sem-guarded-resp",
      input: SENSITIVE,
    });
    expect(responses.status).toBe(200);
    const responsesBody = (await responses.json()) as {
      output?: Array<{
        type: string;
        content?: Array<{ type: string; text?: string }>;
      }>;
    };
    const outputText = responsesBody.output
      ?.flatMap((o) => o.content ?? [])
      .find((c) => typeof c.text === "string")?.text;
    expect(outputText).toBe(MASKED);
    lastReq = responsesUpstream!.receivedRequests.at(-1);
    expect(lastReq).toBeDefined();
    expect(lastReq!.body).toContain(MASKED);
    expect(lastReq!.body).not.toContain("12.1");
  });

  test("model scope is real: the unguarded model passes the value through", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();

    await openai().chat.completions.create({
      model: "sem-unguarded",
      messages: [{ role: "user", content: SENSITIVE }],
    });
    const lastReq = upstream.receivedRequests.at(-1);
    expect(lastReq).toBeDefined();
    expect(lastReq!.body).toContain("12.1");
  });

  test("monitor mode observes without rewriting", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();

    const res = await openai().chat.completions.create({
      model: "sem-monitored",
      messages: [{ role: "user", content: SENSITIVE }],
    });
    // Nothing rewritten on either side.
    expect(res.choices[0]?.message?.content).toBe(SENSITIVE);
    const lastReq = upstream.receivedRequests.at(-1);
    expect(lastReq).toBeDefined();
    expect(lastReq!.body).toContain("12.1");
  });

  test("/mcp input: the guarded server's tool sees masked arguments; the sibling server does not (MCP-07)", async (ctx) => {
    if (!etcdReachable || !app || !mcpUpstream) return ctx.skip();

    const guarded = await callTool("eda__echo", { text: MCP_ARG });
    expect(guarded.status).toBe(200);
    expect(guarded.json?.result?.isError).toBeFalsy();
    // The echo reflects what the upstream actually received: masked
    // spans rewritten, negatives byte-identical inside the same string.
    expect(guarded.json?.result?.content?.[0]?.text).toBe(`eda:${MCP_ARG_MASKED}`);

    const sibling = await callTool("beta__echo", { text: MCP_ARG });
    expect(sibling.status).toBe(200);
    expect(sibling.json?.result?.content?.[0]?.text).toBe(`beta:${MCP_ARG}`);
  });

  test("/mcp output: text, embedded resource and structuredContent masked in place, still JSON", async (ctx) => {
    if (!etcdReachable || !app || !mcpUpstream) return ctx.skip();

    const reply = await callTool("eda__report", { text: "go" });
    expect(reply.status).toBe(200);
    expect(reply.json?.error).toBeUndefined();
    expect(reply.json?.result?.isError).toBeFalsy();

    const result = reply.json?.result;
    expect(result?.content?.[0]?.text).toBe(MCP_SUMMARY_MASKED);
    expect(result?.content?.[1]?.resource?.text).toBe(
      `Compile OK version: ***\n${MCP_NEG}`,
    );
    expect(result?.structuredContent).toEqual({
      log: 'config {"version": "***"} ok',
      cells: 42,
    });
    expect(reply.body).not.toContain("12.1");
    expect(reply.body).not.toContain("2022.4");
  });
});
