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

// E2E: capture-group custom patterns + configurable `replacement`
// (AISIX-Cloud#1334) — the "replace the value, keep the key" semantics.
//
// Rules under test (seeded as kind=pii custom_patterns):
// - plain form  `(?:version|版本)\s*[:：]\s*(\d+(?:\.\d+)+)` → group 1
//   becomes `***`, the keyword and separator stay verbatim; anchoring on
//   the keyword is what keeps dot-separated negatives (durations, sizes,
//   IPs, file:line) untouched.
// - JSON form   `"version"\s*:\s*"([^"]*)"` → only the value inside the
//   quotes is rewritten, so the document still parses.
//
// Acceptance (issue #1334): both forms rewrite on request AND response,
// Chinese and English labels covered, JSON stays parseable (asserted by
// parsing, not string comparison), hard negatives hit zero, non-matched
// content survives byte-for-byte (exact-equality assertions), and the
// whole flow stays 200 — mask never blocks.

const CALLER = "sk-pii-capture-e2e";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

// Hard negatives from the issue: dot-separated numbers that are NOT
// version values (duration, size, process node, file:line, IPv4).
const NEGATIVES =
  "Elapsed: 12.345s Memory: 4.2 GB node 0.13um top.v:12:1 ip 10.2.255.1";

// Mixed prompt: an English hit, the negatives, and a Chinese hit.
const PROMPT = `EDA version: 12.1 ${NEGATIVES} 工具版本：2022.4 结束`;
const PROMPT_MASKED = `EDA version: *** ${NEGATIVES} 工具版本：*** 结束`;

// The model reply embeds the JSON key-value form.
const REPLY_JSON = `{"tool":"vcs","version": "12.1","cells":42}`;
const REPLY = `config ${REPLY_JSON} ok ${NEGATIVES}`;

describe("pii capture-group + replacement e2e", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-capture",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: REPLY },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 8, total_tokens: 13 },
      },
    });

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "pii-capture-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "pii-capture-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createGuardrail({
      name: "pii-capture-guard",
      enabled: true,
      hook_point: "both",
      kind: "pii",
      custom_patterns: [
        {
          name: "eda_version",
          regex: "(?:version|版本)\\s*[:：]\\s*(\\d+(?:\\.\\d+)+)",
          action: "mask",
          replacement: "***",
        },
        {
          name: "eda_version_json",
          regex: '"version"\\s*:\\s*"([^"]*)"',
          action: "mask",
          replacement: "***",
        },
      ],
    });

    // Caller key LAST: once it authenticates, every resource seeded
    // before it (guardrail included) is in the snapshot (AGENTS.md gate
    // rule — the gate must not exercise the behavior under test).
    await seed.createApiKey({
      key_hash: hash(CALLER),
      allowed_models: ["pii-capture-e2e"],
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

  const client = () =>
    new OpenAI({
      apiKey: CALLER,
      baseURL: `${app!.proxyUrl}/v1`,
      maxRetries: 0,
    });

  const chat = (content: string) =>
    client().chat.completions.create({
      model: "pii-capture-e2e",
      messages: [{ role: "user", content }],
    });

  test("request: value replaced in place, key + negatives byte-identical, zh+en", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const res = await chat(PROMPT);
    // Mask never blocks: the call succeeded (a 422 would have thrown).
    expect(res.choices[0]?.message?.content ?? "").not.toBe("");

    // The upstream saw the ENTIRE prompt with only the two anchored
    // values rewritten — exact equality is the byte-for-byte assertion
    // for everything outside the hits (negatives included).
    const sent = JSON.parse(upstream.receivedRequests.at(-1)!.body) as {
      messages: Array<{ content: string }>;
    };
    expect(sent.messages[0].content).toBe(PROMPT_MASKED);
  });

  test("request: a no-hit prompt passes through byte-identical", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    await chat(NEGATIVES);
    const sent = JSON.parse(upstream.receivedRequests.at(-1)!.body) as {
      messages: Array<{ content: string }>;
    };
    expect(sent.messages[0].content).toBe(NEGATIVES);
  });

  test("response: JSON form masked in place and still parseable", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const res = await chat("show me the config");
    const reply = res.choices[0]?.message?.content ?? "";
    // Only the value inside the quotes changed; prefix/suffix and the
    // negatives after the JSON survive byte-for-byte.
    expect(reply).toBe(
      `config {"tool":"vcs","version": "***","cells":42} ok ${NEGATIVES}`,
    );
    // Structural acceptance: parse the embedded JSON, don't just diff it.
    const embedded = reply.slice(reply.indexOf("{"), reply.lastIndexOf("}") + 1);
    const parsed = JSON.parse(embedded) as {
      tool: string;
      version: string;
      cells: number;
    };
    expect(parsed.version).toBe("***");
    expect(parsed.tool).toBe("vcs");
    expect(parsed.cells).toBe(42);
    expect(reply).not.toContain("12.1");
  });
});
