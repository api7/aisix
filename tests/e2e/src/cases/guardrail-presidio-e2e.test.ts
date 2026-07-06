import { createServer, type Server } from "node:http";
import { createHash } from "node:crypto";
import OpenAI, { APIError } from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  pickFreePort,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: the `presidio` guardrail (#52) — self-hosted Microsoft Presidio
// PII detection + anonymization, two-step analyze→anonymize. We stand up
// ONE mock server implementing both `/analyze` (detects EMAIL occurrences
// as EMAIL_ADDRESS with offsets, SSN-shaped digits as US_SSN) and
// `/anonymize` (applies the requested operator: replace → <ENTITY_TYPE>,
// hash → fixed hex marker). Covered: per-entity block; input redaction
// (upstream sees anonymized text); output redaction non-streaming AND
// streaming (span split across chunks — only the buffer_full hold-back
// reassembly catches it); the hash operator; analyzer 5xx fail-open.
//
// References:
// - Presidio API <https://microsoft.github.io/presidio/api-docs/api-docs.html>
// - LiteLLM `presidio` (behavior baseline): per-entity MASK/BLOCK,
//   language, skip-empty-text. Operator selection is our superset.

const CALLER = "sk-presidio-e2e-caller";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

const EMAIL = "alice@example.com";
const SSN = "123-45-6789";
const ERROR_MARKER = "presidiofivehundredmarker";
const HASHED = "5d41402abc4b2a76b9719d911017c592"; // fixed mock hash output

interface PresidioMock {
  baseUrl: string;
  analyzeRequests: Array<{
    text: string;
    language: string;
    entities: string[] | undefined;
    scoreThreshold: number | undefined;
  }>;
  anonymizeRequests: Array<{
    text: string;
    operatorType: string | undefined;
  }>;
  close(): Promise<void>;
}

// One server, both roles: Presidio's analyzer and anonymizer are separate
// containers in production, but the guardrail addresses them by base URL +
// fixed path, so a single mock serving /analyze and /anonymize stands in
// for both.
async function startPresidioMock(): Promise<PresidioMock> {
  const analyzeRequests: PresidioMock["analyzeRequests"] = [];
  const anonymizeRequests: PresidioMock["anonymizeRequests"] = [];
  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      let body: Record<string, unknown> = {};
      try {
        body = JSON.parse(raw);
      } catch {
        // leave defaults
      }
      const text = typeof body.text === "string" ? (body.text as string) : "";

      if (req.url?.startsWith("/analyze")) {
        analyzeRequests.push({
          text,
          language: (body.language as string) ?? "",
          entities: body.entities as string[] | undefined,
          scoreThreshold: body.score_threshold as number | undefined,
        });
        if (text.includes(ERROR_MARKER)) {
          res.statusCode = 503;
          res.end("mock presidio outage");
          return;
        }
        const results: Array<Record<string, unknown>> = [];
        const emailAt = text.indexOf(EMAIL);
        if (emailAt >= 0) {
          results.push({
            entity_type: "EMAIL_ADDRESS",
            start: emailAt,
            end: emailAt + EMAIL.length,
            score: 0.85,
          });
        }
        const ssnAt = text.indexOf(SSN);
        if (ssnAt >= 0) {
          results.push({
            entity_type: "US_SSN",
            start: ssnAt,
            end: ssnAt + SSN.length,
            score: 0.9,
          });
        }
        res.statusCode = 200;
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify(results));
        return;
      }

      if (req.url?.startsWith("/anonymize")) {
        const anonymizers = (body.anonymizers ?? {}) as Record<
          string,
          { type?: string }
        >;
        const operatorType = anonymizers.DEFAULT?.type;
        anonymizeRequests.push({ text, operatorType });
        const results = (body.analyzer_results ?? []) as Array<{
          entity_type: string;
          start: number;
          end: number;
        }>;
        // Apply spans end→start so earlier offsets stay valid.
        let out = text;
        const items: Array<Record<string, unknown>> = [];
        for (const r of [...results].sort((a, b) => b.start - a.start)) {
          const replacement =
            operatorType === "hash" ? HASHED : `<${r.entity_type}>`;
          out = out.slice(0, r.start) + replacement + out.slice(r.end);
          items.push({
            operator: operatorType ?? "replace",
            entity_type: r.entity_type,
            start: r.start,
            end: r.start + replacement.length,
            text: replacement,
          });
        }
        res.statusCode = 200;
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify({ text: out, items }));
        return;
      }

      res.statusCode = 404;
      res.end("unknown path");
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, "127.0.0.1", resolve);
  });
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    analyzeRequests,
    anonymizeRequests,
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

describe("presidio guardrail e2e: block, input/output redaction, operators", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let streamUpstream: OpenAiUpstream | undefined;
  let presidio: PresidioMock | undefined;
  let admin: AdminClient | undefined;
  let guardrailId: string | undefined;
  let etcdReachable = false;

  const guardrailBody = (operator: string) => ({
    name: "presidio-e2e-guard",
    enabled: true,
    hook_point: "both",
    fail_open: true,
    kind: "presidio",
    analyzer_url: presidio!.baseUrl,
    anonymizer_url: presidio!.baseUrl,
    entities: [
      { type: "EMAIL_ADDRESS", action: "mask" },
      { type: "US_SSN", action: "block" },
    ],
    default_action: "mask",
    operator,
    language: "en",
    score_threshold: 0.5,
  });

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    presidio = await startPresidioMock();

    // Non-streaming upstream: echoes a reply CONTAINING an email, so the
    // output redaction has something to rewrite.
    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-presidio",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: `you can reach the customer at ${EMAIL} today`,
            },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 8, total_tokens: 13 },
      },
    });

    // Streaming upstream: the SAME email split across two delta chunks —
    // only the buffer_full hold-back channel reassembly catches the span.
    streamUpstream = await startOpenAiUpstream({
      streamEvents: [
        '{"id":"strm-presidio","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
        '{"id":"strm-presidio","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"mail alice@exam"},"finish_reason":null}]}',
        '{"id":"strm-presidio","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"ple.com now"},"finish_reason":null}]}',
        '{"id":"strm-presidio","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
        "[DONE]",
      ],
      eventDelayMs: 20,
    });

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "presidio-e2e-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "presidio-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    const streamPk = await admin.createProviderKey({
      display_name: "presidio-stream-e2e-pk",
      secret: "sk-mock",
      api_base: `${streamUpstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "presidio-stream-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: streamPk.id,
    });
    await admin.createApiKey({
      key_hash: hash(CALLER),
      allowed_models: ["presidio-e2e", "presidio-stream-e2e"],
    });

    const created = await admin.json<{ id: string }>(
      "POST",
      "/admin/v1/guardrails",
      guardrailBody("replace"),
    );
    guardrailId = created.id;
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await streamUpstream?.close();
    await presidio?.close();
  });

  const client = () =>
    new OpenAI({
      apiKey: CALLER,
      baseURL: `${app!.proxyUrl}/v1`,
      maxRetries: 0,
    });

  // Poll until the replace-operator guardrail is live (the email-bearing
  // mock reply comes back anonymized), so every test stands alone (no
  // hidden dependency on execution order). Once propagated, the first
  // poll returns immediately. The hash-operator test establishes its own
  // baseline with a full PUT + its own propagation probe.
  const ensureGuardrailLive = () =>
    waitConfigPropagation(async () => {
      const r = await client().chat.completions.create({
        model: "presidio-e2e",
        messages: [{ role: "user", content: "probe" }],
      });
      return (r.choices[0]?.message?.content ?? "").includes("<EMAIL_ADDRESS>");
    });

  test("redact: request PII anonymized before the upstream, response PII before the caller", async (ctx) => {
    if (!etcdReachable || !app || !upstream || !presidio) {
      ctx.skip();
      return;
    }

    await ensureGuardrailLive();

    const res = await client().chat.completions.create({
      model: "presidio-e2e",
      messages: [
        { role: "user", content: `contact me at ${EMAIL} about the order` },
      ],
    });

    // Response side: the model's reply had the email; the caller sees the
    // anonymized token and never the value.
    const reply = res.choices[0]?.message?.content ?? "";
    expect(reply).toContain("<EMAIL_ADDRESS>");
    expect(reply).not.toContain(EMAIL);

    // Request side: the upstream received the anonymized prompt — the
    // value never left the gateway.
    const lastReq = upstream.receivedRequests.at(-1);
    expect(lastReq).toBeDefined();
    expect(lastReq!.body).toContain("<EMAIL_ADDRESS>");
    expect(lastReq!.body).toContain("about the order");
    expect(lastReq!.body).not.toContain(EMAIL);

    // The analyzer was consulted with the configured language, entities,
    // and confidence floor.
    const analyzed = presidio.analyzeRequests.at(-1);
    expect(analyzed?.language).toBe("en");
    expect(analyzed?.entities).toEqual(["EMAIL_ADDRESS", "US_SSN"]);
    expect(analyzed?.scoreThreshold).toBe(0.5);
  });

  test("block: a block-action entity rejects with 422 content_filter, value not echoed", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    await ensureGuardrailLive();
    const upstreamBefore = upstream.receivedRequests.length;
    let caught: unknown;
    try {
      await client().chat.completions.create({
        model: "presidio-e2e",
        messages: [{ role: "user", content: `my ssn is ${SSN} ok` }],
      });
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(APIError);
    if (!(caught instanceof APIError)) throw new Error("unreachable");
    expect(caught.status).toBe(422);
    expect((caught.error as { type?: unknown })?.type).toBe("content_filter");
    // The matched value must not leak back to the caller (#153).
    expect(JSON.stringify(caught.error ?? {})).not.toContain(SSN);
    expect(upstream.receivedRequests.length).toBe(upstreamBefore);
  });

  test("streaming output: split-across-chunks PII is anonymized via hold-back", async (ctx) => {
    if (!etcdReachable || !app || !streamUpstream) {
      ctx.skip();
      return;
    }
    await ensureGuardrailLive();
    const stream = await client().chat.completions.create({
      model: "presidio-stream-e2e",
      messages: [{ role: "user", content: "stream me the contact" }],
      stream: true,
    });
    let assembled = "";
    for await (const chunk of stream) {
      assembled += chunk.choices[0]?.delta?.content ?? "";
    }
    expect(assembled).toContain("<EMAIL_ADDRESS>");
    expect(assembled).not.toContain(EMAIL);
  });

  test("analyzer 5xx with fail_open=true → request passes (bypass)", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    await ensureGuardrailLive();
    const before = upstream.receivedRequests.length;
    const res = await client().chat.completions.create({
      model: "presidio-e2e",
      messages: [{ role: "user", content: `hello ${ERROR_MARKER}` }],
    });
    expect(res.choices[0]?.message.role).toBe("assistant");
    expect(upstream.receivedRequests.length).toBe(before + 1);
  });

  test("hash operator: anonymizer is driven with hash config and its output is honored", async (ctx) => {
    if (!etcdReachable || !app || !upstream || !admin || !guardrailId || !presidio) {
      ctx.skip();
      return;
    }
    await admin.json(
      "PUT",
      `/admin/v1/guardrails/${guardrailId}`,
      guardrailBody("hash"),
    );

    // Propagation probe: the anonymized reply flips from <EMAIL_ADDRESS>
    // to the mock's fixed hash output.
    await waitConfigPropagation(async () => {
      const r = await client().chat.completions.create({
        model: "presidio-e2e",
        messages: [{ role: "user", content: "probe" }],
      });
      return (r.choices[0]?.message?.content ?? "").includes(HASHED);
    });

    const res = await client().chat.completions.create({
      model: "presidio-e2e",
      messages: [{ role: "user", content: `mail ${EMAIL} now` }],
    });
    expect(res.choices[0]?.message?.content ?? "").toContain(HASHED);

    // The anonymizer saw the hash operator (sha256 via the DEFAULT slot).
    const anonymized = presidio.anonymizeRequests.at(-1);
    expect(anonymized?.operatorType).toBe("hash");

    // And the upstream-bound request was hashed too, value never leaked.
    const lastReq = upstream.receivedRequests.at(-1);
    expect(lastReq!.body).toContain(HASHED);
    expect(lastReq!.body).not.toContain(EMAIL);
  });
});
