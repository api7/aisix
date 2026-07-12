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

// E2E: `kind: "pii"` guardrail (#932 / AISIX-Cloud#932) — in-process
// sensitive-data detection with per-detector `mask` / `block` actions.
//
// - mask on the REQUEST: the caller's prompt PII is rewritten to
//   [<DETECTOR>_REDACTED] before it reaches the upstream (verified via the
//   mock upstream's received body).
// - mask on the RESPONSE (non-streaming + streaming): the model's reply is
//   rewritten before it reaches the caller; the streaming case splits the
//   value across chunk boundaries to pin the channel-reassembly path.
// - block: a block-action detector rejects with the standard 422
//   content_filter envelope, and the matched value never appears in it.
//
// Detector values below are synthetic: the china_id_card sample is the
// canonical ISO 7064 documentation example, the bank card is the classic
// Luhn test number.

const CALLER = "sk-pii-e2e-caller";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

const EMAIL = "alice@example.com";
const CN_ID = "11010519491231002X"; // valid ISO 7064 MOD 11-2 check digit
const CARD = "4111111111111111"; // passes Luhn

describe("pii guardrail e2e: mask + block on request and response", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let streamUpstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcd: EtcdClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // Non-streaming upstream: echoes a reply CONTAINING an email, so the
    // output mask has something to rewrite.
    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-pii",
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
    // per-chunk masking would miss it; only the hold-back channel
    // reassembly catches the span.
    streamUpstream = await startOpenAiUpstream({
      streamEvents: [
        '{"id":"strm-pii","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
        '{"id":"strm-pii","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"mail alice@exam"},"finish_reason":null}]}',
        '{"id":"strm-pii","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"ple.com now"},"finish_reason":null}]}',
        '{"id":"strm-pii","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
        "[DONE]",
      ],
      eventDelayMs: 20,
    });

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "pii-e2e-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "pii-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: hash(CALLER),
      allowed_models: ["pii-e2e"],
    });

    const streamPk = await seed.createProviderKey({
      display_name: "pii-stream-e2e-pk",
      secret: "sk-mock",
      api_base: `${streamUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "pii-stream-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: streamPk.id,
    });
    await seed.createApiKey({
      key_hash: hash(`${CALLER}-stream`),
      allowed_models: ["pii-stream-e2e"],
    });

    // One env-wide pii guardrail: email masks (redact-and-continue),
    // china_id_card blocks (reject).
    await seed.createGuardrail({
      name: "pii-e2e-guard",
      enabled: true,
      hook_point: "both",
      kind: "pii",
      detectors: [
        { type: "email", action: "mask" },
        { type: "china_id_card", action: "block" },
      ],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await streamUpstream?.close();
  });

  const client = () =>
    new OpenAI({
      apiKey: CALLER,
      baseURL: `${app!.proxyUrl}/v1`,
      maxRetries: 0,
    });

  test("mask: request PII is rewritten before the upstream, response PII before the caller", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    // Propagation probe: once the guardrail is live, the (email-bearing)
    // mock reply comes back masked.
    await waitConfigPropagation(async () => {
      try {
        const r = await client().chat.completions.create({
          model: "pii-e2e",
          messages: [{ role: "user", content: "probe" }],
        });
        return (r.choices[0]?.message?.content ?? "").includes("[EMAIL_REDACTED]");
      } catch {
        return false; // caller key / model still propagating
      }
    });

    const res = await client().chat.completions.create({
      model: "pii-e2e",
      messages: [
        { role: "user", content: `contact me at ${EMAIL} about the order` },
      ],
    });

    // Response side: the model's reply had the email; the caller sees the
    // mask token and never the value.
    const reply = res.choices[0]?.message?.content ?? "";
    expect(reply).toContain("[EMAIL_REDACTED]");
    expect(reply).not.toContain(EMAIL);

    // Request side: the upstream received the MASKED prompt — the value
    // never left the gateway. (Structure preserved: only the span is
    // replaced.)
    const lastReq = upstream.receivedRequests.at(-1);
    expect(lastReq).toBeDefined();
    const upstreamBody = lastReq!.body;
    expect(upstreamBody).toContain("[EMAIL_REDACTED]");
    expect(upstreamBody).toContain("about the order");
    expect(upstreamBody).not.toContain(EMAIL);
  });

  test("block: a block-action detector rejects with 422 content_filter, value not echoed", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const upstreamHitsBefore = upstream.receivedRequests.length;
    let caught: unknown;
    try {
      await client().chat.completions.create({
        model: "pii-e2e",
        messages: [{ role: "user", content: `my id number is ${CN_ID}` }],
      });
    } catch (e) {
      caught = e;
    }

    expect(caught).toBeInstanceOf(APIError);
    if (!(caught instanceof APIError)) throw new Error("unreachable");
    expect(caught.status).toBe(422);
    expect((caught.error as { type?: unknown })?.type).toBe("content_filter");
    // #153 / #932 no-leak: the matched value never appears in the envelope;
    // the guardrail name does (#519 B.4b).
    const blob = JSON.stringify(caught.error ?? {}) + (caught.message ?? "");
    expect(blob).not.toContain(CN_ID);
    expect(blob).toContain("guardrail 'pii-e2e-guard'");
    // Input block fires pre-dispatch: the upstream is never hit.
    expect(upstream.receivedRequests.length).toBe(upstreamHitsBefore);
  });

  test("mask does NOT block: a mask-only match still gets a 200", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    // A bank card would only matter to a block detector — none is
    // configured for it, and email is mask-action, so the request goes
    // through (masked) rather than 422ing.
    const res = await client().chat.completions.create({
      model: "pii-e2e",
      messages: [{ role: "user", content: `card ${CARD} email ${EMAIL}` }],
    });
    expect(res.choices[0]?.message?.content ?? "").toContain("[EMAIL_REDACTED]");
  });

  test("streaming mask: a span split across delta chunks is reassembled and masked (#932)", async (ctx) => {
    if (!etcdReachable || !app || !streamUpstream) {
      ctx.skip();
      return;
    }

    const streamCaller = `${CALLER}-stream`;
    const doStream = () =>
      fetch(`${app!.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${streamCaller}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: "pii-stream-e2e",
          messages: [{ role: "user", content: "innocent prompt" }],
          stream: true,
        }),
      });

    await waitConfigPropagation(async () => {
      const probe = await doStream();
      if (probe.status !== 200) {
        await probe.text();
        return false;
      }
      return (await probe.text()).includes("[EMAIL_REDACTED]");
    });

    const res = await doStream();
    expect(res.status).toBe(200);
    const wire = await res.text();

    // The email was split "alice@exam" + "ple.com" across two chunks —
    // channel reassembly at the hold-back release must still catch it.
    expect(wire).toContain("[EMAIL_REDACTED]");
    expect(wire).not.toContain(EMAIL);
    expect(wire).not.toContain("alice@exam");
    // Clean stream contract: [DONE] present, no error event.
    expect(wire).toContain("data: [DONE]");
    expect(wire).not.toContain("event: error");
    // Non-content fields survive the rewrite (finish_reason intact).
    expect(wire).toContain('"finish_reason":"stop"');
  });

  test("monitor mode: enforcement_mode=monitor observes but does not mask", async (ctx) => {
    if (!etcdReachable) {
      ctx.skip();
      return;
    }

    // Dedicated app instance (its own etcd prefix / env), so the env-wide
    // masking guardrail from the main suite cannot interfere — this env
    // carries ONLY the monitor-mode pii guardrail.
    const monUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-mon",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: `reply with ${EMAIL}` },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 8, total_tokens: 13 },
      },
    });
    const monApp = await spawnApp();
    const monSeed = new SeedClient(new EtcdClient(), monApp.etcdPrefix);
    const monPk = await monSeed.createProviderKey({
      display_name: "pii-mon-pk",
      secret: "sk-mock",
      api_base: `${monUpstream.baseUrl}/v1`,
    });
    await monSeed.createModel({
      display_name: "pii-mon-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: monPk.id,
    });
    const monCaller = `${CALLER}-mon`;
    await monSeed.createApiKey({
      key_hash: hash(monCaller),
      allowed_models: ["pii-mon-e2e"],
    });
    await monSeed.createGuardrail({
      name: "pii-mon-guard",
      enabled: true,
      hook_point: "both",
      enforcement_mode: "monitor",
      kind: "pii",
      detectors: [{ type: "email", action: "mask" }],
    });

    const monClient = new OpenAI({
      apiKey: monCaller,
      baseURL: `${monApp.proxyUrl}/v1`,
      maxRetries: 0,
    });
    await waitConfigPropagation(async () => {
      try {
        const r = await monClient.chat.completions.create({
          model: "pii-mon-e2e",
          messages: [{ role: "user", content: "probe" }],
        });
        return (r.choices[0]?.message?.content ?? "").length > 0;
      } catch {
        return false;
      }
    });

    const res = await monClient.chat.completions.create({
      model: "pii-mon-e2e",
      messages: [{ role: "user", content: `mask me maybe: ${EMAIL}` }],
    });
    // Monitor mode: content flows UNCHANGED in both directions; the
    // would-be mask counts land in ops logs only.
    expect(res.choices[0]?.message?.content ?? "").toContain(EMAIL);
    const lastReq = monUpstream.receivedRequests.at(-1);
    expect(lastReq!.body).toContain(EMAIL);

    await monApp.exit();
    await monUpstream.close();
  });

  test("/v1/messages passthrough: request masked before the Anthropic upstream", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    // Anthropic-native mock upstream: capture the received body, return a
    // minimal non-streaming message whose reply carries an email so the
    // output mask has something to rewrite too.
    const received: unknown[] = [];
    const { createServer } = await import("node:http");
    const anthUpstream = createServer((req, res) => {
      const chunks: Buffer[] = [];
      req.on("data", (c: Buffer) => chunks.push(c));
      req.on("end", () => {
        received.push(JSON.parse(Buffer.concat(chunks).toString()));
        res.writeHead(200, { "content-type": "application/json" });
        res.end(
          JSON.stringify({
            id: "msg_pii",
            type: "message",
            role: "assistant",
            model: "claude-3-5-haiku-20241022",
            content: [{ type: "text", text: `reach them at ${EMAIL} ok` }],
            stop_reason: "end_turn",
            usage: { input_tokens: 4, output_tokens: 6 },
          }),
        );
      });
    });
    await new Promise<void>((resolve) => anthUpstream.listen(0, resolve));
    const anthPort = (anthUpstream.address() as { port: number }).port;

    const anthPk = await seed.createProviderKey({
      display_name: "pii-anth-pk",
      secret: "sk-ant-mock",
      api_base: `http://127.0.0.1:${anthPort}`,
      provider: "anthropic",
      adapter: "anthropic",
    });
    await seed.createModel({
      display_name: "pii-anth-e2e",
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: anthPk.id,
    });
    const anthCaller = `${CALLER}-anth`;
    await seed.createApiKey({
      key_hash: hash(anthCaller),
      allowed_models: ["pii-anth-e2e"],
    });

    const call = () =>
      fetch(`${app!.proxyUrl}/v1/messages`, {
        method: "POST",
        headers: { "content-type": "application/json", "x-api-key": anthCaller },
        body: JSON.stringify({
          model: "pii-anth-e2e",
          max_tokens: 32,
          messages: [
            { role: "user", content: `write to ${EMAIL} please` },
          ],
        }),
      });

    await waitConfigPropagation(async () => {
      const r = await call();
      if (r.status !== 200) {
        await r.text();
        return false;
      }
      const b = (await r.json()) as {
        content?: Array<{ text?: string }>;
      };
      return (b.content?.[0]?.text ?? "").includes("[EMAIL_REDACTED]");
    });

    const res = await call();
    expect(res.status).toBe(200);
    const body = (await res.json()) as { content?: Array<{ text?: string }> };
    // Response side masked for the caller…
    expect(body.content?.[0]?.text ?? "").toContain("[EMAIL_REDACTED]");
    expect(JSON.stringify(body)).not.toContain(EMAIL);
    // …and the request side was masked before the upstream.
    const lastReq = received.at(-1);
    const upstreamBlob = JSON.stringify(lastReq);
    expect(upstreamBlob).toContain("[EMAIL_REDACTED]");
    expect(upstreamBlob).not.toContain(EMAIL);

    await new Promise<void>((resolve, reject) =>
      anthUpstream.close((e) => (e ? reject(e) : resolve())),
    );
  });
});
