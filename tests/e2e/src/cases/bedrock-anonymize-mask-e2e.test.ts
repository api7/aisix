import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  pickFreePort,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: Bedrock guardrail ANONYMIZE write-back (#932 bedrock follow-up).
//
// Bedrock reports `action=GUARDRAIL_INTERVENED` for BOTH a hard block and
// a PII anonymization; the per-entity `assessments[]` actions tell them
// apart. Pre-fix the DP blocked on ANY intervention. Now, on the
// chat-shaped families, an ANONYMIZE disposition masks-and-continues:
//
// - INPUT: the request's text slots go up as one content block each in a
//   single ApplyGuardrail call, and Bedrock's `outputs[i]` replaces slot i
//   before the request reaches the upstream (verified via the mock
//   upstream's received body).
// - OUTPUT (non-streaming + streaming hold-back): the model's reply is
//   masked before it reaches the caller.
// - Hard block (BLOCKED entity) still blocks with the standard 422 /
//   error-frame envelope.
// - Defensive fallback (LiteLLM `_merge_masked_texts` semantics): masked
//   outputs that don't align 1:1 with the input blocks are NOT applied —
//   originals pass through, the request continues.

const CALLER = "sk-bedrock-mask-caller";
const STREAM_CALLER = "sk-bedrock-mask-stream-caller";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

const EMAIL = "alice@example.com";
const EMAIL_RE = /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g;

interface BedrockCall {
  source: string;
  texts: string[];
}

interface MockBedrock {
  url: string;
  calls: BedrockCall[];
  close(): Promise<void>;
}

// Content-driven ApplyGuardrail mock:
// - any block containing "BLOCKME"   → INTERVENED, BLOCKED pii entity
// - any block containing "MISALIGN"  → INTERVENED, ANONYMIZED, but ONE
//   merged output for N input blocks (deliberately misaligned)
// - any block containing an email    → INTERVENED, ANONYMIZED, one output
//   per input block with emails replaced by {EMAIL} (the real contract)
// - otherwise                        → NONE
async function startMockBedrock(): Promise<MockBedrock> {
  const calls: BedrockCall[] = [];
  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      let source = "";
      let texts: string[] = [];
      try {
        const body = JSON.parse(raw) as {
          source?: string;
          content?: Array<{ text?: { text?: string } }>;
        };
        source = body.source ?? "";
        texts = (body.content ?? []).map((c) => c.text?.text ?? "");
      } catch {
        // fall through with empty texts — answered as NONE below
      }
      calls.push({ source, texts });

      const anonymizedAssessment = {
        sensitiveInformationPolicy: {
          piiEntities: [{ match: EMAIL, type: "EMAIL", action: "ANONYMIZED" }],
          regexes: [],
        },
      };
      let payload: unknown;
      if (texts.some((t) => t.includes("BLOCKME"))) {
        payload = {
          action: "GUARDRAIL_INTERVENED",
          outputs: [],
          assessments: [
            {
              sensitiveInformationPolicy: {
                piiEntities: [
                  { match: "BLOCKME", type: "NAME", action: "BLOCKED" },
                ],
                regexes: [],
              },
            },
          ],
        };
      } else if (texts.some((t) => t.includes("MISALIGN"))) {
        payload = {
          action: "GUARDRAIL_INTERVENED",
          outputs: [{ text: texts.join(" ").replace(EMAIL_RE, "{EMAIL}") }],
          assessments: [anonymizedAssessment],
        };
      } else if (texts.some((t) => t.includes("@"))) {
        payload = {
          action: "GUARDRAIL_INTERVENED",
          outputs: texts.map((t) => ({ text: t.replace(EMAIL_RE, "{EMAIL}") })),
          assessments: [anonymizedAssessment],
        };
      } else {
        payload = { action: "NONE", outputs: [] };
      }
      res.statusCode = 200;
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify(payload));
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) =>
    server.listen(port, "127.0.0.1", resolve),
  );
  return {
    url: `http://127.0.0.1:${port}`,
    calls,
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

describe("bedrock guardrail ANONYMIZE write-back (#932 follow-up)", () => {
  let etcdReachable = false;
  let upstream: OpenAiUpstream | undefined;
  let streamUpstream: OpenAiUpstream | undefined;
  let bedrock: MockBedrock | undefined;
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // Non-streaming upstream: the canned reply CONTAINS an email so the
    // output mask has something to rewrite.
    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-bmask",
        object: "chat.completion",
        created: 1,
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

    // Streaming upstream: the email split across two delta chunks — only
    // the hold-back channel reassembly (one OUTPUT scan over the joined
    // text) can catch the span.
    streamUpstream = await startOpenAiUpstream({
      streamEvents: [
        '{"id":"strm-bmask","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
        '{"id":"strm-bmask","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"mail alice@exam"},"finish_reason":null}]}',
        '{"id":"strm-bmask","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"ple.com now"},"finish_reason":null}]}',
        '{"id":"strm-bmask","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
        "[DONE]",
      ],
      eventDelayMs: 10,
    });

    bedrock = await startMockBedrock();
    app = await spawnApp({ extra: { bedrock_endpoint_url: bedrock.url } });
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "bmask-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "bmask-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: hash(CALLER),
      allowed_models: ["bmask-e2e"],
    });

    const streamPk = await seed.createProviderKey({
      display_name: "bmask-stream-pk",
      secret: "sk-mock",
      api_base: `${streamUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "bmask-stream-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: streamPk.id,
    });
    await seed.createApiKey({
      key_hash: hash(STREAM_CALLER),
      allowed_models: ["bmask-stream-e2e"],
    });

    await seed.createGuardrail({
      name: "gr-bedrock-mask",
      enabled: true,
      hook_point: "both",
      fail_open: false,
      kind: "bedrock",
      guardrail_id: "bmaskgr00001",
      guardrail_version: "DRAFT",
      region: "us-east-1",
      aws_credentials: {
        kind: "static",
        access_key_id: "AKIDBEDROCKMASK00001",
        secret_access_key: "secret-bedrock-mask",
      },
      latency_mode: { kind: "serial" },
    });

    // Ready once a clean chat passes AND the guardrail fired on INPUT.
    await waitConfigPropagation(async () => {
      try {
        const r = await chat(CALLER, "bmask-e2e", [
          { role: "user", content: "warmup all clean" },
        ]);
        await r.text();
        return (
          r.status === 200 &&
          bedrock!.calls.some((c) => c.source === "INPUT")
        );
      } catch {
        return false;
      }
    });
  });

  afterAll(async () => {
    await app?.exit();
    await bedrock?.close();
    await upstream?.close();
    await streamUpstream?.close();
  });

  function chat(
    caller: string,
    model: string,
    messages: Array<{ role: string; content: string }>,
    stream = false,
  ) {
    return fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${caller}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({ model, messages, ...(stream ? { stream: true } : {}) }),
    });
  }

  test(
    "ANONYMIZE masks the request per slot and the reply before the caller",
    async (ctx) => {
      if (!etcdReachable) {
        ctx.skip();
        return;
      }
      const upstreamBefore = upstream!.receivedRequests.length;
      const r = await chat(CALLER, "bmask-e2e", [
        { role: "system", content: "be helpful" },
        { role: "user", content: `contact ${EMAIL} please` },
      ]);
      expect(r.status).toBe(200);
      const body = (await r.json()) as {
        choices: Array<{ message: { content: string } }>;
      };

      // INPUT went up as one content block per message (Plan D), not a
      // joined blob.
      const inputCall = bedrock!.calls
        .filter((c) => c.source === "INPUT")
        .at(-1)!;
      expect(inputCall.texts).toEqual([
        "be helpful",
        `contact ${EMAIL} please`,
      ]);

      // The upstream saw the MASKED prompt — the raw email never left.
      const sent = upstream!.receivedRequests.slice(upstreamBefore).at(-1)!;
      expect(sent.body).toContain("contact {EMAIL} please");
      expect(sent.body).not.toContain(EMAIL);

      // The reply's email is masked before the caller sees it.
      expect(body.choices[0].message.content).toBe(
        "you can reach the customer at {EMAIL} today",
      );
      expect(
        bedrock!.calls.some((c) => c.source === "OUTPUT"),
        "output hook must have scanned the reply",
      ).toBe(true);
    },
    60_000,
  );

  test(
    "a BLOCKED entity still blocks before the upstream",
    async (ctx) => {
      if (!etcdReachable) {
        ctx.skip();
        return;
      }
      const upstreamBefore = upstream!.receivedRequests.length;
      const r = await chat(CALLER, "bmask-e2e", [
        { role: "user", content: "please BLOCKME now" },
      ]);
      expect(r.status).toBe(422);
      const body = await r.text();
      expect(body).toContain("content policy");
      expect(body).not.toContain("BLOCKME");
      expect(upstream!.receivedRequests.length).toBe(upstreamBefore);
    },
    60_000,
  );

  test(
    "misaligned masked outputs are not applied — originals pass through",
    async (ctx) => {
      if (!etcdReachable) {
        ctx.skip();
        return;
      }
      const upstreamBefore = upstream!.receivedRequests.length;
      // Two slots, but the mock answers with ONE merged output — the
      // fallback must keep the originals and continue.
      const r = await chat(CALLER, "bmask-e2e", [
        { role: "user", content: `MISALIGN write to bob@example.org` },
        { role: "user", content: "second slot" },
      ]);
      expect(r.status).toBe(200);
      await r.text();
      const sent = upstream!.receivedRequests.slice(upstreamBefore).at(-1)!;
      expect(sent.body).toContain("bob@example.org");
      expect(sent.body).not.toContain("{EMAIL}");
    },
    60_000,
  );

  test(
    "streaming: the reply is masked across chunk boundaries via hold-back",
    async (ctx) => {
      if (!etcdReachable) {
        ctx.skip();
        return;
      }
      const r = await chat(
        STREAM_CALLER,
        "bmask-stream-e2e",
        [{ role: "user", content: "stream something" }],
        true,
      );
      expect(r.status).toBe(200);
      const raw = await r.text();
      const content = raw
        .split("\n")
        .filter((l) => l.startsWith("data: ") && !l.includes("[DONE]"))
        .map((l) => {
          try {
            const j = JSON.parse(l.slice("data: ".length)) as {
              choices?: Array<{ delta?: { content?: string } }>;
            };
            return j.choices?.[0]?.delta?.content ?? "";
          } catch {
            return "";
          }
        })
        .join("");
      expect(content).toBe("mail {EMAIL} now");
      expect(content).not.toContain("alice@");
    },
    60_000,
  );

  test(
    "/v1/messages: the Anthropic-native body is masked before the bridge",
    async (ctx) => {
      if (!etcdReachable) {
        ctx.skip();
        return;
      }
      const upstreamBefore = upstream!.receivedRequests.length;
      const r = await fetch(`${app!.proxyUrl}/v1/messages`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: "bmask-e2e",
          max_tokens: 64,
          messages: [
            { role: "user", content: `write to carol@example.dev today` },
          ],
        }),
      });
      expect(r.status).toBe(200);
      await r.text();
      const sent = upstream!.receivedRequests.slice(upstreamBefore).at(-1)!;
      expect(sent.body).toContain("write to {EMAIL} today");
      expect(sent.body).not.toContain("carol@example.dev");
    },
    60_000,
  );
});
