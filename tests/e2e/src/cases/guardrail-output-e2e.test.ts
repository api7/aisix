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

// E2E: keyword guardrail blocks the assistant's *output* when it
// contains the forbidden pattern. The existing guardrail-keyword-e2e
// covers `hook_point: "input"`; this case covers the symmetric
// `hook_point: "output"` user journey, which is the more
// interesting safety surface — input filtering only stops users from
// asking for forbidden content; output filtering is what stops the
// model from disclosing it (e.g. trained-in PII, jailbreak responses,
// off-policy text the model produced for an innocent-looking prompt).
//
// Reference:
// - OpenAI Chat Completions API spec
//   <https://platform.openai.com/docs/api-reference/chat/create>
// - OpenAI / Azure content-filter convention for the
//   `error.type: "content_filter"` envelope value
//   <https://learn.microsoft.com/azure/ai-services/openai/concepts/content-filter>

const CALLER_PLAINTEXT = "sk-gr-out-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const FORBIDDEN_WORD = "leakedsecret";

describe("output guardrail e2e: model-emitted forbidden text is blocked before reaching caller", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let streamUpstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    // Mock upstream emits an OpenAI-shape completion that CONTAINS
    // the forbidden word. The caller's prompt is innocent; the
    // forbidden content originates from the model's response — that's
    // exactly the case input-only guardrails miss.
    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-leak",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: `Sure, here it is: ${FORBIDDEN_WORD}.`,
            },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 8, total_tokens: 13 },
      },
    });

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "gr-out-e2e-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "gr-out-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["gr-out-e2e"],
    });
    // Output guardrail: runs against the assistant's response after
    // the upstream call returns, before relay to the caller.
    await admin.json("POST", "/admin/v1/guardrails", {
      name: "gr-out-e2e-keyword",
      enabled: true,
      hook_point: "output",
      kind: "keyword",
      patterns: [{ kind: "literal", value: FORBIDDEN_WORD }],
    });

    // Per #204: a parallel streaming-shaped upstream + Model so the
    // streaming-block test below shares the guardrail policy (env-wide
    // by default) with the non-streaming case. The two Models' provider
    // keys point at different upstreams so the streaming wire shape
    // and the non-streaming wire shape don't interfere.
    streamUpstream = await startOpenAiUpstream({
      streamEvents: [
        '{"id":"strm-leak","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
        // Chunk 2 carries the forbidden literal embedded in a longer
        // assistant message; the buffer-then-check at end-of-stream
        // accumulates the full text and the keyword guardrail matches.
        `{"id":"strm-leak","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"sure here it is: ${FORBIDDEN_WORD}"},"finish_reason":null}]}`,
        '{"id":"strm-leak","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
        "[DONE]",
      ],
      eventDelayMs: 50,
    });
    const streamPk = await admin.createProviderKey({
      display_name: "gr-out-stream-e2e-pk",
      secret: "sk-mock",
      api_base: `${streamUpstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "gr-out-stream-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: streamPk.id,
    });
    // Add the streaming Model alias to the existing caller's allow-list.
    await admin.createApiKey({
      key_hash: createHash("sha256")
        .update(`${CALLER_PLAINTEXT}-stream`)
        .digest("hex"),
      allowed_models: ["gr-out-stream-e2e"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await streamUpstream?.close();
  });

  test("upstream emits forbidden word → caller sees content_filter 422, NOT the forbidden text", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // Output guardrails fire AFTER upstream dispatch, so propagation
    // readiness is signaled by the same 422-on-blocked-response
    // pattern. A 200 means the guardrail isn't loaded yet
    // (gateway forwarded the leaked content); keep polling.
    await waitConfigPropagation(async () => {
      try {
        await client.chat.completions.create({
          model: "gr-out-e2e",
          messages: [{ role: "user", content: "innocent question" }],
        });
        return false; // 200 means guardrail not ready
      } catch (e) {
        return e instanceof APIError && e.status === 422;
      }
    });

    const upstreamHitsBefore = upstream.receivedRequests.length;

    let caught: unknown;
    try {
      await client.chat.completions.create({
        model: "gr-out-e2e",
        messages: [{ role: "user", content: "tell me something useful" }],
      });
    } catch (e) {
      caught = e;
    }

    expect(caught).toBeInstanceOf(APIError);
    if (!(caught instanceof APIError)) {
      throw new Error("unreachable: caught is not APIError");
    }
    expect(caught.status).toBe(422);
    // Pin envelope to the content_filter type so a regression that
    // 422'd via a different path (e.g. generic schema validation)
    // would fail; the value `content_filter` is the OpenAI / Azure
    // public taxonomy for this exact case.
    expect((caught.error as { type?: unknown })?.type).toBe("content_filter");

    // The leaked word MUST NOT appear anywhere in the caller-visible
    // error envelope. The whole point of an output guardrail is to
    // prevent the forbidden content from reaching the caller — even
    // echoing it back inside an error message would defeat the
    // purpose (and would be a real, reportable security regression).
    const errorBlob = JSON.stringify(caught.error ?? {});
    const messageBlob = caught.message ?? "";
    expect(errorBlob).not.toContain(FORBIDDEN_WORD);
    expect(messageBlob).not.toContain(FORBIDDEN_WORD);
    // #519 B.4b: the redacted message DOES name which guardrail fired
    // (operator-assigned metadata, not matched content). Pin
    // error.message itself — another field carrying the name must not
    // mask a regression in the client-facing message.
    expect((caught.error as { message?: unknown })?.message).toContain(
      "guardrail 'gr-out-e2e-keyword'",
    );

    // Output guardrails run AFTER the upstream call, so the upstream
    // hit count MUST go up by 1 (the guardrail can only inspect what
    // the upstream returned). A regression that short-circuited
    // pre-dispatch would leave the count flat — that's a *safer*
    // failure mode (no upstream call, no token cost), but it would
    // signal the guardrail's hook_point semantics drifted.
    expect(upstream.receivedRequests.length - upstreamHitsBefore).toBe(1);
  });

  test("streaming output: forbidden text in delta chunks → SSE error event, no [DONE] (#204)", async (ctx) => {
    if (!etcdReachable || !app || !streamUpstream) {
      ctx.skip();
      return;
    }

    // Per #204: pre-fix the streaming path skipped output guardrails
    // entirely — `kind: "keyword"` deny-list could be trivially
    // bypassed by setting `stream: true`. The fix is buffer-then-
    // check at end-of-stream, emitting an SSE `event: error` frame
    // and closing without `[DONE]`. We exercise the wire shape
    // through raw `fetch` (the OpenAI Node SDK abstracts SSE so
    // direct byte-level inspection isn't possible through it).
    const streamCallerPlaintext = `${CALLER_PLAINTEXT}-stream`;

    // Wait for the streaming Model + caller key to propagate. A
    // 200 with no error event means the guardrail isn't loaded yet
    // (we'd see the forbidden literal reach the wire and a clean
    // `[DONE]`); keep polling until we observe the block.
    await waitConfigPropagation(async () => {
      try {
        const probe = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
          method: "POST",
          headers: {
            authorization: `Bearer ${streamCallerPlaintext}`,
            "content-type": "application/json",
          },
          body: JSON.stringify({
            model: "gr-out-stream-e2e",
            messages: [{ role: "user", content: "ready-probe" }],
            stream: true,
          }),
        });
        if (probe.status !== 200) {
          await probe.text();
          return false;
        }
        const text = await probe.text();
        // Block path confirmed when the wire shows the SSE error
        // frame AND no `[DONE]` (matching the contract this test
        // pins below).
        return text.includes("event: error") && !text.includes("data: [DONE]");
      } catch {
        return false;
      }
    });

    const upstreamHitsBefore = streamUpstream.receivedRequests.length;
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${streamCallerPlaintext}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "gr-out-stream-e2e",
        messages: [{ role: "user", content: "tell me something" }],
        stream: true,
      }),
    });

    expect(res.status).toBe(200);
    const wire = await res.text();

    // Per docs §5 + #204: blocked stream MUST close without `[DONE]`
    // so SDKs that key off the sentinel detect the truncation.
    expect(wire).not.toContain("data: [DONE]");
    // SSE `event: error` MUST appear so SDKs see a failure signal.
    expect(wire).toContain("event: error");

    // Locate the error frame's data line and verify the OpenAI
    // envelope shape + content_filter taxonomy + redacted message.
    const errEventIdx = wire.indexOf("event: error\n");
    expect(errEventIdx).toBeGreaterThanOrEqual(0);
    const afterErr = wire.slice(errEventIdx + "event: error\n".length);
    const dataLine = afterErr
      .split("\n")
      .find((l: string) => l.startsWith("data: "));
    expect(dataLine).toBeDefined();
    const jsonPayload = dataLine!.slice("data: ".length);
    const parsed = JSON.parse(jsonPayload) as {
      error?: { type?: unknown; message?: unknown };
    };
    expect(parsed.error?.type).toBe("content_filter");
    // #519 B.4b: the redacted message names WHICH guardrail fired.
    expect(parsed.error?.message).toBe(
      "response blocked by content policy (guardrail 'gr-out-e2e-keyword')",
    );

    // Per #153 (mirrored to streaming): the matched literal MUST NOT
    // appear in the error envelope. (The pre-emitted `data: ...`
    // chunks DO carry the partial content — that's the documented
    // trade-off of buffer-then-check; the security guarantee is the
    // error event + missing `[DONE]`, not byte-perfect prevention
    // of all leakage.)
    expect(jsonPayload).not.toContain(FORBIDDEN_WORD);

    // Output guardrails run AFTER the upstream — the streaming
    // upstream IS hit so the buffer-then-check has content to
    // evaluate. A regression that short-circuited pre-dispatch
    // (skipping streaming entirely, the pre-#204 behavior) would
    // also fail this assertion since the upstream count wouldn't
    // move.
    expect(streamUpstream.receivedRequests.length - upstreamHitsBefore).toBeGreaterThan(0);
  });

  test("streaming Allow path: clean content streams to caller with [DONE], no error event (#204 audit M3)", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    // Per #204 audit M3: a regression that ALWAYS blocks streaming
    // (e.g. an off-by-one `errored = true` after the guardrail
    // check) would not fail the existing block-case test. The
    // Allow companion below pins that the gate opens cleanly when
    // the assistant content does NOT contain a forbidden literal:
    // full content reaches the caller, terminal `[DONE]` appears,
    // and there is no SSE error frame.
    const cleanUpstream = await startOpenAiUpstream({
      streamEvents: [
        '{"id":"strm-clean","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
        // Distinct phrase that does NOT contain FORBIDDEN_WORD —
        // guardrail must Allow.
        '{"id":"strm-clean","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"hello world clean response"},"finish_reason":null}]}',
        '{"id":"strm-clean","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
        "[DONE]",
      ],
      eventDelayMs: 50,
    });
    const cleanPk = await admin.createProviderKey({
      display_name: "gr-out-stream-clean-pk",
      secret: "sk-mock",
      api_base: `${cleanUpstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "gr-out-stream-clean-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: cleanPk.id,
    });
    const cleanCallerPlaintext = `${CALLER_PLAINTEXT}-stream-clean`;
    await admin.createApiKey({
      key_hash: createHash("sha256")
        .update(cleanCallerPlaintext)
        .digest("hex"),
      allowed_models: ["gr-out-stream-clean-e2e"],
    });

    await waitConfigPropagation(async () => {
      try {
        const probe = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
          method: "POST",
          headers: {
            authorization: `Bearer ${cleanCallerPlaintext}`,
            "content-type": "application/json",
          },
          body: JSON.stringify({
            model: "gr-out-stream-clean-e2e",
            messages: [{ role: "user", content: "ready-probe" }],
            stream: true,
          }),
        });
        if (probe.status !== 200) {
          await probe.text();
          return false;
        }
        const text = await probe.text();
        return text.includes("data: [DONE]") && !text.includes("event: error");
      } catch {
        return false;
      }
    });

    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${cleanCallerPlaintext}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "gr-out-stream-clean-e2e",
        messages: [{ role: "user", content: "say hi" }],
        stream: true,
      }),
    });

    expect(res.status).toBe(200);
    const wire = await res.text();
    // Allow path contracts:
    expect(wire).toContain("data: [DONE]");
    expect(wire).not.toContain("event: error");
    // The full assistant content reaches the wire (proves the
    // gateway didn't short-circuit pre-dispatch and didn't strip
    // chunks on the way out).
    expect(wire).toContain("hello world clean response");

    await cleanUpstream.close();
  });
});
