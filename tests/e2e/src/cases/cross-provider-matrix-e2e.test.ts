import { createHash } from "node:crypto";
import OpenAI from "openai";
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

// E2E: cross-provider dispatch matrix for OpenAI-compat upstreams.
// The OpenAI / Gemini / DeepSeek provider bridges all speak the
// OpenAI wire shape upstream, so the test surface is "did the gateway
// dispatch through the right bridge for the right Provider enum?"
// rather than "did wire translation succeed?" (translation is covered
// in `anthropic-upstream-e2e.test.ts` for the Anthropic case).
//
// Existing coverage filled by other test files:
// - openai (non-stream + stream):   sdk-compat (#134)
// - anthropic non-stream:           anthropic-upstream-e2e (#141)
// - anthropic stream:                NOT YET (mock harness's `streamEvents`
//                                   only writes `data:` SSE lines, not the
//                                   `event:`/`data:` typed-event pairs that
//                                   the Anthropic streaming wire requires;
//                                   harness extension needed first).
//
// This file fills the remaining four combos: gemini and deepseek,
// each in non-streaming and streaming form. Together with the existing
// per-provider e2e cases, this nails the matrix at 4×{non,stream} = 8
// for OpenAI-compat upstreams.
//
// Reference: OpenAI Chat Completions wire format
// (https://platform.openai.com/docs/api-reference/chat/create) — both
// Gemini and DeepSeek bridges proxy this shape verbatim.

const CALLER_PLAINTEXT = "sk-matrix-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

type Provider = "gemini" | "deepseek";

interface MatrixCase {
  readonly provider: Provider;
  readonly upstreamModelId: string;
  readonly displayPrefix: string;
  readonly expectedContent: string;
}

const CASES: ReadonlyArray<MatrixCase> = [
  {
    provider: "gemini",
    upstreamModelId: "gemini-2.0-flash",
    displayPrefix: "matrix-gemini",
    expectedContent: "Hello from Gemini!",
  },
  {
    provider: "deepseek",
    upstreamModelId: "deepseek-chat",
    displayPrefix: "matrix-deepseek",
    expectedContent: "Hello from DeepSeek!",
  },
];

describe("cross-provider matrix: OpenAI-compat upstreams", () => {
  let app: SpawnedApp | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;
  // Each test creates its own upstream mock (per provider × streaming
  // variant). Track them so `afterAll` can close every one.
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    // Wildcard `allowed_models: ["*"]` lets the same caller key reach
    // every model the per-test setup creates, without re-creating the
    // ApiKey per case.
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  for (const tc of CASES) {
    test(`${tc.provider} upstream — non-streaming`, async (ctx) => {
      if (!etcdReachable || !app || !admin) {
        ctx.skip();
        return;
      }

      // Upstream returns a canned OpenAI-shape completion. Both the
      // Gemini and DeepSeek bridges forward this shape verbatim to
      // the caller; the test pins that the caller-visible content
      // round-trips byte-for-byte.
      const upstream = await startOpenAiUpstream({
        nonStreamBody: {
          id: `cmpl-${tc.provider}`,
          object: "chat.completion",
          created: Math.floor(Date.now() / 1000),
          model: tc.upstreamModelId,
          choices: [
            {
              index: 0,
              message: {
                role: "assistant",
                content: tc.expectedContent,
              },
              finish_reason: "stop",
            },
          ],
          usage: { prompt_tokens: 5, completion_tokens: 4, total_tokens: 9 },
        },
      });
      upstreams.push(upstream);

      const pk = await admin.createProviderKey({
        display_name: `${tc.displayPrefix}-pk-non-stream`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      const modelName = `${tc.displayPrefix}-non-stream`;
      await admin.createModel({
        display_name: modelName,
        provider: tc.provider,
        model_name: tc.upstreamModelId,
        provider_key_id: pk.id,
      });

      const client = new OpenAI({
        apiKey: CALLER_PLAINTEXT,
        baseURL: `${app.proxyUrl}/v1`,
        maxRetries: 0,
      });

      await waitConfigPropagation(async () => {
        try {
          await client.chat.completions.create({
            model: modelName,
            messages: [{ role: "user", content: "ready-probe" }],
          });
          return true;
        } catch {
          return false;
        }
      });

      const baseline = upstream.receivedRequests.length;
      const completion = await client.chat.completions.create({
        model: modelName,
        messages: [{ role: "user", content: "hi" }],
      });

      expect(completion.object).toBe("chat.completion");
      expect(completion.choices[0]?.message.role).toBe("assistant");
      expect(completion.choices[0]?.message.content).toBe(tc.expectedContent);
      expect(completion.choices[0]?.finish_reason).toBe("stop");
      expect(completion.usage?.total_tokens).toBe(9);

      // Dispatch contract: gateway hit `/v1/chat/completions` exactly
      // once (the OpenAI-compat path the Gemini/DeepSeek bridges use).
      // A regression that mis-routed through Anthropic bridge would
      // hit `/v1/messages` instead; a regression that introduced a
      // retry loop would land >1 matching call here.
      const testCalls = upstream.receivedRequests
        .slice(baseline)
        .filter((r) => r.path === "/v1/chat/completions");
      expect(testCalls).toHaveLength(1);
      const testCall = testCalls[0]!;
      expect(testCall.method).toBe("POST");
      // Auth header: gateway forwards the upstream's secret as
      // `Authorization: Bearer <secret>`. The mock accepts any auth,
      // so without this assertion an upstream-401 regression (header
      // dropped, swapped, or rewritten with the caller's key) would
      // pass against the mock but fail in production.
      expect(testCall.headers["authorization"]).toBe("Bearer sk-mock");
      // Wire-shape contract: body reaches upstream as OpenAI Chat
      // Completions JSON. `model` is rewritten to the upstream's own
      // id (caller-visible name → `upstreamModelId`). `stream` is
      // absent or false on the non-stream path.
      const body = JSON.parse(testCall.body) as {
        model?: string;
        messages?: Array<{ role: string; content: string }>;
        stream?: boolean;
      };
      expect(body.model).toBe(tc.upstreamModelId);
      expect(body.messages?.[0]?.role).toBe("user");
      expect(body.messages?.[0]?.content).toBe("hi");
      expect(body.stream ?? false).toBe(false);
    });

    test(`${tc.provider} upstream — streaming`, async (ctx) => {
      if (!etcdReachable || !app || !admin) {
        ctx.skip();
        return;
      }

      // Three OpenAI-shape SSE chunks — role delta, content delta,
      // finish_reason — followed by `[DONE]`. The OpenAI Node SDK
      // assembles `chunk.choices[0].delta.content` across deltas.
      const sseEvents = [
        `{"id":"c1","object":"chat.completion.chunk","model":"${tc.upstreamModelId}","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`,
        `{"id":"c1","object":"chat.completion.chunk","model":"${tc.upstreamModelId}","choices":[{"index":0,"delta":{"content":${JSON.stringify(tc.expectedContent)}},"finish_reason":null}]}`,
        `{"id":"c1","object":"chat.completion.chunk","model":"${tc.upstreamModelId}","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
        "[DONE]",
      ];
      const upstream = await startOpenAiUpstream({ streamEvents: sseEvents });
      upstreams.push(upstream);

      const pk = await admin.createProviderKey({
        display_name: `${tc.displayPrefix}-pk-stream`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      const modelName = `${tc.displayPrefix}-stream`;
      await admin.createModel({
        display_name: modelName,
        provider: tc.provider,
        model_name: tc.upstreamModelId,
        provider_key_id: pk.id,
      });

      const client = new OpenAI({
        apiKey: CALLER_PLAINTEXT,
        baseURL: `${app.proxyUrl}/v1`,
        maxRetries: 0,
      });

      await waitConfigPropagation(async () => {
        try {
          const probe = await client.chat.completions.create({
            model: modelName,
            messages: [{ role: "user", content: "ready-probe" }],
            stream: true,
          });
          for await (const _chunk of probe) {
            break;
          }
          return true;
        } catch {
          return false;
        }
      });

      const baseline = upstream.receivedRequests.length;
      const stream = await client.chat.completions.create({
        model: modelName,
        messages: [{ role: "user", content: "hi" }],
        stream: true,
      });

      // Capture chunks in arrival order — a regression that reordered
      // SSE events (finish_reason before content, or content split
      // across the wrong chunks) needs the order to be visible, not
      // just summed.
      const chunks: Array<{ content: string; finish: string | null }> = [];
      for await (const chunk of stream) {
        const choice = chunk.choices[0];
        chunks.push({
          content: choice?.delta?.content ?? "",
          finish: choice?.finish_reason ?? null,
        });
      }
      const collected = chunks.map((c) => c.content).join("");
      expect(collected).toBe(tc.expectedContent);
      // finish_reason must arrive on the LAST chunk that carries it,
      // never mid-stream.
      const finishIdx = chunks.findIndex((c) => c.finish === "stop");
      expect(finishIdx).toBeGreaterThanOrEqual(0);
      expect(finishIdx).toBe(chunks.length - 1);

      const testCalls = upstream.receivedRequests
        .slice(baseline)
        .filter((r) => r.path === "/v1/chat/completions");
      expect(testCalls).toHaveLength(1);
      const testCall = testCalls[0]!;
      expect(testCall.method).toBe("POST");
      expect(testCall.headers["authorization"]).toBe("Bearer sk-mock");
      const body = JSON.parse(testCall.body) as {
        model?: string;
        messages?: Array<{ role: string; content: string }>;
        stream?: boolean;
      };
      expect(body.model).toBe(tc.upstreamModelId);
      expect(body.messages?.[0]?.role).toBe("user");
      expect(body.messages?.[0]?.content).toBe("hi");
      expect(body.stream).toBe(true);
    });
  }
});
