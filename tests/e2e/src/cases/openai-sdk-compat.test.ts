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

// E2E client-compatibility check: drive the gateway through the
// official `openai` npm SDK rather than the harness's hand-rolled
// `ProxyClient`. This catches wire mismatches that bypass the internal
// shim — header normalization, retry shape, streaming chunk parsing on
// the client side, async client lifecycle. The harness's `proxy.ts`
// header comment explicitly invites this pattern; this test makes it
// real.
//
// Reference: OpenAI Node SDK source for chat completions parsing
// (https://github.com/openai/openai-node/blob/master/src/resources/chat/completions.ts)
// and the OpenAI Chat Completions API spec
// (https://platform.openai.com/docs/api-reference/chat/create).

const CALLER_PLAINTEXT = "sk-sdk-compat-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("openai SDK compat: drive gateway through real client", () => {
  let app: SpawnedApp | undefined;
  let nonStreamUpstream: OpenAiUpstream | undefined;
  let streamUpstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    nonStreamUpstream = await startOpenAiUpstream();
    streamUpstream = await startOpenAiUpstream({
      streamEvents: [
        '{"id":"u-1","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
        '{"id":"u-1","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}',
        '{"id":"u-1","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}',
        '{"id":"u-1","model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
        "[DONE]",
      ],
    });

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    // Two ProviderKeys → two Models — keeps `receivedRequests` on each
    // mock unambiguous so the assertion below proves the SDK hit the
    // expected upstream rather than leaking across.
    const pkSync = await admin.createProviderKey({
      display_name: "sdk-compat-sync-pk",
      secret: "sk-mock",
      api_base: `${nonStreamUpstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "sdk-compat-sync",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pkSync.id,
    });

    const pkStream = await admin.createProviderKey({
      display_name: "sdk-compat-stream-pk",
      secret: "sk-mock",
      api_base: `${streamUpstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "sdk-compat-stream",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pkStream.id,
    });

    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["sdk-compat-sync", "sdk-compat-stream"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await nonStreamUpstream?.close();
    await streamUpstream?.close();
  });

  test("openai.chat.completions.create() — non-streaming", async (ctx) => {
    if (!etcdReachable || !app || !nonStreamUpstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
    });

    // Snapshot propagation: poll the SDK path itself until it stops
    // erroring (Model + ProviderKey + ApiKey all visible to the
    // dispatcher). Mirrors the pattern in smoke.test.ts.
    await waitConfigPropagation(async () => {
      try {
        await client.chat.completions.create({
          model: "sdk-compat-sync",
          messages: [{ role: "user", content: "ping" }],
        });
        return true;
      } catch {
        return false;
      }
    });

    // Baseline-isolate the propagation probe so the assertion below
    // measures only the effect of the actual test call. Without this,
    // tightening to an absolute count (e.g. `length === 1`) would
    // silently break — the probe consumes one slot in receivedRequests.
    const baseline = nonStreamUpstream.receivedRequests.length;

    const completion = await client.chat.completions.create({
      model: "sdk-compat-sync",
      messages: [{ role: "user", content: "hello" }],
    });

    expect(completion.object).toBe("chat.completion");
    expect(completion.choices[0]?.message.role).toBe("assistant");
    expect(typeof completion.choices[0]?.message.content).toBe("string");
    expect(completion.usage?.total_tokens).toBeGreaterThan(0);

    // Belt-and-suspenders: the test call hit the upstream exactly once
    // (delta from baseline) and that hit landed on the chat-completions
    // path. The absolute-count form rejects regressions that double-fire
    // or short-circuit and silently fall through to a stray route.
    expect(nonStreamUpstream.receivedRequests.length - baseline).toBe(1);
    expect(
      nonStreamUpstream.receivedRequests[baseline]?.path.startsWith(
        "/v1/chat/completions",
      ),
    ).toBe(true);
  });

  test("openai.chat.completions.create({stream:true}) — streaming", async (ctx) => {
    if (!etcdReachable || !app || !streamUpstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
    });

    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "sdk-compat-stream",
          messages: [{ role: "user", content: "ping" }],
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

    const stream = await client.chat.completions.create({
      model: "sdk-compat-stream",
      messages: [{ role: "user", content: "say hello" }],
      stream: true,
    });

    let collected = "";
    let finishReason: string | null | undefined;
    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta;
      if (delta?.content) collected += delta.content;
      finishReason ??= chunk.choices[0]?.finish_reason ?? undefined;
    }

    // The SDK must reconstruct the streamed text correctly — wire
    // mismatches (chunked encoding, content-type, SSE framing, [DONE]
    // sentinel handling) would surface here with truncated `collected`
    // or a missing finish_reason.
    expect(collected).toBe("hello world");
    expect(finishReason).toBe("stop");
    expect(
      streamUpstream.receivedRequests.some((r) =>
        r.path.startsWith("/v1/chat/completions"),
      ),
    ).toBe(true);
  });
});
