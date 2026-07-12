import { createHash } from "node:crypto";
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
import { harnessRequest } from "../harness/http.js";

const CALLER_PLAINTEXT = "sk-request-id-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

describe("request id e2e: gateway-generated request IDs are UUIDs", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let chatUpstream: OpenAiUpstream | undefined;
  let embeddingsUpstream: OpenAiUpstream | undefined;
  let messagesUpstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    chatUpstream = await startOpenAiUpstream();
    embeddingsUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        object: "list",
        data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2] }],
        model: "text-embedding-3-small",
        usage: { prompt_tokens: 1, total_tokens: 1 },
      },
    });
    messagesUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "msg_01",
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: "ok" }],
        model: "claude-3-5-haiku-20241022",
        stop_reason: "end_turn",
        usage: { input_tokens: 1, output_tokens: 1 },
      },
    });

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const chatPk = await seed.createProviderKey({
      display_name: "request-id-chat-pk",
      secret: "sk-mock",
      api_base: `${chatUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "request-id-chat",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: chatPk.id,
    });

    const embeddingsPk = await seed.createProviderKey({
      display_name: "request-id-embeddings-pk",
      secret: "sk-mock",
      api_base: `${embeddingsUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "request-id-embeddings",
      provider: "openai",
      model_name: "text-embedding-3-small",
      provider_key_id: embeddingsPk.id,
    });

    const messagesPk = await seed.createProviderKey({
      display_name: "request-id-messages-pk",
      secret: "sk-ant-mock",
      api_base: messagesUpstream.baseUrl,
    });
    await seed.createModel({
      display_name: "request-id-messages",
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: messagesPk.id,
    });

    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all([
      chatUpstream?.close(),
      embeddingsUpstream?.close(),
      messagesUpstream?.close(),
    ]);
  });

  test("chat, embeddings, and messages forward UUID request IDs upstream", async (ctx) => {
    if (
      !etcdReachable ||
      !app ||
      !chatUpstream ||
      !embeddingsUpstream ||
      !messagesUpstream
    ) {
      ctx.skip();
      return;
    }

    await waitConfigPropagation(async () => {
      try {
        const [chat, embeddings, messages] = await Promise.all([
          postJson(app.proxyUrl, "/v1/chat/completions", {
            model: "request-id-chat",
            messages: [{ role: "user", content: "ready" }],
          }),
          postJson(app.proxyUrl, "/v1/embeddings", {
            model: "request-id-embeddings",
            input: "ready",
          }),
          postJson(app.proxyUrl, "/v1/messages", {
            model: "request-id-messages",
            messages: [{ role: "user", content: "ready" }],
            max_tokens: 16,
          }),
        ]);
        return chat.status === 200 && embeddings.status === 200 && messages.status === 200;
      } catch {
        return false;
      }
    });

    const chatBaseline = chatUpstream.receivedRequests.length;
    const embeddingsBaseline = embeddingsUpstream.receivedRequests.length;
    const messagesBaseline = messagesUpstream.receivedRequests.length;

    await expectOk(
      postJson(app.proxyUrl, "/v1/chat/completions", {
        model: "request-id-chat",
        messages: [{ role: "user", content: "hello" }],
      }),
    );
    await expectOk(
      postJson(app.proxyUrl, "/v1/embeddings", {
        model: "request-id-embeddings",
        input: "hello",
      }),
    );
    await expectOk(
      postJson(app.proxyUrl, "/v1/messages", {
        model: "request-id-messages",
        messages: [{ role: "user", content: "hello" }],
        max_tokens: 16,
      }),
    );

    expectUpstreamRequestId(chatUpstream, chatBaseline, "/v1/chat/completions");
    expectUpstreamRequestId(embeddingsUpstream, embeddingsBaseline, "/v1/embeddings");
    expectUpstreamRequestId(messagesUpstream, messagesBaseline, "/v1/messages");
  });
});

async function postJson(
  baseUrl: string,
  path: string,
  body: unknown,
): Promise<{ status: number; body: unknown }> {
  const res = await harnessRequest(`${baseUrl}${path}`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    },
    body: JSON.stringify(body),
  });
  const text = await res.body.text();
  return {
    status: res.statusCode,
    body: text ? JSON.parse(text) : null,
  };
}

async function expectOk(promise: Promise<{ status: number; body: unknown }>): Promise<void> {
  const res = await promise;
  if (res.status !== 200) {
    throw new Error(`expected 200, got ${res.status}: ${JSON.stringify(res.body)}`);
  }
}

function expectUpstreamRequestId(
  upstream: OpenAiUpstream,
  baseline: number,
  path: string,
): void {
  const calls = upstream.receivedRequests
    .slice(baseline)
    .filter((req) => req.path === path);
  expect(calls).toHaveLength(1);
  expect(calls[0]!.headers["x-aisix-request-id"]).toMatch(UUID_RE);
}
