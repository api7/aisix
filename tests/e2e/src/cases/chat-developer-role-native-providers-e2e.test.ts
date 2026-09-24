import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  waitConfigPropagation,
  type SpawnedApp,
} from "../harness/index.js";

// A Chat Completions `developer` message on the providers whose bridge
// builds its own request shape. None of them has a positional developer
// role, so the instruction goes where each one keeps system instructions:
// Anthropic's top-level `system`, Gemini's `systemInstruction`, Converse's
// `system[]`. It must never be forwarded as a conversation turn.
//
// Reference:
//   - <https://platform.openai.com/docs/api-reference/chat/create>
//   - <https://docs.anthropic.com/en/api/messages>
//   - <https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference>
//   - <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html>

const CALLER_PLAINTEXT = "sk-chat-developer-role-native";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const INSTRUCTION = "Follow application instructions";

interface RecordingUpstream {
  baseUrl: string;
  received: Array<{ path: string; body: string }>;
  close(): Promise<void>;
}

async function startJsonUpstream(reply: unknown): Promise<RecordingUpstream> {
  const received: Array<{ path: string; body: string }> = [];
  const server: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      received.push({
        path: (req.url ?? "/").split("?")[0],
        body: Buffer.concat(chunks).toString("utf8"),
      });
      res.statusCode = 200;
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify(reply));
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return {
    baseUrl: `http://127.0.0.1:${addr.port}`,
    received,
    close: () =>
      new Promise<void>((resolve, reject) =>
        server.close((e) => (e ? reject(e) : resolve())),
      ),
  };
}

describe("chat developer role → native provider instruction slots", () => {
  let app: SpawnedApp | undefined;
  let anthropic: RecordingUpstream | undefined;
  let gemini: RecordingUpstream | undefined;
  let bedrock: RecordingUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    anthropic = await startJsonUpstream({
      id: "msg_dev",
      type: "message",
      role: "assistant",
      model: "claude-3-5-haiku-20241022",
      content: [{ type: "text", text: "ok" }],
      stop_reason: "end_turn",
      usage: { input_tokens: 7, output_tokens: 1 },
    });
    gemini = await startJsonUpstream({
      candidates: [
        { content: { role: "model", parts: [{ text: "ok" }] }, finishReason: "STOP" },
      ],
      usageMetadata: { promptTokenCount: 7, candidatesTokenCount: 1, totalTokenCount: 8 },
    });
    bedrock = await startJsonUpstream({
      output: { message: { role: "assistant", content: [{ text: "ok" }] } },
      stopReason: "end_turn",
      usage: { inputTokens: 7, outputTokens: 1, totalTokens: 8 },
      metrics: { latencyMs: 1 },
    });

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    // The Anthropic adapter appends `/v1/messages` to the bare host.
    const anthropicPk = await seed.createProviderKey({
      display_name: "dev-anthropic-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-mock",
      api_base: anthropic.baseUrl,
    });
    await seed.createModel({
      display_name: "dev-claude",
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: anthropicPk.id,
    });

    const vertexPk = await seed.createProviderKey({
      display_name: "dev-vertex-pk",
      provider: "google",
      adapter: "vertex",
      secret: JSON.stringify({
        access_token: "ya29.developer-e2e",
        project: "proj-e2e",
        region: "us-central1",
      }),
      api_base: gemini.baseUrl,
    });
    await seed.createModel({
      display_name: "dev-gemini",
      provider: "google",
      model_name: "gemini-2.5-flash",
      provider_key_id: vertexPk.id,
    });

    const bedrockPk = await seed.createProviderKey({
      display_name: "dev-bedrock-pk",
      provider: "bedrock",
      adapter: "bedrock",
      secret: JSON.stringify({
        access_key_id: "AKIA-developer-e2e",
        secret_access_key: "sk-developer-e2e",
        region: "us-west-2",
      }),
      api_base: bedrock.baseUrl,
    });
    await seed.createModel({
      display_name: "dev-nova",
      provider: "bedrock",
      model_name: "amazon.nova-pro-v1:0",
      provider_key_id: bedrockPk.id,
    });

    // Seeded last, so this key authenticating implies the whole seed set
    // has reached the gateway's snapshot.
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["dev-claude", "dev-gemini", "dev-nova"],
    });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      await res.text();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await anthropic?.close();
    await gemini?.close();
    await bedrock?.close();
  });

  /** Sends a developer instruction ahead of a user turn and returns the
   * body the upstream received. */
  async function send(
    model: string,
    upstream: RecordingUpstream,
  ): Promise<Record<string, any>> {
    const before = upstream.received.length;
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model,
        messages: [
          { role: "developer", content: INSTRUCTION },
          { role: "user", content: "hello" },
        ],
      }),
    });
    expect(res.status, await res.text()).toBe(200);
    expect(upstream.received.length - before).toBe(1);
    return JSON.parse(upstream.received.at(-1)!.body);
  }

  test("anthropic: developer becomes the top-level system prompt", async (ctx) => {
    if (!etcdReachable || !app || !anthropic) {
      ctx.skip();
      return;
    }
    const body = await send("dev-claude", anthropic);
    expect(body.system).toBe(INSTRUCTION);
    expect(body.messages).toEqual([
      { role: "user", content: [{ type: "text", text: "hello" }] },
    ]);
  });

  test("gemini: developer becomes systemInstruction", async (ctx) => {
    if (!etcdReachable || !app || !gemini) {
      ctx.skip();
      return;
    }
    const body = await send("dev-gemini", gemini);
    expect(body.systemInstruction?.parts?.[0]?.text).toBe(INSTRUCTION);
    expect(body.contents).toEqual([
      { role: "user", parts: [{ text: "hello" }] },
    ]);
  });

  test("bedrock converse: developer becomes a system block", async (ctx) => {
    if (!etcdReachable || !app || !bedrock) {
      ctx.skip();
      return;
    }
    const body = await send("dev-nova", bedrock);
    expect(body.system).toEqual([{ text: INSTRUCTION }]);
    expect(body.messages).toHaveLength(1);
    expect(body.messages[0].role).toBe("user");
  });
});
