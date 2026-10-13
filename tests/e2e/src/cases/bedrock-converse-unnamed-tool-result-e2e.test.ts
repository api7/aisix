import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  eventStream,
  spawnApp,
  waitConfigPropagation,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: a `role:"tool"` message with no `tool_call_id`, sent to a Bedrock
// model answered through Converse, reaches Bedrock as a `toolResult` under a
// generated `toolUseId` rather than being dropped from the conversation;
// Bedrock decides what to make of it.
//
// Converse serves every streamed Bedrock request and every non-streamed
// request to a non-Anthropic publisher.

const CALLER_PLAINTEXT = "sk-bedrock-unnamed-tool-result";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const CONVERSE_REPLY = {
  output: { message: { role: "assistant", content: [{ text: "It is sunny." }] } },
  stopReason: "end_turn",
  usage: { inputTokens: 20, outputTokens: 4, totalTokens: 24 },
  metrics: { latencyMs: 5 },
};
const CONVERSE_STREAM = eventStream([
  ["messageStart", { role: "assistant" }],
  ["contentBlockDelta", { contentBlockIndex: 0, delta: { text: "It is sunny." } }],
  ["contentBlockStop", { contentBlockIndex: 0 }],
  ["messageStop", { stopReason: "end_turn" }],
  ["metadata", { usage: { inputTokens: 20, outputTokens: 4, totalTokens: 24 }, metrics: { latencyMs: 5 } }],
]);

const MESSAGES = [
  { role: "user", content: "Weather in Paris?" },
  {
    role: "assistant",
    content: null,
    tool_calls: [
      { id: "tooluse_1", type: "function", function: { name: "get_weather", arguments: "{}" } },
    ],
  },
  { role: "tool", content: "sunny" },
];
const TOOLS = [
  {
    type: "function",
    function: { name: "get_weather", parameters: { type: "object", properties: {} } },
  },
];

describe("Bedrock Converse: a tool result without tool_call_id", () => {
  let app: SpawnedApp | undefined;
  let upstream: Server | undefined;
  const received: Array<{ path: string; body: any }> = [];
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = createServer((req, res) => {
      let raw = "";
      req.on("data", (c) => (raw += c));
      req.on("end", () => {
        received.push({ path: req.url ?? "", body: JSON.parse(raw || "{}") });
        if (req.url?.endsWith("/converse-stream")) {
          res.writeHead(200, { "content-type": "application/vnd.amazon.eventstream" });
          res.end(CONVERSE_STREAM);
        } else {
          res.writeHead(200, { "content-type": "application/json" });
          res.end(JSON.stringify(CONVERSE_REPLY));
        }
      });
    });
    await new Promise<void>((resolve) => upstream!.listen(0, "127.0.0.1", resolve));
    const addr = upstream.address();
    if (addr === null || typeof addr === "string") throw new Error("no port");

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "bedrock-unnamed-tool-result-pk",
      provider: "bedrock",
      adapter: "bedrock",
      secret: JSON.stringify({
        access_key_id: "AKIA-unnamed-tool-result",
        secret_access_key: "sk-unnamed-tool-result",
        region: "us-east-1",
      }),
      api_base: `http://127.0.0.1:${addr.port}`,
    });
    for (const [alias, name] of [
      ["unnamed-tool-nova", "us.amazon.nova-lite-v1:0"],
      ["unnamed-tool-claude", "us.anthropic.claude-haiku-4-5-20251001-v1:0"],
    ]) {
      await seed.createModel({
        display_name: alias,
        provider: "bedrock",
        model_name: name,
        provider_key_id: pk.id,
      });
    }
    // Seeded last, so this key authenticating implies the whole seed set
    // has reached the gateway's snapshot.
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
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
    await new Promise<void>((resolve) => (upstream ? upstream.close(() => resolve()) : resolve()));
  });

  test.for<[string, boolean]>([
    ["unnamed-tool-nova", false],
    ["unnamed-tool-nova", true],
    ["unnamed-tool-claude", true],
  ])("%s stream=%s forwards the result under a generated id", async ([model, stream], ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const before = received.length;
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({ model, stream, tools: TOOLS, messages: MESSAGES }),
    });
    const text = await res.text();
    expect(res.status, text).toBe(200);
    expect(text).toContain("It is sunny.");

    const sent = received.slice(before);
    expect(sent).toHaveLength(1);
    expect(sent[0].path).toMatch(stream ? /\/converse-stream$/ : /\/converse$/);
    const results = sent[0].body.messages
      .flatMap((m: any) => m.content)
      .filter((b: any) => b.toolResult)
      .map((b: any) => b.toolResult);
    expect(results).toHaveLength(1);
    expect(typeof results[0].toolUseId).toBe("string");
    expect(results[0].toolUseId.length).toBeGreaterThan(0);
    expect(JSON.stringify(results[0].content)).toContain("sunny");
  });
});
