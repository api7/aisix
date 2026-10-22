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

// E2E: the same chat request, sent repeatedly to a Bedrock model answered
// through Converse, reaches Bedrock as the same bytes every time. Bedrock
// prompt caching matches the prompt prefix exactly, with `tools` first, so a
// body that differs between identical requests is never read from cache.
//
// The request carries a tool with several parameters and a replayed tool call
// with several arguments: the two places a JSON object travels inside the
// Converse body.

const CALLER_PLAINTEXT = "sk-bedrock-converse-stable-body";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const REPEATS = 5;

const CONVERSE_REPLY = {
  output: { message: { role: "assistant", content: [{ text: "Found it." }] } },
  stopReason: "end_turn",
  usage: { inputTokens: 20, outputTokens: 3, totalTokens: 23 },
  metrics: { latencyMs: 5 },
};
const CONVERSE_STREAM = eventStream([
  ["messageStart", { role: "assistant" }],
  ["contentBlockDelta", { contentBlockIndex: 0, delta: { text: "Found it." } }],
  ["contentBlockStop", { contentBlockIndex: 0 }],
  ["messageStop", { stopReason: "end_turn" }],
  ["metadata", { usage: { inputTokens: 20, outputTokens: 3, totalTokens: 23 }, metrics: { latencyMs: 5 } }],
]);

const TOOLS = [
  {
    type: "function",
    function: {
      name: "grep",
      description: "Search file contents",
      parameters: {
        type: "object",
        properties: {
          pattern: { type: "string" },
          path: { type: "string" },
          include: { type: "string" },
          context: { type: "integer" },
          ignore_case: { type: "boolean" },
          max_results: { type: "integer" },
        },
        required: ["pattern"],
      },
    },
  },
];
const MESSAGES = [
  { role: "system", content: "You are a coding assistant." },
  { role: "user", content: "Find TODO in src." },
  {
    role: "assistant",
    content: null,
    tool_calls: [
      {
        id: "tooluse_1",
        type: "function",
        function: {
          name: "grep",
          arguments: JSON.stringify({
            pattern: "TODO",
            path: "src",
            include: "*.rs",
            context: 2,
            ignore_case: true,
            max_results: 50,
          }),
        },
      },
    ],
  },
  { role: "tool", tool_call_id: "tooluse_1", content: "src/main.rs:3: // TODO" },
];

describe("Bedrock Converse: identical requests send identical bodies", () => {
  let app: SpawnedApp | undefined;
  let upstream: Server | undefined;
  const received: Array<{ path: string; raw: string }> = [];
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = createServer((req, res) => {
      let raw = "";
      req.on("data", (c) => (raw += c));
      req.on("end", () => {
        received.push({ path: req.url ?? "", raw });
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
      display_name: "bedrock-converse-stable-body-pk",
      provider: "bedrock",
      adapter: "bedrock",
      secret: JSON.stringify({
        access_key_id: "AKIA-converse-stable-body",
        secret_access_key: "sk-converse-stable-body",
        region: "us-east-1",
      }),
      api_base: `http://127.0.0.1:${addr.port}`,
    });
    for (const [alias, name] of [
      ["stable-body-nova", "us.amazon.nova-lite-v1:0"],
      ["stable-body-claude", "us.anthropic.claude-haiku-4-5-20251001-v1:0"],
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
    ["stable-body-nova", false],
    ["stable-body-nova", true],
    ["stable-body-claude", true],
  ])("%s stream=%s", async ([model, stream], ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const before = received.length;
    for (let i = 0; i < REPEATS; i++) {
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
    }

    const sent = received.slice(before);
    expect(sent).toHaveLength(REPEATS);
    for (const s of sent) {
      expect(s.path).toMatch(stream ? /\/converse-stream$/ : /\/converse$/);
    }
    // The schema and the replayed arguments both reached Bedrock.
    expect(sent[0].raw).toContain("max_results");
    expect(sent[0].raw).toContain("*.rs");
    expect(new Set(sent.map((s) => s.raw)).size).toBe(1);
  });
});
