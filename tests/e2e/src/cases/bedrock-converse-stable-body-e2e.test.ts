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
// Converse body. Every client endpoint that can reach Converse is driven, and
// `/v1/messages` to Claude, which Bedrock answers through InvokeModel, is held
// to the same standard.

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

const INVOKE_REPLY = {
  id: "msg_stable_body",
  type: "message",
  role: "assistant",
  model: "claude-haiku-4-5-20251001",
  content: [{ type: "text", text: "Found it." }],
  stop_reason: "end_turn",
  usage: { input_tokens: 20, output_tokens: 3 },
};

const SYSTEM = "You are a coding assistant.";
const QUESTION = "Find TODO in src.";
const TOOL_OUTPUT = "src/main.rs:3: // TODO";
const PARAMETERS = {
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
};
const ARGUMENTS = {
  pattern: "TODO",
  path: "src",
  include: "*.rs",
  context: 2,
  ignore_case: true,
  max_results: 50,
};

type Endpoint = "/v1/chat/completions" | "/v1/messages" | "/v1/responses";

function requestBody(endpoint: Endpoint, model: string, stream: boolean): unknown {
  switch (endpoint) {
    case "/v1/chat/completions":
      return {
        model,
        stream,
        tools: [{ type: "function", function: { name: "grep", description: "Search file contents", parameters: PARAMETERS } }],
        messages: [
          { role: "system", content: SYSTEM },
          { role: "user", content: QUESTION },
          {
            role: "assistant",
            content: null,
            tool_calls: [
              { id: "tooluse_1", type: "function", function: { name: "grep", arguments: JSON.stringify(ARGUMENTS) } },
            ],
          },
          { role: "tool", tool_call_id: "tooluse_1", content: TOOL_OUTPUT },
        ],
      };
    case "/v1/messages":
      return {
        model,
        stream,
        max_tokens: 256,
        system: SYSTEM,
        tools: [{ name: "grep", description: "Search file contents", input_schema: PARAMETERS }],
        messages: [
          { role: "user", content: QUESTION },
          { role: "assistant", content: [{ type: "tool_use", id: "tooluse_1", name: "grep", input: ARGUMENTS }] },
          { role: "user", content: [{ type: "tool_result", tool_use_id: "tooluse_1", content: TOOL_OUTPUT }] },
        ],
      };
    case "/v1/responses":
      return {
        model,
        stream,
        instructions: SYSTEM,
        tools: [{ type: "function", name: "grep", description: "Search file contents", parameters: PARAMETERS }],
        input: [
          { role: "user", content: QUESTION },
          { type: "function_call", call_id: "tooluse_1", name: "grep", arguments: JSON.stringify(ARGUMENTS) },
          { type: "function_call_output", call_id: "tooluse_1", output: TOOL_OUTPUT },
        ],
      };
  }
}

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
        } else if (req.url?.endsWith("/invoke")) {
          res.writeHead(200, { "content-type": "application/json" });
          res.end(JSON.stringify(INVOKE_REPLY));
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

  test.for<[Endpoint, string, boolean, string]>([
    ["/v1/chat/completions", "stable-body-nova", false, "/converse"],
    ["/v1/chat/completions", "stable-body-nova", true, "/converse-stream"],
    ["/v1/chat/completions", "stable-body-claude", true, "/converse-stream"],
    ["/v1/messages", "stable-body-nova", false, "/converse"],
    ["/v1/messages", "stable-body-nova", true, "/converse-stream"],
    ["/v1/messages", "stable-body-claude", false, "/invoke"],
    ["/v1/responses", "stable-body-nova", false, "/converse"],
    ["/v1/responses", "stable-body-nova", true, "/converse-stream"],
    ["/v1/responses", "stable-body-claude", true, "/converse-stream"],
  ])("%s %s stream=%s", async ([endpoint, model, stream, upstreamPath], ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const before = received.length;
    for (let i = 0; i < REPEATS; i++) {
      const res = await fetch(`${app.proxyUrl}${endpoint}`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
          "content-type": "application/json",
        },
        body: JSON.stringify(requestBody(endpoint, model, stream)),
      });
      const text = await res.text();
      expect(res.status, text).toBe(200);
    }

    const sent = received.slice(before);
    expect(sent).toHaveLength(REPEATS);
    for (const s of sent) {
      expect(s.path.endsWith(upstreamPath), s.path).toBe(true);
    }
    // The schema and the replayed arguments both reached Bedrock.
    expect(sent[0].raw).toContain("max_results");
    expect(sent[0].raw).toContain("*.rs");
    expect(new Set(sent.map((s) => s.raw)).size).toBe(1);
  });
});
