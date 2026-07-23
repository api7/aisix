import { createHash } from "node:crypto";
import OpenAI from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: cross-provider translation. The caller speaks OpenAI Chat
// Completions to the gateway; the gateway speaks Anthropic Messages
// to the upstream. The Anthropic-shaped response (`type: "message"`,
// `content: [{type:"text",text}]`, `stop_reason`,
// `usage: {input_tokens, output_tokens}`) must come back to the SDK
// caller as an OpenAI shape (`object: "chat.completion"`,
// `choices[0].message.content`, `choices[0].finish_reason`,
// `usage: {prompt_tokens, completion_tokens, total_tokens}`).
//
// The mock-upstream harness's body is path-agnostic so
// `startOpenAiUpstream` doubles as the Anthropic mock when fed an
// Anthropic-shaped `nonStreamBody`. The Anthropic bridge appends
// `/v1/messages` to the api_base on its own, so we still reach the
// mock — `receivedRequests` confirms it.
//
// Reference:
// - OpenAI Chat Completions API spec:
//   <https://platform.openai.com/docs/api-reference/chat/create>
// - Anthropic Messages API spec:
//   <https://docs.anthropic.com/en/api/messages>

const CALLER_PLAINTEXT = "sk-an-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("anthropic upstream e2e: OpenAI in, Anthropic out, OpenAI back to caller", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "msg_01",
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: "Hello from Claude!" }],
        model: "claude-3-5-haiku-20241022",
        stop_reason: "end_turn",
        usage: { input_tokens: 5, output_tokens: 4 },
      },
    });
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    // The Anthropic bridge appends `/v1/messages` to the api_base, so
    // we point it at the bare host (no `/v1` suffix) — the OpenAI
    // bridge convention is the opposite (host + `/v1`), and getting
    // these mixed up was the lesson from the unit-level matrix tests.
    const pk = await seed.createProviderKey({
      display_name: "an-e2e-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-mock",
      api_base: upstream.baseUrl,
    });
    await seed.createModel({
      display_name: "an-e2e",
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["an-e2e"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("OpenAI client + Anthropic upstream round-trip translates wire shape both ways", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    await waitConfigPropagation(async () => {
      try {
        const r = await client.chat.completions.create({
          model: "an-e2e",
          messages: [{ role: "user", content: "ready-probe" }],
        });
        return r.choices[0]?.message.role === "assistant";
      } catch {
        return false;
      }
    });

    // Baseline-isolate the readiness probe so the request-shape
    // assertions below match the test call's request, not the probe's.
    // The probe also targets `an-e2e` and so also lands on
    // `/v1/messages`; without slicing from baseline, `find()` over
    // `receivedRequests` would return the probe and the assertions
    // would verify the wrong request — a regression that broke ONLY
    // the test call's translation would slip through.
    const baseline = upstream.receivedRequests.length;

    const completion = await client.chat.completions.create({
      model: "an-e2e",
      messages: [
        { role: "system", content: "you are terse" },
        { role: "user", content: "hi" },
      ],
    });

    // Response wire shape: must be OpenAI-shaped on the way out.
    expect(completion.object).toBe("chat.completion");
    expect(completion.choices[0]?.message.role).toBe("assistant");
    expect(completion.choices[0]?.message.content).toBe("Hello from Claude!");
    // Anthropic stop_reason "end_turn" must translate to OpenAI
    // finish_reason "stop". A regression that left "end_turn" through
    // would break every OpenAI-compatible client downstream.
    expect(completion.choices[0]?.finish_reason).toBe("stop");
    // Anthropic input_tokens / output_tokens → OpenAI prompt_tokens /
    // completion_tokens / total_tokens.
    expect(completion.usage?.prompt_tokens).toBe(5);
    expect(completion.usage?.completion_tokens).toBe(4);
    expect(completion.usage?.total_tokens).toBe(9);

    // Confirm the gateway hit /v1/messages on the test call (slice
    // from baseline so the probe's earlier hit cannot stand in).
    const messagesReq = upstream.receivedRequests
      .slice(baseline)
      .find((r) => r.path.startsWith("/v1/messages"));
    expect(messagesReq).toBeDefined();

    // Wire-shape on the request side: the gateway must speak Anthropic
    // Messages, not OpenAI Chat Completions. Without these assertions
    // the mock would happily 200 even on an OpenAI-shaped body, so
    // the response-side translation could pass while the request-side
    // translation was completely broken (the regression that prompted
    // this audit pass).
    const sentBody = JSON.parse(messagesReq!.body) as {
      model?: string;
      max_tokens?: unknown;
      system?: unknown;
      messages?: Array<{ role?: string; content?: unknown }>;
    };
    // model_name from the Model resource (Anthropic's id), not the
    // gateway's display_name "an-e2e".
    expect(sentBody.model).toBe("claude-3-5-haiku-20241022");
    // A plain-string system message must go out in Anthropic's STRING
    // `system` form — byte-identical to what the gateway has always
    // sent. The block-array form is reserved for callers that sent
    // typed blocks themselves (covered by the cache_control scenario
    // below); flipping plain-text callers to array form would change
    // the wire bytes for every existing caller.
    expect(sentBody.system).toBe("you are terse");
    // Anthropic's API requires `max_tokens`; OpenAI's doesn't. The
    // gateway must inject a value when crossing the boundary.
    expect(typeof sentBody.max_tokens).toBe("number");
    // Caller's user message must reach the upstream, role + content
    // intact — pin to the test call's content (`"hi"`), not the probe's
    // (`"ready-probe"`). This doubles as a baseline-isolation check.
    expect(sentBody.messages?.[0]?.role).toBe("user");
    // Anthropic API serialises content as an array of typed blocks
    // (`[{type:"text", text:"hi"}]`), not a bare string — gateway
    // converts on the way out. The OpenAI-side input was a string,
    // so a regression that dropped the content (or sent the wrong
    // text) would no longer match.
    expect(sentBody.messages?.[0]?.content).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: "text", text: "hi" }),
      ]),
    );

    // Auth shape: Anthropic uses `x-api-key` + `anthropic-version`,
    // not `Authorization: Bearer`. A regression that forwarded the
    // OpenAI auth shape to the Anthropic upstream would 401 in
    // production but pass against the permissive mock here without
    // these explicit header assertions.
    expect(messagesReq!.headers["x-api-key"]).toBe("sk-ant-mock");
    expect(messagesReq!.headers["anthropic-version"]).toBeDefined();
  });
});

// E2E (#395): an Anthropic upstream that returns ONLY a `tool_use` block
// (no text block) must surface `choices[0].message.content === null` on
// the OpenAI shape — not `""`. The text-block join produced `""` before
// the fix; we now map empty/absent text to `null`, per the documented
// `string | null` content shape.
const TOOL_CALLER_PLAINTEXT = "sk-an-e2e-toolnull-caller";
const TOOL_CALLER_KEY_HASH = createHash("sha256")
  .update(TOOL_CALLER_PLAINTEXT)
  .digest("hex");

describe("anthropic upstream e2e: tool_use-only response surfaces content:null (#395)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "msg_toolonly_01",
        type: "message",
        role: "assistant",
        // Only a tool_use block, no text block — the bug-class case.
        content: [
          {
            type: "tool_use",
            id: "toolu_abc",
            name: "get_weather",
            input: { location: "SF" },
          },
        ],
        model: "claude-3-5-haiku-20241022",
        stop_reason: "tool_use",
        usage: { input_tokens: 7, output_tokens: 3 },
      },
    });
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "an-e2e-toolnull-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-mock",
      api_base: upstream.baseUrl,
    });
    await seed.createModel({
      display_name: "an-e2e-toolnull",
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: TOOL_CALLER_KEY_HASH,
      allowed_models: ["an-e2e-toolnull"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("tool_use-only Anthropic response → content is exactly null on OpenAI shape", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: TOOL_CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    await waitConfigPropagation(async () => {
      try {
        const r = await client.chat.completions.create({
          model: "an-e2e-toolnull",
          messages: [{ role: "user", content: "ready-probe" }],
        });
        return r.choices[0]?.message.role === "assistant";
      } catch {
        return false;
      }
    });

    const completion = await client.chat.completions.create({
      model: "an-e2e-toolnull",
      messages: [{ role: "user", content: "weather in SF?" }],
    });

    expect(completion.choices[0]?.finish_reason).toBe("tool_calls");
    // The tool_use block translates to OpenAI tool_calls.
    expect(completion.choices[0]?.message.tool_calls?.[0]?.function.name).toBe(
      "get_weather",
    );
    // The fix: content must be exactly null, not "".
    expect(completion.choices[0]?.message.content).toBeNull();
  });
});

// E2E (#906): an Anthropic upstream reports cache_creation_input_tokens /
// cache_read_input_tokens as input classes SEPARATE from input_tokens
// (Anthropic's total input = input + cache_creation + cache_read). The
// OpenAI-shape `total_tokens` the caller sees must fold those in — pre-fix
// it was input + output only, under-counting every cached request.
const CACHE_CALLER_PLAINTEXT = "sk-an-e2e-cachetotal-caller";
const CACHE_CALLER_KEY_HASH = createHash("sha256")
  .update(CACHE_CALLER_PLAINTEXT)
  .digest("hex");

describe("anthropic upstream e2e: cache tokens fold into total_tokens (#906)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "msg_cache_01",
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: "cached hello" }],
        model: "claude-3-5-haiku-20241022",
        stop_reason: "end_turn",
        usage: {
          input_tokens: 10,
          output_tokens: 4,
          cache_creation_input_tokens: 200,
          cache_read_input_tokens: 800,
        },
      },
    });
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "an-e2e-cachetotal-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-mock",
      api_base: upstream.baseUrl,
    });
    await seed.createModel({
      display_name: "an-e2e-cachetotal",
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CACHE_CALLER_KEY_HASH,
      allowed_models: ["an-e2e-cachetotal"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("cache_creation + cache_read fold into total_tokens on the OpenAI shape", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CACHE_CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    await waitConfigPropagation(async () => {
      try {
        const r = await client.chat.completions.create({
          model: "an-e2e-cachetotal",
          messages: [{ role: "user", content: "ready-probe" }],
        });
        return r.choices[0]?.message.role === "assistant";
      } catch {
        return false;
      }
    });

    const completion = await client.chat.completions.create({
      model: "an-e2e-cachetotal",
      messages: [{ role: "user", content: "hi" }],
    });

    // prompt_tokens stays the non-cached input (Option A: cache is NOT
    // folded into prompt_tokens), but total_tokens is the honest sum of
    // every input class + completion: 10 + 4 + 200 + 800 = 1014.
    expect(completion.usage?.prompt_tokens).toBe(10);
    expect(completion.usage?.completion_tokens).toBe(4);
    expect(completion.usage?.total_tokens).toBe(10 + 4 + 200 + 800);
  });
});

// E2E (AISIX-Cloud#1110 Gap A): a caller managing its own provider-side
// prompt caching attaches `cache_control` markers to content blocks (and
// tool definitions) in the OpenAI shape. Those markers MUST reach the
// Anthropic upstream — a translation that flattens blocks to
// concatenated text silently strips them, the upstream caches nothing,
// and the caller pays full input price every turn while believing its
// caching strategy is active. Marker shape per
// <https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching>.
const CC_CALLER_PLAINTEXT = "sk-an-e2e-cachectl-caller";
const CC_CALLER_KEY_HASH = createHash("sha256")
  .update(CC_CALLER_PLAINTEXT)
  .digest("hex");

describe("anthropic upstream e2e: client cache_control markers survive translation", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "msg_cc_01",
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: "ok" }],
        model: "claude-3-5-haiku-20241022",
        stop_reason: "end_turn",
        usage: { input_tokens: 5, output_tokens: 1 },
      },
    });
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "an-e2e-cachectl-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-mock",
      api_base: upstream.baseUrl,
    });
    await seed.createModel({
      display_name: "an-e2e-cachectl",
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CC_CALLER_KEY_HASH,
      allowed_models: ["an-e2e-cachectl"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("system/message/tool cache_control markers reach the upstream body", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const proxy = new ProxyClient(app.proxyUrl, CC_CALLER_PLAINTEXT);

    await waitConfigPropagation(async () => {
      const r = await proxy.chat({
        model: "an-e2e-cachectl",
        messages: [{ role: "user", content: "ready-probe" }],
      });
      return r.status === 200;
    });

    const baseline = upstream.receivedRequests.length;

    const { status } = await proxy.chat({
      model: "an-e2e-cachectl",
      messages: [
        {
          role: "system",
          content: [
            {
              type: "text",
              text: "big stable system prefix",
              cache_control: { type: "ephemeral" },
            },
          ],
        },
        {
          role: "user",
          content: [
            { type: "text", text: "conversation history" },
            {
              type: "text",
              text: "first question",
              cache_control: { type: "ephemeral" },
            },
          ],
        },
        {
          role: "assistant",
          content: [
            {
              type: "text",
              text: "prior answer",
              cache_control: { type: "ephemeral" },
            },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "text",
              text: "latest question",
              // Stray per-part metadata (an OpenAI streaming index
              // replayed from assembled history) must NOT reach the
              // strict Anthropic validator.
              index: 0,
              cache_control: { type: "ephemeral", ttl: "1h" },
            },
          ],
        },
      ],
      tools: [
        {
          type: "function",
          function: {
            name: "get_weather",
            parameters: { type: "object", properties: {} },
          },
          cache_control: { type: "ephemeral" },
        },
      ],
    });
    expect(status).toBe(200);

    const messagesReq = upstream.receivedRequests
      .slice(baseline)
      .find((r) => r.path.startsWith("/v1/messages"));
    expect(messagesReq).toBeDefined();
    const sentBody = JSON.parse(messagesReq!.body) as {
      system?: unknown;
      messages?: Array<{ role?: string; content?: unknown }>;
      tools?: Array<Record<string, unknown>>;
    };

    // System marker: the block-form system prompt must go out as
    // Anthropic's block-array `system`, marker intact — not flattened
    // to a plain string.
    expect(sentBody.system).toEqual([
      {
        type: "text",
        text: "big stable system prefix",
        cache_control: { type: "ephemeral" },
      },
    ]);

    // Message markers on EVERY turn, not just the first: per-block
    // structure preserved (2 blocks stay 2 blocks — a marker means
    // "cache through here", so its position is the payload), markers on
    // user AND replayed-assistant turns, TTL intact, and the stray
    // `index` field stripped before the strict upstream validator.
    expect(sentBody.messages).toHaveLength(3);
    expect(sentBody.messages?.[0]?.role).toBe("user");
    expect(sentBody.messages?.[0]?.content).toEqual([
      { type: "text", text: "conversation history" },
      {
        type: "text",
        text: "first question",
        cache_control: { type: "ephemeral" },
      },
    ]);
    expect(sentBody.messages?.[1]?.role).toBe("assistant");
    expect(sentBody.messages?.[1]?.content).toEqual([
      {
        type: "text",
        text: "prior answer",
        cache_control: { type: "ephemeral" },
      },
    ]);
    expect(sentBody.messages?.[2]?.role).toBe("user");
    expect(sentBody.messages?.[2]?.content).toEqual([
      {
        type: "text",
        text: "latest question",
        cache_control: { type: "ephemeral", ttl: "1h" },
      },
    ]);

    // Tool translation, exact shape: Anthropic's `{name, input_schema}`
    // — no OpenAI `type`/`function` wrapper riding along (the strict
    // upstream rejects unknown tool fields), `input_schema` translated
    // from `parameters`, marker intact. Tool definitions sit first in
    // the prompt-cache prefix hierarchy.
    expect(sentBody.tools).toEqual([
      {
        name: "get_weather",
        input_schema: { type: "object", properties: {} },
        cache_control: { type: "ephemeral" },
      },
    ]);
  });
});
