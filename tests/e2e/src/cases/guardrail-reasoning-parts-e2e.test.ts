import { createHash } from "node:crypto";
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

// E2E: an assistant turn's replayed reasoning can arrive as a content part
// instead of the `reasoning_content` field — Mistral's
// `{type:"thinking", thinking:[{type:"text", text}], signature?}` and
// Databricks' `{type:"reasoning", summary:[{type:"summary_text", text,
// signature?}]}`. Input guardrails must treat it exactly as they treat
// `reasoning_content`:
//
//   - a block rule fires on text in either part, signed or not;
//   - a mask rule rewrites an unsigned part before dispatch, and forwards a
//     signed one byte-identical (rewriting signed reasoning would break the
//     signature the upstream verifies);
//   - a passthrough route scans both parts too.

const CALLER = "sk-reasoning-parts-e2e";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const MODEL = "reasoning-parts-model";
const BLOCKWORD = "reasoningpartforbidden";
const LATEST_TURN_BLOCKWORD = "latestturnreasoningforbidden";
const EMAIL = "carol@example.com";
const MASKED = "[EMAIL_REDACTED]";

const mistralPart = (text: string, signature?: string) => ({
  type: "thinking",
  thinking: [{ type: "text", text }],
  ...(signature === undefined ? {} : { signature }),
});

const databricksPart = (...entries: Array<{ text: string; signature: string }>) => ({
  type: "reasoning",
  summary: entries.map((e) => ({ type: "summary_text", ...e })),
});

const historyWith = (parts: unknown[]) => [
  { role: "user", content: "What is the weather in Paris?" },
  { role: "assistant", content: [...parts, { type: "text", text: "Let me check." }] },
  { role: "user", content: "Go on." },
];

describe("input guardrails read replayed reasoning content parts", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let proxy: ProxyClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "reasoning-parts-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createPassthroughRoute({
      name: "reasoning-parts-tunnel",
      path_prefix: "/reasoning-parts",
      target_url: `${upstream.baseUrl}/v1`,
      provider_key_id: pk.id,
    });
    await seed.createGuardrail({
      name: "reasoning-parts-block",
      enabled: true,
      hook_point: "input",
      kind: "keyword",
      patterns: [{ kind: "literal", value: BLOCKWORD }],
    });
    // Reads only the turn the model has not answered yet — which includes a
    // trailing assistant prefill.
    await seed.createGuardrail({
      name: "reasoning-parts-latest-turn-block",
      enabled: true,
      hook_point: "input",
      kind: "keyword",
      input_messages: "latest_turn",
      patterns: [{ kind: "literal", value: LATEST_TURN_BLOCKWORD }],
    });
    await seed.createGuardrail({
      name: "reasoning-parts-mask",
      enabled: true,
      hook_point: "input",
      kind: "pii",
      detectors: [{ type: "email", action: "mask" }],
    });
    // Caller key last: authenticating with it implies every resource
    // above is in the gateway's snapshot.
    await seed.createApiKey({
      key_hash: CALLER_HASH,
      allowed_models: [MODEL],
      allowed_routes: ["*"],
    });

    proxy = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => (await proxy!.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  async function expectBlocked(parts: unknown[]): Promise<void> {
    const before = upstream!.receivedRequests.length;
    const res = await proxy!.chat({ model: MODEL, messages: historyWith(parts) });
    expect(res.status).toBe(422);
    expect((res.body as { error?: { type?: unknown } }).error?.type).toBe("content_filter");
    expect(JSON.stringify(res.body)).not.toContain(BLOCKWORD);
    expect(upstream!.receivedRequests.length).toBe(before);
  }

  async function dispatchedAssistantContent(parts: unknown[]): Promise<unknown[]> {
    const before = upstream!.receivedRequests.length;
    const res = await proxy!.chat({ model: MODEL, messages: historyWith(parts) });
    expect(res.status).toBe(200);
    expect(upstream!.receivedRequests.length).toBe(before + 1);
    const sent = JSON.parse(upstream!.receivedRequests[before]!.body) as {
      messages: Array<{ role: string; content: unknown }>;
    };
    const assistant = sent.messages.find((m) => m.role === "assistant");
    expect(Array.isArray(assistant?.content)).toBe(true);
    return assistant!.content as unknown[];
  }

  test("a block rule fires on a Mistral thinking part", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    await expectBlocked([mistralPart(`maybe ${BLOCKWORD} helps`)]);
  });

  test("a block rule fires on a signed Databricks reasoning part", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    await expectBlocked([
      databricksPart({ text: `maybe ${BLOCKWORD} helps`, signature: "sig-databricks" }),
    ]);
  });

  test("a mask rule rewrites an unsigned Mistral thinking part and forwards a signed one unchanged", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const signed = mistralPart(`signed note for ${EMAIL}`, "sig-mistral");
    const content = await dispatchedAssistantContent([
      mistralPart(`email ${EMAIL} later`),
      signed,
    ]);
    expect(content[0]).toEqual(mistralPart(`email ${MASKED} later`));
    expect(content[1]).toEqual(signed);
  });

  test("a mask rule rewrites an unsigned Databricks summary entry and forwards a signed one unchanged", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const content = await dispatchedAssistantContent([
      databricksPart(
        { text: `unsigned ${EMAIL}`, signature: "" },
        { text: `signed ${EMAIL}`, signature: "sig-databricks" },
      ),
    ]);
    expect(content[0]).toEqual(
      databricksPart(
        { text: `unsigned ${MASKED}`, signature: "" },
        { text: `signed ${EMAIL}`, signature: "sig-databricks" },
      ),
    );
  });

  test("a latest_turn block rule fires on a signed part in a trailing assistant prefill", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const before = upstream!.receivedRequests.length;
    const res = await proxy!.chat({
      model: MODEL,
      messages: [
        { role: "user", content: "What is the weather in Paris?" },
        {
          role: "assistant",
          content: [mistralPart(`maybe ${LATEST_TURN_BLOCKWORD}`, "sig-mistral")],
        },
      ],
    });
    expect(res.status).toBe(422);
    expect((res.body as { error?: { type?: unknown } }).error?.type).toBe("content_filter");
    expect(upstream!.receivedRequests.length).toBe(before);
  });

  async function passthrough(parts: unknown[]): Promise<Response> {
    return fetch(`${app!.proxyUrl}/reasoning-parts/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify({ model: "gpt-4o-mini", messages: historyWith(parts) }),
    });
  }

  test("a passthrough route scans both reasoning part shapes", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    for (const part of [
      mistralPart(`maybe ${BLOCKWORD}`),
      databricksPart({ text: `maybe ${BLOCKWORD}`, signature: "sig-databricks" }),
    ]) {
      const before = upstream!.receivedRequests.length;
      const res = await passthrough([part]);
      const body = (await res.json()) as { error?: { type?: unknown } };
      expect(res.status, JSON.stringify(part)).toBe(422);
      expect(body.error?.type).toBe("content_filter");
      expect(upstream!.receivedRequests.length).toBe(before);
    }
    // A clean Mistral part is relayed, not refused as unscannable.
    const clean = await passthrough([mistralPart("nothing to see")]);
    await clean.text();
    expect(clean.status).toBe(200);
  });
});
