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

// E2E for AISIX-Cloud#936: OpenAI's reasoning models (o-series, gpt-5 and
// later) reject `max_tokens` on Chat Completions and take the cap as
// `max_completion_tokens`. A caller that sends the long-standing
// `max_tokens` to one of them through an `openai` or `azure-openai` key
// must reach the upstream with `max_completion_tokens` instead — from every
// inbound protocol that ends in an upstream Chat Completions call — while
// every other parameter, and every non-reasoning model, is forwarded as
// sent. An operator's own `param_renames` on the key still decides last.

const CALLER_PLAINTEXT = "sk-reasoning-token-cap-936";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const auth = { authorization: `Bearer ${CALLER_PLAINTEXT}` };

const CHAT_BODY = {
  id: "chatcmpl-936",
  object: "chat.completion",
  created: 1,
  model: "upstream",
  choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
  usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
};
const CHAT_STREAM = [
  JSON.stringify({
    id: "chatcmpl-936",
    object: "chat.completion.chunk",
    model: "upstream",
    choices: [{ index: 0, delta: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
    usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
  }),
  "[DONE]",
];

type Key =
  | "openai"
  | "openai-stream"
  | "openai-bridged-responses"
  | "openai-operator-rename"
  | "openai-default-cap"
  | "azure"
  | "azure-default-cap";
type Inbound = "chat" | "chat-stream" | "messages" | "responses";

interface Case {
  label: string;
  key: Key;
  inbound: Inbound;
  /** The upstream model name — for Azure, the deployment name. */
  upstreamModel: string;
  /** Extra caller fields; `chat` and `messages` send `max_tokens: 50`. */
  extra?: Record<string, unknown>;
  /** The token-cap fields the upstream must receive, and nothing else of the two. */
  want: { max_tokens?: number; max_completion_tokens?: number };
}

const cases: Case[] = [
  { label: "openai o-series", key: "openai", inbound: "chat", upstreamModel: "o3-mini", want: { max_completion_tokens: 50 } },
  { label: "openai gpt-5", key: "openai", inbound: "chat", upstreamModel: "gpt-5-mini", want: { max_completion_tokens: 50 } },
  {
    label: "openai gpt-5, both caps sent",
    key: "openai",
    inbound: "chat",
    upstreamModel: "gpt-5-nano",
    extra: { max_completion_tokens: 80 },
    want: { max_completion_tokens: 80 },
  },
  { label: "openai gpt-5-chat", key: "openai", inbound: "chat", upstreamModel: "gpt-5-chat-latest", want: { max_tokens: 50 } },
  { label: "openai gpt-4o", key: "openai", inbound: "chat", upstreamModel: "gpt-4o", want: { max_tokens: 50 } },
  { label: "openai streamed", key: "openai-stream", inbound: "chat-stream", upstreamModel: "gpt-5", want: { max_completion_tokens: 50 } },
  { label: "openai via /v1/messages", key: "openai", inbound: "messages", upstreamModel: "o1", want: { max_completion_tokens: 50 } },
  {
    label: "openai via /v1/responses",
    key: "openai-bridged-responses",
    inbound: "responses",
    upstreamModel: "gpt-5.1",
    want: { max_completion_tokens: 40 },
  },
  {
    label: "openai, operator renames the cap back",
    key: "openai-operator-rename",
    inbound: "chat",
    upstreamModel: "o3",
    want: { max_tokens: 50 },
  },
  { label: "azure o-series deployment", key: "azure", inbound: "chat", upstreamModel: "prod-o3-deploy", want: { max_completion_tokens: 50 } },
  { label: "azure gpt-5-chat", key: "azure", inbound: "chat", upstreamModel: "gpt-5-chat", want: { max_completion_tokens: 50 } },
  { label: "azure gpt-4o deployment", key: "azure", inbound: "chat", upstreamModel: "gpt-4o-prod", want: { max_tokens: 50 } },
  // A key's `default_body_fields` `max_tokens` fills in behind the caller: it
  // must not come back beside a converted cap, and it is converted itself.
  {
    label: "openai, key default max_tokens, caller sent max_tokens",
    key: "openai-default-cap",
    inbound: "chat",
    upstreamModel: "o4-mini",
    want: { max_completion_tokens: 50 },
  },
  {
    label: "openai, key default max_tokens, caller sent no cap",
    key: "openai-default-cap",
    inbound: "chat",
    upstreamModel: "gpt-5-pro",
    extra: { max_tokens: undefined },
    want: { max_completion_tokens: 64 },
  },
  {
    label: "openai non-reasoning, key default max_tokens, caller sent no cap",
    key: "openai-default-cap",
    inbound: "chat",
    upstreamModel: "gpt-4o-mini",
    extra: { max_tokens: undefined },
    want: { max_tokens: 64 },
  },
  {
    label: "azure, key default max_tokens, caller sent max_tokens",
    key: "azure-default-cap",
    inbound: "chat",
    upstreamModel: "o4-default-deploy",
    want: { max_completion_tokens: 50 },
  },
  {
    label: "azure, key default max_tokens, caller sent no cap",
    key: "azure-default-cap",
    inbound: "chat",
    upstreamModel: "gpt-5-default-deploy",
    extra: { max_tokens: undefined },
    want: { max_completion_tokens: 64 },
  },
];

describe("reasoning models receive max_completion_tokens (AISIX-Cloud#936)", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let plain: OpenAiUpstream | undefined;
  let sse: OpenAiUpstream | undefined;

  const post = (path: string, body: Record<string, unknown>): Promise<Response> =>
    fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: { ...auth, "content-type": "application/json" },
      body: JSON.stringify(body),
    });

  function call(c: Case, model: string): Promise<Response> {
    const messages = [{ role: "user", content: "hi" }];
    switch (c.inbound) {
      case "chat":
        return post("/v1/chat/completions", { model, messages, max_tokens: 50, temperature: 0.5, ...c.extra });
      case "chat-stream":
        return post("/v1/chat/completions", { model, messages, stream: true, max_tokens: 50, temperature: 0.5 });
      case "messages":
        return post("/v1/messages", { model, messages, max_tokens: 50, temperature: 0.5 });
      case "responses":
        return post("/v1/responses", { model, input: "hi", max_output_tokens: 40, temperature: 0.5 });
    }
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    plain = await startOpenAiUpstream({ nonStreamBody: CHAT_BODY });
    sse = await startOpenAiUpstream({ streamEvents: CHAT_STREAM });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const keys: Record<Key, string> = {
      openai: (await seed.createProviderKey({ display_name: "k936-openai", secret: "sk-mock", api_base: `${plain.baseUrl}/v1` })).id,
      "openai-stream": (
        await seed.createProviderKey({ display_name: "k936-openai-stream", secret: "sk-mock", api_base: `${sse.baseUrl}/v1` })
      ).id,
      // No native Responses surface, so /v1/responses is bridged to Chat Completions.
      "openai-bridged-responses": (
        await seed.createProviderKey({
          display_name: "k936-openai-bridged",
          secret: "sk-mock",
          api_base: `${plain.baseUrl}/v1`,
          apis: {},
        })
      ).id,
      "openai-operator-rename": (
        await seed.createProviderKey({
          display_name: "k936-openai-rename",
          secret: "sk-mock",
          api_base: `${plain.baseUrl}/v1`,
          request: { param_renames: { max_completion_tokens: "max_tokens" } },
        })
      ).id,
      "openai-default-cap": (
        await seed.createProviderKey({
          display_name: "k936-openai-default-cap",
          secret: "sk-mock",
          api_base: `${plain.baseUrl}/v1`,
          request: { default_body_fields: { max_tokens: 64 } },
        })
      ).id,
      "azure-default-cap": (
        await seed.createProviderKey({
          display_name: "k936-azure-default-cap",
          secret: "azure-mock",
          api_base: plain.baseUrl,
          provider: "azure",
          adapter: "azure-openai",
          request: { default_body_fields: { max_tokens: 64 } },
        })
      ).id,
      azure: (
        await seed.createProviderKey({
          display_name: "k936-azure",
          secret: "azure-mock",
          api_base: plain.baseUrl,
          provider: "azure",
          adapter: "azure-openai",
        })
      ).id,
    };
    for (const c of cases) {
      await seed.createModel({
        display_name: `m936-${c.upstreamModel}`,
        provider: c.key.startsWith("azure") ? "azure" : "openai",
        model_name: c.upstreamModel,
        provider_key_id: keys[c.key],
      });
    }
    // Seeded last: the key authenticating implies the whole set is live.
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, { headers: auth });
      await res.arrayBuffer();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await plain?.close();
    await sse?.close();
  });

  test.for(cases)("$label", { timeout: 30_000 }, async (c, ctx) => {
    if (!etcdReachable || !app || !plain || !sse) return ctx.skip();
    const res = await call(c, `m936-${c.upstreamModel}`);
    const text = await res.text();
    expect(res.status, `${c.label}: ${text}`).toBe(200);

    const upstream = c.key === "openai-stream" ? sse : plain;
    const sent = upstream.receivedRequests
      .filter((r) => r.path.includes("/chat/completions"))
      .map((r) => JSON.parse(r.body) as Record<string, unknown>)
      .filter((b) => b.model === c.upstreamModel);
    expect(sent, `${c.label}: upstream chat requests for ${c.upstreamModel}`).toHaveLength(1);
    const body = sent[0];
    expect(
      { max_tokens: body.max_tokens, max_completion_tokens: body.max_completion_tokens },
      `${c.label}: token cap fields of ${JSON.stringify(body)}`,
    ).toEqual({ max_tokens: undefined, max_completion_tokens: undefined, ...c.want });
    // Nothing else about the request changes.
    expect(body.temperature, `${c.label}: temperature`).toBe(0.5);
  });
});
