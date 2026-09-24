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

// E2E: a Cohere Provider Key serves chat, embeddings and rerank whichever
// of its two documented base URLs it was configured with — the bare host
// (`https://api.cohere.com`) or Cohere's OpenAI-compatibility base
// (`https://api.cohere.ai/compatibility/v1`). Chat and embeddings belong on
// Cohere's OpenAI-compatible surface under `/compatibility/v1`; rerank has
// no counterpart there and belongs on the native `/v2/rerank`.
//
// References: <https://docs.cohere.com/docs/compatibility-api>,
// <https://docs.cohere.com/reference/rerank>.

const CALLER_PLAINTEXT = "sk-cohere-upstream-urls";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const auth = { authorization: `Bearer ${CALLER_PLAINTEXT}`, "content-type": "application/json" };

// One body each route's response handling accepts.
const BODY = {
  id: "cohere-mock",
  object: "chat.completion",
  created: 1,
  model: "cohere-mock",
  choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
  data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2] }],
  results: [{ index: 0, relevance_score: 0.9 }],
  meta: { billed_units: { search_units: 1 } },
  usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
};

type BaseForm = "bare host" | "compatibility base";
const forms: BaseForm[] = ["bare host", "compatibility base"];

const routes = [
  {
    route: "chat",
    path: "/v1/chat/completions",
    upstreamPath: "/compatibility/v1/chat/completions",
    model: "command-a-03-2025",
    body: (model: string) => ({ model, messages: [{ role: "user", content: "hi" }] }),
  },
  {
    route: "embeddings",
    path: "/v1/embeddings",
    upstreamPath: "/compatibility/v1/embeddings",
    model: "embed-v4.0",
    body: (model: string) => ({ model, input: "hi" }),
  },
  {
    route: "rerank",
    path: "/v1/rerank",
    upstreamPath: "/v2/rerank",
    model: "rerank-v3.5",
    body: (model: string) => ({ model, query: "q", documents: ["a", "b"] }),
  },
] as const;

const cases = forms.flatMap((form) => routes.map((r) => ({ ...r, form, label: `${r.route}, ${form}` })));

describe("Cohere Provider Key upstream URLs", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  const upstreams = new Map<BaseForm, OpenAiUpstream>();

  const alias = (c: (typeof cases)[number]) => `cohere-${c.route}-${c.form.replace(" ", "-")}`;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    for (const form of forms) {
      const upstream = await startOpenAiUpstream({ nonStreamBody: BODY });
      upstreams.set(form, upstream);
      const pk = await seed.createProviderKey({
        display_name: `cohere-${form}`,
        secret: "cohere-mock-key",
        provider: "cohere",
        adapter: "openai",
        api_base: form === "bare host" ? upstream.baseUrl : `${upstream.baseUrl}/compatibility/v1`,
      });
      for (const c of cases.filter((c) => c.form === form)) {
        await seed.createModel({
          display_name: alias(c),
          provider: "cohere",
          model_name: c.model,
          provider_key_id: pk.id,
        });
      }
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
    for (const u of upstreams.values()) await u.close();
  });

  test.for(cases)("$label", { timeout: 30_000 }, async (c, ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const upstream = upstreams.get(c.form)!;
    const before = upstream.receivedRequests.length;
    const res = await fetch(`${app.proxyUrl}${c.path}`, {
      method: "POST",
      headers: auth,
      body: JSON.stringify(c.body(alias(c))),
    });
    const text = await res.text();
    expect(res.status, `${c.label}: ${text}`).toBe(200);

    const sent = upstream.receivedRequests.slice(before);
    expect(
      sent.map((r) => r.path),
      `${c.label}: the upstream paths requested`,
    ).toEqual([c.upstreamPath]);
    expect(sent[0].headers.authorization).toBe("Bearer cohere-mock-key");
    expect((JSON.parse(sent[0].body) as { model?: string }).model).toBe(c.model);
  });
});
