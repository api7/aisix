import { execFile } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { createServer, type Server } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
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
import { pickFreePort } from "../harness/ports.js";

const execFileP = promisify(execFile);

// E2E: every place a document points at a Model may name it by resource
// id (`<field>_id`) instead of by display name. The id decides when
// present, resolves against whatever name the model currently carries,
// and — when it names no model — degrades exactly the way a dangling
// name already does at that site.
//
// The five sites, and what each case observes:
//   routing target          `x-aisix-served-by`
//   ensemble panel + judge  the upstream model each sub-call asked for
//   semantic router         `x-aisix-served-by` / `x-aisix-route`
//   cache policy scope      `x-aisix-cache`
//   semantic guardrail      the 422 a fail-closed screen produces
//
// Reference: OpenAI Chat Completions API spec
// (https://platform.openai.com/docs/api-reference/chat/create).

const CALLER = "sk-model-ref-ids-e2e";
/**
 * A resource id no model carries. Used as BOTH an id and a name in the
 * paired "names no model" cases, so the two spellings produce the same
 * reference string and their errors are comparable byte for byte.
 */
const DANGLING = "11111111-2222-3333-4444-555555555555";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");

/** Deterministic one-hot vectors, so every similarity is exactly 0 or 1. */
function keywordVector(text: string): number[] {
  const t = text.toLowerCase();
  if (t.includes("jailbreak")) return [1, 0, 0, 0];
  if (t.includes("contract")) return [0, 1, 0, 0];
  if (t.includes("weather")) return [0, 0, 1, 0];
  return [0, 0, 0, 1];
}

interface EmbeddingMock {
  baseUrl: string;
  close(): Promise<void>;
}

async function startEmbeddingMock(
  opts: { fail?: boolean } = {},
): Promise<EmbeddingMock> {
  const server: Server = createServer((req, res) => {
    res.on("error", () => {});
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      if (!req.url?.includes("/embeddings")) {
        res.statusCode = 404;
        res.end("{}");
        return;
      }
      if (opts.fail) {
        res.statusCode = 500;
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify({ error: { message: "embedding upstream down" } }));
        return;
      }
      let body: { input?: string | string[] };
      try {
        body = JSON.parse(raw || "{}") as { input?: string | string[] };
      } catch {
        res.statusCode = 400;
        res.end("{}");
        return;
      }
      const inputs = Array.isArray(body.input) ? body.input : [body.input ?? ""];
      res.statusCode = 200;
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          object: "list",
          model: "embed-mock",
          data: inputs.map((text, index) => ({
            object: "embedding",
            index,
            embedding: keywordVector(text),
          })),
          usage: { prompt_tokens: inputs.length, total_tokens: inputs.length },
        }),
      );
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

describe("model references by resource id", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let chatUp: OpenAiUpstream | undefined;
  let embedMock: EmbeddingMock | undefined;
  let failingEmbedMock: EmbeddingMock | undefined;
  let etcdReachable = false;

  /** Model display name → resource id, for every model this suite seeds. */
  const ids: Record<string, string> = {};
  /** Model display name → the upstream model name its provider sees. */
  const upstreamNames: Record<string, string> = {};
  let chatPkId = "";
  let embedPkId = "";

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    chatUp = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-mr",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "answered" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    });
    embedMock = await startEmbeddingMock();
    failingEmbedMock = await startEmbeddingMock({ fail: true });

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const chatPk = await seed.createProviderKey({
      display_name: "mr-chat-pk",
      secret: "sk-mock",
      api_base: `${chatUp.baseUrl}/v1`,
    });
    chatPkId = chatPk.id;
    const embedPk = await seed.createProviderKey({
      display_name: "mr-embed-pk",
      secret: "sk-mock",
      api_base: `${embedMock.baseUrl}/v1`,
    });
    const failPk = await seed.createProviderKey({
      display_name: "mr-embed-fail-pk",
      secret: "sk-mock",
      api_base: `${failingEmbedMock.baseUrl}/v1`,
    });

    // Each direct model asks its upstream for a DIFFERENT model name, so
    // the recorded request bodies say which target actually served —
    // the ensemble sub-calls carry no response header to read.
    const direct = async (name: string) => {
      const upstreamName = `up-${name}`;
      const created = await seed!.createModel({
        display_name: name,
        provider: "openai",
        model_name: upstreamName,
        provider_key_id: chatPkId,
      });
      ids[name] = created.id;
      upstreamNames[name] = upstreamName;
    };
    const embedding = async (name: string, pkId: string) => {
      const created = await seed!.createModel({
        display_name: name,
        provider: "openai",
        model_name: `embed-${name}`,
        provider_key_id: pkId,
        embedding: { dimensions: 4, normalize: true },
      });
      ids[name] = created.id;
    };

    for (const name of [
      "mr-alpha",
      "mr-beta",
      "mr-judge",
      "mr-panelist",
      "mr-rename-route",
      "mr-rename-ensemble",
      "mr-rename-semantic",
      "mr-cache-scoped",
      "mr-cache-other",
      "mr-cache-renamed",
      "mr-guarded",
      "mr-guarded-rename",
      "mr-sentinel",
    ]) {
      await direct(name);
    }
    await embedding("mr-embed", embedPk.id);
    await embedding("mr-embed-renamed", embedPk.id);
    await embedding("mr-embed-guard-renamed", embedPk.id);
    await embedding("mr-embed-fail", failPk.id);
    embedPkId = embedPk.id;

    // ---- routing groups ----
    await seed.createModel({
      display_name: "mr-group-by-id",
      routing: { strategy: "failover", targets: [{ model_id: ids["mr-alpha"] }] },
    });
    await seed.createModel({
      display_name: "mr-group-conflict",
      routing: {
        strategy: "failover",
        targets: [{ model: "mr-beta", model_id: ids["mr-alpha"] }],
      },
    });
    await seed.createModel({
      display_name: "mr-group-rename",
      routing: {
        strategy: "failover",
        targets: [{ model_id: ids["mr-rename-route"] }],
      },
    });
    await seed.createModel({
      display_name: "mr-group-dangling-id",
      routing: { strategy: "failover", targets: [{ model_id: DANGLING }] },
    });
    await seed.createModel({
      display_name: "mr-group-dangling-name",
      routing: { strategy: "failover", targets: [{ model: DANGLING }] },
    });

    // ---- ensembles (single-member panels: min_responses clamps to 1) ----
    await seed.createModel({
      display_name: "mr-ensemble-by-id",
      ensemble: {
        panel: [{ model_id: ids["mr-panelist"] }],
        judge: { model_id: ids["mr-judge"] },
      },
    });
    await seed.createModel({
      display_name: "mr-ensemble-conflict",
      ensemble: {
        panel: [{ model: "mr-beta", model_id: ids["mr-panelist"] }],
        judge: { model: "mr-beta", model_id: ids["mr-judge"] },
      },
    });
    await seed.createModel({
      display_name: "mr-ensemble-rename",
      ensemble: {
        panel: [{ model_id: ids["mr-panelist"] }],
        judge: { model_id: ids["mr-rename-ensemble"] },
      },
    });
    await seed.createModel({
      display_name: "mr-ensemble-dangling-id",
      ensemble: {
        panel: [{ model_id: ids["mr-panelist"] }],
        judge: { model_id: DANGLING },
      },
    });
    await seed.createModel({
      display_name: "mr-ensemble-dangling-name",
      ensemble: {
        panel: [{ model_id: ids["mr-panelist"] }],
        judge: { model: DANGLING },
      },
    });

    // ---- semantic routers ----
    await seed.createModel({
      display_name: "mr-semantic-by-id",
      semantic: {
        embedding_model_id: ids["mr-embed"],
        routes: [
          {
            name: "legal",
            target_id: ids["mr-alpha"],
            examples: ["analyze this contract"],
            threshold: 0.5,
          },
        ],
        default_id: ids["mr-beta"],
        match: { threshold: 0.5 },
      },
    });
    await seed.createModel({
      display_name: "mr-semantic-conflict",
      semantic: {
        embedding_model: "mr-embed-fail",
        embedding_model_id: ids["mr-embed"],
        routes: [
          {
            name: "legal",
            target: "mr-beta",
            target_id: ids["mr-alpha"],
            examples: ["analyze this contract"],
            threshold: 0.5,
          },
        ],
        default: "mr-alpha",
        default_id: ids["mr-beta"],
        match: { threshold: 0.5 },
      },
    });
    await seed.createModel({
      display_name: "mr-semantic-rename",
      semantic: {
        embedding_model_id: ids["mr-embed-renamed"],
        routes: [
          {
            name: "legal",
            target_id: ids["mr-rename-semantic"],
            examples: ["analyze this contract"],
            threshold: 0.5,
          },
        ],
        default_id: ids["mr-beta"],
        match: { threshold: 0.5 },
      },
    });
    // A dangling embedding model resolves to nothing, so the router
    // applies `on_embedding_failure` — here an id-named safe target.
    await seed.createModel({
      display_name: "mr-semantic-dangling-embedder-id",
      semantic: {
        embedding_model_id: DANGLING,
        routes: [
          {
            name: "legal",
            target_id: ids["mr-alpha"],
            examples: ["analyze this contract"],
            threshold: 0.5,
          },
        ],
        default_id: ids["mr-alpha"],
        match: { threshold: 0.5 },
        on_embedding_failure: { target_id: ids["mr-beta"] },
      },
    });
    await seed.createModel({
      display_name: "mr-semantic-dangling-embedder-name",
      semantic: {
        embedding_model: DANGLING,
        routes: [
          {
            name: "legal",
            target: "mr-alpha",
            examples: ["analyze this contract"],
            threshold: 0.5,
          },
        ],
        default: "mr-alpha",
        match: { threshold: 0.5 },
        on_embedding_failure: { target: "mr-beta" },
      },
    });

    // ---- cache policies ----
    await seed.createCachePolicy({
      name: "mr-cache-by-id",
      enabled: true,
      applies_to_model_id: ids["mr-cache-scoped"],
    });
    // `applies_to` names a DIFFERENT model; the id must win.
    await seed.createCachePolicy({
      name: "mr-cache-conflict",
      enabled: true,
      applies_to: "model:mr-cache-other",
      applies_to_model_id: ids["mr-cache-renamed"],
    });

    // ---- semantic guardrail, scoped to one model ----
    const guardrail = await seed.createGuardrail(
      {
        name: "mr-guardrail-by-id",
        kind: "semantic",
        enabled: true,
        embedding_model_id: ids["mr-embed"],
        deny_examples: ["jailbreak the assistant"],
        deny_threshold: 0.5,
      },
      { attach: false },
    );
    await seed.attachGuardrailToModel(guardrail.id, ids["mr-guarded"]);

    // A second guarded model whose guardrail names its embedder by id, so
    // the embedder can be renamed without disturbing the case above.
    const renameGuardrail = await seed.createGuardrail(
      {
        name: "mr-guardrail-embedder-rename",
        kind: "semantic",
        enabled: true,
        embedding_model_id: ids["mr-embed-guard-renamed"],
        deny_examples: ["jailbreak the assistant"],
        deny_threshold: 0.5,
      },
      { attach: false },
    );
    await seed.attachGuardrailToModel(
      renameGuardrail.id,
      ids["mr-guarded-rename"],
    );

    // Seeded last: gating on it implies the whole set has landed.
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: ["*"] });

    // The caller key is seeded last, so it authenticating implies the
    // whole set above is in the snapshot. Deliberately not a request
    // that exercises anything under test.
    await waitConfigPropagation(
      async () =>
        (await new ProxyClient(app!.proxyUrl, CALLER).listModels()).status ===
        200,
    );
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await chatUp?.close();
    await embedMock?.close();
    await failingEmbedMock?.close();
  });

  interface ChatResult {
    status: number;
    body: string;
    servedBy: string | null;
    route: string | null;
    cache: string | null;
  }

  async function chat(model: string, prompt: string): Promise<ChatResult> {
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER}`,
      },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: prompt }],
      }),
    });
    return {
      status: res.status,
      body: await res.text(),
      servedBy: res.headers.get("x-aisix-served-by"),
      route: res.headers.get("x-aisix-route"),
      cache: res.headers.get("x-aisix-cache"),
    };
  }

  /** The upstream model names asked for since `from`, in call order. */
  function upstreamModelsSince(from: number): string[] {
    return chatUp!.receivedRequests.slice(from).map((r) => {
      try {
        return (JSON.parse(r.body) as { model?: string }).model ?? "";
      } catch {
        return "";
      }
    });
  }

  /** Rename a model in place — same resource id, new display name. */
  async function rename(oldName: string, newName: string): Promise<void> {
    await seed!.update("models", ids[oldName], {
      display_name: newName,
      provider: "openai",
      model_name: upstreamNames[oldName],
      provider_key_id: chatPkId,
    });
    upstreamNames[newName] = upstreamNames[oldName];
    ids[newName] = ids[oldName];
    await waitConfigPropagation(async () => {
      const r = await chat(newName, "hello");
      return r.status === 200;
    });
  }

  // ───────────────────────── routing targets ─────────────────────────

  test("a routing target named by id dispatches to that model", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const r = await chat("mr-group-by-id", "hello");
    expect(r.status).toBe(200);
    expect(r.servedBy).toBe("mr-alpha");
  });

  test("routing target: the id decides and the name beside it is ignored", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const r = await chat("mr-group-conflict", "hello");
    expect(r.status).toBe(200);
    expect(r.servedBy).toBe("mr-alpha");
  });

  test("routing target: an id naming no model fails exactly as a name naming no model does", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const byId = await chat("mr-group-dangling-id", "hello");
    const byName = await chat("mr-group-dangling-name", "hello");
    expect(byId.status).toBe(byName.status);
    expect(byId.body).toBe(byName.body);
    // Not a 5xx and not a dropped row: the group still resolves as a
    // model, it just has no target to dispatch to.
    expect(byId.status).toBeLessThan(500);
  });

  // ───────────────────────── ensemble members ────────────────────────

  test("an ensemble panel member and judge named by id are the ones called", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const before = chatUp!.receivedRequests.length;
    const r = await chat("mr-ensemble-by-id", "hello");
    expect(r.status).toBe(200);
    const called = upstreamModelsSince(before);
    expect(called).toContain(upstreamNames["mr-panelist"]);
    expect(called).toContain(upstreamNames["mr-judge"]);
    expect(called).not.toContain(upstreamNames["mr-beta"]);
  });

  test("ensemble: the ids decide and the names beside them are ignored", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const before = chatUp!.receivedRequests.length;
    const r = await chat("mr-ensemble-conflict", "hello");
    expect(r.status).toBe(200);
    const called = upstreamModelsSince(before);
    expect(called).toContain(upstreamNames["mr-panelist"]);
    expect(called).toContain(upstreamNames["mr-judge"]);
    expect(called).not.toContain(upstreamNames["mr-beta"]);
  });

  test("ensemble: a judge id naming no model fails as a judge name naming no model does", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const byId = await chat("mr-ensemble-dangling-id", "hello");
    const byName = await chat("mr-ensemble-dangling-name", "hello");
    expect(byId.status).toBe(byName.status);
    expect(byId.body).toBe(byName.body);
    expect(byId.status).not.toBe(200);
  });

  // ───────────────────────── semantic routers ────────────────────────

  test("a semantic router named entirely by ids embeds, routes and falls through", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const matched = await chat("mr-semantic-by-id", "review this contract");
    expect(matched.status).toBe(200);
    expect(matched.route).toBe("legal");
    expect(matched.servedBy).toBe("mr-alpha");

    // Orthogonal prompt: no route clears its threshold → `default_id`.
    const fellThrough = await chat("mr-semantic-by-id", "hello there");
    expect(fellThrough.status).toBe(200);
    expect(fellThrough.route).toBeNull();
    expect(fellThrough.servedBy).toBe("mr-beta");
  });

  test("semantic: every id decides over the name beside it", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    // `embedding_model` names the FAILING embedder and `target` names
    // the wrong model; both are ignored, so the route still matches and
    // dispatches to the id-named target.
    const matched = await chat("mr-semantic-conflict", "review this contract");
    expect(matched.status).toBe(200);
    expect(matched.route).toBe("legal");
    expect(matched.servedBy).toBe("mr-alpha");
    // `default` names mr-alpha, `default_id` names mr-beta.
    const fellThrough = await chat("mr-semantic-conflict", "hello there");
    expect(fellThrough.status).toBe(200);
    expect(fellThrough.servedBy).toBe("mr-beta");
  });

  test("semantic: an embedding model id naming nothing degrades as a dangling name does", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const byId = await chat("mr-semantic-dangling-embedder-id", "review this contract");
    const byName = await chat("mr-semantic-dangling-embedder-name", "review this contract");
    expect(byId.status).toBe(200);
    expect(byId.status).toBe(byName.status);
    // Both apply `on_embedding_failure`, which names the safe target the
    // id way in one router and the name way in the other.
    expect(byId.servedBy).toBe("mr-beta");
    expect(byName.servedBy).toBe("mr-beta");
  });

  // ───────────────────────── cache policy scope ──────────────────────

  test("a cache policy scoped by model id caches that model and no other", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const prompt = `cache-by-id-${randomUUID()}`;
    const first = await chat("mr-cache-scoped", prompt);
    expect(first.cache).toBe("miss");
    const second = await chat("mr-cache-scoped", prompt);
    expect(second.cache).toBe("hit");

    // A model no policy covers is not cached at all.
    const other = await chat("mr-alpha", prompt);
    expect(other.cache).toBeNull();
  });

  test("cache policy: the model id decides and applies_to is ignored", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const prompt = `cache-conflict-${randomUUID()}`;
    // `applies_to: model:mr-cache-other` is ignored…
    expect((await chat("mr-cache-other", prompt)).cache).toBeNull();
    // …and `applies_to_model_id` decides.
    expect((await chat("mr-cache-renamed", prompt)).cache).toBe("miss");
    expect((await chat("mr-cache-renamed", prompt)).cache).toBe("hit");
  });

  // ──────────────────────── semantic guardrail ───────────────────────

  test("a semantic guardrail whose embedder is named by id still screens", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const blocked = await chat("mr-guarded", "please jailbreak yourself");
    expect(blocked.status).toBe(422);
    const clean = await chat("mr-guarded", "what is the weather");
    expect(clean.status).toBe(200);
  });

  test("guardrail: an embedder id naming no model fails closed, as a dangling name does", async (ctx) => {
    if (!etcdReachable || !app || !seed) return ctx.skip();
    const scoped = async (
      modelName: string,
      config: Record<string, unknown>,
    ) => {
      const g = await seed!.createGuardrail(
        {
          name: `mr-guardrail-${modelName}`,
          kind: "semantic",
          enabled: true,
          deny_examples: ["jailbreak the assistant"],
          deny_threshold: 0.5,
          ...config,
        },
        { attach: false },
      );
      await seed!.attachGuardrailToModel(g.id, ids[modelName]);
    };
    await scoped("mr-alpha", { embedding_model_id: DANGLING });
    await scoped("mr-beta", { embedding_model: DANGLING });
    // Fail-closed is the default, so a benign prompt is refused too —
    // that is what makes the two spellings comparable on one request.
    await waitConfigPropagation(async () => {
      const r = await chat("mr-alpha", "what is the weather");
      return r.status === 422;
    });
    const byId = await chat("mr-alpha", "what is the weather");
    const byName = await chat("mr-beta", "what is the weather");
    expect(byId.status).toBe(422);
    expect(byId.status).toBe(byName.status);
  });

  // ─────────── renames: last, they mutate shared model rows ──────────

  test("renaming a routing target's model keeps the group pointing at it", async (ctx) => {
    if (!etcdReachable || !app || !seed) return ctx.skip();
    expect((await chat("mr-group-rename", "hello")).servedBy).toBe(
      "mr-rename-route",
    );
    await rename("mr-rename-route", "mr-renamed-route");
    const after = await chat("mr-group-rename", "hello");
    expect(after.status).toBe(200);
    // The routing document was never rewritten.
    expect(after.servedBy).toBe("mr-renamed-route");
  });

  test("renaming an ensemble judge keeps the ensemble calling it", async (ctx) => {
    if (!etcdReachable || !app || !seed) return ctx.skip();
    expect((await chat("mr-ensemble-rename", "hello")).status).toBe(200);
    await rename("mr-rename-ensemble", "mr-renamed-ensemble");
    const before = chatUp!.receivedRequests.length;
    const after = await chat("mr-ensemble-rename", "hello");
    expect(after.status).toBe(200);
    expect(upstreamModelsSince(before)).toContain(
      upstreamNames["mr-renamed-ensemble"],
    );
  });

  test("renaming a semantic route target and its embedder keeps the router working", async (ctx) => {
    if (!etcdReachable || !app || !seed) return ctx.skip();
    expect(
      (await chat("mr-semantic-rename", "review this contract")).servedBy,
    ).toBe("mr-rename-semantic");

    await rename("mr-rename-semantic", "mr-renamed-semantic");
    // The embedding model is renamed through the raw document, since it
    // carries an `embedding` block the direct-model helper does not.
    await seed.update("models", ids["mr-embed-renamed"], {
      display_name: "mr-embed-renamed-v2",
      provider: "openai",
      model_name: "embed-mr-embed-renamed",
      provider_key_id: (await seed.createProviderKey({
        display_name: `mr-embed-pk-${randomUUID()}`,
        secret: "sk-mock",
        api_base: `${embedMock!.baseUrl}/v1`,
      })).id,
      embedding: { dimensions: 4, normalize: true },
    });

    await waitConfigPropagation(async () => {
      const r = await chat("mr-semantic-rename", "review this contract");
      return r.status === 200 && r.servedBy === "mr-renamed-semantic";
    });
    const after = await chat("mr-semantic-rename", "review this contract");
    expect(after.status).toBe(200);
    expect(after.route).toBe("legal");
    expect(after.servedBy).toBe("mr-renamed-semantic");
  });

  test("renaming a semantic guardrail's embedder keeps the row screening", async (ctx) => {
    if (!etcdReachable || !app || !seed) return ctx.skip();
    // Screening works before the rename…
    expect(
      (await chat("mr-guarded-rename", "please jailbreak yourself")).status,
    ).toBe(422);
    expect((await chat("mr-guarded-rename", "what is the weather")).status).toBe(
      200,
    );

    // …and only the EMBEDDING MODEL's row is rewritten across it —
    // neither the guardrail nor its attachment is touched. That is the
    // point of the case, not incidental setup: the guardrail chain is
    // cached and rebuilt only when the guardrail or attachment tables
    // change, so a build-time resolution would survive this rename and
    // the case would pass for the wrong reason. Do not "fix" a future
    // failure here by also rewriting the guardrail row.
    await seed.update("models", ids["mr-embed-guard-renamed"], {
      display_name: "mr-embed-guard-renamed-v2",
      provider: "openai",
      model_name: "embed-mr-embed-guard-renamed",
      provider_key_id: embedPkId,
      embedding: { dimensions: 4, normalize: true },
    });
    await waitConfigPropagation(async () => {
      const listed = await new ProxyClient(app!.proxyUrl, CALLER).listModels();
      if (listed.status !== 200) return false;
      const names = (listed.body as { data: { id: string }[] }).data.map(
        (m) => m.id,
      );
      return names.includes("mr-embed-guard-renamed-v2");
    });

    expect(
      (await chat("mr-guarded-rename", "please jailbreak yourself")).status,
    ).toBe(422);
    // Still screening rather than merely failing closed on everything —
    // an embedder that stopped resolving would refuse this one too.
    expect((await chat("mr-guarded-rename", "what is the weather")).status).toBe(
      200,
    );
  });

  test("renaming a cache policy's scoped model keeps the policy on it", async (ctx) => {
    if (!etcdReachable || !app || !seed) return ctx.skip();
    await rename("mr-cache-renamed", "mr-cache-renamed-v2");
    const prompt = `cache-rename-${randomUUID()}`;
    expect((await chat("mr-cache-renamed-v2", prompt)).cache).toBe("miss");
    expect((await chat("mr-cache-renamed-v2", prompt)).cache).toBe("hit");
  });
});

// The id spelling is a control-plane projection: a resources file derives
// its ids from its own entry names, so an id written there could never
// resolve. `aisix validate` refuses each of them by name rather than
// loading a file whose references silently point at nothing.
describe("resources file: every model-reference id spelling is refused", () => {
  const BIN_PATH =
    process.env.AISIX_BIN ??
    join(process.cwd(), "..", "..", "target", "debug", "aisix");

  const PRELUDE = [
    '_format_version: "1"',
    "provider_keys:",
    "  - display_name: pk",
    "    api_key: sk-x",
    "models:",
    "  - display_name: m",
    "    provider: openai",
    "    model_name: gpt-4o",
    "    provider_key: pk",
    "  - display_name: e",
    "    provider: openai",
    "    model_name: embed",
    "    provider_key: pk",
    "    embedding:",
    "      dimensions: 4",
  ].join("\n");

  const ID = "11111111-1111-1111-1111-111111111111";

  const CASES: { label: string; field: string; withId: string; without: string }[] =
    [
      {
        label: "routing target",
        field: "model_id",
        withId: `\n  - display_name: g\n    routing:\n      targets:\n        - model: m\n          model_id: ${ID}\n`,
        without: `\n  - display_name: g\n    routing:\n      targets:\n        - model: m\n`,
      },
      {
        label: "ensemble panel member",
        field: "model_id",
        withId: `\n  - display_name: p\n    ensemble:\n      panel:\n        - model: m\n          model_id: ${ID}\n      judge:\n        model: m\n`,
        without: `\n  - display_name: p\n    ensemble:\n      panel:\n        - model: m\n      judge:\n        model: m\n`,
      },
      {
        label: "ensemble judge",
        field: "model_id",
        withId: `\n  - display_name: j\n    ensemble:\n      panel:\n        - model: m\n      judge:\n        model: m\n        model_id: ${ID}\n`,
        without: `\n  - display_name: j\n    ensemble:\n      panel:\n        - model: m\n      judge:\n        model: m\n`,
      },
      {
        label: "semantic embedding model",
        field: "embedding_model_id",
        withId: `\n  - display_name: s\n    semantic:\n      embedding_model: e\n      embedding_model_id: ${ID}\n      routes:\n        - name: r\n          target: m\n          examples: ["hi"]\n      default: m\n      match:\n        threshold: 0.5\n`,
        without: `\n  - display_name: s\n    semantic:\n      embedding_model: e\n      routes:\n        - name: r\n          target: m\n          examples: ["hi"]\n      default: m\n      match:\n        threshold: 0.5\n`,
      },
      {
        label: "semantic default",
        field: "default_id",
        withId: `\n  - display_name: s\n    semantic:\n      embedding_model: e\n      routes:\n        - name: r\n          target: m\n          examples: ["hi"]\n      default: m\n      default_id: ${ID}\n      match:\n        threshold: 0.5\n`,
        without: `\n  - display_name: s\n    semantic:\n      embedding_model: e\n      routes:\n        - name: r\n          target: m\n          examples: ["hi"]\n      default: m\n      match:\n        threshold: 0.5\n`,
      },
      {
        label: "semantic route target",
        field: "target_id",
        withId: `\n  - display_name: s\n    semantic:\n      embedding_model: e\n      routes:\n        - name: r\n          target: m\n          target_id: ${ID}\n          examples: ["hi"]\n      default: m\n      match:\n        threshold: 0.5\n`,
        without: `\n  - display_name: s\n    semantic:\n      embedding_model: e\n      routes:\n        - name: r\n          target: m\n          examples: ["hi"]\n      default: m\n      match:\n        threshold: 0.5\n`,
      },
      {
        label: "semantic on_embedding_failure target",
        field: "target_id",
        withId: `\n  - display_name: s\n    semantic:\n      embedding_model: e\n      routes:\n        - name: r\n          target: m\n          examples: ["hi"]\n      default: m\n      match:\n        threshold: 0.5\n      on_embedding_failure:\n        target: m\n        target_id: ${ID}\n`,
        without: `\n  - display_name: s\n    semantic:\n      embedding_model: e\n      routes:\n        - name: r\n          target: m\n          examples: ["hi"]\n      default: m\n      match:\n        threshold: 0.5\n      on_embedding_failure:\n        target: m\n`,
      },
      {
        label: "cache policy model scope",
        field: "applies_to_model_id",
        withId: `\ncache_policies:\n  - name: c\n    applies_to: all\n    applies_to_model_id: ${ID}\n`,
        without: `\ncache_policies:\n  - name: c\n    applies_to: all\n`,
      },
      {
        label: "cache policy similarity embedder",
        field: "embedding_model_id",
        withId: `\ncache_policies:\n  - name: c\n    semantic:\n      embedding_model: e\n      embedding_model_id: ${ID}\n      threshold: 0.9\n`,
        without: `\ncache_policies:\n  - name: c\n    semantic:\n      embedding_model: e\n      threshold: 0.9\n`,
      },
      {
        label: "semantic guardrail embedder",
        field: "embedding_model_id",
        withId: `\nguardrails:\n  - name: g\n    kind: semantic\n    embedding_model: e\n    embedding_model_id: ${ID}\n    deny_examples: ["x"]\n    deny_threshold: 0.8\n`,
        without: `\nguardrails:\n  - name: g\n    kind: semantic\n    embedding_model: e\n    deny_examples: ["x"]\n    deny_threshold: 0.8\n`,
      },
    ];

  test("each id field is rejected by name, and the same file without it validates", async () => {
    const dir = await mkdtemp(join(tmpdir(), "aisix-model-ref-ids-"));
    try {
      for (const { label, field, withId, without } of CASES) {
        const bad = join(dir, `bad-${label.replace(/\W+/g, "-")}.yaml`);
        await writeFile(bad, PRELUDE + withId, "utf8");
        let failure: (Error & { code?: number; stderr?: string }) | undefined;
        try {
          await execFileP(BIN_PATH, ["validate", "--resources", bad]);
        } catch (e) {
          failure = e as Error & { code?: number; stderr?: string };
        }
        if (!failure) throw new Error(`${label}: expected \`aisix validate\` to fail`);
        expect(failure.code, label).toBe(1);
        expect(String(failure.stderr), label).toContain(
          `does not accept \`${field}\``,
        );

        // The same file with the id field removed validates, so the
        // refusal is that field and not anything else in the fixture.
        const good = join(dir, `good-${label.replace(/\W+/g, "-")}.yaml`);
        await writeFile(good, PRELUDE + without, "utf8");
        const ok = await execFileP(BIN_PATH, ["validate", "--resources", good]);
        expect(ok.stdout, label).toContain("OK:");
      }
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  }, 60_000);
});
