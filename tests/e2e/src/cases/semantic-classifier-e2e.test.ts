import { createHash } from "node:crypto";
import { createServer, type IncomingHttpHeaders, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForLogLine,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";
import { pickFreePort } from "../harness/ports.js";

// E2E: a semantic router with a `classifier` block decides by one call to
// a hosted decision model (TypeSafe's jev) instead of by embeddings, and
// every semantic router — either backend — says on its access-log line how
// it decided (`semantic_route`, `semantic_score`, `semantic_fallback`).
//
// Real `aisix` binary + etcd + a mock TypeSafe endpoint + mock chat and
// embedding upstreams. The mock decision model answers by keyword in the
// prompt, so every decision is fixed:
//   "python"   -> code, confidence 0.93
//   "integral" -> math, confidence 0.9   (math's target excludes the caller)
//   "weather"  -> none_of_the_above, 0.88
//   "maybe"    -> code, confidence 0.3   (below min_confidence 0.5)
//   "boom"     -> HTTP 500
//   "slow"     -> answers after 2s        (timeout_ms is 500)
//   "garbage"  -> picks "poetry", which was never offered

const CALLER_PLAINTEXT = "sk-semantic-classifier-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");
const TYPESAFE_SECRET = "ts-secret-e2e";

const INSTRUCTIONS =
  "Which route should handle this user request to an AI assistant?";
const NONE_DESCRIPTION =
  "anything not covered by the routes above (e.g. weather, travel booking, medical advice, image generation, news)";

interface DecisionCall {
  path: string;
  headers: IncomingHttpHeaders;
  body: {
    state?: string;
    model?: string;
    questions?: Record<
      string,
      { type?: string; instructions?: string; criteria?: Record<string, string> }
    >;
  };
}

interface DecisionMock {
  baseUrl: string;
  calls: DecisionCall[];
  close(): Promise<void>;
}

async function startDecisionMock(): Promise<DecisionMock> {
  const calls: DecisionCall[] = [];
  const server: Server = createServer((req, res) => {
    res.on("error", () => {});
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      const body = JSON.parse(raw || "{}") as DecisionCall["body"];
      calls.push({ path: req.url ?? "", headers: req.headers, body });
      const state = (body.state ?? "").toLowerCase();
      const answer = (choice: string, confidence: number) => {
        res.statusCode = 200;
        res.setHeader("content-type", "application/json");
        res.end(
          JSON.stringify({
            model: "jev-1.13.0",
            answers: {
              route: {
                type: "choice",
                choice,
                probabilities: { [choice]: confidence },
                confidence,
              },
            },
            usage: { input_tokens: 300, output_tokens: 20 },
          }),
        );
      };
      if (state.includes("boom")) {
        res.statusCode = 500;
        res.end(JSON.stringify({ error: "internal" }));
      } else if (state.includes("slow")) {
        setTimeout(() => answer("code", 0.99), 2_000);
      } else if (state.includes("garbage")) {
        answer("poetry", 0.99);
      } else if (state.includes("python")) {
        answer("code", 0.93);
      } else if (state.includes("integral")) {
        answer("math", 0.9);
      } else if (state.includes("weather")) {
        answer("none_of_the_above", 0.88);
      } else if (state.includes("maybe")) {
        answer("code", 0.3);
      } else {
        res.statusCode = 422;
        res.end(JSON.stringify({ error: "unexpected state in e2e" }));
      }
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    calls,
    async close() {
      server.closeAllConnections();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    },
  };
}

/** Keyword `/v1/embeddings` mock; `fail` answers every call with 500. */
async function startEmbeddingMock(
  opts: { fail?: boolean } = {},
): Promise<{ baseUrl: string; close(): Promise<void> }> {
  const vector = (text: string): number[] => {
    const t = text.toLowerCase();
    if (t.includes("python")) return [1, 0, 0];
    if (t.includes("integral")) return [0, 1, 0];
    return [0, 0, 1];
  };
  const server: Server = createServer((req, res) => {
    res.on("error", () => {});
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      res.setHeader("content-type", "application/json");
      if (opts.fail) {
        res.statusCode = 500;
        res.end(JSON.stringify({ error: { message: "embedding upstream down" } }));
        return;
      }
      const body = JSON.parse(raw || "{}") as { input?: string | string[] };
      const inputs = Array.isArray(body.input) ? body.input : [body.input ?? ""];
      res.statusCode = 200;
      res.end(
        JSON.stringify({
          object: "list",
          model: "embed-mock",
          data: inputs.map((text, index) => ({
            object: "embedding",
            index,
            embedding: vector(text),
          })),
          usage: { prompt_tokens: 1, total_tokens: 1 },
        }),
      );
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    async close() {
      await new Promise<void>((resolve) => server.close(() => resolve()));
    },
  };
}

function chatUpstreamReplying(content: string): Promise<OpenAiUpstream> {
  return startOpenAiUpstream({
    nonStreamBody: {
      id: `cmpl-${content}`,
      object: "chat.completion",
      created: 0,
      model: "gpt-4o-mini",
      choices: [
        { index: 0, message: { role: "assistant", content }, finish_reason: "stop" },
      ],
      usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
    },
  });
}

interface ChatResult {
  status: number;
  content: string | undefined;
  route: string | null;
  requestId: string;
}

/** The fields of one access-log line, as the text formatter writes them. */
function field(line: string, name: string): string | undefined {
  const m = new RegExp(`\\b${name}=("(?:[^"\\\\]|\\\\.)*"|\\S+)`).exec(line);
  if (!m) return undefined;
  return m[1].startsWith('"') ? (JSON.parse(m[1]) as string) : m[1];
}

describe("semantic classifier e2e", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  let decisions: DecisionMock | undefined;
  const closers: Array<() => Promise<void>> = [];
  const rejectedIds: Record<string, string> = {};

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp({ logLevel: "info" });
    seed = new SeedClient(etcd, app.etcdPrefix);

    decisions = await startDecisionMock();
    closers.push(() => decisions!.close());
    const embed = await startEmbeddingMock();
    const brokenEmbed = await startEmbeddingMock({ fail: true });
    closers.push(() => embed.close(), () => brokenEmbed.close());

    const typesafe = await seed.createProviderKey({
      display_name: "typesafe-pk",
      secret: TYPESAFE_SECRET,
      provider: "typesafe",
      adapter: undefined,
      api_base: decisions.baseUrl,
    });
    // A usable key for the wrong provider: the classifier must refuse it
    // rather than send its credential to the decision endpoint.
    const wrongProvider = await seed.createProviderKey({
      display_name: "not-typesafe-pk",
      secret: "sk-wrong",
      api_base: decisions.baseUrl,
    });

    async function direct(name: string, extra: Record<string, unknown> = {}) {
      const upstream = await chatUpstreamReplying(`served-by-${name}`);
      closers.push(() => upstream.close());
      const pk = await seed!.createProviderKey({
        display_name: `${name}-pk`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      await seed!.createModel({
        display_name: name,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
        ...extra,
      });
    }
    await direct("code-model");
    await direct("default-model");
    await direct("safe-model");
    // Streams its answer, so a routed request's line is written by the
    // stream's terminal emitter rather than by the handler.
    const streaming = await startOpenAiUpstream({
      streamEvents: [
        JSON.stringify({
          id: "cmpl-stream",
          object: "chat.completion.chunk",
          created: 0,
          model: "gpt-4o-mini",
          choices: [{ index: 0, delta: { role: "assistant", content: "streamed" }, finish_reason: null }],
        }),
        JSON.stringify({
          id: "cmpl-stream",
          object: "chat.completion.chunk",
          created: 0,
          model: "gpt-4o-mini",
          choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        }),
        "[DONE]",
      ],
    });
    closers.push(() => streaming.close());
    const streamPk = await seed.createProviderKey({
      display_name: "stream-model-pk",
      secret: "sk-mock",
      api_base: `${streaming.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "stream-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: streamPk.id,
    });
    // Reachable from nowhere the suite calls from, so a route to it is
    // displaced to `default` by the member gate.
    await direct("fenced-model", { allowed_cidrs: ["10.255.255.0/24"] });

    for (const [name, pk, embedModel] of [
      ["embed-mock", "embed", embed],
      ["embed-broken", "embed-broken", brokenEmbed],
    ] as const) {
      const key = await seed.createProviderKey({
        display_name: `${pk}-pk`,
        secret: "sk-mock",
        api_base: `${embedModel.baseUrl}/v1`,
      });
      await seed.createModel({
        display_name: name,
        provider: "openai",
        model_name: "embed-mock",
        provider_key_id: key.id,
        embedding: { dimensions: 3, normalize: true },
      });
    }

    const jevRoutes = [
      { name: "code", target: "code-model", description: "programming questions" },
      { name: "math", target: "fenced-model", description: "math problems" },
    ];
    const jevRouter = (
      name: string,
      classifier: Record<string, unknown>,
    ): Promise<unknown> =>
      seed!.createModel({
        display_name: name,
        semantic: {
          classifier: {
            type: "jev",
            provider_key_id: typesafe.id,
            timeout_ms: 500,
            ...classifier,
          },
          routes: jevRoutes,
          default: "default-model",
        },
      });
    await jevRouter("jev-router", {});
    await seed.createModel({
      display_name: "jev-stream",
      semantic: {
        classifier: { type: "jev", provider_key_id: typesafe.id },
        routes: [{ name: "code", target: "stream-model", description: "programming questions" }],
        default: "default-model",
      },
    });
    await jevRouter("jev-fail", { on_failure: "fail" });
    await jevRouter("jev-target", { on_failure: { target: "safe-model" } });
    await jevRouter("jev-wrong-key", { provider_key_id: wrongProvider.id });
    await jevRouter("jev-missing-key", { provider_key_id: "no-such-provider-key" });

    for (const [name, embeddingModel] of [
      ["emb-router", "embed-mock"],
      ["emb-broken", "embed-broken"],
    ]) {
      await seed.createModel({
        display_name: name,
        semantic: {
          embedding_model: embeddingModel,
          routes: [
            { name: "code", target: "code-model", examples: ["python please"] },
            { name: "math", target: "fenced-model", examples: ["an integral"] },
          ],
          default: "default-model",
          match: { threshold: 0.5 },
        },
      });
    }

    // Classifier rows the gateway must refuse to load, never load with
    // the offending field ignored.
    const valid = () => ({
      classifier: { type: "jev", provider_key_id: typesafe.id },
      routes: [{ name: "code", target: "code-model", description: "code" }],
      default: "default-model",
    });
    const bad: Record<string, Record<string, unknown>> = {
      embedding_model: { ...valid(), embedding_model: "embed-mock" },
      embedding_timeout_ms: { ...valid(), embedding_timeout_ms: 100 },
      on_embedding_failure: { ...valid(), on_embedding_failure: "fail" },
      match: { ...valid(), match: { threshold: 0.5 } },
      examples: {
        ...valid(),
        routes: [{ name: "code", target: "code-model", description: "c", examples: ["x"] }],
      },
      threshold: {
        ...valid(),
        routes: [{ name: "code", target: "code-model", description: "c", threshold: 0.5 }],
      },
      no_description: { ...valid(), routes: [{ name: "code", target: "code-model" }] },
      reserved_name: {
        ...valid(),
        routes: [{ name: "none_of_the_above", target: "code-model", description: "c" }],
      },
    };
    for (const [why, semantic] of Object.entries(bad)) {
      const row = await seed.createModel({ display_name: `bad-${why}`, semantic });
      rejectedIds[why] = row.id;
    }

    // Seeded last: its authenticating implies every row above has landed.
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(closers.map((c) => c()));
  });

  async function chat(model: string, prompt: string): Promise<ChatResult> {
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
      },
      body: JSON.stringify({ model, messages: [{ role: "user", content: prompt }] }),
    });
    let content: string | undefined;
    if (res.status === 200) {
      const json = (await res.json()) as { choices?: { message?: { content?: string } }[] };
      content = json.choices?.[0]?.message?.content;
    } else {
      await res.text();
    }
    return {
      status: res.status,
      content,
      route: res.headers.get("x-aisix-route"),
      requestId: res.headers.get("x-aisix-request-id") ?? "",
    };
  }

  async function accessLine(requestId: string): Promise<string> {
    expect(requestId).not.toBe("");
    return waitForLogLine(
      app!,
      (l) => l.includes("proxy request completed") && l.includes(`"${requestId}"`),
      `access-log line for ${requestId}`,
    );
  }

  async function expectDecision(
    r: ChatResult,
    want: { route?: string; score?: string; fallback?: string },
  ): Promise<void> {
    const line = await accessLine(r.requestId);
    expect(field(line, "semantic_route"), line).toBe(want.route);
    expect(field(line, "semantic_score"), line).toBe(want.score);
    expect(field(line, "semantic_fallback"), line).toBe(want.fallback);
  }

  test("routes to the classifier's pick and asks the question the contract names", async (ctx) => {
    if (!etcdReachable || !app || !decisions) return ctx.skip();
    const before = decisions.calls.length;
    const r = await chat("jev-router", "fix my python script");
    expect(r.status).toBe(200);
    expect(r.content).toBe("served-by-code-model");
    expect(r.route).toBe("code");
    await expectDecision(r, { route: "code", score: "0.93" });

    expect(decisions.calls.length).toBe(before + 1);
    const call = decisions.calls[decisions.calls.length - 1];
    // The endpoint is the Provider Key's api_base, not the vendor default.
    expect(call.path).toBe("/v1/systemone");
    expect(call.headers.authorization).toBe(`Bearer ${TYPESAFE_SECRET}`);
    expect(call.body.state).toBe("fix my python script");
    expect(call.body.model).toBe("jev-latest");
    const questions = Object.values(call.body.questions ?? {});
    expect(questions).toHaveLength(1);
    expect(questions[0].type).toBe("choice");
    expect(questions[0].instructions).toBe(INSTRUCTIONS);
    // Every route by name, in configured order, then the gateway's own
    // option last.
    expect(Object.entries(questions[0].criteria ?? {})).toEqual([
      ["code", "programming questions"],
      ["math", "math problems"],
      ["none_of_the_above", NONE_DESCRIPTION],
    ]);
  });

  test("a streamed response's line carries the decision too", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
      },
      body: JSON.stringify({
        model: "jev-stream",
        stream: true,
        messages: [{ role: "user", content: "fix my python script" }],
      }),
    });
    expect(res.status).toBe(200);
    expect(await res.text()).toContain("streamed");
    await expectDecision(
      { status: 200, content: undefined, route: null, requestId: res.headers.get("x-aisix-request-id") ?? "" },
      { route: "code", score: "0.93" },
    );
  });

  test("none_of_the_above and a low-confidence pick both go to default", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const none = await chat("jev-router", "what is the weather in Paris");
    expect(none.status).toBe(200);
    expect(none.content).toBe("served-by-default-model");
    expect(none.route).toBeNull();
    await expectDecision(none, { score: "0.88", fallback: "none_of_the_above" });

    const low = await chat("jev-router", "maybe something about code");
    expect(low.status).toBe(200);
    expect(low.content).toBe("served-by-default-model");
    expect(low.route).toBeNull();
    await expectDecision(low, { score: "0.3", fallback: "low_confidence" });
  });

  test("a pick whose target excludes the caller is served by default", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const r = await chat("jev-router", "solve this integral");
    expect(r.status).toBe(200);
    expect(r.content).toBe("served-by-default-model");
    expect(r.route).toBeNull();
    await expectDecision(r, { score: "0.9", fallback: "target_unavailable" });
  });

  test("an empty prompt goes to default without calling the classifier", async (ctx) => {
    if (!etcdReachable || !app || !decisions) return ctx.skip();
    const before = decisions.calls.length;
    const r = await chat("jev-router", "   ");
    expect(r.status).toBe(200);
    expect(r.content).toBe("served-by-default-model");
    await expectDecision(r, { fallback: "empty_prompt" });
    expect(decisions.calls.length).toBe(before);
  });

  for (const [what, prompt] of [
    ["an upstream error", "boom goes the classifier"],
    ["a timeout", "slow classifier please"],
    ["an option it never offered", "garbage answer"],
  ]) {
    test(`on_failure applies to ${what}`, async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const byDefault = await chat("jev-router", prompt);
      expect(byDefault.status).toBe(200);
      expect(byDefault.content).toBe("served-by-default-model");
      expect(byDefault.route).toBeNull();
      await expectDecision(byDefault, { fallback: "decision_failed" });

      const failed = await chat("jev-fail", prompt);
      expect(failed.status).toBe(503);
      await expectDecision(failed, { fallback: "decision_failed" });

      const toTarget = await chat("jev-target", prompt);
      expect(toTarget.status).toBe(200);
      expect(toTarget.content).toBe("served-by-safe-model");
      expect(toTarget.route).toBeNull();
      await expectDecision(toTarget, { fallback: "decision_failed" });
    });
  }

  test("a provider key that is missing or not typesafe is a decision failure", async (ctx) => {
    if (!etcdReachable || !app || !decisions) return ctx.skip();
    const before = decisions.calls.length;
    for (const router of ["jev-wrong-key", "jev-missing-key"]) {
      const r = await chat(router, "fix my python script");
      expect(r.status).toBe(200);
      expect(r.content).toBe("served-by-default-model");
      await expectDecision(r, { fallback: "decision_failed" });
    }
    // The wrong-provider key's credential never reached the endpoint.
    expect(decisions.calls.length).toBe(before);
  });

  test("classifier rows carrying an embedding-mode field, or missing a description, are rejected", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await fetch(`${app.metricsUrl}/status/config`);
    expect(res.status).toBe(200);
    const cfg = (await res.json()) as {
      rejected: Array<{ resource_kind: string; resource_id: string }>;
    };
    for (const [why, id] of Object.entries(rejectedIds)) {
      expect(
        cfg.rejected.some((r) => r.resource_kind === "models" && r.resource_id === id),
        `row with ${why} was not rejected: ${JSON.stringify(cfg.rejected)}`,
      ).toBe(true);
    }
    // Rejected, so not servable either.
    const r = await chat("bad-match", "fix my python script");
    expect(r.status).not.toBe(200);
  });

  test("an embedding router's line carries the same three fields", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const matched = await chat("emb-router", "python please");
    expect(matched.status).toBe(200);
    expect(matched.route).toBe("code");
    await expectDecision(matched, { route: "code", score: "1.0" });

    const missed = await chat("emb-router", "hello there");
    expect(missed.status).toBe(200);
    expect(missed.content).toBe("served-by-default-model");
    await expectDecision(missed, { score: "0.0", fallback: "no_match" });

    const displaced = await chat("emb-router", "an integral");
    expect(displaced.status).toBe(200);
    expect(displaced.content).toBe("served-by-default-model");
    await expectDecision(displaced, { score: "1.0", fallback: "target_unavailable" });

    const empty = await chat("emb-router", "   ");
    expect(empty.status).toBe(200);
    await expectDecision(empty, { fallback: "empty_prompt" });

    const failed = await chat("emb-broken", "python please");
    expect(failed.status).toBe(200);
    expect(failed.content).toBe("served-by-default-model");
    await expectDecision(failed, { fallback: "decision_failed" });
  });

  test("a request to a model that is not a semantic router carries none of them", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const r = await chat("code-model", "python please");
    expect(r.status).toBe(200);
    const line = await accessLine(r.requestId);
    expect(line).not.toContain("semantic_");
  });
});
