import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";
import { pickFreePort } from "../harness/ports.js";

// E2E: semantic routing (#641). A request to a `semantic` virtual model
// embeds the latest user message, scores it against each route's example
// embeddings, and dispatches to the best route's target — or to `default`
// when none clears its threshold. No CP involved: real `aisix` binary +
// etcd + a deterministic mock embedding endpoint + mock chat upstreams.
//
// The mock embedding endpoint maps each input string to a one-hot vector
// by keyword, so routing decisions are fully deterministic and asserted:
//   contains "contract"/"legal"/"nda" -> [0,0,1,0]  (legal route)
//   contains "python"/"code"          -> [0,1,0,0]  (code route)
//   contains "translate"              -> [1,0,0,0]  (translate route)
//   anything else                     -> [0,0,0,1]  (orthogonal -> default)
// A route example sharing a keyword with the prompt yields cosine 1.0;
// an unrelated prompt yields cosine 0 against every example -> default.

const CALLER_PLAINTEXT = "sk-semantic-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

function keywordVector(text: string): number[] {
  const t = text.toLowerCase();
  if (t.includes("translate")) return [1, 0, 0, 0];
  if (t.includes("python") || t.includes("code")) return [0, 1, 0, 0];
  if (t.includes("contract") || t.includes("legal") || t.includes("nda"))
    return [0, 0, 1, 0];
  return [0, 0, 0, 1];
}

interface EmbeddingMock {
  baseUrl: string;
  callCount(): number;
  close(): Promise<void>;
}

/**
 * A minimal OpenAI-compatible `/v1/embeddings` mock that returns a
 * deterministic keyword vector per input. When `fail` is set, every
 * embeddings call returns 500 (to exercise `on_embedding_failure`).
 */
async function startEmbeddingMock(opts: { fail?: boolean } = {}): Promise<EmbeddingMock> {
  let calls = 0;
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
      calls++;
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
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify({ error: { message: "invalid JSON payload" } }));
        return;
      }
      const inputs = Array.isArray(body.input)
        ? body.input
        : [body.input ?? ""];
      const data = inputs.map((text, index) => ({
        object: "embedding",
        index,
        embedding: keywordVector(text),
      }));
      res.statusCode = 200;
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          object: "list",
          model: "embed-mock",
          data,
          usage: { prompt_tokens: inputs.length, total_tokens: inputs.length },
        }),
      );
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    callCount: () => calls,
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

function chatUpstreamReplying(content: string): Promise<OpenAiUpstream> {
  return startOpenAiUpstream({
    nonStreamBody: {
      id: `cmpl-${content}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: "gpt-4o-mini",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content },
          finish_reason: "stop",
        },
      ],
      usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
    },
  });
}

describe("semantic routing e2e", () => {
  let app: SpawnedApp | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];
  const embedMocks: EmbeddingMock[] = [];

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });

    // Shared `prod-chat` fixture for the matched + default-fallthrough
    // tests, so neither depends on the other's execution order.
    const embed = await startEmbeddingMock();
    const legal = await chatUpstreamReplying("served-by-legal");
    const fallback = await chatUpstreamReplying("served-by-default");
    embedMocks.push(embed);
    upstreams.push(legal, fallback);

    await createEmbeddingModel("bge-mock", embed);
    await createDirectModel("legal-model", legal);
    await createDirectModel("default-model", fallback);
    await admin.createModel({
      display_name: "prod-chat",
      semantic: {
        embedding_model: "bge-mock",
        routes: [
          {
            name: "legal",
            target: "legal-model",
            examples: ["analyze this contract for legal risk"],
            threshold: 0.5,
          },
        ],
        default: "default-model",
        match: { threshold: 0.5 },
      },
    });

    await waitConfigPropagation(async () => {
      try {
        const r = await chat("prod-chat", "review the nda contract clauses");
        return r.status === 200 && r.content === "served-by-legal";
      } catch {
        return false;
      }
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
    await Promise.all(embedMocks.map((m) => m.close()));
  });

  async function createDirectModel(
    displayName: string,
    upstream: OpenAiUpstream,
  ): Promise<void> {
    if (!admin) throw new Error("admin not ready");
    const pk = await admin.createProviderKey({
      display_name: `${displayName}-pk`,
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: displayName,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
  }

  async function createEmbeddingModel(
    displayName: string,
    mock: EmbeddingMock,
  ): Promise<void> {
    if (!admin) throw new Error("admin not ready");
    const pk = await admin.createProviderKey({
      display_name: `${displayName}-pk`,
      secret: "sk-mock",
      api_base: `${mock.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: displayName,
      provider: "openai",
      model_name: "embed-mock",
      provider_key_id: pk.id,
      embedding: { dimensions: 4, normalize: true },
    });
  }

  interface ChatResult {
    status: number;
    content: string | undefined;
    route: string | null;
    servedBy: string | null;
  }

  async function chat(model: string, prompt: string): Promise<ChatResult> {
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
      },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: prompt }],
      }),
    });
    let content: string | undefined;
    if (res.status === 200) {
      const json = (await res.json()) as {
        choices?: { message?: { content?: string } }[];
      };
      content = json.choices?.[0]?.message?.content;
    } else {
      await res.text();
    }
    return {
      status: res.status,
      content,
      route: res.headers.get("x-aisix-route"),
      servedBy: res.headers.get("x-aisix-served-by"),
    };
  }

  test("routes a matching prompt to its route target and sets x-aisix-route", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }
    // `prod-chat` is provisioned in beforeAll (shared, order-independent).

    const r = await chat("prod-chat", "please review the contract for risk");
    expect(r.status).toBe(200);
    expect(r.content).toBe("served-by-legal");
    expect(r.route).toBe("legal");
    expect(r.servedBy).toBe("legal-model");
  });

  test("falls through to default when no route clears its threshold (no x-aisix-route)", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }
    // Reuses the prod-chat router from the previous test. A prompt with no
    // route keyword embeds orthogonal to every example -> default.
    const r = await chat("prod-chat", "what time is the meeting tomorrow");
    expect(r.status).toBe(200);
    expect(r.content).toBe("served-by-default");
    // No route matched -> the header is absent.
    expect(r.route).toBeNull();
  });

  test("on_embedding_failure: default routes to the default model when embedding errors", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }
    const brokenEmbed = await startEmbeddingMock({ fail: true });
    const fallback = await chatUpstreamReplying("served-by-safe-default");
    embedMocks.push(brokenEmbed);
    upstreams.push(fallback);

    await createEmbeddingModel("broken-embed", brokenEmbed);
    await createDirectModel("safe-default-model", fallback);
    await admin.createModel({
      display_name: "degrading-router",
      semantic: {
        embedding_model: "broken-embed",
        routes: [
          {
            name: "legal",
            target: "safe-default-model",
            examples: ["analyze this contract"],
            threshold: 0.5,
          },
        ],
        default: "safe-default-model",
        match: { threshold: 0.5 },
        on_embedding_failure: "default",
      },
    });

    await waitConfigPropagation(async () => {
      try {
        const r = await chat("degrading-router", "anything at all");
        return r.status === 200 && r.content === "served-by-safe-default";
      } catch {
        return false;
      }
    });

    const r = await chat("degrading-router", "contract review please");
    expect(r.status).toBe(200);
    expect(r.content).toBe("served-by-safe-default");
    // Embedding failed, so no route matched -> no route header.
    expect(r.route).toBeNull();
  });

  test("on_embedding_failure: fail returns 503 when embedding errors", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }
    const brokenEmbed = await startEmbeddingMock({ fail: true });
    const target = await chatUpstreamReplying("should-not-be-reached");
    embedMocks.push(brokenEmbed);
    upstreams.push(target);

    await createEmbeddingModel("broken-embed-2", brokenEmbed);
    await createDirectModel("unreached-model", target);
    await admin.createModel({
      display_name: "strict-router",
      semantic: {
        embedding_model: "broken-embed-2",
        routes: [
          {
            name: "legal",
            target: "unreached-model",
            examples: ["analyze this contract"],
            threshold: 0.5,
          },
        ],
        default: "unreached-model",
        match: { threshold: 0.5 },
        on_embedding_failure: "fail",
      },
    });

    await waitConfigPropagation(async () => {
      try {
        const r = await chat("strict-router", "contract review");
        return r.status === 503;
      } catch {
        return false;
      }
    });

    const r = await chat("strict-router", "contract review please");
    expect(r.status).toBe(503);
  });
});
