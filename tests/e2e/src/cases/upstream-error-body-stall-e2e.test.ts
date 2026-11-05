import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  waitConfigPropagation,
  waitForLogLine,
  type SpawnedApp,
} from "../harness/index.js";

// An upstream that answers a failure status, sends part of the body and then
// goes silent must not hold the caller past the model's budget.

const CALLER = "sk-error-body-stall-caller";
const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");
const BUDGET_MS = 800;
const CLIENT_GIVES_UP_MS = 8000;

describe("an upstream that stalls its error body", () => {
  let app: SpawnedApp | undefined;
  let etcdReachable = false;
  let upstream: Server | undefined;
  let rateLimited: Server | undefined;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    upstream = createServer((req, res) => {
      req.resume();
      res.writeHead(500, { "content-type": "application/json", "content-length": "1000" });
      res.write('{"error":{"mess');
    });
    await new Promise<void>((r) => upstream!.listen(0, "127.0.0.1", r));
    const base = `http://127.0.0.1:${(upstream.address() as { port: number }).port}`;

    rateLimited = createServer((req, res) => {
      req.resume();
      res.writeHead(429, { "content-type": "application/json", "content-length": "1000", "retry-after": "7" });
      res.write('{"error":{"mess');
    });
    await new Promise<void>((r) => rateLimited!.listen(0, "127.0.0.1", r));
    const rateLimitedBase = `http://127.0.0.1:${(rateLimited.address() as { port: number }).port}/v1`;

    app = await spawnApp({ extraEnv: { RUST_LOG: "info" } });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const names: string[] = [];
    for (const [name, provider, modelName] of [
      ["stall-openai", "openai", "m"],
      ["stall-429", "openai", "m"],
      ["stall-azure", "azure-openai", "m"],
      ["stall-vertex-gemini", "google", "gemini-2.5-flash"],
      ["stall-vertex-claude", "google", "claude-sonnet-4-5"],
      ["stall-anthropic", "anthropic", "m"],
      ["stall-cohere", "cohere", "m"],
      ["stall-bedrock-converse", "bedrock", "anthropic.claude-3-5-haiku-20241022-v1:0"],
      ["stall-bedrock-chat", "bedrock", "global.openai.gpt-6.1-sol"],
    ] as const) {
      const bedrock = provider === "bedrock";
      const vertex = provider === "google";
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        secret: bedrock
          ? JSON.stringify({ access_key_id: "AKIA-stall", secret_access_key: "sk-stall", region: "us-east-1" })
          : vertex
            ? JSON.stringify({ access_token: "ya29.stall", project: "proj-e2e", region: "us-central1" })
            : "sk-mock",
        provider,
        ...(provider === "cohere" ? { adapter: "openai" } : {}),
        ...(bedrock ? { adapter: "bedrock" } : {}),
        ...(vertex ? { adapter: "vertex" } : {}),
        ...(provider === "azure-openai" ? { adapter: "azure-openai" } : {}),
        api_base: name === "stall-429" ? rateLimitedBase : provider === "openai" ? `${base}/v1` : base,
      });
      await seed.createModel({
        display_name: name,
        provider,
        model_name: modelName,
        provider_key_id: pk.id,
        timeout: BUDGET_MS,
        stream_timeout: BUDGET_MS,
        cooldown: { enabled: false },
      });
      names.push(name);
    }
    await seed.createApiKey({ key_hash: sha256(CALLER), allowed_routes: ["*"], allowed_models: names });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER}` },
      });
      const { data } = (await res.json()) as { data?: unknown[] };
      return data?.length === names.length;
    });
  });

  afterAll(async () => {
    await app?.exit();
    upstream?.closeAllConnections();
    upstream?.close();
    rateLimited?.closeAllConnections();
    rateLimited?.close();
  });

  const json = (body: unknown): RequestInit => ({
    method: "POST",
    headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const multipart = (fields: Record<string, string>): RequestInit => {
    const form = new FormData();
    for (const [k, v] of Object.entries(fields)) form.set(k, v);
    form.set("file", new Blob([new Uint8Array(16)]), "input.png");
    return { method: "POST", headers: { authorization: `Bearer ${CALLER}` }, body: form };
  };
  const messages = [{ role: "user", content: "hi" }];

  test.each<[string, string, () => RequestInit]>([
    ["streaming native responses", "/v1/responses", () => json({ model: "stall-openai", input: "hi", stream: true })],
    ["native responses", "/v1/responses", () => json({ model: "stall-openai", input: "hi" })],
    ["speech", "/v1/audio/speech", () => json({ model: "stall-openai", input: "hi", voice: "alloy" })],
    ["transcriptions", "/v1/audio/transcriptions", () => multipart({ model: "stall-openai" })],
    ["streaming transcriptions", "/v1/audio/transcriptions", () => multipart({ model: "stall-openai", stream: "true" })],
    ["translations", "/v1/audio/translations", () => multipart({ model: "stall-openai" })],
    ["rerank", "/v1/rerank", () => json({ model: "stall-openai", query: "q", documents: ["a"] })],
    ["cohere rerank", "/v1/rerank", () => json({ model: "stall-cohere", query: "q", documents: ["a"] })],
    ["image edits", "/v1/images/edits", () => multipart({ model: "stall-openai", prompt: "hi" })],
    ["chat", "/v1/chat/completions", () => json({ model: "stall-openai", messages })],
    ["streaming chat", "/v1/chat/completions", () => json({ model: "stall-openai", messages, stream: true })],
    ["streaming chat to anthropic", "/v1/chat/completions", () => json({ model: "stall-anthropic", messages, stream: true })],
    ["streaming messages", "/v1/messages", () => json({ model: "stall-openai", messages, max_tokens: 5, stream: true })],
    ["streaming native messages", "/v1/messages", () => json({ model: "stall-anthropic", messages, max_tokens: 5, stream: true })],
    ["bridged streaming responses", "/v1/responses", () => json({ model: "stall-anthropic", input: "hi", stream: true })],
    ["count_tokens", "/v1/messages/count_tokens", () => json({ model: "stall-anthropic", messages })],
    ["completions", "/v1/completions", () => json({ model: "stall-openai", prompt: "hi" })],
    ["embeddings", "/v1/embeddings", () => json({ model: "stall-openai", input: "hi" })],
    ["image generations", "/v1/images/generations", () => json({ model: "stall-openai", prompt: "hi" })],
    ["decisions", "/v1/decisions", () => json({ model: "stall-openai", input: "hi", questions: [{ type: "predicate", name: "q", instructions: "i" }] })],
    ["videos", "/v1/videos", () => json({ model: "stall-openai", prompt: "hi" })],
    ["streaming chat to azure", "/v1/chat/completions", () => json({ model: "stall-azure", messages, stream: true })],
    ["streaming chat to vertex gemini", "/v1/chat/completions", () => json({ model: "stall-vertex-gemini", messages, stream: true })],
    ["streaming chat to vertex claude", "/v1/chat/completions", () => json({ model: "stall-vertex-claude", messages, stream: true })],
    ["streaming native messages on vertex", "/v1/messages", () => json({ model: "stall-vertex-claude", messages, max_tokens: 5, stream: true })],
    ["streaming bedrock converse", "/v1/chat/completions", () => json({ model: "stall-bedrock-converse", messages, stream: true })],
    ["bedrock converse", "/v1/chat/completions", () => json({ model: "stall-bedrock-converse", messages })],
    ["streaming bedrock chat endpoint", "/v1/chat/completions", () => json({ model: "stall-bedrock-chat", messages, stream: true })],
    ["bedrock chat endpoint", "/v1/chat/completions", () => json({ model: "stall-bedrock-chat", messages })],
    ["streaming bedrock native messages", "/v1/messages", () => json({ model: "stall-bedrock-converse", messages, max_tokens: 5, stream: true })],
  ])("%s answers within the budget", async (_what, path, init) => {
    if (!etcdReachable) return;
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      ...init(),
      signal: AbortSignal.timeout(CLIENT_GIVES_UP_MS),
    });
    await res.text();
    expect(res.status).toBeGreaterThanOrEqual(500);
  }, 20000);

  test("what the response head said survives the stalled body, and the record names the stall", async () => {
    if (!etcdReachable) return;
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      ...json({ model: "stall-429", messages, stream: true }),
      signal: AbortSignal.timeout(CLIENT_GIVES_UP_MS),
    });
    await res.text();
    expect(res.status).toBe(429);
    expect(res.headers.get("retry-after")).toBe("7");
    const requestId = res.headers.get("x-aisix-request-id")!;
    await waitForLogLine(
      app!,
      (l) =>
        l.includes(requestId) &&
        l.includes(`upstream returned HTTP 429: failed to read response body: timed out after ${BUDGET_MS}ms`),
      "the stalled request's access-log line",
    );
  }, 20000);
});
