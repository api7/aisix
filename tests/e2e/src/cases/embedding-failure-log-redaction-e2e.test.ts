import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  pickFreePort,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForLogLine,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: when the embedding call behind a semantic router or a semantic
// cache policy fails, its warning says why without repeating what the
// embedding upstream answered. The embedded text is the caller's prompt,
// and an upstream can quote it back — in an error message, or as a value
// whose wrong type makes the response undecodable.
//
// Two embedding upstreams, each quoting the request it received: an
// OpenAI-shaped one that answers 400 with the request in its message, and
// a Bedrock Titan one that answers 200 with the request where the vector
// belongs. Each backs one router (falling back to its default) and one
// cache policy (proceeding uncached), so the caller is served throughout
// and the warning is the only trace.

const CALLER_PLAINTEXT = "sk-embed-redaction-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");
const PROMPT_MARKER = "embedredactionpromptmarker";
const ROUTE_EXAMPLE = "embedredactionroutexample";

function quotingServer(respond: (raw: string) => { status: number; body: unknown }): Server {
  return createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      const { status, body } = respond(raw);
      res.statusCode = status;
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify(body));
    });
  });
}

async function listen(server: Server): Promise<number> {
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return port;
}

describe("embedding failure logs keep the upstream's echo of the prompt out", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let proxy: ProxyClient | undefined;
  const servers: Server[] = [];
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-redaction",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "ok" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 1, total_tokens: 6 },
      },
    });

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "redaction-chat-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    const chatModel = (name: string) =>
      seed.createModel({
        display_name: name,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
    await chatModel("redaction-default");

    const echo = quotingServer((raw) => ({
      status: 400,
      body: { error: { message: `invalid input: ${raw}` } },
    }));
    const titan = quotingServer((raw) => ({
      status: 200,
      body: { embedding: raw, inputTextTokenCount: 1 },
    }));
    servers.push(echo, titan);
    const echoPk = await seed.createProviderKey({
      display_name: "redaction-echo-pk",
      secret: "sk-mock",
      api_base: `http://127.0.0.1:${await listen(echo)}/v1`,
    });
    await seed.createModel({
      display_name: "redaction-echo-embed",
      provider: "openai",
      model_name: "embed-mock",
      provider_key_id: echoPk.id,
      embedding: { dimensions: 4, normalize: true },
    });
    const titanPk = await seed.createProviderKey({
      display_name: "redaction-titan-pk",
      provider: "bedrock",
      adapter: "bedrock",
      secret: JSON.stringify({
        access_key_id: "AKIA-redaction-e2e",
        secret_access_key: "sk-redaction-e2e",
        region: "us-west-2",
      }),
      api_base: `http://127.0.0.1:${await listen(titan)}`,
    });
    await seed.createModel({
      display_name: "redaction-titan-embed",
      provider: "bedrock",
      model_name: "amazon.titan-embed-text-v1",
      provider_key_id: titanPk.id,
      embedding: { dimensions: 4, normalize: true },
    });

    for (const embed of ["echo", "titan"]) {
      await seed.createModel({
        display_name: `redaction-router-${embed}`,
        semantic: {
          embedding_model: `redaction-${embed}-embed`,
          routes: [
            {
              name: "route-topic",
              target: "redaction-default",
              examples: [ROUTE_EXAMPLE],
              threshold: 0.9,
            },
          ],
          default: "redaction-default",
          match: { threshold: 0.9 },
          on_embedding_failure: "default",
        },
      });
      await chatModel(`redaction-cached-${embed}`);
      await seed.createCachePolicy({
        name: `redaction-cache-${embed}`,
        backend: "memory",
        applies_to: `model:redaction-cached-${embed}`,
        ttl_seconds: 600,
        semantic: { embedding_model: `redaction-${embed}-embed`, threshold: 0.85 },
      });
    }

    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(
      async () => (await proxy!.listModels()).status === 200,
    );
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    for (const s of servers) {
      await new Promise<void>((resolve) => s.close(() => resolve()));
    }
  });

  async function failureLine(
    model: string,
    matches: (line: string) => boolean,
  ): Promise<string> {
    const res = await proxy!.chat({
      model,
      messages: [{ role: "user", content: `hello ${PROMPT_MARKER}` }],
    });
    // The router falls back to its default and the cache proceeds
    // uncached: the caller is served despite the failed embedding.
    expect(res.status).toBe(200);
    return waitForLogLine(app!, matches, `the ${model} embedding failure line`);
  }

  function expectNoEcho(line: string) {
    expect(line).not.toContain("invalid input");
    expect(line).not.toContain(PROMPT_MARKER);
    expect(line).not.toContain(ROUTE_EXAMPLE);
  }

  const routerLine = (router: string) => (l: string) =>
    l.includes("semantic embedding call failed") && l.includes(`router=${router}`);
  const cacheLine = (policy: string) => (l: string) =>
    l.includes("cache semantic embedding call failed") &&
    l.includes(`policy_name=${policy}`);

  test("router: an upstream error that quotes the prompt logs only its status", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const line = await failureLine("redaction-router-echo", routerLine("redaction-router-echo"));
    expect(line).toContain("upstream returned HTTP 400");
    expectNoEcho(line);
  });

  test("router: an undecodable response that quotes the prompt logs only the decode failure", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const line = await failureLine("redaction-router-titan", routerLine("redaction-router-titan"));
    expect(line).toContain("could not be decoded");
    expectNoEcho(line);
  });

  test("cache: an upstream error that quotes the prompt logs only its status", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const line = await failureLine("redaction-cached-echo", cacheLine("redaction-cache-echo"));
    expect(line).toContain("upstream returned HTTP 400");
    expectNoEcho(line);
  });

  test("cache: an undecodable response that quotes the prompt logs only the decode failure", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const line = await failureLine("redaction-cached-titan", cacheLine("redaction-cache-titan"));
    expect(line).toContain("could not be decoded");
    expectNoEcho(line);
  });
});
