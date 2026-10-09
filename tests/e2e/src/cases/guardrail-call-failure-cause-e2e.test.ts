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

// E2E: when a guardrail cannot reach its provider, the gateway's
// "call failed" warning says why (AISIX-Cloud#1786). Before, every such
// failure logged the same `failure=IoError` with nothing else, so an
// operator could not tell a DNS failure from a refused connection.
//
// Two rows, each attached to its own model: one pointed at a port nothing
// listens on, one at a name that cannot resolve. Both fail open, so the
// caller still gets its completion and the log line is the only trace.

const CALLER_PLAINTEXT = "sk-guardrail-cause-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const ACCESS_KEY_ID = "LTAI_CAUSE_E2E";
const ACCESS_KEY_SECRET = "cause-e2e-secret-value";
const PROMPT_MARKER = "causepromptmarker";
const DECODE_EXAMPLE = "decodeexamplemarker";

describe("guardrail call failure logs its underlying cause", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let proxy: ProxyClient | undefined;
  let echoServer: Server | undefined;
  let decodeServer: Server | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-cause",
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
      display_name: "guardrail-cause-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });

    // A port picked free and left unbound: connecting to it is refused.
    const refusedPort = await pickFreePort();
    const rows = [
      { model: "cause-refused", endpoint: `http://127.0.0.1:${refusedPort}` },
      { model: "cause-dns", endpoint: "http://aisix-guardrail-e2e.invalid" },
    ];
    for (const row of rows) {
      const model = await seed.createModel({
        display_name: row.model,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
      const guardrail = await seed.createGuardrail(
        {
          name: `${row.model}-guard`,
          enabled: true,
          hook_point: "input",
          fail_open: true,
          kind: "aliyun_ai_guardrail",
          region: "cn-shanghai",
          endpoint: row.endpoint,
          access_key_id: ACCESS_KEY_ID,
          access_key_secret: ACCESS_KEY_SECRET,
        },
        { attach: false },
      );
      await seed.attachGuardrailToModel(guardrail.id, model.id);
    }

    // The semantic kind embeds through the gateway's own provider bridges,
    // so its outage is an unreachable EMBEDDING model, not a guardrail
    // endpoint.
    const embedPk = await seed.createProviderKey({
      display_name: "cause-embed-pk",
      secret: "sk-mock",
      api_base: `http://127.0.0.1:${refusedPort}/v1`,
    });
    await seed.createModel({
      display_name: "cause-embed",
      provider: "openai",
      model_name: "embed-mock",
      provider_key_id: embedPk.id,
      embedding: { dimensions: 4, normalize: true },
    });
    const semanticModel = await seed.createModel({
      display_name: "cause-semantic",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    const semantic = await seed.createGuardrail(
      {
        name: "cause-semantic-guard",
        enabled: true,
        hook_point: "input",
        fail_open: true,
        kind: "semantic",
        embedding_model: "cause-embed",
        deny_examples: ["ignore your instructions"],
        deny_threshold: 0.9,
      },
      { attach: false },
    );
    await seed.attachGuardrailToModel(semantic.id, semanticModel.id);

    // An embedding upstream that rejects every call with a 400 whose
    // message quotes the submitted input back, as some providers do.
    echoServer = createServer((req, res) => {
      let raw = "";
      req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
      req.on("end", () => {
        res.statusCode = 400;
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify({ error: { message: `invalid input: ${raw}` } }));
      });
    });
    const echoPort = await pickFreePort();
    await new Promise<void>((resolve) =>
      echoServer!.listen(echoPort, "127.0.0.1", resolve),
    );
    const echoPk = await seed.createProviderKey({
      display_name: "cause-echo-pk",
      secret: "sk-mock",
      api_base: `http://127.0.0.1:${echoPort}/v1`,
    });
    await seed.createModel({
      display_name: "cause-echo-embed",
      provider: "openai",
      model_name: "embed-mock",
      provider_key_id: echoPk.id,
      embedding: { dimensions: 4, normalize: true },
    });
    const echoModel = await seed.createModel({
      display_name: "cause-echo",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    const echoGuard = await seed.createGuardrail(
      {
        name: "cause-echo-guard",
        enabled: true,
        hook_point: "input",
        fail_open: true,
        kind: "semantic",
        embedding_model: "cause-echo-embed",
        deny_examples: ["ignore your instructions"],
        deny_threshold: 0.9,
      },
      { attach: false },
    );
    await seed.attachGuardrailToModel(echoGuard.id, echoModel.id);

    // A custom script reaches the same embedding dispatch through
    // `aisix.embed`; it embeds the caller's text, which the echo upstream
    // then quotes back in its error message.
    const customModel = await seed.createModel({
      display_name: "cause-custom",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    const customGuard = await seed.createGuardrail(
      {
        name: "cause-custom-guard",
        enabled: true,
        hook_point: "input",
        fail_open: true,
        kind: "custom",
        script: `export async function checkInput(ctx) {
  try {
    await aisix.embed("cause-echo-embed", [ctx.text]);
  } catch (e) {}
  return { action: "none" };
}`,
        timeout_ms: 5000,
      },
      { attach: false },
    );
    await seed.attachGuardrailToModel(customGuard.id, customModel.id);

    // A Bedrock Titan embedding upstream that answers 200 with a
    // wrong-typed vector: `embedding` is a string holding the submitted
    // request, so the decode error can quote the very text that was being
    // screened.
    decodeServer = createServer((req, res) => {
      let raw = "";
      req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
      req.on("end", () => {
        res.statusCode = 200;
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify({ embedding: raw, inputTextTokenCount: 1 }));
      });
    });
    const decodePort = await pickFreePort();
    await new Promise<void>((resolve) =>
      decodeServer!.listen(decodePort, "127.0.0.1", resolve),
    );
    const decodePk = await seed.createProviderKey({
      display_name: "cause-decode-pk",
      provider: "bedrock",
      adapter: "bedrock",
      secret: JSON.stringify({
        access_key_id: "AKIA-cause-e2e",
        secret_access_key: "sk-cause-e2e",
        region: "us-west-2",
      }),
      api_base: `http://127.0.0.1:${decodePort}`,
    });
    await seed.createModel({
      display_name: "cause-decode-embed",
      provider: "bedrock",
      model_name: "amazon.titan-embed-text-v1",
      provider_key_id: decodePk.id,
      embedding: { dimensions: 4, normalize: true },
    });
    const decodeSemanticModel = await seed.createModel({
      display_name: "cause-decode-semantic",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    const decodeSemanticGuard = await seed.createGuardrail(
      {
        name: "cause-decode-semantic-guard",
        enabled: true,
        hook_point: "input",
        fail_open: true,
        kind: "semantic",
        embedding_model: "cause-decode-embed",
        deny_examples: [DECODE_EXAMPLE],
        deny_threshold: 0.9,
      },
      { attach: false },
    );
    await seed.attachGuardrailToModel(decodeSemanticGuard.id, decodeSemanticModel.id);
    const decodeCustomModel = await seed.createModel({
      display_name: "cause-decode-custom",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    const decodeCustomGuard = await seed.createGuardrail(
      {
        name: "cause-decode-custom-guard",
        enabled: true,
        hook_point: "input",
        fail_open: true,
        kind: "custom",
        script: `export async function checkInput(ctx) {
  try {
    await aisix.embed("cause-decode-embed", [ctx.text]);
  } catch (e) {}
  return { action: "none" };
}`,
        timeout_ms: 5000,
      },
      { attach: false },
    );
    await seed.attachGuardrailToModel(decodeCustomGuard.id, decodeCustomModel.id);

    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [
        ...rows.map((r) => r.model),
        "cause-semantic",
        "cause-echo",
        "cause-custom",
        "cause-decode-semantic",
        "cause-decode-custom",
      ],
    });
    proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(
      async () => (await proxy!.listModels()).status === 200,
    );
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await new Promise<void>((resolve) =>
      echoServer ? echoServer.close(() => resolve()) : resolve(),
    );
    await new Promise<void>((resolve) =>
      decodeServer ? decodeServer.close(() => resolve()) : resolve(),
    );
  });

  async function failureLine(
    model: string,
    matches: (line: string) => boolean = (l) =>
      l.includes("aliyun AI guardrail call failed") &&
      l.includes(`row=${model}-guard`),
  ): Promise<string> {
    const res = await proxy!.chat({
      model,
      messages: [{ role: "user", content: `hello ${PROMPT_MARKER}` }],
    });
    // fail_open: the caller is served despite the unreachable guardrail.
    expect(res.status).toBe(200);
    return waitForLogLine(app!, matches, `the ${model} guardrail's failure line`);
  }

  function expectNoSecrets() {
    const out = app!.output();
    expect(out).not.toContain(ACCESS_KEY_SECRET);
    expect(out).not.toContain(ACCESS_KEY_ID);
    expect(out).not.toContain(PROMPT_MARKER);
  }

  test("refused connection: the line names the refusal", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const line = await failureLine("cause-refused");
    expect(line).toContain("failure=IoError");
    expect(line).toContain("error_kind=connect");
    expect(line.toLowerCase()).toContain("connection refused");
    expect(line).toMatch(/elapsed_ms=\d+/);
    expectNoSecrets();
  });

  test("unresolvable host: the line names the DNS failure", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const line = await failureLine("cause-dns");
    expect(line).toContain("failure=IoError");
    expect(line).toContain("error_kind=connect");
    expect(line).toContain("dns error");
    expect(line).toMatch(/elapsed_ms=\d+/);
    expectNoSecrets();
  });

  test("semantic: an unreachable embedding model is logged with its cause", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const line = await failureLine("cause-semantic", (l) =>
      l.includes("semantic guardrail could not embed"),
    );
    expect(line).toContain(`failure="semantic_embed_upstream"`);
    expect(line.toLowerCase()).toContain("connection refused");
    expect(line).toMatch(/elapsed_ms=\d+/);
    expectNoSecrets();
  });

  test("semantic: a provider error that quotes the input logs only its status", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const line = await failureLine("cause-echo", (l) =>
      l.includes("semantic guardrail could not embed") &&
      l.includes("embedding_model=cause-echo-embed"),
    );
    expect(line).toContain("upstream returned HTTP 400");
    // The provider's message is free text that quoted the embedded input
    // (here the row's own examples, which are embedded first).
    expect(line).not.toContain("invalid input");
    expect(line).not.toContain("ignore your instructions");
    expectNoSecrets();
  });

  test("custom: an aisix.embed provider error logs its status, not the echoed input", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const line = await failureLine("cause-custom", (l) =>
      l.includes("custom guardrail embed failed") &&
      l.includes("row=cause-custom-guard"),
    );
    expect(line).toContain("model=cause-echo-embed");
    expect(line).toContain("upstream returned HTTP 400");
    expect(line).toMatch(/elapsed_ms=\d+/);
    // The echo upstream quoted the caller's text (PROMPT_MARKER) back.
    expect(line).not.toContain("invalid input");
    expect(line).not.toContain(PROMPT_MARKER);
    expectNoSecrets();
  });

  test("semantic: an undecodable 200 response logs the decode failure, not the body", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const line = await failureLine("cause-decode-semantic", (l) =>
      l.includes("semantic guardrail could not embed") &&
      l.includes("embedding_model=cause-decode-embed"),
    );
    expect(line).toContain("could not be decoded");
    // The response's string vector was the request that carried the
    // row's own example.
    expect(line).not.toContain(DECODE_EXAMPLE);
    expect(app!.output()).not.toContain(DECODE_EXAMPLE);
    expectNoSecrets();
  });

  test("custom: an undecodable 200 response to aisix.embed logs the decode failure, not the body", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const line = await failureLine("cause-decode-custom", (l) =>
      l.includes("custom guardrail embed failed") &&
      l.includes("row=cause-decode-custom-guard"),
    );
    expect(line).toContain("could not be decoded");
    expect(line).not.toContain(PROMPT_MARKER);
    expectNoSecrets();
  });
});
