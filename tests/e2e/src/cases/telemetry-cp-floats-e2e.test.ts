import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startMockControlPlane,
  startOpenAiUpstream,
  waitConfigPropagation,
  type MockControlPlane,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";
import { pickFreePort } from "../harness/ports.js";

// E2E: a number configured on a resource reaches `/dp/telemetry` as the
// operator wrote it. A semantic guardrail configured with a 0.9 threshold
// must be reported with `"threshold":0.9` in the bytes the managed gateway
// posts, not a binary approximation of it such as 0.8999999761581421, which
// a control plane comparing against the configured row reads as a
// different value.

const KEY = "sk-tcf-e2e-caller";
const hash = (plaintext: string) => createHash("sha256").update(plaintext).digest("hex");
const EMBED_MODEL = "tcf-embed";
const CHAT_MODEL = "tcf-chat";
const GUARDRAIL = "tcf-topic-guard";
const DENY_THRESHOLD = 0.9;

interface EmbeddingMock {
  baseUrl: string;
  close(): Promise<void>;
}

/** OpenAI-compatible `/v1/embeddings` answering one fixed vector per input. */
async function startEmbeddingMock(): Promise<EmbeddingMock> {
  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      const body = JSON.parse(raw || "{}") as { input?: string | string[] };
      const inputs = Array.isArray(body.input) ? body.input : [body.input ?? ""];
      res.writeHead(200, { "content-type": "application/json" }).end(
        JSON.stringify({
          object: "list",
          model: "embed-mock",
          data: inputs.map((_, index) => ({ object: "embedding", index, embedding: [1, 0, 0, 0] })),
          usage: { prompt_tokens: inputs.length, total_tokens: inputs.length },
        }),
      );
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    close: () => new Promise<void>((resolve, reject) => server.close((e) => (e ? reject(e) : resolve()))),
  };
}

describe("usage events sent to the control plane carry configured numbers as written", () => {
  let cp: MockControlPlane | undefined;
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let embed: EmbeddingMock | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    embed = await startEmbeddingMock();
    cp = await startMockControlPlane();
    app = await spawnApp({ extra: { managed: cp.managed }, awaitProxyListener: false });
    const seed = new SeedClient(etcd, `${app.etcdPrefix}/${cp.envId}`);

    const embedPk = await seed.createProviderKey({ display_name: "tcf-embed-pk", api_key: "sk-mock", api_base: `${embed.baseUrl}/v1` });
    await seed.createModel({
      display_name: EMBED_MODEL,
      provider: "openai",
      model_name: "embed-mock",
      provider_key_id: embedPk.id,
      embedding: { dimensions: 4, normalize: true },
    });
    const chatPk = await seed.createProviderKey({ display_name: "tcf-chat-pk", api_key: "sk-mock", api_base: `${upstream.baseUrl}/v1` });
    const chat = await seed.createModel({
      display_name: CHAT_MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: chatPk.id,
    });
    const guardrail = await seed.createGuardrail(
      {
        enabled: true,
        name: GUARDRAIL,
        hook_point: "input",
        enforcement_mode: "monitor",
        kind: "semantic",
        embedding_model: EMBED_MODEL,
        deny_examples: ["anything at all"],
        deny_threshold: DENY_THRESHOLD,
      },
      { attach: false },
    );
    await seed.attachGuardrailToModel(guardrail.id, chat.id);
    await seed.createApiKey({ key_hash: hash(KEY), allowed_models: ["*"] });

    await waitConfigPropagation(async () => {
      try {
        return (await new ProxyClient(app!.proxyUrl, KEY).listModels()).status === 200;
      } catch (e) {
        // The proxy listener binds once the first configuration applies.
        const err = e as { code?: string; cause?: { code?: string } };
        if ((err.code ?? err.cause?.code) === "ECONNREFUSED") return false;
        throw e;
      }
    });
  });

  afterAll(async () => {
    await app?.exit();
    await cp?.close();
    await upstream?.close();
    await embed?.close();
  });

  test("a semantic guardrail's threshold is posted as the configured 0.9", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await new ProxyClient(app.proxyUrl, KEY).chat({
      model: CHAT_MODEL,
      messages: [{ role: "user", content: "hello" }],
    });
    expect(res.status).toBe(200);

    const carrying = () => cp!.telemetryBodies.filter((body) => body.includes(`"guardrail_name":"${GUARDRAIL}"`));
    await waitConfigPropagation(async () => carrying().length > 0, 20_000);
    const body = carrying()[0]!;
    const event = (JSON.parse(body) as { events: { guardrail_scores?: { threshold: number }[] }[] }).events.find(
      (e) => e.guardrail_scores?.length,
    );
    expect(event?.guardrail_scores?.[0]?.threshold).toBe(DENY_THRESHOLD);
    // The bytes, not just the parsed number: the control plane decodes the
    // literal it receives.
    expect(body).toContain(`"threshold":${DENY_THRESHOLD}`);
  }, 60_000);
});
