import { createHash } from "node:crypto";
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

describe("guardrail call failure logs its underlying cause", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let proxy: ProxyClient | undefined;
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

    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: rows.map((r) => r.model),
    });
    proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(
      async () => (await proxy!.listModels()).status === 200,
    );
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  async function failureLine(model: string): Promise<string> {
    const res = await proxy!.chat({
      model,
      messages: [{ role: "user", content: `hello ${PROMPT_MARKER}` }],
    });
    // fail_open: the caller is served despite the unreachable guardrail.
    expect(res.status).toBe(200);
    return waitForLogLine(
      app!,
      (l) =>
        l.includes("aliyun AI guardrail call failed") &&
        l.includes(`row=${model}-guard`),
      `the ${model} guardrail's failure line`,
    );
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
});
