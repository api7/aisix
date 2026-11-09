import { createHash } from "node:crypto";
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

// E2E: a gateway with no configured log level writes WARN and above, plus its
// access log, which only `observability.access_log` turns off. A lower level
// or `RUST_LOG` doesn't touch the access log, and `info` still brings back the
// lines the default hides.

const ACCESS_LINE = "proxy request completed";
const PROVIDER_CALL_LINE = "provider call completed";
const BOOT_LINE = "tracing initialised";
const CALLER_PLAINTEXT = "sk-default-log-level-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const MODEL = "default-log-level-e2e";

const CHAT_COMPLETION = {
  id: "chatcmpl-default-log-level",
  object: "chat.completion",
  model: "gpt-4o-mini",
  choices: [{ index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "stop" }],
  usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
};

/** One request; the id it was answered under. */
async function chat(app: SpawnedApp): Promise<string> {
  const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: { authorization: `Bearer ${CALLER_PLAINTEXT}`, "content-type": "application/json" },
    body: JSON.stringify({ model: MODEL, messages: [{ role: "user", content: "hello" }] }),
  });
  await res.text();
  expect(res.status).toBe(200);
  const id = res.headers.get("x-aisix-request-id");
  expect(id, "every response carries x-aisix-request-id").toBeTruthy();
  return id!;
}

describe("the gateway's default log level", () => {
  let upstream: OpenAiUpstream | undefined;
  const apps: Record<string, SpawnedApp> = {};
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({ nonStreamBody: CHAT_COMPLETION });
    apps.byDefault = await spawnApp({ binaryDefaultLogLevel: true });
    apps.switchedOff = await spawnApp({ binaryDefaultLogLevel: true, accessLog: false });
    apps.errorOnly = await spawnApp({ logLevel: "error" });
    apps.info = await spawnApp({ logLevel: "info" });

    for (const app of Object.values(apps)) {
      const seed = new SeedClient(etcd, app.etcdPrefix);
      const pk = await seed.createProviderKey({
        display_name: `${MODEL}-pk`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      await seed.createModel({
        display_name: MODEL,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
      await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: [MODEL] });
      const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
      await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
    }
  });

  afterAll(async () => {
    for (const app of Object.values(apps)) await app.exit();
    await upstream?.close();
  });

  test("no configured level: the access log is written, INFO lines are not", async (ctx) => {
    if (!etcdReachable) {
      ctx.skip();
      return;
    }
    const app = apps.byDefault!;
    const id = await chat(app);
    const line = await waitForLogLine(
      app,
      (l) => l.includes(ACCESS_LINE) && l.includes(`request_id="${id}"`),
      "the access-log line",
    );
    expect(line).toContain("status_code=200");
    // Who connected comes from the request's context, which the default
    // level must not strip from the lines it does write.
    expect(line).toMatch(/peer=127\.0\.0\.1:\d+/);

    // A later request's access line proves the writer has passed the first
    // request's lines, however they were ordered.
    const later = await chat(app);
    await waitForLogLine(
      app,
      (l) => l.includes(ACCESS_LINE) && l.includes(`request_id="${later}"`),
      "the second access-log line",
    );
    expect(app.output()).not.toContain(PROVIDER_CALL_LINE);
    expect(app.output()).not.toContain(BOOT_LINE);
  });

  test("access_log: false turns the access log off at the default level", async (ctx) => {
    if (!etcdReachable) {
      ctx.skip();
      return;
    }
    const app = apps.switchedOff!;
    await chat(app);
    // Nothing at this level marks the end of the request's lines, so give
    // what the exit flushed time to come through the pipe.
    await app.stop();
    await new Promise((r) => setTimeout(r, 200));
    expect(app.output()).not.toContain(ACCESS_LINE);
  });

  test("a stricter level and RUST_LOG leave the access log on", async (ctx) => {
    if (!etcdReachable) {
      ctx.skip();
      return;
    }
    const app = apps.errorOnly!;
    const id = await chat(app);
    await waitForLogLine(
      app,
      (l) => l.includes(ACCESS_LINE) && l.includes(`request_id="${id}"`),
      "the access-log line at error",
    );
  });

  test("log_level: info brings the INFO lines back", async (ctx) => {
    if (!etcdReachable) {
      ctx.skip();
      return;
    }
    const app = apps.info!;
    const id = await chat(app);
    await waitForLogLine(
      app,
      (l) => l.includes(PROVIDER_CALL_LINE) && l.includes(`request_id="${id}"`),
      "the provider-call line",
    );
    expect(app.output()).toContain(BOOT_LINE);
  });
});
