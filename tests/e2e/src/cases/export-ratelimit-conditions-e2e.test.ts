import { execFile } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { join } from "node:path";
import { promisify } from "node:util";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  awaitWindowHeadroom,
  EtcdClient,
  etcdEndpoint,
  ProxyClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// A conditional rate-limit policy whose condition leaves name a model and
// an API key must survive `aisix export` into a resources file: the
// file-mode gateway loads it and throttles exactly what the etcd-backed
// gateway throttles. The control plane writes those leaves as resource
// ids; a resources file names resources by their entry names.

const execFileP = promisify(execFile);
const BIN_PATH =
  process.env.AISIX_BIN ?? join(process.cwd(), "..", "..", "target", "debug", "aisix");

const LIMITED_KEY = "sk-export-rlc-limited";
const FREE_KEY = "sk-export-rlc-free";
const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

const LIMITED_MODEL = "erlc-limited";
const FREE_MODEL = "erlc-free";

describe("aisix export: a conditional rate-limit policy on model and api_key leaves", () => {
  let etcdApp: SpawnedApp | undefined;
  let fileApp: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let exported = "";
  let exportStderr = "";

  const chat = (app: SpawnedApp, key: string, model: string) =>
    new ProxyClient(app.proxyUrl, key).chat({
      model,
      messages: [{ role: "user", content: "hello" }],
    });

  beforeAll(async () => {
    const etcd = new EtcdClient();
    expect(await etcd.ping(), "this e2e requires etcd").toBe(true);
    upstream = await startOpenAiUpstream();
    etcdApp = await spawnApp();
    const seed = new SeedClient(etcd, etcdApp.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "erlc-pk",
      secret: "sk-upstream",
      api_base: `${upstream.baseUrl}/v1`,
    });
    const models: Record<string, string> = {};
    for (const name of [LIMITED_MODEL, FREE_MODEL]) {
      const m = await seed.createModel({
        display_name: name,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
      models[name] = m.id;
    }
    const limitedKeyId = randomUUID();
    await seed.update("rate_limit_policies", randomUUID(), {
      name: "erlc-policy",
      conditions: [
        { dimension: "model", operator: "in", value: [models[LIMITED_MODEL]] },
        { dimension: "api_key", operator: "==", value: limitedKeyId },
      ],
      group_by: ["model"],
      limits: { rpm: 1 },
    });
    await seed.update("api_keys", limitedKeyId, {
      key_hash: sha256(LIMITED_KEY),
      allowed_models: ["*"],
    });
    await seed.createApiKey({ key_hash: sha256(FREE_KEY), allowed_models: ["*"] });

    const probe = new ProxyClient(etcdApp.proxyUrl, FREE_KEY);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);

    const { stdout, stderr } = await execFileP(BIN_PATH, [
      "export",
      "--etcd",
      etcdEndpoint(),
      "--prefix",
      etcdApp.etcdPrefix,
      "--reveal-secrets",
    ], { timeout: 30_000 });
    exported = stdout;
    exportStderr = stderr;
  });

  afterAll(async () => {
    await fileApp?.exit();
    await etcdApp?.exit();
    await upstream?.close();
  });

  test("the exported file loads and throttles what the etcd gateway throttles", { timeout: 60_000 }, async () => {
    fileApp = await spawnApp({ resourcesFile: exported });
    for (const app of [etcdApp!, fileApp]) {
      // `rpm` counts in a wall-clock minute; keep the pair inside one.
      await awaitWindowHeadroom(10);
      expect((await chat(app, LIMITED_KEY, LIMITED_MODEL)).status).toBe(200);
      expect((await chat(app, LIMITED_KEY, LIMITED_MODEL)).status).toBe(429);
      // Neither leaf alone matches.
      expect((await chat(app, FREE_KEY, LIMITED_MODEL)).status).toBe(200);
      expect((await chat(app, LIMITED_KEY, FREE_MODEL)).status).toBe(200);
    }
    expect(exportStderr).not.toContain("erlc-policy");
  });
});
