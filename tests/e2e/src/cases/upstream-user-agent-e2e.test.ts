import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { join } from "node:path";
import { promisify } from "node:util";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

const CALLER_KEY = "sk-user-agent-e2e";
const routes = [
  {
    path: "/v1/chat/completions",
    body: { messages: [{ role: "user", content: "hi" }] },
  },
  {
    path: "/v1/messages",
    body: { max_tokens: 16, messages: [{ role: "user", content: "hi" }] },
  },
  {
    path: "/v1/responses",
    body: { input: "hi" },
  },
];

describe.each([false, true])("upstream User-Agent (threadPerCore=%s)", (threadPerCore) => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let version: string;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    if (!(await etcd.ping())) return;

    const binary =
      process.env.AISIX_BIN ?? join(process.cwd(), "../../target/debug/aisix");
    const { stdout } = await promisify(execFile)(binary, ["--version"]);
    expect(stdout.trim()).toMatch(/^aisix \S+$/);
    version = stdout.trim().slice("aisix ".length);

    upstream = await startOpenAiUpstream();
    app = await spawnApp({ threadPerCore });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    for (const pool of ["default", "provider-key"]) {
      const pk = await seed.createProviderKey({
        display_name: `ua-${pool}`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
        // Select the per-key client even on loopback HTTP.
        ...(pool === "provider-key" ? { tls: { verify: false } } : {}),
      });
      await seed.createModel({
        display_name: `ua-${pool}`,
        provider: "openai",
        model_name: "mock-model",
        provider_key_id: pk.id,
      });
    }
    await seed.createApiKey({
      key_hash: createHash("sha256").update(CALLER_KEY).digest("hex"),
      allowed_models: ["*"],
    });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_KEY}` },
      });
      await res.arrayBuffer();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  describe.each(["default", "provider-key"])("%s pool", (pool) => {
    test.for(routes)("$path reports the binary's version upstream", async (route, ctx) => {
      if (!app || !upstream) {
        ctx.skip();
        return;
      }
      const before = upstream.receivedRequests.length;
      const res = await fetch(`${app.proxyUrl}${route.path}`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${CALLER_KEY}`,
          "anthropic-version": "2023-06-01",
          "user-agent": "test-client/1.0",
        },
        body: JSON.stringify({ model: `ua-${pool}`, ...route.body }),
      });
      await res.arrayBuffer();
      expect(res.status).toBe(200);
      expect(res.headers.get("server")).toBe(`AISIX/${version}`);
      expect(upstream.receivedRequests).toHaveLength(before + 1);
      expect(upstream.receivedRequests[before].headers["user-agent"]).toBe(
        `aisix/${version}`,
      );
    });
  });
});
