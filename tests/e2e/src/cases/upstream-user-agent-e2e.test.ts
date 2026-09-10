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
    provider: "openai",
    path: "/v1/chat/completions",
    body: { messages: [{ role: "user", content: "hi" }] },
  },
  {
    provider: "openai",
    path: "/v1/messages",
    body: { max_tokens: 16, messages: [{ role: "user", content: "hi" }] },
  },
  {
    provider: "openai",
    path: "/v1/responses",
    body: { input: "hi" },
  },
  {
    provider: "anthropic",
    path: "/v1/chat/completions",
    body: { messages: [{ role: "user", content: "hi" }] },
  },
];

describe.each([false, true])("upstream User-Agent (threadPerCore=%s)", (threadPerCore) => {
  let app: SpawnedApp | undefined;
  const upstreams: Record<string, OpenAiUpstream> = {};
  let version: string;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    if (!(await etcd.ping())) return;

    const binary =
      process.env.AISIX_BIN ?? join(process.cwd(), "../../target/debug/aisix");
    const { stdout } = await promisify(execFile)(binary, ["--version"]);
    expect(stdout.trim()).toMatch(/^aisix \S+$/);
    version = stdout.trim().slice("aisix ".length);

    upstreams.openai = await startOpenAiUpstream();
    upstreams.anthropic = await startOpenAiUpstream({
      nonStreamBody: {
        id: "msg-user-agent",
        type: "message",
        role: "assistant",
        model: "mock-model",
        content: [{ type: "text", text: "hi" }],
        stop_reason: "end_turn",
        stop_sequence: null,
        usage: { input_tokens: 5, output_tokens: 1 },
      },
    });
    app = await spawnApp({ threadPerCore });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    for (const [provider, upstream] of Object.entries(upstreams)) {
      for (const pool of ["default", "provider-key"]) {
        const pk = await seed.createProviderKey({
          display_name: `ua-${provider}-${pool}`,
          provider,
          adapter: provider,
          secret: "sk-mock",
          api_base: upstream.baseUrl + (provider === "openai" ? "/v1" : ""),
          // Select the per-key client even on loopback HTTP.
          ...(pool === "provider-key" ? { tls: { verify: false } } : {}),
        });
        await seed.createModel({
          display_name: `ua-${provider}-${pool}`,
          provider,
          model_name: "mock-model",
          provider_key_id: pk.id,
        });
      }
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
    await Promise.all(Object.values(upstreams).map((upstream) => upstream.close()));
  });

  describe.each(["default", "provider-key"])("%s pool", (pool) => {
    test.for(routes)("$provider $path reports the binary's version upstream", async (route, ctx) => {
      const upstream = upstreams[route.provider];
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
        body: JSON.stringify({ model: `ua-${route.provider}-${pool}`, ...route.body }),
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
