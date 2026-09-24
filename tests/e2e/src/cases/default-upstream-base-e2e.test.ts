import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  waitConfigPropagation,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: a Provider Key with no `api_base` whose vendor has a built-in
// default reaches that default on every route, not only on chat.
//
// The defaults are the real public hosts, which a spec cannot stand up.
// The gateway is instead started behind a forward proxy
// (`HTTPS_PROXY`) that records the CONNECT target and refuses the
// tunnel: the recorded host is the upstream the gateway chose, and no
// request ever leaves the machine. Before the fix these routes answered
// 400 ("has no api_base configured") without dialing anything.

const CALLER_PLAINTEXT = "sk-default-upstream-base";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

interface ConnectRecorder {
  url: string;
  targets: string[];
  close(): Promise<void>;
}

async function startConnectRecorder(): Promise<ConnectRecorder> {
  const targets: string[] = [];
  const server: Server = createServer((_req, res) => {
    res.writeHead(502).end();
  });
  server.on("connect", (req, socket) => {
    targets.push(req.url ?? "");
    socket.end("HTTP/1.1 502 Bad Gateway\r\ncontent-length: 0\r\n\r\n");
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${port}`,
    targets,
    close: () =>
      new Promise<void>((resolve) => {
        server.closeAllConnections();
        server.close(() => resolve());
      }),
  };
}

describe("default upstream base on the direct-HTTP routes e2e", () => {
  let app: SpawnedApp | undefined;
  let proxy: ConnectRecorder | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    proxy = await startConnectRecorder();
    app = await spawnApp({
      extraEnv: {
        HTTPS_PROXY: proxy.url,
        https_proxy: proxy.url,
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const anthropicPk = await seed.createProviderKey({
      display_name: "anthropic-no-base",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-e2e",
    });
    const legacyAnthropicPk = await seed.createProviderKey({
      display_name: "legacy-anthropic-no-base",
      provider: "",
      adapter: "anthropic",
      secret: "sk-ant-legacy-e2e",
    });
    const legacyOpenaiPk = await seed.createProviderKey({
      display_name: "legacy-openai-no-base",
      provider: "",
      adapter: "openai",
      secret: "sk-oai-legacy-e2e",
    });
    await seed.createModel({
      display_name: "anthropic-default",
      provider: "anthropic",
      model_name: "claude-sonnet-4-5",
      provider_key_id: anthropicPk.id,
    });
    await seed.createModel({
      display_name: "legacy-anthropic-default",
      provider: "anthropic",
      model_name: "claude-sonnet-4-5",
      provider_key_id: legacyAnthropicPk.id,
    });
    await seed.createModel({
      display_name: "legacy-openai-default",
      provider: "openai",
      model_name: "gpt-4o-mini-tts",
      provider_key_id: legacyOpenaiPk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [
        "anthropic-default",
        "legacy-anthropic-default",
        "legacy-openai-default",
      ],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await proxy?.close();
  });

  async function ready(): Promise<SpawnedApp> {
    const probe = new ProxyClient(app!.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      return res.status === 200;
    });
    return app!;
  }

  async function post(path: string, body: unknown): Promise<number> {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
      },
      body: JSON.stringify(body),
    });
    await res.arrayBuffer();
    return res.status;
  }

  const messages = [{ role: "user", content: "hi" }];

  test.each([
    ["/v1/messages", "anthropic-default", { max_tokens: 8, messages }],
    ["/v1/messages/count_tokens", "anthropic-default", { messages }],
    ["/v1/messages", "legacy-anthropic-default", { max_tokens: 8, messages }],
    ["/v1/messages/count_tokens", "legacy-anthropic-default", { messages }],
  ])(
    "%s on %s dials api.anthropic.com",
    async (path, model, body, ctx) => {
      if (!etcdReachable || !app) {
        ctx.skip();
        return;
      }
      await ready();
      const before = proxy!.targets.length;
      const status = await post(path, { model, ...body });
      expect(status).not.toBe(400);
      expect(proxy!.targets.slice(before)).toContain("api.anthropic.com:443");
    },
  );

  test("legacy empty-vendor openai-adapter key dials api.openai.com on /v1/audio/speech", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    await ready();
    const before = proxy!.targets.length;
    const status = await post("/v1/audio/speech", {
      model: "legacy-openai-default",
      input: "hi",
      voice: "alloy",
    });
    expect(status).not.toBe(400);
    expect(proxy!.targets.slice(before)).toContain("api.openai.com:443");
  });
});
