import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { harnessRequest } from "../harness/http.js";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

const KEY = "sk-url-rewrite-routing";
const BODY = { model: "chat-alias", messages: [{ role: "user", content: "hello" }] };

describe("URL rewrite precedes every route and respects host scope", () => {
  let app: SpawnedApp | undefined;
  let agent: OpenAiUpstream | undefined;
  let llm: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    agent = await startOpenAiUpstream({ nonStreamBody: { servedBy: "agent" } });
    llm = await startOpenAiUpstream();
    app = await spawnApp({
      extraEnv: {
        AISIX_PROXY__URL_REWRITES: JSON.stringify([
          { hosts: ["gw.example.com"], match: "^/(?:pjt/)?chat$", rewrite: "/v1/chat/completions" },
          { hosts: ["*.tenant.example.com"], match: "^/chat$", rewrite: "/rewritten/chat" },
          { match: "^/legacy/(.*)$", rewrite: "/rewritten/$1" },
          { match: "^/rewritten/.*$", rewrite: "/must-not-cascade" },
        ]),
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "rewrite-provider", api_key: "upstream-secret", api_base: `${llm.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "chat-alias", provider: "openai", model_name: "upstream-model",
      provider_key_id: pk.id,
    });
    for (const [prefix, mount] of [["/pjt", "/path"], ["/rewritten", "/path"], ["/chat", "/unchanged"]]) {
      await seed.createPassthroughRoute({
        name: `path-${prefix.slice(1)}`, path_prefix: prefix,
        target_url: `${agent.baseUrl}${mount}`, credential_mode: "forward_client",
      });
    }
    await seed.createPassthroughRoute({
      name: "host-route",
      hosts: ["relay.example.com", "other.example.com", "*.tenant.example.com"],
      target_url: `${agent.baseUrl}/host`, credential_mode: "forward_client",
    });
    await seed.createPassthroughRoute({
      name: "combined-route", hosts: ["combo.example.com"], path_prefix: "/rewritten",
      target_url: `${agent.baseUrl}/combined`, credential_mode: "forward_client",
    });
    await seed.createApiKey({
      key_hash: createHash("sha256").update(KEY).digest("hex"),
      allowed_models: ["*"], allowed_routes: ["*"],
    });
    await waitConfigPropagation(async () => {
      const res = await harnessRequest(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${KEY}` },
      });
      await res.body.text();
      return res.statusCode === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await agent?.close();
    await llm?.close();
  });

  for (const path of ["/chat", "/pjt/chat"]) {
    test(`${path} on the gateway host uses the standard Chat pipeline`, async (ctx) => {
      if (!etcdReachable || !app || !agent || !llm) return ctx.skip();
      const beforeAgent = agent.receivedRequests.length;
      const beforeLlm = llm.receivedRequests.length;
      const res = await harnessRequest(`${app.proxyUrl}${path}`, {
        method: "POST",
        headers: {
          host: "GW.Example.COM:8443", authorization: `Bearer ${KEY}`,
          "content-type": "application/json", "x-aisix-request-id": "rewrite-chat",
        },
        body: JSON.stringify(BODY),
      });
      const responseBody = await res.body.json();
      expect(res.statusCode, JSON.stringify(responseBody)).toBe(200);
      expect(responseBody).toMatchObject({
        choices: [{ message: { content: "mock reply" } }],
        usage: { prompt_tokens: 5, completion_tokens: 3 },
      });
      expect(res.headers["x-aisix-request-id"]).toBe("rewrite-chat");
      expect(agent.receivedRequests).toHaveLength(beforeAgent);
      expect(llm.receivedRequests).toHaveLength(beforeLlm + 1);
      expect(llm.receivedRequests.at(-1)).toMatchObject({
        path: "/v1/chat/completions",
        headers: { authorization: "Bearer upstream-secret" },
      });
      expect(JSON.parse(llm.receivedRequests.at(-1)!.body)).toMatchObject({ model: "upstream-model" });
    });

  }

  for (const [host, path, expected] of [
    ["relay.example.com", "/legacy/item?probe=a%2Fb", "/host/rewritten/item?probe=a%2Fb"],
    ["combo.example.com", "/legacy/item?probe=1", "/combined/item?probe=1"],
    ["gw.example.com", "/legacy/item", "/path/item"],
    ["other.example.com", "/chat", "/host/chat"],
    ["other.example.com", "/pjt/chat", "/host/pjt/chat"],
    ["A.tenant.EXAMPLE.com:8080", "/chat", "/host/rewritten/chat"],
    ["tenant.example.com", "/chat", "/unchanged"],
    ["two.one.tenant.example.com", "/chat", "/unchanged"],
  ]) {
    test(`${host}${path} reaches the expected passthrough path`, async (ctx) => {
      if (!etcdReachable || !app || !agent || !llm) return ctx.skip();
      const beforeAgent = agent.receivedRequests.length;
      const beforeLlm = llm.receivedRequests.length;
      const res = await harnessRequest(`${app.proxyUrl}${path}`, {
        method: "POST",
        headers: { host, authorization: `Bearer ${KEY}`, "content-type": "application/json" },
        body: JSON.stringify(BODY),
      });
      expect(res.statusCode).toBe(200);
      expect(await res.body.json()).toEqual({ servedBy: "agent" });
      expect(llm.receivedRequests).toHaveLength(beforeLlm);
      expect(agent.receivedRequests).toHaveLength(beforeAgent + 1);
      expect(agent.receivedRequests.at(-1)).toMatchObject({ path: expected, body: JSON.stringify(BODY) });
    });

  }

  test("rewritten Chat and host passthrough requests still require authentication", async (ctx) => {
    if (!etcdReachable || !app || !agent || !llm) return ctx.skip();
    const beforeAgent = agent.receivedRequests.length;
    const beforeLlm = llm.receivedRequests.length;
    for (const [host, path] of [["gw.example.com", "/chat"], ["relay.example.com", "/legacy/chat"]]) {
      const res = await harnessRequest(`${app.proxyUrl}${path}`, {
        method: "POST", headers: { host, "content-type": "application/json" },
        body: JSON.stringify(BODY),
      });
      await res.body.text();
      expect(res.statusCode).toBe(401);
    }
    expect(agent.receivedRequests).toHaveLength(beforeAgent);
    expect(llm.receivedRequests).toHaveLength(beforeLlm);
  });
});
