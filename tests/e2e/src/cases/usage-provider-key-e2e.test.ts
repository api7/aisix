import { createHash } from "node:crypto";
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

const CALLER_KEY = "sk-provider-key-usage-caller";
const PRIMARY = "pk-usage-primary-model";
const FALLBACK = "pk-usage-fallback-model";
const GROUP = "pk-usage-routing-group";
const BLOCKED_INPUT = "hide-this-input";

interface UsageEvent {
  request_id: string;
  api_key_id: string;
  provider_key_id?: string;
  status_code: number;
  attempt_index: number;
  is_final_attempt: boolean;
}

describe("Provider Key IDs in managed usage telemetry", () => {
  let app: SpawnedApp | undefined;
  let cp: MockControlPlane | undefined;
  const upstreams: OpenAiUpstream[] = [];
  let callerId: string;
  let primaryKeyId: string;
  let fallbackKeyId: string;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    if (!(await etcd.ping())) return;

    const primary = await startOpenAiUpstream({ status: 502 });
    upstreams.push(primary);
    const fallback = await startOpenAiUpstream();
    upstreams.push(fallback);
    cp = await startMockControlPlane();
    app = await spawnApp({
      extra: { managed: cp.managed },
      awaitProxyListener: false,
    });
    const seed = new SeedClient(etcd, `${app.etcdPrefix}/${cp.envId}`);

    const addModel = async (name: string, upstream: OpenAiUpstream) => {
      const pk = await seed.createProviderKey({
        display_name: `${name}-credential`,
        api_key: `sk-${name}`,
        api_base: `${upstream.baseUrl}/v1`,
      });
      await seed.createModel({
        display_name: name,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
      return pk.id;
    };
    primaryKeyId = await addModel(PRIMARY, primary);
    fallbackKeyId = await addModel(FALLBACK, fallback);
    await seed.createModel({
      display_name: GROUP,
      routing: {
        strategy: "failover",
        targets: [{ model: PRIMARY }, { model: FALLBACK }],
        retries: 0,
        max_fallbacks: 1,
      },
    });
    await seed.createGuardrail({
      name: "pk-usage-input-guardrail",
      kind: "keyword",
      hook_point: "input",
      patterns: [{ kind: "literal", value: BLOCKED_INPUT }],
    });
    await seed.createCachePolicy({
      name: "pk-usage-cache",
      enabled: true,
      applies_to: `model:${FALLBACK}`,
    });
    const caller = await seed.createApiKey({
      key_hash: createHash("sha256").update(CALLER_KEY).digest("hex"),
      allowed_models: [GROUP, FALLBACK],
    });
    callerId = caller.id;
    const proxy = new ProxyClient(app.proxyUrl, CALLER_KEY);
    await waitConfigPropagation(async () => {
      try {
        return (await proxy.listModels()).status === 200;
      } catch (e) {
        const err = e as { code?: string; cause?: { code?: string } };
        if ((err.code ?? err.cause?.code) === "ECONNREFUSED") return false;
        throw e;
      }
    });
  });

  afterAll(async () => {
    await app?.exit();
    await cp?.close();
    await Promise.all(upstreams.map((upstream) => upstream.close()));
  });

  async function usageFor(requestId: string): Promise<UsageEvent[]> {
    const events = () => cp!.telemetryBodies
      .flatMap((body) => (JSON.parse(body) as { events: UsageEvent[] }).events)
      .filter((event) => event.request_id === requestId);
    await waitConfigPropagation(async () => events().some((event) => event.is_final_attempt), 20_000);
    return events();
  }

  function chat(model: string, content: string): Promise<Response> {
    return fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER_KEY}`, "content-type": "application/json" },
      body: JSON.stringify({ model, messages: [{ role: "user", content }] }),
    });
  }

  test("each fallback attempt carries the ID of the credential it used", async (ctx) => {
    if (!app) return ctx.skip();
    const res = await chat(GROUP, "hi");
    expect(res.status).toBe(200);
    expect(upstreams.map((upstream) => upstream.receivedRequests.length)).toEqual([1, 1]);
    expect(upstreams[0]!.receivedRequests[0]?.headers.authorization).toBe(`Bearer sk-${PRIMARY}`);
    expect(upstreams[1]!.receivedRequests[0]?.headers.authorization).toBe(`Bearer sk-${FALLBACK}`);
    const requestId = res.headers.get("x-aisix-request-id");
    expect(requestId).toBeTruthy();
    expect(await usageFor(requestId!)).toMatchObject([
      { api_key_id: callerId, provider_key_id: primaryKeyId, status_code: 502, attempt_index: 0, is_final_attempt: false },
      { api_key_id: callerId, provider_key_id: fallbackKeyId, status_code: 200, attempt_index: 1, is_final_attempt: true },
    ]);
  });

  test("a local rejection on a model with a configured key omits the key ID", async (ctx) => {
    if (!app) return ctx.skip();
    const before = upstreams.map((upstream) => upstream.receivedRequests.length);
    const res = await chat(FALLBACK, BLOCKED_INPUT);
    expect(res.status).toBe(422);
    expect(upstreams.map((upstream) => upstream.receivedRequests.length)).toEqual(before);
    const requestId = res.headers.get("x-aisix-request-id");
    expect(requestId).toBeTruthy();
    const events = await usageFor(requestId!);
    expect(events).toMatchObject([{ api_key_id: callerId, status_code: 422, is_final_attempt: true }]);
    expect(events[0]).not.toHaveProperty("provider_key_id");
  });

  test("a cache hit omits the configured key ID after the miss reports it", async (ctx) => {
    if (!app) return ctx.skip();
    const fallback = upstreams[1]!;
    const before = fallback.receivedRequests.length;
    const miss = await chat(FALLBACK, "cache this reply");
    expect(miss.status).toBe(200);
    const body = await miss.text();
    const hit = await chat(FALLBACK, "cache this reply");
    expect(hit.status).toBe(200);
    expect(await hit.text()).toBe(body);
    expect(fallback.receivedRequests).toHaveLength(before + 1);

    expect(await usageFor(miss.headers.get("x-aisix-request-id")!)).toMatchObject([
      { api_key_id: callerId, provider_key_id: fallbackKeyId, is_final_attempt: true },
    ]);
    const events = await usageFor(hit.headers.get("x-aisix-request-id")!);
    expect(events).toMatchObject([{ api_key_id: callerId, is_final_attempt: true }]);
    expect(events[0]).not.toHaveProperty("provider_key_id");
  });
});
