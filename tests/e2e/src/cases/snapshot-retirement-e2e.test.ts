import { createHash, randomUUID } from "node:crypto";
import { createServer, type ServerResponse } from "node:http";
import { once } from "node:events";
import { expect, test } from "vitest";
import {
  EtcdClient, ProxyClient, spawnApp, waitConfigPropagation, type SpawnedApp,
} from "../harness/index.js";

test("configuration replacement preserves held requests and accepts cancellation", async (ctx) => {
  const etcd = new EtcdClient();
  if (!(await etcd.ping())) {
    ctx.skip();
    return;
  }
  const held: ServerResponse[] = [];
  const upstream = createServer((req, res) => {
    req.resume();
    req.on("end", () => held.push(res));
  });
  upstream.listen(0, "127.0.0.1");
  await once(upstream, "listening");
  const address = upstream.address();
  if (!address || typeof address === "string") throw new Error("missing upstream port");
  let app: SpawnedApp | undefined;
  const controllers: AbortController[] = [];
  const finish = (response: ServerResponse) => {
    response.writeHead(200, { "content-type": "application/json" });
    response.end(JSON.stringify({
      id: "held-response", object: "chat.completion", model: "gpt-4o-mini",
      choices: [{ index: 0, message: { role: "assistant", content: "complete" }, finish_reason: "stop" }],
      usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
    }));
  };
  try {
    app = await spawnApp();
    const prefix = app.etcdPrefix;
    const provider = randomUUID();
    const modelId = randomUUID();
    await etcd.put(`${prefix}/provider_keys/${provider}`, JSON.stringify({
      provider: "openai", adapter: "openai", display_name: "held-provider",
      secret: "sk-mock", api_base: `http://127.0.0.1:${address.port}/v1`,
    }));
    const putModel = (revision: number) => etcd.put(`${prefix}/models/${modelId}`, JSON.stringify({
      display_name: `held-model-${revision}`, provider: "openai",
      model_name: "gpt-4o-mini", provider_key_id: provider,
    }));
    const secret = `sk-retire-${randomUUID()}`;
    const caller = `${prefix}/api_keys/${randomUUID()}`;
    const callerConfig = JSON.stringify({
      key_hash: createHash("sha256").update(secret).digest("hex"), allowed_models: ["*"],
    });
    await putModel(0);
    await etcd.put(caller, callerConfig);
    const proxy = new ProxyClient(app.proxyUrl, secret);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);

    const requests = [];
    for (let revision = 0; revision < 3; revision++) {
      const controller = new AbortController();
      controllers.push(controller);
      requests.push(fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST", signal: controller.signal,
        headers: { authorization: `Bearer ${secret}`, "content-type": "application/json" },
        body: JSON.stringify({ model: `held-model-${revision}`, messages: [{ role: "user", content: "hold" }] }),
      }).then(async (response) => ({ status: response.status, body: await response.json() }))
        .catch((error: Error) => ({ error: error.name })));
      await expect.poll(() => held.length).toBe(revision + 1);

      // Revocation must become visible while older requests are still alive.
      await etcd.delete(caller);
      await waitConfigPropagation(async () => (await proxy.listModels()).status === 401);
      await putModel(revision + 1);
      await etcd.put(caller, callerConfig);
      await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
    }
    controllers[1].abort();
    expect(await requests[1]).toEqual({ error: "AbortError" });
    for (const i of [2, 0]) {
      finish(held[i]);
      expect(await requests[i]).toMatchObject({
        status: 200, body: { choices: [{ message: { content: "complete" } }] },
      });
    }
    const models = await proxy.listModels();
    expect(models.status).toBe(200);
    const body = models.body as { data: Array<{ id: string }> };
    expect(body.data.map((model) => model.id)).toEqual(["held-model-3"]);
    const live = await fetch(`${app.proxyUrl}/livez`);
    expect(live.status).toBe(200);
    await live.text();
    await etcd.deletePrefix(prefix);
  } finally {
    for (const controller of controllers) controller.abort();
    for (const response of held) response.destroy();
    await app?.exit();
    upstream.closeAllConnections();
    await new Promise<void>((resolve, reject) => upstream.close((error) => error ? reject(error) : resolve()));
  }
});
