import { createHash, randomUUID } from "node:crypto";
import { expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type SpawnedApp,
} from "../harness/index.js";

test("policy candidates follow creation, identity edits, fallback conditions and deletion", async (ctx) => {
  const etcd = new EtcdClient();
  if (!(await etcd.ping())) {
    ctx.skip();
    return;
  }
  const upstream = await startOpenAiUpstream();
  let app: SpawnedApp | undefined;
  try {
    app = await spawnApp();
    const prefix = app.etcdPrefix;
    const provider = randomUUID();
    const model = randomUUID();
    const policy = `${prefix}/rate_limit_policies/${randomUUID()}`;
    const callers = [randomUUID(), randomUUID()];
    const secrets = callers.map((id) => `sk-${id}`);
    const key = (secret: string, member: string) => JSON.stringify({
      key_hash: createHash("sha256").update(secret).digest("hex"),
      allowed_models: ["*"], user_id: member,
    });
    await etcd.put(`${prefix}/provider_keys/${provider}`, JSON.stringify({
      provider: "openai", adapter: "openai", display_name: "candidate-provider",
      secret: "sk-mock", api_base: `${upstream.baseUrl}/v1`,
    }));
    await etcd.put(`${prefix}/models/${model}`, JSON.stringify({
      display_name: "candidate-model", provider: "openai", model_name: "gpt-4o-mini",
      provider_key_id: provider,
    }));
    const conditional = (conditions: unknown[]) => JSON.stringify({
      name: "candidate-policy", conditions, limits: { rpd: 1 },
    });
    const member = (value: string) => ({ dimension: "member", operator: "==", value });
    await etcd.put(policy, conditional([member("unrelated")]));
    for (let i = 0; i < callers.length; i++) {
      await etcd.put(`${prefix}/api_keys/${callers[i]}`, key(secrets[i], callers[i]));
    }
    // A fresh caller written after each mutation observes its complete prefix
    // without consuming the policy's quota while waiting for propagation.
    const synced = async () => {
      const id = randomUUID();
      const secret = `sk-marker-${id}`;
      await etcd.put(`${prefix}/api_keys/${id}`, key(secret, "readiness"));
      const client = new ProxyClient(app!.proxyUrl, secret);
      await waitConfigPropagation(async () => (await client.listModels()).status === 200);
    };
    const request = async (caller: number, status: number) => {
      const client = new ProxyClient(app!.proxyUrl, secrets[caller]);
      const response = await client.chat({
        model: "candidate-model", messages: [{ role: "user", content: "hello" }],
      });
      expect(response.status, JSON.stringify(response.body)).toBe(status);
    };
    await synced();
    await request(0, 200);
    await request(1, 200);

    await etcd.put(policy, conditional([member(callers[0])]));
    await synced();
    await request(0, 200);
    await request(0, 429);
    await request(1, 200);

    await etcd.put(policy, conditional([member(callers[1])]));
    await synced();
    await request(0, 200);
    // The unchanged policy id retains its shared counter across the edit.
    await request(1, 429);

    await etcd.put(policy, conditional([
      { logic: "or", children: [member(callers[0]), member("unrelated")] },
    ]));
    await synced();
    await request(0, 429);
    await request(1, 200);
    await etcd.delete(policy);
    await synced();
    await request(0, 200);

    const classic = (caller: number) => JSON.stringify({
      name: "classic-candidate", scope: "api_key", scope_ref: callers[caller],
      window: "day", max_requests: 1,
    });
    await etcd.put(policy, classic(0));
    await synced();
    await request(0, 200);
    await request(0, 429);
    await request(1, 200);
    await etcd.put(policy, classic(1));
    await synced();
    await request(0, 200);
    await request(1, 200);
    await request(1, 429);
    await etcd.delete(policy);
    await synced();
    await request(1, 200);
  } finally {
    if (app) {
      await app.stop();
      await etcd.deletePrefix(app.etcdPrefix);
    }
    await upstream.close();
  }
}, 30_000);
