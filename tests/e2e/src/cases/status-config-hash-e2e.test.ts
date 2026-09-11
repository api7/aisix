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

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value !== null && typeof value === "object") {
    return `{${Object.entries(value)
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
      .map(([key, child]) => `${JSON.stringify(key)}:${canonical(child)}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

function digest(rows: Map<string, string>): string {
  const hash = createHash("sha256");
  for (const key of [...rows.keys()].sort()) {
    const raw = rows.get(key)!;
    let value: string;
    try {
      value = canonical(JSON.parse(raw));
    } catch {
      value = raw;
    }
    hash.update(`${key}\0${value}\n`);
  }
  return hash.digest("hex");
}

test("configuration digests match the written rows through updates, rejection, deletion and resync", async (ctx) => {
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
    const rows = new Map<string, string>();
    const provider = randomUUID();
    const modelKey = (i: number) =>
      `${prefix}/models/00000000-0000-4000-8000-${String(i).padStart(12, "0")}`;
    const model = (i: number) => JSON.stringify({
      display_name: `hash-model-${i}`,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: provider,
    });
    rows.set(`${prefix}/provider_keys/${provider}`, JSON.stringify({
      provider: "openai",
      adapter: "openai",
      display_name: "hash-provider",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    }));
    for (let i = 0; i < 256; i++) rows.set(modelKey(i), model(i));
    await etcd.putMany([...rows]);
    const plaintext = `sk-hash-${randomUUID()}`;
    const caller = `${prefix}/api_keys/${randomUUID()}`;
    rows.set(caller, JSON.stringify({
      key_hash: createHash("sha256").update(plaintext).digest("hex"),
      allowed_models: ["*"],
    }));
    await etcd.put(caller, rows.get(caller)!);
    const proxy = () => new ProxyClient(app!.proxyUrl, plaintext);
    await waitConfigPropagation(async () => (await proxy().listModels()).status === 200);

    const check = async (served = rows, rejected = 0) => {
      const sourceHash = digest(rows);
      const servedHash = digest(served);
      await waitConfigPropagation(async () => {
        const res = await fetch(`${app!.metricsUrl}/status/config`);
        expect(res.status).toBe(200);
        const status = await res.json() as {
          source: { source_hash: string };
          applied?: { config_hash: string };
          rejected: unknown[];
        };
        return (
          status.source.source_hash === sourceHash &&
          status.applied?.config_hash === servedHash &&
          status.rejected.length === rejected
        );
      });
    };
    const put = async (key: string, raw: string) => {
      rows.set(key, raw);
      await etcd.put(key, raw);
    };
    const remove = async (key: string) => {
      rows.delete(key);
      await etcd.delete(key);
    };
    await check();
    for (const i of [254, 127, 0]) {
      await put(modelKey(i), JSON.stringify({ ...JSON.parse(model(i)), model_name: "gpt-4o" }));
      await check();
    }
    await remove(modelKey(127));
    await check();
    await put(modelKey(256), model(256));
    await check();
    // A canonical-equivalent write still advances etcd while retaining
    // exactly the digest of the compact original document.
    await put(modelKey(0), JSON.stringify(JSON.parse(rows.get(modelKey(0))!), null, 2));
    await put(modelKey(1), JSON.stringify({ ...JSON.parse(model(1)), model_name: "gpt-4o" }));
    await check();

    const served = new Map(rows);
    await put(modelKey(190), "not JSON");
    await put(modelKey(257), "{}");
    await check(served, 2);
    const chat = await proxy().chat({
      model: "hash-model-190",
      messages: [{ role: "user", content: "last good model still serves" }],
    });
    expect(chat.status, JSON.stringify(chat.body)).toBe(200);
    await remove(modelKey(190));
    served.delete(modelKey(190));
    await check(served, 1);
    await put(modelKey(257), model(257));
    await check();

    // Spaced updates and key deletion must publish the same final bytes
    // regardless of how many watch events the gateway groups together.
    for (let i = 0; i < 24; i++) {
      await put(modelKey(i), JSON.stringify({ ...JSON.parse(model(i)), model_name: "gpt-4o" }));
      await new Promise((resolve) => setTimeout(resolve, 40));
    }
    const callerConfig = rows.get(caller)!;
    await remove(caller);
    await check();
    expect((await proxy().listModels()).status).toBe(401);
    await put(caller, callerConfig);
    await check();
    expect((await proxy().listModels()).status).toBe(200);

    await app.stop();
    await remove(modelKey(0));
    await put(modelKey(128), JSON.stringify({ ...JSON.parse(model(128)), model_name: "gpt-4o" }));
    app = await spawnApp({ etcdPrefix: prefix });
    await check();
    const afterResync = await proxy().chat({
      model: "hash-model-255",
      messages: [{ role: "user", content: "resync complete" }],
    });
    expect(afterResync.status, JSON.stringify(afterResync.body)).toBe(200);
    await etcd.deletePrefix(prefix);
    rows.clear();
    await check();
  } finally {
    await app?.exit();
    await upstream.close();
  }
});
