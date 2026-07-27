import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  pickFreePort,
  spawnApp,
  waitConfigPropagation,
  type SpawnedApp,
} from "../harness/index.js";

// E2E for `upstream.pool_idle_timeout_secs` (AISIX-Cloud#1122, #1126).
//
// The setting exists because a hop between the gateway and the provider
// (load balancer, NAT gateway, service mesh) closes idle connections on
// its own schedule. If the gateway's pool outlives that schedule, it
// eventually hands out a connection the far end has already dropped and
// the request fails with an opaque transport error.
//
// The failure itself only reproduces deterministically when the far end
// disappears *silently* (a dropped NAT mapping, no FIN) — hyper detects a
// clean FIN and discards the connection before reusing it. What is
// testable, and what the fix actually turns on, is whether the pool
// expires a connection at the configured deadline at all: with the knob
// set, a request after the deadline must open a new connection; without
// it, reqwest's own 90s default keeps the old one. An upstream that never
// closes first isolates the gateway's behaviour from the peer's.

const CALLER_PLAINTEXT = "sk-pool-idle-1126";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const POOL_IDLE_S = 1;
const IDLE_GAP_MS = POOL_IDLE_S * 1000 + 1500;

interface CountingUpstream {
  baseUrl: string;
  /** TCP connections accepted so far. */
  connections(): number;
  close(): Promise<void>;
}

/** An OpenAI-shaped upstream that counts accepted TCP connections and
 *  never closes an idle one first. */
async function startCountingUpstream(): Promise<CountingUpstream> {
  let connections = 0;
  const server: Server = createServer((_req, res) => {
    res.statusCode = 200;
    res.setHeader("content-type", "application/json");
    res.end(
      JSON.stringify({
        id: "cmpl-pool",
        object: "chat.completion",
        created: 0,
        model: "gpt-4o-mini",
        choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      }),
    );
  });
  // Well above the idle gap: whichever side closes, it is not this one.
  server.keepAliveTimeout = 120_000;
  server.headersTimeout = 125_000;
  server.on("connection", () => {
    connections += 1;
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    connections: () => connections,
    close: () =>
      new Promise<void>((resolve, reject) => server.close((e) => (e ? reject(e) : resolve()))),
  };
}

async function seedInto(app: SpawnedApp, etcd: EtcdClient, apiBase: string): Promise<void> {
  const seed = new SeedClient(etcd, app.etcdPrefix);
  const pk = (
    await seed.createProviderKey({
      display_name: "pool-idle-pk",
      secret: "sk-mock",
      api_base: `${apiBase}/v1`,
    })
  ).id;
  await seed.createModel({
    display_name: "pool-idle-model",
    provider: "openai",
    model_name: "gpt-4o-mini",
    provider_key_id: pk,
  });
  await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["pool-idle-model"] });
  await waitConfigPropagation(async () => {
    const res = await fetch(`${app.proxyUrl}/v1/models`, {
      headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
    });
    if (!res.ok) return false;
    const body = (await res.json()) as { data?: Array<{ id: string }> };
    return !!body.data?.some((m) => m.id === "pool-idle-model");
  });
}

async function callOnce(app: SpawnedApp): Promise<void> {
  const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: "pool-idle-model",
      messages: [{ role: "user", content: "hi" }],
    }),
  });
  expect(res.status).toBe(200);
  await res.json();
}

describe("upstream pool idle timeout", () => {
  let expiring: SpawnedApp | undefined;
  let holding: SpawnedApp | undefined;
  let expiringUpstream: CountingUpstream | undefined;
  let holdingUpstream: CountingUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    expiringUpstream = await startCountingUpstream();
    holdingUpstream = await startCountingUpstream();

    expiring = await spawnApp({ extra: { upstream: { pool_idle_timeout_secs: POOL_IDLE_S } } });
    // 0 switches the knob off, leaving reqwest's own 90s pool lifetime.
    holding = await spawnApp({ extra: { upstream: { pool_idle_timeout_secs: 0 } } });

    await seedInto(expiring, etcd, expiringUpstream.baseUrl);
    await seedInto(holding, etcd, holdingUpstream.baseUrl);
  });

  afterAll(async () => {
    await Promise.all([expiring?.exit(), holding?.exit()]);
    await Promise.all([expiringUpstream?.close(), holdingUpstream?.close()]);
  });

  test("a connection idle past the configured deadline is not reused", async (ctx) => {
    if (!etcdReachable || !expiring || !expiringUpstream) {
      ctx.skip();
      return;
    }
    await callOnce(expiring);
    expect(expiringUpstream.connections()).toBe(1);
    await new Promise((r) => setTimeout(r, IDLE_GAP_MS));
    await callOnce(expiring);
    expect(expiringUpstream.connections()).toBe(2);
  }, 30_000);

  // The control: the same gap without the knob keeps the pooled
  // connection, so the assertion above is measuring the setting and not
  // something the upstream or the runtime does on its own.
  test("with the knob off, the same gap reuses the pooled connection", async (ctx) => {
    if (!etcdReachable || !holding || !holdingUpstream) {
      ctx.skip();
      return;
    }
    await callOnce(holding);
    expect(holdingUpstream.connections()).toBe(1);
    await new Promise((r) => setTimeout(r, IDLE_GAP_MS));
    await callOnce(holding);
    expect(holdingUpstream.connections()).toBe(1);
  }, 30_000);
});
