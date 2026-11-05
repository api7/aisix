import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startA2aUpstream,
  waitConfigPropagation,
  type A2aUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// The agent card a gateway serves advertises what X-Forwarded-Proto says, and
// https when it says nothing usable.

const KEY = "sk-a2a-card-scheme";
const AGENT = "scheme-agent";

describe("the served agent card's scheme", () => {
  let app: SpawnedApp | undefined;
  let upstream: A2aUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    upstream = await startA2aUpstream({ cardMount: "origin" });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.update("a2a_agents", randomUUID(), {
      name: AGENT,
      url: upstream.url,
      protocol_version: "1.0",
      auth_type: "none",
      enabled: true,
    });
    await seed.createApiKey({
      key_hash: createHash("sha256").update(KEY).digest("hex"),
      allowed_models: [],
      allowed_agents: ["*"],
    });
    // Seeded last, so its key authenticating implies the agent has arrived too.
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${KEY}` },
      });
      return res.status === 200;
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  async function cardUrl(forwarded?: string) {
    const res = await fetch(`${app!.proxyUrl}/a2a/${AGENT}/.well-known/agent-card.json`, {
      headers: {
        authorization: `Bearer ${KEY}`,
        ...(forwarded === undefined ? {} : { "x-forwarded-proto": forwarded }),
      },
    });
    expect(res.status).toBe(200);
    return ((await res.json()) as { url: string }).url;
  }

  test.each<[string, string | undefined, string]>([
    ["absent", undefined, "https"],
    ["http", "http", "http"],
    ["https", "https", "https"],
    ["http in capitals", "HTTP", "http"],
    ["a chain of proxies, by its first hop", "http, https", "http"],
    ["empty", "", "https"],
    ["no scheme it can advertise", "ftp", "https"],
  ])("X-Forwarded-Proto %s", async (_what, forwarded, scheme) => {
    if (!etcdReachable || !app) return;
    expect(await cardUrl(forwarded)).toBe(`${scheme}://${new URL(app.proxyUrl).host}/a2a/${AGENT}`);
  });
});
