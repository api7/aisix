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

// E2E: the `/a2a/{agent}` gateway endpoint against a real gateway + etcd + real
// upstream A2A agents. The A2A MVP (#717) deferred endpoint-level e2e; this is
// it, written around the two faults that made the endpoint unusable in
// practice.
//
// Pinned contract:
//   - every upstream call announces the agent's pinned wire version in
//     `A2A-Version`, on the card fetch as well as the JSON-RPC call (#911) —
//     without it an agent must read the call as 0.3 and reject a 1.0 body;
//   - a `0.3`-pinned agent is announced as 0.3, so the pin is what is sent
//     rather than a constant;
//   - an agent whose card is published under its own path prefix resolves, and
//     the catch-all 405 its platform returns at the origin is not mistaken for
//     the agent's answer (#913);
//   - the served card carries NO upstream address: the top-level `url` and
//     every `supportedInterfaces[].url` point back at the gateway, so a 1.0
//     caller (which reads its endpoint out of `supportedInterfaces`) cannot
//     route around the gateway;
//   - the upstream credential is presented by the gateway and never reaches
//     the caller; the caller's own key never reaches the upstream;
//   - per-agent ACL and unknown/disabled agents still gate the endpoint.

const KEY_ALLOWED = "sk-a2a-e2e-allowed";
const KEY_NO_AGENTS = "sk-a2a-e2e-no-agents";

const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

const messageSend = (id: string) => ({
  jsonrpc: "2.0",
  id,
  method: "message/send",
  params: {
    message: {
      role: "user",
      parts: [{ kind: "text", text: "invoice 42" }],
      messageId: "m-1",
    },
  },
});

describe("a2a gateway e2e: /a2a/{agent}", () => {
  let app: SpawnedApp | undefined;
  let rootHosted: A2aUpstream | undefined;
  let pathHosted: A2aUpstream | undefined;
  let etcdReachable = false;
  let seed: SeedClient;

  // The gate responses (401 / 403 / 404) are plain text, not JSON-RPC
  // envelopes, so the body is parsed opportunistically.
  const readBody = (text: string): Record<string, any> | undefined => {
    if (!text) return undefined;
    try {
      return JSON.parse(text) as Record<string, any>;
    } catch {
      return undefined;
    }
  };

  const call = async (path: string, token: string, body: unknown) => {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${token}`,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    return { status: res.status, raw: text, json: readBody(text) };
  };

  const fetchCard = async (agent: string, token: string) => {
    const res = await fetch(
      `${app!.proxyUrl}/a2a/${agent}/.well-known/agent-card.json`,
      { headers: { authorization: `Bearer ${token}` } },
    );
    const text = await res.text();
    return { status: res.status, raw: text, json: readBody(text) };
  };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    rootHosted = await startA2aUpstream({
      cardMount: "origin",
      token: "upstream-secret-tok",
    });
    pathHosted = await startA2aUpstream({ cardMount: "path" });
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    await seed.update("a2a_agents", randomUUID(), {
      name: "invoices",
      url: rootHosted.url,
      protocol_version: "1.0",
      auth_type: "bearer",
      secret: "upstream-secret-tok",
      enabled: true,
    });
    await seed.update("a2a_agents", randomUUID(), {
      name: "legacy",
      url: rootHosted.url,
      protocol_version: "0.3",
      auth_type: "bearer",
      secret: "upstream-secret-tok",
      enabled: true,
    });
    await seed.update("a2a_agents", randomUUID(), {
      name: "tenant",
      url: pathHosted.url,
      protocol_version: "1.0",
      auth_type: "none",
      enabled: true,
    });
    await seed.update("a2a_agents", randomUUID(), {
      name: "retired",
      url: rootHosted.url,
      protocol_version: "1.0",
      auth_type: "none",
      enabled: false,
    });

    await seed.createApiKey({
      key_hash: sha256(KEY_ALLOWED),
      allowed_models: [],
      allowed_agents: ["*"],
    });
    await seed.createApiKey({
      key_hash: sha256(KEY_NO_AGENTS),
      allowed_models: [],
    });

    // The gate proves only that the seeded rows reached the DP snapshot, and
    // deliberately asserts none of the behaviour under test: a 404 would mean
    // the agent row has not landed, a 401 that the key row has not. Anything
    // else — including a 403 or a 502 — means both are present, and any defect
    // in discovery, the version header, the card rewrite or the ACL then fails
    // its own test by name instead of surfacing as a 60s propagation timeout.
    await waitConfigPropagation(async () => {
      const probe = await call("/a2a/invoices", KEY_ALLOWED, messageSend("gate"));
      return probe.status !== 404 && probe.status !== 401;
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await rootHosted?.close();
    await pathHosted?.close();
  });

  test("announces the pinned wire version on every upstream call", async (ctx) => {
    if (!etcdReachable || !app || !rootHosted) return ctx.skip();
    rootHosted.requests.length = 0;

    await fetchCard("invoices", KEY_ALLOWED);
    const reply = await call("/a2a/invoices", KEY_ALLOWED, messageSend("e2e-1"));

    expect(reply.status).toBe(200);
    expect(reply.json?.result?.sawVersion).toBe("1.0");
    expect(reply.json?.result?.sawMethod).toBe("message/send");
    // A card fetch is an A2A request like any other, so EVERY hop carries the
    // version — including the candidate probes that miss.
    expect(rootHosted.requests.map((r) => r.version)).not.toContain(null);
    expect(rootHosted.requests.every((r) => r.version === "1.0")).toBe(true);
    expect(rootHosted.requests.some((r) => r.httpMethod === "GET")).toBe(true);
    expect(rootHosted.requests.some((r) => r.httpMethod === "POST")).toBe(true);
  });

  test("announces 0.3 for an agent pinned to 0.3", async (ctx) => {
    if (!etcdReachable || !app || !rootHosted) return ctx.skip();
    rootHosted.requests.length = 0;

    const reply = await call("/a2a/legacy", KEY_ALLOWED, messageSend("e2e-2"));

    expect(reply.status).toBe(200);
    expect(reply.json?.result?.sawVersion).toBe("0.3");
    expect(rootHosted.requests.at(-1)?.version).toBe("0.3");
  });

  test("reaches an agent whose card is published under a path prefix", async (ctx) => {
    if (!etcdReachable || !app || !pathHosted) return ctx.skip();
    pathHosted.requests.length = 0;

    const card = await fetchCard("tenant", KEY_ALLOWED);
    expect(card.status).toBe(200);
    expect(card.json?.name).toBe("Invoice Processor");

    // Resolved on the first candidate: the agent's own path prefix is tried
    // before the origin, so the platform's catch-all 405 at the origin is never
    // even reached — which is exactly what used to be mistaken for the agent's
    // answer when the origin was the ONLY candidate.
    const cardPaths = pathHosted.requests
      .filter((r) => r.httpMethod === "GET")
      .map((r) => r.path);
    expect(cardPaths).toEqual([
      "/v3/agents/serve/tenant-42/.well-known/agent-card.json",
    ]);

    const reply = await call("/a2a/tenant", KEY_ALLOWED, messageSend("e2e-3"));
    expect(reply.status).toBe(200);
    expect(reply.json?.result?.id).toBe("task-e2e-1");
  });

  test("the served card points every endpoint back at the gateway", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    const card = await fetchCard("invoices", KEY_ALLOWED);

    expect(card.status).toBe(200);
    expect(card.raw).not.toContain("internal.upstream.invalid");
    expect(card.json?.url).toMatch(/\/a2a\/invoices$/);
    for (const iface of card.json?.supportedInterfaces ?? []) {
      expect(iface.url).toMatch(/\/a2a\/invoices$/);
    }
    // Everything the gateway does not own survives the rewrite.
    expect(card.json?.skills?.[0]?.id).toBe("invoice");
    expect(card.json?.version).toBe("2.1.0");
  });

  test("the gateway holds the upstream credential and the caller never sees it", async (ctx) => {
    if (!etcdReachable || !app || !rootHosted) return ctx.skip();
    rootHosted.requests.length = 0;

    const reply = await call("/a2a/invoices", KEY_ALLOWED, messageSend("e2e-4"));

    expect(reply.status).toBe(200);
    const seen = rootHosted.requests.at(-1);
    expect(seen?.authorization).toBe("Bearer upstream-secret-tok");
    // The caller's own AISIX key must never be forwarded as the upstream token.
    expect(seen?.authorization).not.toContain(KEY_ALLOWED);
    expect(JSON.stringify(reply.json)).not.toContain("upstream-secret-tok");
  });

  test("gates on the per-agent ACL and on agent existence", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    const denied = await call("/a2a/invoices", KEY_NO_AGENTS, messageSend("x"));
    expect(denied.status).toBe(403);

    const unknown = await call("/a2a/ghost", KEY_ALLOWED, messageSend("x"));
    expect(unknown.status).toBe(404);

    const disabled = await call("/a2a/retired", KEY_ALLOWED, messageSend("x"));
    expect(disabled.status).toBe(404);

    const unauthenticated = await fetch(`${app.proxyUrl}/a2a/invoices`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(messageSend("x")),
    });
    expect(unauthenticated.status).toBe(401);
  });
});
