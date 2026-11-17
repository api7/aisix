import { createHash } from "node:crypto";
import { WebSocket as WsClient, WebSocketServer, type WebSocket as WsSocket } from "ws";
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

// E2E: an `azure-openai` Provider Key's `api_version` also picks the route
// for realtime sessions and the files / batches surface:
//
// - unset: the dated routes with their own defaults —
//   `/openai/realtime?api-version=2024-10-01-preview&deployment=<d>` and
//   `/openai/files?api-version=2024-10-21`;
// - a dated value: the same routes with that version;
// - `v1`: `/openai/v1/realtime?model=<d>` and `/openai/v1/files`, no
//   api-version.
//
// Azure v1 API: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle
// Realtime over WebSockets: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/realtime-audio-websockets

const CALLER = "sk-azure-rt-jobs";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const SECRET = "az-resource-key";
const DATED = "2025-03-01-preview";
const DEPLOYMENT = "dep-realtime";

const VERSIONS = { unset: undefined, dated: DATED, v1: "v1" } as const;
type Version = keyof typeof VERSIONS;
const versions = Object.keys(VERSIONS) as Version[];

interface Handshake {
  url: string;
  headers: Record<string, string>;
}

async function startRealtimeUpstream(): Promise<{ port: number; handshakes: Handshake[]; close(): Promise<void> }> {
  const handshakes: Handshake[] = [];
  const wss = new WebSocketServer({ host: "127.0.0.1", port: 0 });
  wss.on("connection", (socket: WsSocket, req) => {
    handshakes.push({
      url: req.url ?? "",
      headers: Object.fromEntries(
        Object.entries(req.headers).map(([k, v]) => [k.toLowerCase(), Array.isArray(v) ? v.join(",") : String(v ?? "")]),
      ),
    });
    socket.on("message", () => socket.send(JSON.stringify({ type: "session.updated" })));
  });
  await new Promise<void>((resolve) => wss.on("listening", resolve));
  const addr = wss.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return {
    port: addr.port,
    handshakes,
    close: () => new Promise<void>((resolve, reject) => wss.close((e) => (e ? reject(e) : resolve()))),
  };
}

describe("azure-openai api_version on realtime and the files/batches surface", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let realtime: Awaited<ReturnType<typeof startRealtimeUpstream>> | undefined;
  let jobs: OpenAiUpstream | undefined;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    realtime = await startRealtimeUpstream();
    jobs = await startOpenAiUpstream({ nonStreamBody: { object: "list", data: [], has_more: false } });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    for (const [version, apiVersion] of Object.entries(VERSIONS) as [Version, string | undefined][]) {
      for (const [kind, base] of [
        ["rt", `http://127.0.0.1:${realtime.port}`],
        ["jobs", jobs.baseUrl],
      ] as const) {
        const pk = await seed.createProviderKey({
          display_name: `az-${kind}-${version}-pk`,
          provider: "azure",
          adapter: "azure-openai",
          api_key: SECRET,
          api_base: base,
          ...(apiVersion ? { api_version: apiVersion } : {}),
        });
        await seed.createModel({
          display_name: `az-${kind}-${version}`,
          provider: "azure",
          model_name: DEPLOYMENT,
          provider_key_id: pk.id,
        });
      }
    }
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, { headers: { authorization: `Bearer ${CALLER}` } });
      await res.text();
      return res.status === 200;
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await realtime?.close();
    await jobs?.close();
  });

  test.each(versions)("realtime session (%s)", async (version) => {
    if (!etcdReachable || !app || !realtime) return;
    const before = realtime.handshakes.length;
    const c = new WsClient(`${app.proxyUrl.replace("http://", "ws://")}/v1/realtime?model=az-rt-${version}`, {
      headers: { authorization: `Bearer ${CALLER}` },
    });
    const relayed = await new Promise<string>((resolve, reject) => {
      c.on("open", () => c.send(JSON.stringify({ type: "session.update" })));
      c.on("message", (d) => resolve(d.toString()));
      c.on("unexpected-response", (_q, res) => reject(new Error(`upgrade refused: ${res.statusCode}`)));
      c.on("error", (e) => reject(e));
    });
    c.terminate();
    expect(JSON.parse(relayed).type).toBe("session.updated");

    const hs = realtime.handshakes[before]!;
    const expected =
      version === "v1"
        ? `/openai/v1/realtime?model=${DEPLOYMENT}`
        : `/openai/realtime?api-version=${version === "dated" ? DATED : "2024-10-01-preview"}&deployment=${DEPLOYMENT}`;
    expect(hs.url).toBe(expected);
    expect(hs.headers["api-key"]).toBe(SECRET);
    expect(hs.headers["authorization"]).toBeUndefined();
  });

  for (const route of ["files", "batches"] as const) {
    test.each(versions)(`${route} list (%s)`, async (version) => {
      if (!etcdReachable || !app || !jobs) return;
      const before = jobs.receivedRequests.length;
      const res = await fetch(`${app.proxyUrl}/v1/${route}?model=az-jobs-${version}&limit=5`, {
        headers: { authorization: `Bearer ${CALLER}` },
      });
      const text = await res.text();
      expect(res.status, text).toBe(200);
      const sent = jobs.receivedRequests.slice(before);
      expect(sent).toHaveLength(1);
      const expected =
        version === "v1"
          ? `/openai/v1/${route}?limit=5`
          : `/openai/${route}?api-version=${version === "dated" ? DATED : "2024-10-21"}&limit=5`;
      expect(sent[0]!.path).toBe(expected);
      expect(sent[0]!.headers["api-key"]).toBe(SECRET);
      expect(sent[0]!.headers["authorization"]).toBeUndefined();
    });
  }
});
