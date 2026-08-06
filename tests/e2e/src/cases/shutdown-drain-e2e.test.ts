import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  waitConfigPropagation,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: graceful shutdown drains in-flight requests, in whichever serving
// mode the suite leg selects.
//
// Pinned contract: a request the gateway has accepted before SIGTERM
// receives its complete response, and only then does the process exit.
// In thread-per-core mode every worker drains on its own clock, and a
// worker with no connections finishes instantly — that fast drain must
// not end the process while a sibling still holds an in-flight upstream
// call (the regression this file guards).

const KEY = "sk-shutdown-drain-e2e";
const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");
// Long enough that SIGTERM reliably lands mid-flight, short enough to
// finish inside the harness's 3s SIGTERM→SIGKILL escalation window.
const UPSTREAM_DELAY_MS = 2_000;

describe("graceful shutdown drains in-flight requests", () => {
  let app: SpawnedApp | undefined;
  let upstream: Server | undefined;
  let etcdReachable = false;
  let sawRequest = () => {};
  const upstreamGotRequest = new Promise<void>((resolve) => {
    sawRequest = resolve;
  });

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // A deliberately slow upstream: answers every POST with a valid
    // chat completion after UPSTREAM_DELAY_MS, so one request is
    // reliably in flight when SIGTERM lands.
    upstream = createServer((req, res) => {
      req.resume();
      req.on("end", () => {
        sawRequest();
        setTimeout(() => {
          res.writeHead(200, { "content-type": "application/json" });
          res.end(
            JSON.stringify({
              id: "chatcmpl-drain",
              object: "chat.completion",
              created: 1,
              model: "gpt-4o-mini",
              choices: [
                {
                  index: 0,
                  message: { role: "assistant", content: "drained" },
                  finish_reason: "stop",
                },
              ],
              usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
            }),
          );
        }, UPSTREAM_DELAY_MS);
      });
    });
    await new Promise<void>((resolve) => upstream!.listen(0, "127.0.0.1", resolve));
    const upstreamPort = (upstream!.address() as { port: number }).port;

    app = await spawnApp({});
    const seed = new SeedClient(new EtcdClient(), app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "drain-pk",
      secret: "sk-mock",
      api_base: `http://127.0.0.1:${upstreamPort}/v1`,
    });
    await seed.createModel({
      display_name: "drain-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: sha256(KEY),
      allowed_models: ["drain-model"],
    });

    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${KEY}` },
      });
      if (res.status !== 200) return false;
      const body = (await res.json()) as { data?: Array<{ id?: string }> };
      return body.data?.some((m) => m.id === "drain-model") ?? false;
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    if (upstream) await new Promise<void>((resolve) => upstream!.close(() => resolve()));
  });

  test("a request in flight at SIGTERM still completes", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    const inFlight = fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${KEY}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "drain-model",
        messages: [{ role: "user", content: "hold the line" }],
      }),
    });

    // Only signal once the request has actually reached the upstream —
    // from here the gateway holds it in flight for UPSTREAM_DELAY_MS.
    await upstreamGotRequest;
    app.signal("SIGTERM");

    // The drain contract: the response completes despite the shutdown.
    // A process that exits from under the request turns this await into
    // a socket error instead of a response.
    const res = await inFlight;
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      choices?: Array<{ message?: { content?: string } }>;
    };
    expect(body.choices?.[0]?.message?.content).toBe("drained");
  }, 20_000);
});
