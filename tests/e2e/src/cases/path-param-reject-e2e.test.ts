import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: a `:param` segment that fails path extraction (valid percent-encoding,
// invalid UTF-8 after decoding — `%ff`) answers the caller envelope instead
// of axum's bare 400, across the `:param` route family (#880). The access
// log + metrics side rides the same `reject_before_dispatch` call every
// other pre-dispatch rejection uses; the envelope is the e2e-observable
// contract. Authentication still precedes the path parse.

const KEY = "sk-path-reject-e2e";
const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

describe("path-param rejection e2e: :param routes answer the envelope", () => {
  let app: SpawnedApp | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createApiKey({
      key_hash: sha256(KEY),
      allowed_models: ["*"],
    });

    // Propagation probe: the key authenticates (any status but 401).
    for (let i = 0; i < 100; i += 1) {
      const res = await fetch(`${app.proxyUrl}/v1/files/probe`, {
        headers: { authorization: `Bearer ${KEY}` },
      });
      if (res.status !== 401) return;
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    throw new Error("seeded key never became active");
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
  });

  test("malformed :param answers the OpenAI envelope on every family member", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    for (const [method, path] of [
      ["POST", "/a2a/%ff"],
      ["GET", "/mcp/%ff"],
      ["GET", "/v1/files/%ff"],
      ["GET", "/v1/batches/%ff"],
      ["GET", "/v1/fine_tuning/jobs/%ff"],
      ["GET", "/v1/videos/%ff"],
      // `/passthrough/*` left the :param family with the implicit tunnel's
      // removal — unclaimed paths there answer the 410 tombstone instead.
    ] as const) {
      const res = await fetch(`${app.proxyUrl}${path}`, {
        method,
        headers: { authorization: `Bearer ${KEY}` },
      });
      expect(res.status, `${method} ${path}`).toBe(400);
      const body = (await res.json()) as {
        error?: { type?: string; message?: string };
      };
      expect(body.error?.type, `${method} ${path}`).toBe(
        "invalid_request_error",
      );
    }
  });

  test("multipart content-type mismatch answers the envelope too", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // Same silent class one extractor over: JSON sent to a multipart
    // endpoint used to get axum's bare 400.
    for (const path of [
      "/v1/audio/transcriptions",
      "/v1/audio/translations",
      "/v1/files",
    ]) {
      const res = await fetch(`${app.proxyUrl}${path}`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${KEY}`,
          "content-type": "application/json",
        },
        body: "{}",
      });
      expect(res.status, path).toBe(400);
      const body = (await res.json()) as { error?: { type?: string } };
      expect(body.error?.type, path).toBe("invalid_request_error");
    }
  });

  test("an unauthenticated malformed :param is still 401 first", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    const res = await fetch(`${app.proxyUrl}/v1/files/%ff`);
    expect(res.status).toBe(401);
  });
});
