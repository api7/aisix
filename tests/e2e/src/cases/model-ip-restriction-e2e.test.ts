import { createHash } from "node:crypto";
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

// E2E (api7/AISIX-Cloud#557): model-level client-IP CIDR allowlist.
//
// A model with `allowed_cidrs` only serves requests whose resolved client
// IP falls inside one of the ranges; everyone else gets 403 before the
// upstream is ever contacted. An unrestricted model is unaffected. The
// gateway resolves the real client IP from `x-forwarded-for` using the
// nginx-style trusted-proxy chain (#492 plumbing), so the loopback e2e
// client must be a trusted proxy for the forwarded IP to be honoured.
//
// Coverage:
//   AC-1 allow  — in-range XFF → 200, upstream hit.
//   AC-1 block  — out-of-range XFF → 403 `code: "ip_restricted"`, upstream
//                 untouched (rejected pre-dispatch).
//   AC-2 isolation — same external IP: restricted model 403, unrestricted 200.

const CALLER_PLAINTEXT = "sk-model-ip-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const RESTRICTED_MODEL = "ip-restricted-model";
const OPEN_MODEL = "ip-open-model";

async function chat(
  app: SpawnedApp,
  model: string,
  forwardedFor: string,
): Promise<Response> {
  return fetch(`${app.proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
      "x-forwarded-for": forwardedFor,
    },
    body: JSON.stringify({
      model,
      messages: [{ role: "user", content: "hello" }],
    }),
  });
}

describe("model IP restriction e2e (#557): allowed_cidrs gate before upstream", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    // 127.0.0.1 (the loopback e2e client) is the trusted proxy, so the
    // gateway honours `x-forwarded-for` and treats the forwarded value as
    // the real client IP.
    app = await spawnApp({
      realIp: { trusted_proxies: ["127.0.0.1/32"], recursive: true },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "model-ip-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    // Restricted model: only the 10.0.0.0/8 range may call it.
    await seed.createModel({
      display_name: RESTRICTED_MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
      allowed_cidrs: ["10.0.0.0/8"],
    });
    // Open model: no restriction, same upstream.
    await seed.createModel({
      display_name: OPEN_MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [RESTRICTED_MODEL, OPEN_MODEL],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test(
    "in-range allowed, out-of-range 403 (upstream untouched), unrestricted model unaffected",
    async (ctx) => {
      if (!etcdReachable || !app || !upstream) {
        ctx.skip();
        return;
      }

      // Readiness probe doubles as the AC-1 allow case: a client in
      // 10.0.0.0/8 reaches the restricted model and gets a 200.
      await waitConfigPropagation(async () => {
        try {
          const r = await chat(app!, RESTRICTED_MODEL, "10.1.2.3");
          await r.text();
          return r.status === 200;
        } catch {
          return false;
        }
      });

      // AC-1 allow: in-range request reaches upstream.
      const allowed = await chat(app, RESTRICTED_MODEL, "10.255.255.254");
      expect(allowed.status).toBe(200);
      await allowed.text();

      const upstreamHitsBeforeBlock = upstream.receivedRequests.length;

      // AC-1 block: out-of-range request → 403 with the stable
      // `ip_restricted` code, and the upstream is never contacted.
      const blocked = await chat(app, RESTRICTED_MODEL, "114.114.114.114");
      expect(blocked.status).toBe(403);
      const body = (await blocked.json()) as {
        error?: { type?: string; code?: string; message?: string };
      };
      expect(body.error?.code).toBe("ip_restricted");
      expect(typeof body.error?.message).toBe("string");
      // The block fires before dispatch — no new upstream request.
      expect(upstream.receivedRequests.length).toBe(upstreamHitsBeforeBlock);

      // AC-2 isolation: the SAME out-of-range IP reaches the unrestricted
      // model with a 200.
      const openOk = await chat(app, OPEN_MODEL, "114.114.114.114");
      expect(openOk.status).toBe(200);
      await openOk.text();
      expect(upstream.receivedRequests.length).toBe(upstreamHitsBeforeBlock + 1);
    },
    60_000,
  );
});
