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
//
// AISIX-Cloud#1087 follow-up: the same per-model allowlist must hold when
// the model is reached as a Model Group target. Pre-fix a group bypassed
// its members' `allowed_cidrs` entirely — only the named alias was checked
// — so adding a restricted model to a group silently published it to every
// caller. Post-fix an out-of-range target drops out of the candidate set
// (the group still serves from the remaining targets), and a group whose
// every target excludes the caller returns 403.

const CALLER_PLAINTEXT = "sk-model-ip-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const RESTRICTED_MODEL = "ip-restricted-model";
const OPEN_MODEL = "ip-open-model";
// Groups for the #1087 follow-up: one that can fall back to an
// unrestricted member, one whose every member excludes the caller.
const MIXED_GROUP = "ip-mixed-group";
const ALL_RESTRICTED_GROUP = "ip-all-restricted-group";
const RESTRICTED_MODEL_2 = "ip-restricted-model-2";
// The mixed group's open member gets its OWN upstream returning a marker
// string. Sharing the restricted member's upstream would make the test pass
// pre-fix too: both members answer 200, so only the response content can
// prove WHICH member the group actually dispatched to.
const GROUP_OPEN_MODEL = "ip-group-open-model";
const GROUP_OPEN_MARKER = "served-by-open-member";

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
  let groupOpenUpstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    groupOpenUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-group-open",
        object: "chat.completion",
        created: 0,
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: GROUP_OPEN_MARKER },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    });
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
    // Second restricted model on a range the test client is never in, so a
    // group of two restricted members has no reachable target.
    await seed.createModel({
      display_name: RESTRICTED_MODEL_2,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
      allowed_cidrs: ["192.168.0.0/16"],
    });
    const groupOpenPk = await seed.createProviderKey({
      display_name: "group-open-pk",
      secret: "sk-mock",
      api_base: `${groupOpenUpstream!.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: GROUP_OPEN_MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: groupOpenPk.id,
    });
    // Restricted member FIRST so an out-of-range caller only succeeds if the
    // member's allowlist actually removed it from the candidate set.
    await seed.createModel({
      display_name: MIXED_GROUP,
      routing: {
        strategy: "failover",
        targets: [{ model: RESTRICTED_MODEL }, { model: GROUP_OPEN_MODEL }],
      },
    });
    await seed.createModel({
      display_name: ALL_RESTRICTED_GROUP,
      routing: {
        strategy: "failover",
        targets: [{ model: RESTRICTED_MODEL }, { model: RESTRICTED_MODEL_2 }],
      },
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [
        RESTRICTED_MODEL,
        RESTRICTED_MODEL_2,
        OPEN_MODEL,
        GROUP_OPEN_MODEL,
        MIXED_GROUP,
        ALL_RESTRICTED_GROUP,
      ],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await groupOpenUpstream?.close();
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

  test(
    "model group: an out-of-range member drops out and the group serves from the rest",
    async (ctx) => {
      if (!etcdReachable || !app || !upstream) {
        ctx.skip();
        return;
      }
      // Probe from IN range so readiness doesn't depend on the very
      // exclusion this test is about.
      await waitConfigPropagation(async () => {
        const r = await chat(app!, MIXED_GROUP, "10.1.2.3");
        await r.text();
        return r.status === 200;
      });

      // Out-of-range caller: the restricted FIRST target is excluded, so the
      // group is served by the open member instead. Pre-fix the group ignored
      // the member allowlist and dispatched straight to the restricted one —
      // which also answers 200, so the marker in the body is what makes this
      // assertion discriminate between fixed and broken.
      const restrictedHitsBefore = upstream!.receivedRequests.length;
      const served = await chat(app, MIXED_GROUP, "114.114.114.114");
      expect(served.status).toBe(200);
      const body = (await served.json()) as {
        choices?: Array<{ message?: { content?: string } }>;
      };
      expect(body.choices?.[0]?.message?.content).toBe(GROUP_OPEN_MARKER);
      // The excluded member is never attempted — not attempted-then-failed-over.
      expect(upstream!.receivedRequests.length).toBe(restrictedHitsBefore);
    },
    60_000,
  );

  test(
    "model group: 403 when every member excludes the caller, upstream untouched",
    async (ctx) => {
      if (!etcdReachable || !app || !upstream) {
        ctx.skip();
        return;
      }
      await waitConfigPropagation(async () => {
        const r = await chat(app!, ALL_RESTRICTED_GROUP, "10.1.2.3");
        await r.text();
        return r.status === 200;
      });

      const hitsBefore = upstream.receivedRequests.length;
      const blocked = await chat(app, ALL_RESTRICTED_GROUP, "114.114.114.114");
      expect(blocked.status).toBe(403);
      const body = (await blocked.json()) as {
        error?: { code?: string; message?: string };
      };
      expect(body.error?.code).toBe("ip_restricted");
      // Same generic envelope as the direct-model rejection: no model name
      // and no CIDR reaches the caller, so a probe can't enumerate which
      // members exist or what ranges they allow (the #557 rule, which the
      // group path must not weaken by naming the target it excluded).
      const message = body.error?.message ?? "";
      expect(message).toBe(
        "Access denied: your client IP is not allowed to access this model",
      );
      for (const internal of [RESTRICTED_MODEL, RESTRICTED_MODEL_2, "192.168"]) {
        expect(message).not.toContain(internal);
      }
      expect(upstream.receivedRequests.length).toBe(hitsBefore);
    },
    60_000,
  );
});
