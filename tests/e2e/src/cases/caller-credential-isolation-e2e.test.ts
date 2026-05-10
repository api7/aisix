import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: caller credentials never reach the upstream.
//
// The proxy authenticates the caller with one secret (the caller's
// ApiKey plaintext, presented as `Authorization: Bearer sk-aisix-…`
// or `x-api-key`) and then dispatches upstream with a completely
// separate secret (the matching ProviderKey's `secret`, presented
// as `Authorization: Bearer sk-mock-…` for OpenAI-shape providers).
// The two are isolated by design — the gateway is the only entity
// that knows the caller's plaintext, and the upstream provider
// must never see it.
//
// Three contracts pinned here:
//
//   1. The upstream's `Authorization` header carries the
//      ProviderKey's secret, not the caller's. A regression that
//      leaked the caller's bearer to upstream would leak every
//      tenant's auth token to a third party (OpenAI, Anthropic,
//      Bedrock, etc.) — a high-severity security incident.
//
//   2. The caller's plaintext does not appear ANYWHERE in the
//      upstream request — not in any header value, not in the
//      body. Catches a regression that copied the caller's bearer
//      into a non-Authorization header (e.g. Cookie, X-Caller-Auth)
//      or echoed it into a body field.
//
//   3. Hop-by-hop headers the caller may try to forge (Host,
//      X-Forwarded-For) do not influence the upstream request's
//      authentication or routing. The Host header on the upstream
//      request must be the upstream's own host, NOT whatever the
//      caller put in their Host header.
//
// Reference:
//   - `docs/api-proxy.md` §1 (auth) and §6 (provider auth shapes)
//   - OpenAI Python SDK pin a fresh Bearer per request
//     <https://github.com/openai/openai-python/blob/main/src/openai/_client.py>

const CALLER_PLAINTEXT = "sk-cred-isolation-caller-PLAINTEXT-MARKER";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const PROVIDER_SECRET = "sk-mock-provider-secret";

describe("caller credential isolation e2e: caller key never reaches upstream", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "cred-isolation-pk",
      secret: PROVIDER_SECRET,
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "cred-isolation-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["cred-isolation-model"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test(
    "caller plaintext absent from upstream headers + body; ProviderKey used instead",
    async (ctx) => {
      if (!etcdReachable || !app || !upstream) {
        ctx.skip();
        return;
      }

      const proxyUrl = `${app.proxyUrl}/v1/chat/completions`;
      const body = JSON.stringify({
        model: "cred-isolation-model",
        messages: [{ role: "user", content: "hello" }],
      });

      // Snapshot propagation through the same path the test uses.
      await waitConfigPropagation(async () => {
        try {
          const r = await fetch(proxyUrl, {
            method: "POST",
            headers: {
              authorization: `Bearer ${CALLER_PLAINTEXT}`,
              "content-type": "application/json",
            },
            body: JSON.stringify({
              model: "cred-isolation-model",
              messages: [{ role: "user", content: "ready-probe" }],
            }),
          });
          await r.text();
          return r.status === 200;
        } catch {
          return false;
        }
      });

      const baseline = upstream.receivedRequests.length;

      // Send the asserted call carrying the caller's bearer plus a
      // grab-bag of forged hop-by-hop / metadata headers a hostile
      // caller might try. None of these should influence the
      // upstream's auth or routing.
      const res = await fetch(proxyUrl, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
          "content-type": "application/json",
          // Forged hop-by-hop / metadata headers — gateway must
          // not blindly forward these to upstream.
          "x-forwarded-for": "203.0.113.99",
          "x-real-ip": "203.0.113.99",
          // A second auth-shape header that tries to slip the
          // caller's plaintext under a different name. Catches a
          // regression that whitelisted any "auth-shaped" header
          // for forwarding.
          "x-api-key": CALLER_PLAINTEXT,
        },
        body,
      });
      expect(res.status).toBe(200);
      await res.text();

      const sent = upstream.receivedRequests.slice(baseline);
      expect(sent).toHaveLength(1);
      const sentReq = sent[0]!;

      // (1) Upstream Authorization is the ProviderKey's secret.
      expect(sentReq.headers.authorization).toBe(
        `Bearer ${PROVIDER_SECRET}`,
      );

      // (2) Caller plaintext appears NOWHERE in the upstream request.
      // Walk every header value AND the body. A regression that
      // copied the caller's bearer into a non-Authorization header
      // (e.g. Cookie, X-Caller-Auth) would surface here — and a
      // regression that echoed it into the body would too. The
      // CALLER_PLAINTEXT constant has a "PLAINTEXT-MARKER" suffix
      // chosen so a substring search cannot accidentally collide
      // with any unrelated bytes the gateway emits.
      for (const [headerName, headerVal] of Object.entries(
        sentReq.headers,
      )) {
        expect(
          headerVal,
          `caller plaintext leaked in upstream header "${headerName}"`,
        ).not.toContain(CALLER_PLAINTEXT);
      }
      expect(
        sentReq.body,
        "caller plaintext leaked in upstream request body",
      ).not.toContain(CALLER_PLAINTEXT);

      // (3) Hop-by-hop / metadata headers the caller forged are
      // either absent on the upstream side, or overwritten by the
      // gateway's own values. Specifically the upstream Host
      // header must be the upstream's own host (127.0.0.1:<port>),
      // never whatever the caller may have set. node http
      // lowercases all header names; the mock keeps that shape.
      const upstreamHostHeader = sentReq.headers.host ?? "";
      expect(upstreamHostHeader.startsWith("127.0.0.1:")).toBe(true);
      // X-Real-IP / X-Forwarded-For: gateway should either drop
      // them or re-set to its own observed client. In any case,
      // they MUST NOT carry the value the caller forged. Pinning
      // "not equal to the forged value" is the contract.
      expect(sentReq.headers["x-forwarded-for"]).not.toBe(
        "203.0.113.99",
      );
      expect(sentReq.headers["x-real-ip"]).not.toBe("203.0.113.99");
    },
    60_000,
  );
});
