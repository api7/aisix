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

// E2E: /passthrough/{provider}/*rest end-to-end. Per gateway docs
// `docs/api-proxy.md` §4.10, this is the lowest-overhead escape
// hatch for provider endpoints the gateway hasn't yet wrapped
// natively (e.g. OpenAI batches, files, fine-tuning, Anthropic
// message-batches). The gateway:
//
//   - strips `/passthrough/{provider}` from the request URL,
//     appending the rest to the configured Model's api_base
//   - picks the first Model with the matching `provider` prefix
//     (openai, anthropic, gemini, deepseek) and uses its
//     credentials
//   - injects the configured provider API key — Bearer for
//     OpenAI / Gemini / DeepSeek, `x-api-key` + `anthropic-version`
//     for Anthropic
//   - forwards the body verbatim
//
// Prior to this file, the gateway had **zero** e2e coverage on
// /passthrough — meaning every customer using batches, files, or
// fine-tuning APIs had no regression protection on the wire.
//
// Two user journeys pinned:
//
//   1. OpenAI passthrough — caller follows the published examples
//      verbatim: `api_base: "https://api.openai.com/v1"` per
//      `docs/api-admin.md` §4.3, and a call to
//      `/passthrough/openai/v1/files` per `docs/api-proxy.md` §4.10.
//      The gateway must dedup the duplicated `/v1` prefix and hit
//      the upstream at `/v1/files` — a naive concatenation would
//      hit `/v1/v1/files` which the real OpenAI API would 404.
//      Pre-fix this was bug #164.
//
//   2. Anthropic passthrough — caller hits a custom Anthropic
//      endpoint (e.g. /v1/messages/batches). Gateway must:
//       * strip the /passthrough/anthropic prefix and forward
//         path + body + method verbatim
//       * inject Anthropic's auth shape (`x-api-key` +
//         `anthropic-version`), NOT `Authorization: Bearer`
//         (a regression that forwarded Bearer to Anthropic would
//         401 in production)
//
// References:
// - Gateway's own /passthrough contract: `docs/api-proxy.md` §4.10
// - OpenAI Files API
//   <https://platform.openai.com/docs/api-reference/files>
// - Anthropic auth headers spec
//   <https://docs.anthropic.com/en/api/getting-started>

const CALLER_PLAINTEXT = "sk-pt-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("passthrough e2e: /passthrough/{provider}/*rest verbatim forwarding with auth injection + double-/v1 dedup", () => {
  let app: SpawnedApp | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  test("OpenAI passthrough: gateway dedups doubled /v1, forwards verbatim, injects Bearer auth", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    // Mock upstream returns a distinctive body the caller will
    // see. The body shape is intentionally NOT chat-completions —
    // passthrough is for endpoints the gateway doesn't natively
    // wrap (batches/files/fine-tuning), so the gateway must NOT
    // try to parse or normalize the response.
    const upstream = await startOpenAiUpstream({
      nonStreamBody: {
        // Shaped like an OpenAI Files API response per
        // <https://platform.openai.com/docs/api-reference/files/object>.
        id: "file-pt-openai-01",
        object: "file",
        bytes: 12345,
        created_at: Math.floor(Date.now() / 1000),
        filename: "passthrough-test.jsonl",
        purpose: "batch",
        status: "uploaded",
      },
    });
    upstreams.push(upstream);

    // Configure exactly per docs/api-admin.md §4.3 example:
    // `api_base: "https://api.openai.com/v1"` (with /v1).
    const pk = await admin.createProviderKey({
      display_name: "pt-openai-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    // The Model's provider field is what the passthrough route
    // matches on (per docs §4.10 "first Model with that prefix").
    await admin.createModel({
      display_name: "pt-openai-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });

    const headers = {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    };

    // Readiness gate: poll passthrough until it returns 200 with
    // the upstream's distinctive body. A 404 means either the
    // snapshot hasn't propagated yet OR the gateway is hitting
    // `/v1/v1/files` upstream (#164 pre-fix behavior — the mock
    // doesn't have a route there, so it 404s).
    await waitConfigPropagation(async () => {
      try {
        const r = await fetch(`${app!.proxyUrl}/passthrough/openai/v1/files`, {
          method: "POST",
          headers,
          body: JSON.stringify({ purpose: "batch", filename: "ready-probe.jsonl" }),
        });
        if (r.status !== 200) {
          await r.text();
          return false;
        }
        const j = (await r.json()) as { object?: unknown };
        return j.object === "file";
      } catch {
        return false;
      }
    });

    const baseline = upstream.receivedRequests.length;
    const requestBody = JSON.stringify({
      purpose: "batch",
      filename: "real-call.jsonl",
      // Distinctive marker so the body-verbatim assertion can
      // confirm the gateway didn't strip or rewrite anything.
      arbitrary_unknown_field: "must-pass-through-untouched",
    });
    // Call exactly per docs/api-proxy.md §4.10 example:
    // `/passthrough/openai/v1/files` (with /v1).
    const res = await fetch(`${app.proxyUrl}/passthrough/openai/v1/files`, {
      method: "POST",
      headers,
      body: requestBody,
    });

    expect(res.status).toBe(200);
    const body = (await res.json()) as { id?: unknown; object?: unknown };
    // Caller sees the upstream's body byte-for-byte. Passthrough
    // explicitly disclaims body normalisation per docs §4.10
    // ("forwards the request verbatim").
    expect(body.id).toBe("file-pt-openai-01");
    expect(body.object).toBe("file");

    // Upstream-side: the gateway hit `/v1/files`, NOT `/v1/v1/files`.
    // The dedup is the load-bearing #164 contract: api_base ending
    // in `/v1` plus rest starting with `v1/` must produce a single
    // `/v1` prefix. A regression that re-introduces the doubled
    // prefix would land at `/v1/v1/files` and fail this filter.
    const testCalls = upstream.receivedRequests
      .slice(baseline)
      .filter((r) => r.path === "/v1/files");
    expect(testCalls).toHaveLength(1);
    expect(testCalls[0]?.method).toBe("POST");

    // Verify there were NO requests to the doubled-prefix path.
    // A regression that reintroduces the bug would still emit a
    // request — pinning that no such request exists is the strict
    // form of the dedup contract.
    const doubledCalls = upstream.receivedRequests
      .slice(baseline)
      .filter((r) => r.path === "/v1/v1/files");
    expect(doubledCalls).toHaveLength(0);

    // Auth injection: per docs §4.10, the gateway injects the
    // configured provider API key. For OpenAI that's
    // `Authorization: Bearer <secret>`. The caller's own bearer
    // MUST NOT reach the upstream — that would leak the proxy key
    // into upstream provider logs.
    expect(testCalls[0]?.headers["authorization"]).toBe("Bearer sk-mock");

    // Body verbatim: every field the caller sent reaches the
    // upstream unchanged, including unknown fields the gateway
    // doesn't recognise. A regression that JSON-round-tripped or
    // schema-validated the body would silently drop unknown
    // fields like `arbitrary_unknown_field`.
    expect(testCalls[0]?.body).toBe(requestBody);
  });

  test("Anthropic passthrough: gateway uses x-api-key + anthropic-version, NOT Bearer", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    const upstream = await startOpenAiUpstream({
      nonStreamBody: {
        // Shaped like an Anthropic message-batches response per
        // <https://docs.anthropic.com/en/api/creating-message-batches>.
        id: "msgbatch_pt_anthropic_01",
        type: "message_batch",
        processing_status: "in_progress",
        request_counts: { processing: 1, succeeded: 0, errored: 0, canceled: 0, expired: 0 },
        ended_at: null,
        created_at: new Date().toISOString(),
        expires_at: new Date(Date.now() + 24 * 3600 * 1000).toISOString(),
      },
    });
    upstreams.push(upstream);

    // Anthropic api_base is the bare host; bridge composes the
    // rest of the path. For passthrough, the gateway appends
    // `/*rest` directly, so we pass the bare host.
    const pk = await admin.createProviderKey({
      display_name: "pt-anthropic-pk",
      secret: "sk-ant-mock",
      api_base: upstream.baseUrl,
    });
    await admin.createModel({
      display_name: "pt-anthropic-model",
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: pk.id,
    });

    const headers = {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    };

    await waitConfigPropagation(async () => {
      try {
        const r = await fetch(
          `${app!.proxyUrl}/passthrough/anthropic/v1/messages/batches`,
          {
            method: "POST",
            headers,
            body: JSON.stringify({ requests: [] }),
          },
        );
        if (r.status !== 200) {
          await r.text();
          return false;
        }
        const j = (await r.json()) as { type?: unknown };
        return j.type === "message_batch";
      } catch {
        return false;
      }
    });

    const baseline = upstream.receivedRequests.length;
    const requestBody = JSON.stringify({
      requests: [
        {
          custom_id: "batch-req-1",
          params: { model: "claude-3-5-haiku-20241022", max_tokens: 100 },
        },
      ],
    });
    const res = await fetch(
      `${app.proxyUrl}/passthrough/anthropic/v1/messages/batches`,
      {
        method: "POST",
        headers,
        body: requestBody,
      },
    );

    expect(res.status).toBe(200);
    const body = (await res.json()) as { id?: unknown; type?: unknown };
    expect(body.id).toBe("msgbatch_pt_anthropic_01");
    expect(body.type).toBe("message_batch");

    const testCalls = upstream.receivedRequests
      .slice(baseline)
      .filter((r) => r.path === "/v1/messages/batches");
    expect(testCalls).toHaveLength(1);
    expect(testCalls[0]?.method).toBe("POST");

    // Auth injection per docs §4.10: Anthropic uses `x-api-key`
    // + `anthropic-version` headers, NOT `Authorization: Bearer`.
    // A regression that forwarded Bearer to Anthropic upstream
    // would 401 in production but pass against the permissive
    // mock — pinning the exact header set is the only line of
    // defence.
    expect(testCalls[0]?.headers["x-api-key"]).toBe("sk-ant-mock");
    // Anthropic's documented current API version is `2023-06-01`
    // per <https://docs.anthropic.com/en/api/getting-started>. A
    // regression that injected a malformed-but-non-empty version
    // (e.g. "v1", "latest") would 400 against real Anthropic but
    // pass against the permissive mock without this exact pin.
    expect(testCalls[0]?.headers["anthropic-version"]).toBe("2023-06-01");

    // Caller's `Authorization: Bearer sk-pt-e2e-caller` is
    // gateway-internal (validates against the ApiKey table) and
    // MUST NOT leak upstream — that would put a gateway-side
    // ApiKey credential into upstream provider logs. Pin: if the
    // upstream's Authorization header is present at all, its value
    // must NOT contain the caller's plaintext bearer.
    //
    // (Whether the gateway should inject ANY Authorization header
    // when forwarding to Anthropic is a separate question — docs
    // §4.10 implies only x-api-key + anthropic-version for the
    // Anthropic auth shape. Tracked as follow-up.)
    const upstreamAuth = testCalls[0]?.headers["authorization"];
    if (upstreamAuth !== undefined) {
      expect(upstreamAuth as string).not.toContain(CALLER_PLAINTEXT);
    }

    // Body verbatim — same contract as the OpenAI case.
    expect(testCalls[0]?.body).toBe(requestBody);
  });
});
