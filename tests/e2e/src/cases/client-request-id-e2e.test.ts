import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  pickFreePort,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: a caller may hand the gateway its own request id and the gateway
// uses THAT id everywhere (AISIX-Cloud#1288).
//
// The point of the feature is that one id spans systems: the caller's
// business logs already carry it, so the response header, the telemetry
// the operator searches /logs by, and the id the provider sees must all be
// the same value. Before this, the gateway minted a fresh UUID per request
// and the caller had to maintain a second mapping to find anything.
//
// The industry baseline is the same shape: LiteLLM reads its own
// `x-litellm-call-id` off the request and only generates one when absent;
// Portkey and Kong's correlation-id plugin behave the same way for their
// own headers. Like LiteLLM we honour only OUR namespaced header by
// default — `x-request-id` is opt-in precisely because everything in front
// of a gateway stamps it.
//
// Asserted at every place the id surfaces, since they are populated by
// different code: the response headers, the upstream request, the usage
// telemetry (via a mock OTLP receiver, where the per-attempt spans also
// prove retry/failover share one id), and the pre-dispatch rejection path
// that never reaches a handler at all.

const CALLER_PLAINTEXT = "sk-client-reqid-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

interface OtlpReceiver {
  url: string;
  spans: Array<Record<string, string>>;
  close(): Promise<void>;
}

async function startOtlpReceiver(): Promise<OtlpReceiver> {
  const spans: Array<Record<string, string>> = [];
  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      try {
        const body = JSON.parse(raw);
        for (const rs of body.resourceSpans ?? []) {
          for (const ss of rs.scopeSpans ?? []) {
            for (const span of ss.spans ?? []) {
              const attrs: Record<string, string> = {};
              for (const a of span.attributes ?? []) {
                const v = a.value ?? {};
                attrs[a.key] =
                  v.stringValue ?? String(v.intValue ?? v.boolValue ?? "");
              }
              spans.push(attrs);
            }
          }
        }
      } catch {
        // ignore malformed bodies — assertions fail on missing spans
      }
      res.statusCode = 200;
      res.end("{}");
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) =>
    server.listen(port, "127.0.0.1", resolve),
  );
  return {
    url: `http://127.0.0.1:${port}/v1/traces`,
    spans,
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

async function waitForSpans(
  recv: OtlpReceiver,
  requestId: string,
  count: number,
  timeoutMs = 15_000,
): Promise<Array<Record<string, string>>> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    // Attempt carriers only: since AISIX-Cloud#1279 the export also ships
    // the trace's structural SERVER / logical spans, which share
    // `aisix.request_id` but carry no per-attempt usage attributes.
    const hits = recv.spans.filter(
      (a) =>
        a["aisix.request_id"] === requestId &&
        a["aisix.attempt_index"] !== undefined,
    );
    if (hits.length >= count) return hits;
    if (Date.now() >= deadline) {
      throw new Error(
        `expected ${count} usage span(s) for request_id=${requestId}, saw ${hits.length}`,
      );
    }
    await new Promise((r) => setTimeout(r, 50));
  }
}

const OK_BODY = {
  id: "cmpl-client-reqid",
  object: "chat.completion",
  created: 1,
  model: "gpt-4o-mini",
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: "ok" },
      finish_reason: "stop",
    },
  ],
  usage: { prompt_tokens: 3, completion_tokens: 4, total_tokens: 7 },
};

describe("client-supplied request id (AISIX-Cloud#1288)", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let otlp: OtlpReceiver | undefined;
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);
    otlp = await startOtlpReceiver();
    await seed.createObservabilityExporter({
      name: "client-reqid-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
    await otlp?.close();
  });

  async function createModel(
    displayName: string,
    upstream: OpenAiUpstream,
  ): Promise<void> {
    if (!seed) throw new Error("seed client not initialized");
    const pk = await seed.createProviderKey({
      display_name: `${displayName}-pk`,
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: displayName,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
  }

  /** Seed a throwaway key AFTER the config under test, then poll until it authenticates. */
  async function awaitPropagation(tag: string): Promise<void> {
    const canary = `sk-canary-reqid-${tag}-${Date.now()}`;
    await seed!.createApiKey({
      key_hash: createHash("sha256").update(canary).digest("hex"),
      allowed_models: ["*"],
    });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${canary}` },
      });
      return res.status === 200;
    });
  }

  function chat(
    model: string,
    headers: Record<string, string>,
    body: Record<string, unknown> = {},
  ): Promise<Response> {
    return fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
        ...headers,
      },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: "hello" }],
        ...body,
      }),
    });
  }

  test(
    "the caller's id is what the response, the upstream and the telemetry all carry",
    async (ctx) => {
      if (!etcdReachable || !app || !seed || !otlp) {
        ctx.skip();
        return;
      }
      const upstream = await startOpenAiUpstream({ nonStreamBody: OK_BODY });
      upstreams.push(upstream);
      await createModel("reqid-basic", upstream);
      await awaitPropagation("basic");

      // Deliberately NOT a UUID: the whole point is that a caller keeps
      // its own id shape. This one used to be rejected by cp-api's
      // telemetry ingest and the request vanished from billing and /logs.
      const callerId = "req_abc123-orders-svc";
      const res = await chat("reqid-basic", {
        "x-aisix-request-id": callerId,
      });
      expect(res.status).toBe(200);
      await res.text();

      expect(res.headers.get("x-aisix-request-id")).toBe(callerId);
      // The chat success path sets a second, older alias header off the
      // same id; it must not drift into a different value.
      expect(res.headers.get("x-aisix-call-id")).toBe(callerId);

      const sent = upstream.receivedRequests.at(-1)!;
      expect(sent.headers["x-aisix-request-id"]).toBe(callerId);

      const [span] = await waitForSpans(otlp, callerId, 1);
      expect(span["aisix.request_id"]).toBe(callerId);
    },
    30_000,
  );

  test(
    "no client id keeps the old behaviour: a freshly minted UUID",
    async (ctx) => {
      if (!etcdReachable || !app || !seed) {
        ctx.skip();
        return;
      }
      const upstream = await startOpenAiUpstream({ nonStreamBody: OK_BODY });
      upstreams.push(upstream);
      await createModel("reqid-minted", upstream);
      await awaitPropagation("minted");

      const res = await chat("reqid-minted", {});
      expect(res.status).toBe(200);
      await res.text();

      const id = res.headers.get("x-aisix-request-id");
      expect(id).toMatch(UUID_RE);
      expect(upstream.receivedRequests.at(-1)!.headers["x-aisix-request-id"]).toBe(
        id,
      );
    },
    30_000,
  );

  test(
    "a streamed response carries the caller's id too",
    async (ctx) => {
      if (!etcdReachable || !app || !seed) {
        ctx.skip();
        return;
      }
      const upstream = await startOpenAiUpstream({
        streamEvents: [
          JSON.stringify({
            id: "chatcmpl-reqid",
            object: "chat.completion.chunk",
            created: 1,
            model: "gpt-4o-mini",
            choices: [
              { index: 0, delta: { content: "hi" }, finish_reason: null },
            ],
          }),
          JSON.stringify({
            id: "chatcmpl-reqid",
            object: "chat.completion.chunk",
            created: 1,
            model: "gpt-4o-mini",
            choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
            usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
          }),
          "[DONE]",
        ],
      });
      upstreams.push(upstream);
      await createModel("reqid-stream", upstream);
      await awaitPropagation("stream");

      const callerId = "req_stream_01HZY";
      const res = await chat(
        "reqid-stream",
        { "x-aisix-request-id": callerId },
        { stream: true },
      );
      expect(res.status).toBe(200);
      // Headers are sent before the body: the id must be on them, not
      // deferred to end-of-stream.
      expect(res.headers.get("x-aisix-request-id")).toBe(callerId);
      await res.text();

      expect(
        upstream.receivedRequests.at(-1)!.headers["x-aisix-request-id"],
      ).toBe(callerId);
    },
    30_000,
  );

  test(
    "every failover attempt shares the caller's id",
    async (ctx) => {
      if (!etcdReachable || !app || !seed || !otlp) {
        ctx.skip();
        return;
      }
      const primary = await startOpenAiUpstream({
        status: 502,
        errorBody: { error: { message: "primary down", type: "server_error" } },
      });
      const secondary = await startOpenAiUpstream({ nonStreamBody: OK_BODY });
      upstreams.push(primary, secondary);

      await createModel("reqid-primary", primary);
      await createModel("reqid-secondary", secondary);
      await seed.createModel({
        display_name: "reqid-virtual",
        routing: {
          strategy: "failover",
          targets: [{ model: "reqid-primary" }, { model: "reqid-secondary" }],
          retries: 0,
          max_fallbacks: 1,
        },
      });
      await awaitPropagation("failover");

      const callerId = "req_failover_7f3a";
      const res = await chat("reqid-virtual", {
        "x-aisix-request-id": callerId,
      });
      expect(res.status).toBe(200);
      await res.text();
      expect(res.headers.get("x-aisix-request-id")).toBe(callerId);

      // Both upstreams were reached, and both saw the caller's id — an
      // operator correlating provider-side logs finds the same value on
      // the failed attempt and the winning one.
      expect(primary.receivedRequests.length).toBe(1);
      expect(secondary.receivedRequests.length).toBe(1);
      expect(primary.receivedRequests[0]!.headers["x-aisix-request-id"]).toBe(
        callerId,
      );
      expect(secondary.receivedRequests[0]!.headers["x-aisix-request-id"]).toBe(
        callerId,
      );

      // #655 emits one usage event per attempt, all keyed on request_id:
      // querying /logs by the caller's id must return the whole chain.
      const attempts = await waitForSpans(otlp, callerId, 2);
      const kinds = attempts
        .map((a) => a["aisix.attempt_kind"])
        .sort();
      expect(kinds).toEqual(["fallback", "initial"]);
    },
    45_000,
  );

  test(
    "an unusable id degrades to a minted UUID rather than failing the request",
    async (ctx) => {
      if (!etcdReachable || !app || !seed) {
        ctx.skip();
        return;
      }
      const upstream = await startOpenAiUpstream({ nonStreamBody: OK_BODY });
      upstreams.push(upstream);
      await createModel("reqid-degrade", upstream);
      await awaitPropagation("degrade");

      // A space is legal in an HTTP header value but outside the id
      // charset; 300 bytes is past the 256-byte ceiling. Neither may cost
      // the caller their request — nor be persisted, which is why the
      // charset and the ceiling match cp-api's ingest rule exactly.
      for (const bad of ["req abc 123", "x".repeat(300)]) {
        const res = await chat("reqid-degrade", {
          "x-aisix-request-id": bad,
        });
        expect(res.status).toBe(200);
        await res.text();

        const id = res.headers.get("x-aisix-request-id");
        expect(id).not.toBe(bad);
        expect(id).toMatch(UUID_RE);
        expect(
          upstream.receivedRequests.at(-1)!.headers["x-aisix-request-id"],
        ).toBe(id);
      }
    },
    30_000,
  );

  test(
    "a request rejected before dispatch still comes back with the caller's id",
    async (ctx) => {
      if (!etcdReachable || !app) {
        ctx.skip();
        return;
      }
      // Bad credential: refused by the auth extractor, so no handler and
      // no upstream is involved — the id has to come from the middleware.
      const callerId = "req_rejected_9c2";
      const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          authorization: "Bearer sk-not-a-real-key",
          "content-type": "application/json",
          "x-aisix-request-id": callerId,
        },
        body: JSON.stringify({
          model: "reqid-basic",
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      expect(res.status).toBe(401);
      await res.text();
      expect(res.headers.get("x-aisix-request-id")).toBe(callerId);
    },
    30_000,
  );

  test(
    "x-request-id is ignored on the default config",
    async (ctx) => {
      if (!etcdReachable || !app || !seed) {
        ctx.skip();
        return;
      }
      const upstream = await startOpenAiUpstream({ nonStreamBody: OK_BODY });
      upstreams.push(upstream);
      await createModel("reqid-default", upstream);
      await awaitPropagation("default");

      const res = await chat("reqid-default", {
        "x-request-id": "stamped-by-the-ingress",
      });
      expect(res.status).toBe(200);
      await res.text();
      expect(res.headers.get("x-aisix-request-id")).not.toBe(
        "stamped-by-the-ingress",
      );
      expect(res.headers.get("x-aisix-request-id")).toMatch(UUID_RE);
    },
    30_000,
  );
});

// A second gateway, configured to also accept the de-facto standard
// header. Separate describe because `accept_headers` is boot config.
describe("client-supplied request id: x-request-id opted in", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    app = await spawnApp({
      requestId: { accept_headers: ["x-aisix-request-id", "x-request-id"] },
    });
    seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  test(
    "x-request-id is honoured, and the gateway's own header still outranks it",
    async (ctx) => {
      if (!etcdReachable || !app || !seed) {
        ctx.skip();
        return;
      }
      const upstream = await startOpenAiUpstream({ nonStreamBody: OK_BODY });
      upstreams.push(upstream);
      const pk = await seed.createProviderKey({
        display_name: "reqid-optin-pk",
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      await seed.createModel({
        display_name: "reqid-optin",
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
      const canary = `sk-canary-optin-${Date.now()}`;
      await seed.createApiKey({
        key_hash: createHash("sha256").update(canary).digest("hex"),
        allowed_models: ["*"],
      });
      await waitConfigPropagation(async () => {
        const res = await fetch(`${app!.proxyUrl}/v1/models`, {
          headers: { authorization: `Bearer ${canary}` },
        });
        return res.status === 200;
      });

      const send = (headers: Record<string, string>) =>
        fetch(`${app!.proxyUrl}/v1/chat/completions`, {
          method: "POST",
          headers: {
            authorization: `Bearer ${CALLER_PLAINTEXT}`,
            "content-type": "application/json",
            ...headers,
          },
          body: JSON.stringify({
            model: "reqid-optin",
            messages: [{ role: "user", content: "hello" }],
          }),
        });

      let res = await send({ "x-request-id": "from-the-ingress" });
      expect(res.status).toBe(200);
      await res.text();
      expect(res.headers.get("x-aisix-request-id")).toBe("from-the-ingress");
      expect(
        upstream.receivedRequests.at(-1)!.headers["x-aisix-request-id"],
      ).toBe("from-the-ingress");

      // Configured order is priority order.
      res = await send({
        "x-request-id": "from-the-ingress",
        "x-aisix-request-id": "from-the-caller",
      });
      expect(res.status).toBe(200);
      await res.text();
      expect(res.headers.get("x-aisix-request-id")).toBe("from-the-caller");
    },
    45_000,
  );
});
