import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  pickFreePort,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// A latency the gateway measured is never reported as 0.
//
// Latency travels as whole milliseconds and 0 means "not measured" all the
// way to the console, which stores it as no latency at all. A response
// served from the exact-match cache routinely completes in under a
// millisecond, so before this it was reported as if nothing had been
// measured. A measured figure now reads as at least 1 ms.
//
// Observed through the per-env OTLP/HTTP fan-out, where a 0
// `downstream_latency_ms` is dropped from the span exactly as it is dropped
// from the control-plane row.

const CALLER_PLAINTEXT = "sk-cache-hit-latency-floor";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

/** Enough hits that a sub-millisecond one is all but certain. */
const HITS = 30;

interface OtlpReceiver {
  url: string;
  spanAttrs: Array<Record<string, string>>;
  /** Exports that did not parse — answered 400 and surfaced by `waitSpan`. */
  errors: string[];
  close(): Promise<void>;
}

async function startOtlpReceiver(): Promise<OtlpReceiver> {
  const spanAttrs: Array<Record<string, string>> = [];
  const errors: string[] = [];
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
                  v.stringValue ??
                  String(v.intValue ?? v.doubleValue ?? v.boolValue ?? "");
              }
              spanAttrs.push(attrs);
            }
          }
        }
      } catch (err) {
        errors.push(String(err));
        res.statusCode = 400;
        res.end("{}");
        return;
      }
      res.statusCode = 200;
      res.end("{}");
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    url: `http://127.0.0.1:${port}/v1/traces`,
    spanAttrs,
    errors,
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

async function waitSpan(
  recv: OtlpReceiver,
  requestId: string,
  timeoutMs = 10_000,
): Promise<Record<string, string>> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (recv.errors.length > 0) {
      throw new Error(`malformed OTLP export: ${recv.errors[0]}`);
    }
    const hit = recv.spanAttrs.find(
      (a) =>
        a["aisix.request_id"] === requestId &&
        a["aisix.attempt_index"] !== undefined,
    );
    if (hit) return hit;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(`no usage span for request_id=${requestId}`);
}

describe("cache-hit latency is never reported as unmeasured", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let otlp: OtlpReceiver | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    otlp = await startOtlpReceiver();
    await seed.createObservabilityExporter({
      name: "latency-floor-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });
    const pk = await seed.createProviderKey({
      display_name: "latency-floor-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "floor-chat",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createCachePolicy({
      name: "latency-floor-policy",
      enabled: true,
      applies_to: "all",
    });
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  }, 90_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await otlp?.close();
  });

  const post = async (path: string, body: unknown) => {
    const resp = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    });
    await resp.text();
    expect(resp.status).toBe(200);
    return {
      cache: resp.headers.get("x-aisix-cache") ?? "",
      requestId: resp.headers.get("x-aisix-request-id") ?? "",
    };
  };

  // The exact-match response cache serves `/v1/chat/completions` only.
  test("every chat-completions cache hit carries a latency", async (ctx) => {
    if (!etcdReachable || !app || !otlp) {
      ctx.skip();
      return;
    }
    const body = { model: "floor-chat", messages: [{ role: "user", content: "floor" }] };
    expect((await post("/v1/chat/completions", body)).cache).toBe("miss");
    const hits = [];
    for (let i = 0; i < HITS; i++) hits.push(await post("/v1/chat/completions", body));
    for (const hit of hits) {
      expect(hit.cache).toBe("hit");
      const span = await waitSpan(otlp, hit.requestId);
      expect(
        Number(span["aisix.downstream_latency_ms"] ?? 0),
        `${hit.requestId}: a measured wait is at least 1 ms, never "not measured"`,
      ).toBeGreaterThanOrEqual(1);
    }
  });
});
